# MMA : le MÊME kernel tourne sur le chemin hardware (WMMA) et sur le fallback
# portable, selon la config. Helpers (to_device / from_device / warpsz) : harness.jl.

import KernelIntrinsics.MMA as MMA

# bf16 SANS dépendre de BFloat16s : `Core.BFloat16` est un type de Base, mais ses
# conversions vivent dans BFloat16s. Or un bf16 est exactement les 16 bits de poids
# fort d'un Float32 — donc on construit/relit par bits, en pur Base. Le sens
# bf16→Float32 est exact (élargissement), ce qui donne une référence CPU fiable.
_as(::Type{T}, x::Float32) where {T} = T(x)
_as(::Type{Core.BFloat16}, x::Float32) = reinterpret(Core.BFloat16, (reinterpret(UInt32, x) >> 16) % UInt16)
_f32(x) = Float32(x)
_f32(b::Core.BFloat16) = reinterpret(Float32, UInt32(reinterpret(UInt16, b)) << 16)

# Un warp calcule une tuile 16×16×16 : D = A·B, accumulateur initialisé à `fillv`.
# Workgroupsize = warpsz ⇒ un workgroup = un warp (lockstep requis par WMMA).
@kernel function mma_tile!(cfg, C, A, B, fillv)
    a = MMA.load_a(cfg, A, 1, MMA.ColMajor)
    b = MMA.load_b(cfg, B, 1, MMA.ColMajor)
    c = MMA.fill_c(cfg, fillv)
    d = MMA.mma(cfg, a, b, c)
    MMA.store_d!(cfg, C, 1, d, MMA.ColMajor)
end

run_tile(cfg, C, A, B, fillv) =
    (mma_tile!(backend, Int(warpsz))(cfg, C, A, B, fillv; ndrange = Int(warpsz)); synchronize(backend))

# GEMM tuilé : C[16×16] = A[16×K]·B[K×16], K = 16·Kb, un warp, accumulation
# sur la boucle-K. C'est le motif que KF réutilisera. Tuile MMA fixée à K=16.
@kernel function mma_kloop!(cfg, C, A, B, fillv, ::Val{Kb}) where {Kb}
    c = MMA.fill_c(cfg, fillv)
    for kb in 0:(Kb - 1)
        a = MMA.load_a(cfg, A, 1 + kb * 16 * size(A, 1), MMA.ColMajor)
        b = MMA.load_b(cfg, B, 1 + kb * 16, MMA.ColMajor)
        c = MMA.mma(cfg, a, b, c)
    end
    MMA.store_d!(cfg, C, 1, c, MMA.ColMajor)
end

run_kloop(cfg, C, A, B, fillv, Kb) =
    (mma_kloop!(backend, Int(warpsz))(cfg, C, A, B, fillv, Val(Kb); ndrange = Int(warpsz)); synchronize(backend))

@testset "MMA" begin
    # Fallback : warpsize 32 (CUDA/RDNA/Metal) ou 64 (CDNA/MI300, via l'override
    # _mma_wave dans l'ext AMDGPU). Le lane-grid s'adapte à la warpsize.
    if warpsz in (32, 64)
        M = N = K = 16

        # ── Fallback portable (régime correction), Float32 : pas de path HW ──
        @testset "fallback GEMM (MulAdd, Float32)" begin
            A = to_device(rand(Float32, M, K)); B = to_device(rand(Float32, K, N))
            C = to_device(zeros(Float32, M, N))
            cfg = MMA.MMAConfig{M,N,K,Float32,Float32,MMA.MulAdd}()
            @test MMA.mma_supported(cfg) == false          # pas de tensor core pour fp32
            run_tile(cfg, C, A, B, 0.0f0)
            @test from_device(C) ≈ from_device(A) * from_device(B) rtol = 1.0f-4
        end

        # ── Type non-HW : FP64 (régime « perf » du fallback, ici correction) ──
        @testset "fallback GEMM (MulAdd, Float64)" begin
            A = to_device(rand(Float64, M, K)); B = to_device(rand(Float64, K, N))
            C = to_device(zeros(Float64, M, N))
            cfg = MMA.MMAConfig{M,N,K,Float64,Float64,MMA.MulAdd}()
            @test MMA.mma_supported(cfg) == false
            run_tile(cfg, C, A, B, 0.0)
            @test from_device(C) ≈ from_device(A) * from_device(B) rtol = 1.0e-12
        end

        # ── Entiers : Int8→Int32 (fallback partout, MFMA sur MI300). Arithmétique
        #    exacte ⇒ on compare avec == , pas ≈. Régression : le produit doit se
        #    faire en Int32, sinon Int8*Int8 déborde.
        # K=32 : c'est la forme i8 de gfx942 (le K=16 y est absent). Sur CUDA le
        # fallback encaisse n'importe quel K, donc le même test couvre les deux.
        @testset "GEMM Int8→Int32 16×16×32 (MFMA/fallback)" begin
            Ki = 32
            A = to_device(rand(Int8(-100):Int8(100), M, Ki))
            B = to_device(rand(Int8(-100):Int8(100), Ki, N))
            C = to_device(zeros(Int32, M, N))
            cfg = MMA.MMAConfig{M,N,Ki,Int8,Int32,MMA.MulAdd}()
            run_tile(cfg, C, A, B, Int32(0))
            @test from_device(C) == Int32.(from_device(A)) * Int32.(from_device(B))
        end

        # ── Structure complexe : ComplexF32 (fallback-only, via muladd) ──
        @testset "fallback GEMM (MulAdd, ComplexF32)" begin
            A = to_device(rand(ComplexF32, M, K)); B = to_device(rand(ComplexF32, K, N))
            C = to_device(zeros(ComplexF32, M, N))
            cfg = MMA.MMAConfig{M,N,K,ComplexF32,ComplexF32,MMA.MulAdd}()
            run_tile(cfg, C, A, B, 0.0f0 + 0.0f0im)
            @test from_device(C) ≈ from_device(A) * from_device(B) rtol = 1.0f-4
        end

        # ── FP64 16×16×4 : MFMA tensor-core sur MI300, fallback ailleurs. Même kernel. ──
        @testset "GEMM fp64 16×16×4 (MFMA/fallback)" begin
            A = to_device(rand(Float64, 16, 4)); B = to_device(rand(Float64, 4, 16))
            C = to_device(zeros(Float64, 16, 16))
            cfg = MMA.MMAConfig{16,16,4,Float64,Float64,MMA.MulAdd}()
            run_tile(cfg, C, A, B, 0.0)
            @test from_device(C) ≈ from_device(A) * from_device(B) rtol = 1.0e-12
        end

        # ── Opérateur custom : semi-anneau tropical (fallback-only, HW incapable) ──
        @testset "fallback tropical (max-plus)" begin
            A = to_device(rand(Float32, M, K)); B = to_device(rand(Float32, K, N))
            C = to_device(fill(-Inf32, M, N))
            cfg = MMA.MMAConfig{M,N,K,Float32,Float32,MMA.Tropical}()
            run_tile(cfg, C, A, B, -Inf32)
            Ah = from_device(A); Bh = from_device(B)
            ref = [maximum(Ah[i, k] + Bh[k, j] for k in 1:K) for i in 1:M, j in 1:N]
            @test from_device(C) ≈ ref rtol = 1.0f-5
        end

        # ── Boucle-K (composition fill_c → mma… → store_d!), fallback fp32 ──
        @testset "fallback GEMM tuilé K=64 (fp32)" begin
            Kb = 4; Kfull = 16 * Kb
            A = to_device(rand(Float32, M, Kfull)); B = to_device(rand(Float32, Kfull, N))
            C = to_device(zeros(Float32, M, N))
            cfg = MMA.MMAConfig{M,N,16,Float32,Float32,MMA.MulAdd}()
            run_kloop(cfg, C, A, B, 0.0f0, Kb)
            @test from_device(C) ≈ from_device(A) * from_device(B) rtol = 1.0f-4
        end

        # ── Chemin HARDWARE fp16→fp32, MÊME kernel (WMMA sur CUDA, MFMA sur MI300) ──
        # On BALAYE toutes les formes candidates et on teste celles que le backend
        # ANNONCE supportées : `mma_supported` ne doit jamais promettre sans preuve.
        for (CT, AccT) in ((Float16, Float32), (Core.BFloat16, Float32)),
            (Mh, Nh, Kh) in ((16, 16, 16), (8, 32, 16), (32, 8, 16), (32, 32, 8))

            cfgh = MMA.MMAConfig{Mh,Nh,Kh,CT,AccT,MMA.MulAdd}()
            MMA.mma_supported(cfgh) || continue
            @testset "HW GEMM $(Mh)×$(Nh)×$(Kh) ($CT→$AccT)" begin
                A = to_device(_as.(CT, rand(Float32, Mh, Kh)))
                B = to_device(_as.(CT, rand(Float32, Kh, Nh)))
                C = to_device(zeros(AccT, Mh, Nh))
                run_tile(cfgh, C, A, B, zero(AccT))
                ref = _f32.(from_device(A)) * _f32.(from_device(B))
                # bf16 n'a que 8 bits de mantisse (vs 11 pour fp16) → tolérance plus large.
                @test from_device(C) ≈ ref rtol = (CT === Float16 ? 1.0f-2 : 5.0f-2)
            end
        end

        if MMA.mma_supported(MMA.MMAConfig{16,16,16,Float16,Float32,MMA.MulAdd}())
            @testset "HW GEMM tuilé K=64 (fp16→fp32)" begin
                Kb = 4; Kfull = 16 * Kb
                A = to_device(rand(Float16, M, Kfull)); B = to_device(rand(Float16, Kfull, N))
                C = to_device(zeros(Float32, M, N))
                cfg = MMA.MMAConfig{M,N,16,Float16,Float32,MMA.MulAdd}()
                run_kloop(cfg, C, A, B, 0.0f0, Kb)
                ref = Float32.(from_device(A)) * Float32.(from_device(B))
                @test from_device(C) ≈ ref rtol = 1.0f-2
            end
        end
    end
end
