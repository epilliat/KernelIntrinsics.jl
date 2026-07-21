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

# Même GEMM, mais via le point d'entrée (row, col) : l'origine de tuile s'écrit
# directement, sans le `1 + kb*16*size(A,1)` — et sans les divisions entières que
# le décodage de l'index linéaire impose à l'exécution. `acc_identity(cfg)` donne
# le neutre de l'opérateur porté par la config, et `mma_shape` évite de re-passer
# K en paramètre : le kernel est générique en `cfg`.
@kernel function mma_kloop_rc!(cfg, C, A, B, ::Val{Kb}) where {Kb}
    Ktile = MMA.mma_shape(cfg).K
    c = MMA.fill_c(cfg, MMA.acc_identity(cfg))
    for kb in 0:(Kb - 1)
        a = MMA.load_a(cfg, A, 1, 1 + kb * Ktile, MMA.ColMajor)
        b = MMA.load_b(cfg, B, 1 + kb * Ktile, 1, MMA.ColMajor)
        c = MMA.mma(cfg, a, b, c)
    end
    MMA.store_d!(cfg, C, 1, 1, c, MMA.ColMajor)
end

run_kloop_rc(cfg, C, A, B, Kb) =
    (mma_kloop_rc!(backend, Int(warpsz))(cfg, C, A, B, Val(Kb); ndrange = Int(warpsz)); synchronize(backend))

# ── RÉGRESSION : accumulateur porté par une VRAIE boucle-K ────────────────────
#
# Ce kernel existe pour une seule raison : les tests ci-dessus NE PEUVENT PAS voir
# le bug d'accumulateur `undef`. `mma_kloop!`/`mma_kloop_rc!` bornent la boucle-K
# par `Val{Kb}` avec Kb=4 ⇒ LLVM la DÉROULE COMPLÈTEMENT, donc il n'existe aucun
# phi porté par la boucle, donc rien qui puisse valoir `undef`. Historiquement, les
# six points d'entrée MMA étaient installés par des OVERLAYS ; l'inférence ne
# voyait pas à travers, le fragment revenait imprécisément typé, SROA échouait, et
# le préheader émettait `phi float [ undef, … ]` — soit 0, soit du bruit, selon la
# version de LLVM (cf. src/mma.jl, section « Jeton matériel »).
#
# Le motif doit donc réunir TOUS les axes qui manquaient :
#   • borne de boucle DYNAMIQUE (`nkt` calculé à l'exécution) ⇒ phi réel ;
#   • ≥ 2 warps par workgroup ;
#   • ≥ 2 `mma` par panneau chargé (NKS) ;
#   • opérandes mis en scène dans `@localmem` À L'INTÉRIEUR de la boucle, avec des
#     `@synchronize` que l'accumulateur doit traverser ;
#   • à faire tourner sous `--check-bounds=yes` (ce que fait déjà runtests).
# C'est exactement la forme que KernelForge utilise dans son GEMM.
@kernel unsafe_indices = true function mma_gemm_tiled!(cfg, C, A, B, M, N, K,
                                                       ::Val{WS}, ::Val{NW}, ::Val{NKS}) where {WS,NW,NKS}
    @uniform begin
        BM = 16 * NW; BN = 16; BK = 16 * NKS
        wg = NW * WS; nbx = cld(M, BM); nkt = cld(K, BK)   # nkt : borne DYNAMIQUE
        CT = MMA.compute_type(cfg); AccT = MMA.acc_type(cfg)
    end
    lid = Int(@index(Local)); gid = Int(@index(Group))
    brow = (gid - 1) % nbx; bcol = (gid - 1) ÷ nbx
    warp = (lid - 1) ÷ WS; m0 = warp * 16

    sA = @localmem CT (BM, BK)
    sB = @localmem CT (BK, BN)
    sD = @localmem AccT (BM, BN)

    c = MMA.fill_c(cfg, MMA.acc_identity(cfg))
    kt = 0
    while kt < nkt
        p = lid
        while p <= BM * BK
            e = p - 1; row = e % BM; kk = e ÷ BM
            m = brow * BM + row + 1; k = kt * BK + kk + 1
            @inbounds sA[row + 1, kk + 1] = (m <= M && k <= K) ? CT(A[m, k]) : zero(CT)
            p += wg
        end
        p = lid
        while p <= BK * BN
            e = p - 1; kk = e % BK; col = e ÷ BK
            k = kt * BK + kk + 1; n = bcol * BN + col + 1
            @inbounds sB[kk + 1, col + 1] = (k <= K && n <= N) ? CT(B[k, n]) : zero(CT)
            p += wg
        end
        @synchronize
        for ks in 0:(NKS - 1)
            a = MMA.load_a(cfg, sA, m0 + 1, ks * 16 + 1, MMA.ColMajor)
            b = MMA.load_b(cfg, sB, ks * 16 + 1, 1, MMA.ColMajor)
            c = MMA.mma(cfg, a, b, c)
        end
        @synchronize
        kt += 1
    end
    MMA.store_d!(cfg, sD, m0 + 1, 1, c, MMA.ColMajor)
    @synchronize
    p = lid
    while p <= BM * BN
        e = p - 1; r = e % BM; col = e ÷ BM
        m = brow * BM + r + 1; n = bcol * BN + col + 1
        if m <= M && n <= N
            @inbounds C[m, n] = sD[r + 1, col + 1]
        end
        p += wg
    end
end

function run_gemm_tiled(cfg, C, A, B, M, N, K, NW, NKS)
    WS = Int(warpsz); BM = 16 * NW; wg = NW * WS
    mma_gemm_tiled!(backend, wg)(cfg, C, A, B, M, N, K, Val(WS), Val(NW), Val(NKS);
                                 ndrange = wg * cld(M, BM) * cld(N, 16))
    synchronize(backend)
end

# Tuile avec layout par opérande. `la`/`lb` sont des TYPES de tag passés en
# argument (pas des paramètres de MMAConfig) : la config reste inchangée.
@kernel function mma_tile_lay!(cfg, C, A, B, fillv, la, lb)
    a = MMA.load_a(cfg, A, 1, 1, la)
    b = MMA.load_b(cfg, B, 1, 1, lb)
    c = MMA.fill_c(cfg, fillv)
    d = MMA.mma(cfg, a, b, c)
    MMA.store_d!(cfg, C, 1, 1, d, MMA.ColMajor)
end

run_tile_lay(cfg, C, A, B, fillv, la, lb) =
    (mma_tile_lay!(backend, Int(warpsz))(cfg, C, A, B, fillv, la, lb; ndrange = Int(warpsz));
     synchronize(backend))

# Fragments chargés depuis la MÉMOIRE PARTAGÉE (@localmem) — le cas d'usage réel
# d'un GEMM tuilé, et le seul que le harnais ne couvrait pas. Attention : les
# backends n'ont pas le même contrat sur l'argument tableau (CUDA prend
# `pointer(A, idx)` + `size(A,1)`, AMD et le fallback font du getindex 2D), donc
# c'est précisément ici qu'une divergence se verrait.
@kernel unsafe_indices = true function mma_lds!(cfg, C, A, B, ::Val{Mt}, ::Val{Nt},
                                                ::Val{Kt}, ::Val{WS}) where {Mt,Nt,Kt,WS}
    lid = @index(Local, Linear)
    sA = @localmem eltype(A) (Mt, Kt)
    sB = @localmem eltype(B) (Kt, Nt)
    for p in lid:WS:(Mt * Kt)
        @inbounds sA[(p - 1) % Mt + 1, (p - 1) ÷ Mt + 1] = A[(p - 1) % Mt + 1, (p - 1) ÷ Mt + 1]
    end
    for p in lid:WS:(Kt * Nt)
        @inbounds sB[(p - 1) % Kt + 1, (p - 1) ÷ Kt + 1] = B[(p - 1) % Kt + 1, (p - 1) ÷ Kt + 1]
    end
    @synchronize
    a = MMA.load_a(cfg, sA, 1, 1, MMA.ColMajor)
    b = MMA.load_b(cfg, sB, 1, 1, MMA.ColMajor)
    c = MMA.fill_c(cfg, MMA.acc_identity(cfg))
    d = MMA.mma(cfg, a, b, c)
    MMA.store_d!(cfg, C, 1, 1, d, MMA.ColMajor)
end

run_lds(cfg, C, A, B, Mt, Nt, Kt) =
    (mma_lds!(backend, Int(warpsz))(cfg, C, A, B, Val(Mt), Val(Nt), Val(Kt), Val(Int(warpsz));
                                    ndrange = Int(warpsz)); synchronize(backend))

# Tuile à une origine (row, col) quelconque dans une matrice plus grande : vérifie
# les décalages en M et en N, alors que la boucle-K ne décale que le long de K.
@kernel function mma_offset!(cfg, C, A, B, r, c0, kc)
    a = MMA.load_a(cfg, A, r, kc, MMA.ColMajor)
    b = MMA.load_b(cfg, B, kc, c0, MMA.ColMajor)
    acc = MMA.fill_c(cfg, MMA.acc_identity(cfg))
    d = MMA.mma(cfg, a, b, acc)
    MMA.store_d!(cfg, C, r, c0, d, MMA.ColMajor)
end

run_offset(cfg, C, A, B, r, c0, kc) =
    (mma_offset!(backend, Int(warpsz))(cfg, C, A, B, r, c0, kc; ndrange = Int(warpsz));
     synchronize(backend))

# load_c : lit l'accumulateur DEPUIS la mémoire, puis fusionne A·B par-dessus —
# le motif d'épilogue « D = A·B + C » d'un GEMM tuilé. C'est le SEUL point d'entrée
# public (+ ses 3 overrides fallback/WMMA/MFMA) qu'aucun test n'exerçait : un bug
# de layout dans _load_c y passait muet.
@kernel function mma_loadc!(cfg, D, A, B, Cin)
    a = MMA.load_a(cfg, A, 1, 1, MMA.ColMajor)
    b = MMA.load_b(cfg, B, 1, 1, MMA.ColMajor)
    c = MMA.load_c(cfg, Cin, 1, 1, MMA.ColMajor)
    d = MMA.mma(cfg, a, b, c)
    MMA.store_d!(cfg, D, 1, 1, d, MMA.ColMajor)
end

run_loadc(cfg, D, A, B, Cin) =
    (mma_loadc!(backend, Int(warpsz))(cfg, D, A, B, Cin; ndrange = Int(warpsz)); synchronize(backend))

@testset "MMA" begin
    # Fallback : warpsize 32 (CUDA/RDNA/Metal) ou 64 (CDNA/MI300, via l'override
    # _mma_wave dans l'ext AMDGPU). Le lane-grid s'adapte à la warpsize.
    if warpsz in (32, 64)
        M = N = K = 16

        # ── Accesseurs de config (hôte, tout à la compilation) ──
        @testset "accesseurs de config" begin
            cfg = MMA.MMAConfig{8,32,16,Float16,Float32,MMA.MulAdd}()
            @test MMA.mma_shape(cfg) === (M = 8, N = 32, K = 16)
            @test MMA.compute_type(cfg) === Float16
            @test MMA.acc_type(cfg) === Float32
            @test MMA.acc_identity(cfg) === 0.0f0
            # L'identité SUIT l'opérateur : neutre de max-plus = -Inf, pas 0.
            @test MMA.acc_identity(MMA.MMAConfig{16,16,16,Float32,Float32,MMA.Tropical}()) === -Inf32
        end

        # ── Fallback portable (régime correction), Float32 : pas de path HW ──
        @testset "fallback GEMM (MulAdd, Float32)" begin
            A = to_device(rand(Float32, M, K)); B = to_device(rand(Float32, K, N))
            C = to_device(zeros(Float32, M, N))
            cfg = MMA.MMAConfig{M,N,K,Float32,Float32,MMA.MulAdd}()
            @test MMA.mma_supported(backend, cfg) == false          # pas de tensor core pour fp32
            run_tile(cfg, C, A, B, 0.0f0)
            @test from_device(C) ≈ from_device(A) * from_device(B) rtol = 1.0f-4
        end

        # ── Type non-HW : FP64 (régime « perf » du fallback, ici correction) ──
        @testset "fallback GEMM (MulAdd, Float64)" begin
            A = to_device(rand(Float64, M, K)); B = to_device(rand(Float64, K, N))
            C = to_device(zeros(Float64, M, N))
            cfg = MMA.MMAConfig{M,N,K,Float64,Float64,MMA.MulAdd}()
            @test MMA.mma_supported(backend, cfg) == false
            run_tile(cfg, C, A, B, 0.0)
            @test from_device(C) ≈ from_device(A) * from_device(B) rtol = 1.0e-12
        end

        # ── Entiers : Int8→Int32 (fallback partout, MFMA sur MI300). Arithmétique
        #    exacte ⇒ on compare avec == , pas ≈. Régression : le produit doit se
        #    faire en Int32, sinon Int8*Int8 déborde.
        # K=32 : c'est la forme i8 de gfx942 (le K=16 y est absent). Sur CUDA le
        # fallback encaisse n'importe quel K, donc le même test couvre les deux.
        for (Mi, Ni, Ki) in ((16, 16, 32), (32, 32, 16))
            cfgi = MMA.MMAConfig{Mi,Ni,Ki,Int8,Int32,MMA.MulAdd}()
            # Le 32×32 A BIEN un chemin fallback : _lane_grid(32,32,·) = (4,8) en
            # warpsize 32 et (8,8) en warpsize 64. Le `continue` qui l'excluait
            # reposait sur un commentaire faux (« grille lane 16×16 seulement ») et
            # sautait une couverture parfaitement valide.
            @testset "GEMM Int8→Int32 $(Mi)×$(Ni)×$(Ki) (MFMA/fallback)" begin
                A = to_device(rand(Int8(-100):Int8(100), Mi, Ki))
                B = to_device(rand(Int8(-100):Int8(100), Ki, Ni))
                C = to_device(zeros(Int32, Mi, Ni))
                run_tile(cfgi, C, A, B, Int32(0))
                @test from_device(C) == Int32.(from_device(A)) * Int32.(from_device(B))
            end
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

        # ── Point d'entrée (row, col) : MÊME résultat que la forme linéaire,
        #    sur le fallback comme sur le hardware. Vérifie que les deux entrées
        #    ne divergent pas et que acc_identity/mma_shape pilotent le kernel.
        @testset "boucle-K via (row, col) (fp32, fallback)" begin
            Kb = 4; Kfull = 16 * Kb
            A = to_device(rand(Float32, M, Kfull)); B = to_device(rand(Float32, Kfull, N))
            C = to_device(zeros(Float32, M, N))
            cfg = MMA.MMAConfig{M,N,16,Float32,Float32,MMA.MulAdd}()
            run_kloop_rc(cfg, C, A, B, Kb)
            @test from_device(C) ≈ from_device(A) * from_device(B) rtol = 1.0f-4
        end

        if MMA.mma_supported(backend, MMA.MMAConfig{16,16,16,Float16,Float32,MMA.MulAdd}())
            @testset "boucle-K via (row, col) (fp16→fp32, HW)" begin
                Kb = 4; Kfull = 16 * Kb
                A = to_device(rand(Float16, M, Kfull)); B = to_device(rand(Float16, Kfull, N))
                C = to_device(zeros(Float32, M, N))
                cfg = MMA.MMAConfig{M,N,16,Float16,Float32,MMA.MulAdd}()
                run_kloop_rc(cfg, C, A, B, Kb)
                ref = Float32.(from_device(A)) * Float32.(from_device(B))
                @test from_device(C) ≈ ref rtol = 1.0f-2
            end
        end

        # ── RowMajor : le MÊME contrat sur le fallback, sur WMMA et sur MFMA.
        #    Une matrice logique M×K stockée row-major se présente comme un
        #    tableau Julia K×M ; on la charge avec RowMajor et on doit retrouver
        #    A·B. C'est la réponse aux transposées (A'·B), et elle DOIT être
        #    identique sur les trois chemins — avant, RowMajor n'existait que sur
        #    CUDA-WMMA et levait une MethodError sur AMD.
        @testset "RowMajor $tag ($CTr)" for (tag, CTr, AccTr, rtolr) in
            (("fallback fp32", Float32, Float32, 1.0f-4),
             ("fallback ComplexF32", ComplexF32, ComplexF32, 1.0f-4))

            Ah = rand(CTr, M, K); Bh = rand(CTr, K, N)
            # Stockage row-major de A ⇒ tableau K×M ; idem pour B ⇒ tableau N×K.
            Art = to_device(Matrix(transpose(Ah)))
            Brt = to_device(Matrix(transpose(Bh)))
            Ac = to_device(Ah); Bc = to_device(Bh)
            cfg = MMA.MMAConfig{M,N,K,CTr,AccTr,MMA.MulAdd}()
            ref = Ah * Bh

            C1 = to_device(zeros(AccTr, M, N))
            run_tile_lay(cfg, C1, Art, Bc, zero(AccTr), MMA.RowMajor, MMA.ColMajor)
            @test from_device(C1) ≈ ref rtol = rtolr

            C2 = to_device(zeros(AccTr, M, N))
            run_tile_lay(cfg, C2, Ac, Brt, zero(AccTr), MMA.ColMajor, MMA.RowMajor)
            @test from_device(C2) ≈ ref rtol = rtolr

            C3 = to_device(zeros(AccTr, M, N))
            run_tile_lay(cfg, C3, Art, Brt, zero(AccTr), MMA.RowMajor, MMA.RowMajor)
            @test from_device(C3) ≈ ref rtol = rtolr
        end

        # Même chose sur le chemin HARDWARE (WMMA sur CUDA, MFMA sur MI300) : c'est
        # la preuve que fallback et hardware s'accordent sur la sémantique du tag.
        if MMA.mma_supported(backend, MMA.MMAConfig{16,16,16,Float16,Float32,MMA.MulAdd}())
            @testset "RowMajor HW (fp16→fp32)" begin
                Ah = rand(Float16, M, K); Bh = rand(Float16, K, N)
                Art = to_device(Matrix(transpose(Ah)))
                Brt = to_device(Matrix(transpose(Bh)))
                Ac = to_device(Ah); Bc = to_device(Bh)
                cfg = MMA.MMAConfig{M,N,K,Float16,Float32,MMA.MulAdd}()
                ref = Float32.(Ah) * Float32.(Bh)

                C1 = to_device(zeros(Float32, M, N))
                run_tile_lay(cfg, C1, Art, Bc, 0.0f0, MMA.RowMajor, MMA.ColMajor)
                @test from_device(C1) ≈ ref rtol = 1.0f-2

                C2 = to_device(zeros(Float32, M, N))
                run_tile_lay(cfg, C2, Ac, Brt, 0.0f0, MMA.ColMajor, MMA.RowMajor)
                @test from_device(C2) ≈ ref rtol = 1.0f-2

                C3 = to_device(zeros(Float32, M, N))
                run_tile_lay(cfg, C3, Art, Brt, 0.0f0, MMA.RowMajor, MMA.RowMajor)
                @test from_device(C3) ≈ ref rtol = 1.0f-2
            end
        end

        # RowMajor sur les formes HW ASYMÉTRIQUES (M≠N). Le `_swap` échange les
        # composantes d'indice ; un bug RowMajor propre à une tuile non carrée —
        # exactement le cas `A'` sur tuile asymétrique — ne se verrait pas sur
        # 16×16. Ces formes n'existent qu'en WMMA (MFMA gfx942 est carré).
        for (Mh, Nh, Kh) in ((8, 32, 16), (32, 8, 16))
            cfgh = MMA.MMAConfig{Mh,Nh,Kh,Float16,Float32,MMA.MulAdd}()
            MMA.mma_supported(backend, cfgh) || continue
            @testset "RowMajor HW asymétrique $(Mh)×$(Nh)×$(Kh)" begin
                Ah = rand(Float16, Mh, Kh); Bh = rand(Float16, Kh, Nh)
                Art = to_device(Matrix(transpose(Ah)))   # A row-major ⇒ tableau Kh×Mh
                Brt = to_device(Matrix(transpose(Bh)))   # B row-major ⇒ tableau Nh×Kh
                Ac = to_device(Ah); Bc = to_device(Bh)
                ref = Float32.(Ah) * Float32.(Bh)
                C1 = to_device(zeros(Float32, Mh, Nh))
                run_tile_lay(cfgh, C1, Art, Bc, 0.0f0, MMA.RowMajor, MMA.ColMajor)
                @test from_device(C1) ≈ ref rtol = 1.0f-2
                C2 = to_device(zeros(Float32, Mh, Nh))
                run_tile_lay(cfgh, C2, Ac, Brt, 0.0f0, MMA.ColMajor, MMA.RowMajor)
                @test from_device(C2) ≈ ref rtol = 1.0f-2
            end
        end

        # ── Mémoire partagée : les fragments se chargent depuis @localmem, sur le
        #    fallback ET sur le hardware. C'est le motif réel d'un GEMM tuilé.
        @testset "fragments depuis @localmem (fp32, fallback)" begin
            A = to_device(rand(Float32, M, K)); B = to_device(rand(Float32, K, N))
            C = to_device(zeros(Float32, M, N))
            cfg = MMA.MMAConfig{M,N,K,Float32,Float32,MMA.MulAdd}()
            run_lds(cfg, C, A, B, M, N, K)
            @test from_device(C) ≈ from_device(A) * from_device(B) rtol = 1.0f-4
        end

        if MMA.mma_supported(backend, MMA.MMAConfig{16,16,16,Float16,Float32,MMA.MulAdd}())
            @testset "fragments depuis @localmem (fp16→fp32, HW)" begin
                A = to_device(rand(Float16, M, K)); B = to_device(rand(Float16, K, N))
                C = to_device(zeros(Float32, M, N))
                cfg = MMA.MMAConfig{M,N,K,Float16,Float32,MMA.MulAdd}()
                run_lds(cfg, C, A, B, M, N, K)
                ref = Float32.(from_device(A)) * Float32.(from_device(B))
                @test from_device(C) ≈ ref rtol = 1.0f-2
            end
        end

        # ── load_c : lecture de l'accumulateur depuis la mémoire (fusion d'épilogue) ──
        @testset "load_c fusion D=A·B+C (fp32, fallback)" begin
            Ah = rand(Float32, M, K); Bh = rand(Float32, K, N); Ch = rand(Float32, M, N)
            D = to_device(zeros(Float32, M, N))
            cfg = MMA.MMAConfig{M,N,K,Float32,Float32,MMA.MulAdd}()
            run_loadc(cfg, D, to_device(Ah), to_device(Bh), to_device(Ch))
            @test from_device(D) ≈ Ah * Bh + Ch rtol = 1.0f-4
        end

        if MMA.mma_supported(backend, MMA.MMAConfig{16,16,16,Float16,Float32,MMA.MulAdd}())
            @testset "load_c fusion D=A·B+C (fp16→fp32, HW)" begin
                Ah = rand(Float16, M, K); Bh = rand(Float16, K, N); Ch = rand(Float32, M, N)
                D = to_device(zeros(Float32, M, N))
                cfg = MMA.MMAConfig{M,N,K,Float16,Float32,MMA.MulAdd}()
                run_loadc(cfg, D, to_device(Ah), to_device(Bh), to_device(Ch))
                ref = Float32.(Ah) * Float32.(Bh) + Ch
                @test from_device(D) ≈ ref rtol = 1.0f-2
            end
        end

        # ── Décalages en M et en N (pas seulement le long de K) ──
        @testset "offsets M/N $tag" for (tag, CTo, AccTo, rt, hw) in
            (("fp32 (fallback)", Float32, Float32, 1.0f-4, false),
             ("fp16→fp32 (HW)", Float16, Float32, 1.0f-2, true))

            hw && !MMA.mma_supported(backend, MMA.MMAConfig{M,N,K,CTo,AccTo,MMA.MulAdd}()) && continue
            # A est 2M×2K, B est 2K×2N : on calcule la tuile (2e bloc de lignes,
            # 2e bloc de colonnes) en prenant la 2e tranche de K.
            Ah = rand(CTo, 2M, 2K); Bh = rand(CTo, 2K, 2N)
            A = to_device(Ah); B = to_device(Bh)
            C = to_device(zeros(AccTo, 2M, 2N))
            cfg = MMA.MMAConfig{M,N,K,CTo,AccTo,MMA.MulAdd}()
            run_offset(cfg, C, A, B, M + 1, N + 1, K + 1)
            ref = _f32.(Ah[(M + 1):2M, (K + 1):2K]) * _f32.(Bh[(K + 1):2K, (N + 1):2N])
            got = from_device(C)[(M + 1):2M, (N + 1):2N]
            @test _f32.(got) ≈ ref rtol = rt
            # Le reste de C ne doit pas avoir été touché.
            @test all(iszero, from_device(C)[1:M, 1:N])
        end

        # Offsets M/N sur les formes HW ASYMÉTRIQUES : origine (Mh+1, Nh+1), tranche
        # K décalée. Vérifie que le décodage (row, col) tient quand M≠N sur le HW.
        for (Mh, Nh, Kh) in ((8, 32, 16), (32, 8, 16))
            cfgh = MMA.MMAConfig{Mh,Nh,Kh,Float16,Float32,MMA.MulAdd}()
            MMA.mma_supported(backend, cfgh) || continue
            @testset "offsets HW asymétrique $(Mh)×$(Nh)×$(Kh)" begin
                Ah = rand(Float16, 2Mh, 2Kh); Bh = rand(Float16, 2Kh, 2Nh)
                C = to_device(zeros(Float32, 2Mh, 2Nh))
                run_offset(cfgh, C, to_device(Ah), to_device(Bh), Mh + 1, Nh + 1, Kh + 1)
                ref = Float32.(Ah[(Mh + 1):2Mh, (Kh + 1):2Kh]) * Float32.(Bh[(Kh + 1):2Kh, (Nh + 1):2Nh])
                @test from_device(C)[(Mh + 1):2Mh, (Nh + 1):2Nh] ≈ ref rtol = 1.0f-2
                @test all(iszero, from_device(C)[1:Mh, 1:Nh])
            end
        end

        # ── Adjoint / Transpose de Julia sur les chemins getindex ──
        # MESURÉ (pas supposé) : le fallback et MFMA lisent par getindex, donc un
        # `A'` de Julia y fonctionne tel quel — et pour un complexe la CONJUGAISON
        # vient gratuitement, puisque c'est getindex sur Adjoint qui la fait.
        # Sur WMMA en revanche, `pointer` n'existe pas sur un Adjoint : l'appel
        # échoue à la compilation (InvalidIRError), BRUYAMMENT. Aucun chemin ne
        # produit de chiffres faux. La forme portable, valable partout, reste
        # RowMajor sur le parent (testée plus haut).
        @testset "Adjoint/Transpose sur le fallback" begin
            Ap = rand(Float32, K, M); Bh = rand(Float32, K, N)
            cfg = MMA.MMAConfig{M,N,K,Float32,Float32,MMA.MulAdd}()
            C = to_device(zeros(Float32, M, N))
            run_tile_lay(cfg, C, to_device(Ap)', to_device(Bh), 0.0f0,
                         MMA.ColMajor, MMA.ColMajor)
            @test from_device(C) ≈ transpose(Ap) * Bh rtol = 1.0f-4

            # Complexe : A' conjugue, et le fallback doit le refléter.
            Apc = rand(ComplexF32, K, M); Bhc = rand(ComplexF32, K, N)
            cfgc = MMA.MMAConfig{M,N,K,ComplexF32,ComplexF32,MMA.MulAdd}()
            Cc = to_device(zeros(ComplexF32, M, N))
            run_tile_lay(cfgc, Cc, to_device(Apc)', to_device(Bhc), 0.0f0 + 0.0f0im,
                         MMA.ColMajor, MMA.ColMajor)
            @test from_device(Cc) ≈ adjoint(Apc) * Bhc rtol = 1.0f-4
        end

        # ── Dégradation gracieuse : une forme fp16 SANS chemin hardware doit
        #    retomber sur le fallback portable, pas casser à la compilation.
        #    16×16×8 n'est ni une forme WMMA (CUDA) ni une forme MFMA (gfx942) —
        #    régression : tant que les overrides laissaient M,N,K libres, cette
        #    config capturait l'override du bon type et mourait sur un
        #    `_tuple_error` au lieu de dégrader.
        @testset "fallback fp16 16×16×8 (forme hors table)" begin
            cfg8 = MMA.MMAConfig{16,16,8,Float16,Float32,MMA.MulAdd}()
            @test MMA.mma_supported(backend, cfg8) == false
            A = to_device(rand(Float16, 16, 8)); B = to_device(rand(Float16, 8, 16))
            C = to_device(zeros(Float32, 16, 16))
            run_tile(cfg8, C, A, B, 0.0f0)
            ref = Float32.(from_device(A)) * Float32.(from_device(B))
            @test from_device(C) ≈ ref rtol = 1.0f-2
        end

        # ── mma_shapes(backend) : l'énumération DOIT être exacte ──
        # Le testset balaye ce que le backend annonce et fait tourner un vrai GEMM
        # sur CHAQUE entrée. C'est l'invariant « ne jamais annoncer une capacité
        # non testée », rendu automatique : ajouter une ligne à une table backend
        # sans qu'elle marche fait échouer les tests, sans rien écrire ici.
        @testset "mma_shapes() énumère exactement le hardware" begin
            shapes = MMA.mma_shapes(backend)
            @test shapes isa Tuple
            for s in shapes
                cfg = MMA.MMAConfig{s.M,s.N,s.K,s.compute,s.acc,MMA.MulAdd}()
                # 1) cohérence avec la query ponctuelle
                @test MMA.mma_supported(backend, cfg)
                # 2) et ça calcule juste
                if s.compute === Int8
                    Ah = rand(Int8(-100):Int8(100), s.M, s.K)
                    Bh = rand(Int8(-100):Int8(100), s.K, s.N)
                    C = to_device(zeros(s.acc, s.M, s.N))
                    run_tile(cfg, C, to_device(Ah), to_device(Bh), zero(s.acc))
                    @test from_device(C) == Int32.(Ah) * Int32.(Bh)
                else
                    Ah = _as.(s.compute, rand(Float32, s.M, s.K))
                    Bh = _as.(s.compute, rand(Float32, s.K, s.N))
                    C = to_device(zeros(s.acc, s.M, s.N))
                    run_tile(cfg, C, to_device(Ah), to_device(Bh), zero(s.acc))
                    ref = _f32.(Ah) * _f32.(Bh)
                    # fp8/bf8 : sizeof 1 dans la branche flottante (Int8 passe par la
                    # branche entière ci-dessus). ≤3 bits de mantisse, mais la référence
                    # décode les MÊMES octets fp8 que le hardware, donc l'écart n'est que
                    # l'accumulation f32 — 5e-2 est large et sûr.
                    rt = s.compute === Core.BFloat16 ? 5.0f-2 :
                         s.compute === Float64 ? 1.0f-6 :
                         sizeof(s.compute) == 1 ? 5.0f-2 : 1.0f-2
                    @test _f32.(from_device(C)) ≈ ref rtol = rt
                end
            end
        end

        # ── Chemin HARDWARE fp16→fp32, MÊME kernel (WMMA sur CUDA, MFMA sur MI300) ──
        # On BALAYE toutes les formes candidates et on teste celles que le backend
        # ANNONCE supportées : `mma_supported` ne doit jamais promettre sans preuve.
        for (CT, AccT) in ((Float16, Float32), (Core.BFloat16, Float32)),
            (Mh, Nh, Kh) in ((16, 16, 16), (8, 32, 16), (32, 8, 16), (32, 32, 8))

            cfgh = MMA.MMAConfig{Mh,Nh,Kh,CT,AccT,MMA.MulAdd}()
            MMA.mma_supported(backend, cfgh) || continue
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

        if MMA.mma_supported(backend, MMA.MMAConfig{16,16,16,Float16,Float32,MMA.MulAdd}())
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

        # ── RÉGRESSION accumulateur `undef` (cf. mma_gemm_tiled! plus haut) ────
        # Boucle-K à borne dynamique, 1 et 2 warps, 1 et 2 mma par panneau, mise
        # en scène `@localmem` dans la boucle. Sur la version overlay de KI, ce
        # motif produisait `phi float [ undef, %preheader ]` sur le chemin HW :
        # selon LLVM, soit des NaN/1e34, soit — pire — des chiffres justes qui
        # redeviennent faux au prochain changement de build. Ne pas remplacer la
        # borne dynamique par un `Val{}` : cela déroulerait la boucle et le test
        # perdrait toute valeur.
        if MMA.mma_supported(backend, MMA.MMAConfig{16,16,16,Float16,Float32,MMA.MulAdd}())
            @testset "HW GEMM tuilé, boucle-K dynamique (NW=$NW, NKS=$NKS)" for
                    (NW, NKS) in ((1, 1), (2, 1), (1, 2), (2, 2))
                cfg = MMA.MMAConfig{16,16,16,Float16,Float32,MMA.MulAdd}()
                for (Mg, Ng, Kg) in ((16, 16, 64), (48, 32, 96), (33, 17, 70))
                    Ah = rand(Float16, Mg, Kg); Bh = rand(Float16, Kg, Ng)
                    C = to_device(zeros(Float32, Mg, Ng))
                    run_gemm_tiled(cfg, C, to_device(Ah), to_device(Bh), Mg, Ng, Kg, NW, NKS)
                    ref = Float32.(Ah) * Float32.(Bh)
                    got = from_device(C)
                    @test all(isfinite, got)
                    @test got ≈ ref rtol = 1.0f-2
                end
            end

            # ── Le garde-fou QUI MORD : inspection de l'IR ────────────────────
            # Le test numérique ci-dessus est nécessaire mais PAS suffisant : un
            # `undef` peut se matérialiser en 0 et les chiffres sont alors justes
            # PAR CHANCE (c'est exactement ce qui se passait sur CUDA.jl 5.11.3 /
            # LLVM 20 — le motif de la boucle-K sortait `phi float [ undef, … ]`
            # tout en donnant des résultats corrects). Le seul signal fiable est
            # donc l'IR elle-même : sur le chemin hardware, AUCUN phi flottant ne
            # doit être initialisé à `undef` dans le préheader de la boucle-K.
            # Si ce test casse, chercher un overlay réintroduit sur une fonction
            # qui produit un fragment (cf. src/mma.jl, « Jeton matériel »).
            # Ce garde-fou vaut pour TOUT backend hardware, pas seulement CUDA :
            # le chemin MFMA (AMD) était installé par le même mécanisme d'overlay
            # et souffrait donc du même `undef`. Validé sur MI300A (gfx942) le
            # 2026-07-21 : 12/12 numérique ET zéro phi `undef`.
            @testset "boucle-K : accumulateur initialisé (pas d'undef dans l'IR)" begin
                cfg = MMA.MMAConfig{16,16,16,Float16,Float32,MMA.MulAdd}()
                Mg, Ng, Kg = 48, 32, 96
                C = to_device(zeros(Float32, Mg, Ng))
                A = to_device(rand(Float16, Mg, Kg)); B = to_device(rand(Float16, Kg, Ng))
                ir = @capture_llvm run_gemm_tiled(cfg, C, A, B, Mg, Ng, Kg, 2, 2)
                bad = [l for l in split(ir, '\n')
                       if occursin("phi float", l) && occursin("undef", l)]
                @test isempty(bad)
            end
        end
    end
end
