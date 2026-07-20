# Chemin AMD pour KernelIntrinsics.MMA.
#
#   1) FALLBACK wave64 — override _mma_wave() = Val{64}() (le fallback portable de
#      src/mma.jl devient correct sur CDNA/MI300, tous types).
#   2) HARDWARE MFMA (CDNA/gfx942) — chemin tensor-core, généré par `@eval` depuis
#      la table _MFMA_OPS ci-dessous (même style que ext/CUDA/shuffle_vote.jl).
#
# Ajouter une forme = ajouter UNE ligne à la table… puis LA VALIDER SUR MI300 :
# les layouts diffèrent réellement d'une op à l'autre (cf. blocked vs interleaved
# ci-dessous, découvert à l'exécution). Ne jamais les supposer.
#
# TODO : famille 32×32 (nc=16, layout distinct), bf16/i8/fp8, RDNA3 wave32/WMMA.

import KernelIntrinsics.MMA: MMAConfig, ColMajor, MatrixA, MatrixB, Accumulator, MulAdd
import KernelIntrinsics.MMA: _load_a, _load_b, _load_c, _fill_c, _mma, _store_d!, _mma_wave, mma_supported

# ── 1) Fallback wave64 ───────────────────────────────────────────────────────
@amdgpu_overlay @inline _mma_wave() = Val{64}()

# ── 2) MFMA ──────────────────────────────────────────────────────────────────
# Fragment MFMA opaque : NTuple de VecElement, layout interne CDNA non exposé.
struct MFMAFrag{Use,L,T}
    x::NTuple{L,VecElement{T}}
end

@inline _lane0() = Int(AMDGPU.Device.activelane())   # lane 0..63

@inline function _base(A, idx)
    ld = size(A, 1); b0 = idx - 1
    return (b0 % ld, b0 ÷ ld)
end

# Table des intrinsics MFMA supportés.
#   (M, N, K, CT, AccT, intrinsic, na, nc, acc_layout)
#     na = éléments A/B par lane   (na==1 ⇒ l'intrinsic prend un SCALAIRE)
#     nc = éléments accumulateur par lane
#   Layouts (g = lane÷M, j = index d'élément 0-based) — VALIDÉS sur MI300 :
#     opérandes  : A[m,k] m = lane%M, k = na*g + j ;  B[k,n] n = lane%N, k = na*g + j
#     accumulateur, n = lane%N, et
#       :blocked     → m = nc*g + j   (accum f32, validé via f16 16×16×16)
#       :interleaved → m = nc*j + g   (accum f64, validé via f64 16×16×4)
const _MFMA_OPS = (
    (16, 16, 16, Float16, Float32, "llvm.amdgcn.mfma.f32.16x16x16f16", 4, 4, :blocked),
    (16, 16, 4, Float64, Float64, "llvm.amdgcn.mfma.f64.16x16x4f64", 1, 4, :interleaved),
    (32, 32, 8, Float16, Float32, "llvm.amdgcn.mfma.f32.32x32x8f16", 4, 16, :blk4),
)

for (M, N, K, CT, AccT, INTR, na, nc, acclay) in _MFMA_OPS
    AV = na == 1 ? CT : NTuple{na,VecElement{CT}}   # type d'opérande vu par l'intrinsic
    CV = NTuple{nc,VecElement{AccT}}                # type accumulateur
    unwrap = na == 1 ? :(f.x[1].value) : :(f.x)     # frag → opérande intrinsic
    # Ligne accumulateur (g = lane÷M, j = index élément 0-based).
    accm = if acclay === :blocked
        :($nc * g + j)                                  # 16×16, accum f32
    elseif acclay === :interleaved
        :($nc * j + g)                                  # 16×16, accum f64
    elseif acclay === :blk4
        :($(M ÷ (nc ÷ 4)) * (j ÷ 4) + 4 * g + j % 4)   # 32×32 : 4 blocs de 4 lignes
    else
        error("layout accumulateur inconnu : $acclay")
    end

    @eval begin
        @inline _mfma_call(::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}, a::$AV, b::$AV, c::$CV) =
            ccall($INTR, llvmcall, $CV, ($AV, $AV, $CV, Int32, Int32, Int32),
                  a, b, c, Int32(0), Int32(0), Int32(0))

        @inline _operand(::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}, f::MFMAFrag) = $unwrap

        @amdgpu_overlay @inline function _load_a(::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}, A, idx, ::Type{ColMajor})
            r0, c0 = _base(A, idx); lane = _lane0()
            m = lane % $M; g = lane ÷ $M
            x = ntuple(Val($na)) do e
                @inbounds VecElement($CT(A[r0 + m + 1, c0 + $na * g + (e - 1) + 1]))
            end
            MFMAFrag{MatrixA,$na,$CT}(x)
        end

        @amdgpu_overlay @inline function _load_b(::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}, B, idx, ::Type{ColMajor})
            r0, c0 = _base(B, idx); lane = _lane0()
            n = lane % $N; g = lane ÷ $N
            x = ntuple(Val($na)) do e
                @inbounds VecElement($CT(B[r0 + $na * g + (e - 1) + 1, c0 + n + 1]))
            end
            MFMAFrag{MatrixB,$na,$CT}(x)
        end

        @amdgpu_overlay @inline function _load_c(::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}, C, idx, ::Type{ColMajor})
            r0, c0 = _base(C, idx); lane = _lane0()
            n = lane % $N; g = lane ÷ $M
            x = ntuple(Val($nc)) do e
                j = e - 1
                @inbounds VecElement($AccT(C[r0 + $accm + 1, c0 + n + 1]))
            end
            MFMAFrag{Accumulator,$nc,$AccT}(x)
        end

        @amdgpu_overlay @inline _fill_c(::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}, v) =
            MFMAFrag{Accumulator,$nc,$AccT}(ntuple(_ -> VecElement($AccT(v)), Val($nc)))

        @amdgpu_overlay @inline function _mma(cfg::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd},
                                              a::MFMAFrag{MatrixA}, b::MFMAFrag{MatrixB},
                                              c::MFMAFrag{Accumulator})
            MFMAFrag{Accumulator,$nc,$AccT}(_mfma_call(cfg, _operand(cfg, a), _operand(cfg, b), c.x))
        end

        @amdgpu_overlay @inline function _store_d!(::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}, C, idx,
                                                   d::MFMAFrag{Accumulator}, ::Type{ColMajor})
            r0, c0 = _base(C, idx); lane = _lane0()
            n = lane % $N; g = lane ÷ $M
            for e in 1:$nc
                j = e - 1
                @inbounds C[r0 + $accm + 1, c0 + n + 1] = d.x[e].value
            end
            nothing
        end

        # Query hôte. TODO : gater sur l'isa réelle (gfx942) plutôt qu'inconditionnel.
        mma_supported(::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}) = true
    end
end
