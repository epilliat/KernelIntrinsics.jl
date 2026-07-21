# Extension MULTI-DÉCLENCHEUR : ne charge que si AMDGPU **et** DLFP8Types sont tous
# deux présents. C'est ainsi que le chemin hardware fp8/bf8 (MFMA CDNA3) est ajouté
# SANS que KI possède un type fp8 ni dépende de DLFP8Types en dur — le fallback
# portable, lui, gère déjà fp8 pour n'importe quel type bits (cf. src/mma.jl).
#
# Les intrinsics fp8 de gfx942 sont mécaniquement identiques aux lignes i8 (opérandes
# 8 bits packées dans un i64, accumulateur f32, layouts blocked/blk4) : on réutilise
# l'émetteur partagé `MMA._emit_mfma_row`. `fp8` = E4M3FNUZ, `bf8` = E5M2FNUZ (les
# variantes fnuz réellement implémentées par CDNA3). La config à un seul CT interdit
# les combos mixtes (fp8×bf8) — seulement A=B ici.

module KernelIntrinsicsAMDGPUDLFP8TypesExt

using AMDGPU
using DLFP8Types: Float8_E4M3FNUZ, Float8_E5M2FNUZ

import KernelIntrinsics.MMA: MMAConfig, ColMajor, RowMajor, MatrixA, MatrixB, Accumulator, MulAdd
import KernelIntrinsics.MMA: MFMAFrag, _emit_mfma_row, _MFMA_VALIDATED, _register_ext_shape!
import KernelIntrinsics.MMA: _load_a, _load_b, _load_c, _fill_c, _mma, _store_d!, mma_supported

# @amdgpu_overlay est un macro local à l'extension AMDGPU de base ; on en garde une
# copie ici (3 lignes) plutôt que de risquer un partage inter-extensions fragile.
# Le code émis par _emit_mfma_row l'appelle non qualifié : il se résout dans CE module.
macro amdgpu_overlay(expr)
    return esc(:(Base.Experimental.@overlay AMDGPU.method_table $expr))
end

# CDNA3 : gfx942 (validé) et gfx950 (ISA seule, non sondé → filtré par _MFMA_VALIDATED).
const _CDNA3_FP8 = ("gfx942", "gfx950")

# (M, N, K, CT, AccT, ST, intrinsic, na, nc, acc_layout, archs) — même format que
# _MFMA_OPS. fp8/bf8 : 8 valeurs 8-bit packées dans un i64 (ST=Int64, na=8, pk=8),
# accumulateur f32, layouts identiques aux lignes i8 de même forme.
const _MFMA_OPS_FP8 = (
    (16, 16, 32, Float8_E4M3FNUZ, Float32, Int64, "llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8", 8, 4,  :blocked, _CDNA3_FP8),
    (32, 32, 16, Float8_E4M3FNUZ, Float32, Int64, "llvm.amdgcn.mfma.f32.32x32x16.fp8.fp8", 8, 16, :blk4,    _CDNA3_FP8),
    (16, 16, 32, Float8_E5M2FNUZ, Float32, Int64, "llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8", 8, 4,  :blocked, _CDNA3_FP8),
    (32, 32, 16, Float8_E5M2FNUZ, Float32, Int64, "llvm.amdgcn.mfma.f32.32x32x16.bf8.bf8", 8, 16, :blk4,    _CDNA3_FP8),
)

for row in _MFMA_OPS_FP8
    _emit_mfma_row(@__MODULE__, row...)
end

# Rend les formes fp8 visibles à `mma_shapes` (via le registre du package principal).
# À l'__init__ (chargement), pas à la précompilation : la mutation d'un global
# d'un autre module ne doit pas être figée dans l'image précompilée.
function __init__()
    for (M, N, K, CT, AccT, ST, INTR, na, nc, acclay, ARCHS) in _MFMA_OPS_FP8
        _register_ext_shape!(AMDGPU.ROCBackend, (M = M, N = N, K = K, compute = CT, acc = AccT),
                             () -> (g = first(split(AMDGPU.device().gcn_arch, ':'));
                                    g in ARCHS && g in _MFMA_VALIDATED))
    end
end

end # module KernelIntrinsicsAMDGPUDLFP8TypesExt
