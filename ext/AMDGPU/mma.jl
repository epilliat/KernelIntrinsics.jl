# Chemin AMD pour KernelIntrinsics.MMA.
#
#   1) FALLBACK wave64 — override _mma_wave() = Val{64}() (le fallback portable de
#      src/mma.jl devient correct sur CDNA/MI300, tous types).
#   2) HARDWARE MFMA (CDNA/gfx942) — chemin tensor-core. La table _MFMA_OPS
#      ci-dessous est passée ligne par ligne à `MMA._emit_mfma_row` (codegen
#      partagé dans src/mma_amd.jl), qui émet ici les surcharges device. Le même
#      émetteur sert à l'extension fp8 (AMDGPU + DLFP8Types), d'où sa place dans le
#      package principal : `MFMAFrag` doit être un type UNIQUE partagé.
#
# Ajouter une forme = ajouter UNE ligne à la table… puis LA VALIDER SUR MI300 :
# les layouts diffèrent réellement d'une op à l'autre (blocked / interleaved /
# blk4). Ne jamais les supposer — cf. local/mma/proto/probe_layout.jl.
#
# TODO : 32×32×16 i8 (fait), SMFMAC (sparse), RDNA3 wave32/WMMA. fp8/bf8 : voir
# l'extension KernelIntrinsicsAMDGPUDLFP8TypesExt.

import KernelIntrinsics.MMA: MMAConfig, ColMajor, RowMajor, MatrixA, MatrixB, Accumulator, MulAdd
import KernelIntrinsics.MMA: _load_a, _load_b, _load_c, _fill_c, _mma, _store_d!, _mma_wave, mma_supported, mma_shapes
import KernelIntrinsics.MMA: MFMAFrag, _emit_mfma_row, _MFMA_VALIDATED, _ext_shapes

# ── 1) Fallback wave64 ───────────────────────────────────────────────────────
@amdgpu_overlay @inline _mma_wave() = Val{64}()

# ── 2) MFMA ──────────────────────────────────────────────────────────────────
# Table des intrinsics MFMA de base (types natifs, sans dépendance).
#   nc = éléments accumulateur par lane.
#   Layouts (g = lane÷M, j = index d'élément 0-based) — tous VALIDÉS sur MI300 :
#     opérandes  : A[m,k] m = lane%M, k = na*g + j ;  B[k,n] n = lane%N, k = na*g + j
#     accumulateur, n = lane%N, et
#       :blocked     → m = nc*g + j                   (accum f32/i32)
#       :interleaved → m = nc*j + g                   (accum f64)
#       :blk4        → m = 8*(j÷4) + 4*g + j%4        (famille 32×32)
#   ST = type de STOCKAGE de l'opérande vu par l'intrinsic (peut différer de CT :
#        bf16→<4 x i16> donc ST=Int16 ; i8 → 8 packés dans i64 donc ST=Int64).
#   na = éléments de CT par lane ; nw = na÷pk mots ST réellement passés.
#   archs = architectures GCN où l'op EXISTE d'après l'ISA CDNA (MFMA = CDNA only).
#
# ⚠️ « Exister » ≠ « même layout ». Erreur d'archi sur l'INTRINSIC → erreur de
# compilation bruyante (« Cannot select ») ; erreur d'archi sur le LAYOUT → chiffres
# silencieusement faux. Donc `mma_supported` croise l'ISA avec _MFMA_VALIDATED.
const _CDNA1 = ("gfx908", "gfx90a", "gfx942", "gfx950")   # MFMA de base
const _CDNA2 = ("gfx90a", "gfx942", "gfx950")             # + bf16 .1k, f64
const _CDNA3 = ("gfx942", "gfx950")                       # + i8 K=32

const _MFMA_OPS = (
    # (M, N, K, CT, AccT, ST, intrinsic, na, nc, acc_layout, archs)
    (16, 16, 16, Float16, Float32, Float16, "llvm.amdgcn.mfma.f32.16x16x16f16", 4, 4, :blocked, _CDNA1),
    (16, 16, 4, Float64, Float64, Float64, "llvm.amdgcn.mfma.f64.16x16x4f64", 1, 4, :interleaved, _CDNA2),
    (32, 32, 8, Float16, Float32, Float16, "llvm.amdgcn.mfma.f32.32x32x8f16", 4, 16, :blk4, _CDNA1),
    (16, 16, 16, Core.BFloat16, Float32, Int16, "llvm.amdgcn.mfma.f32.16x16x16bf16.1k", 4, 4, :blocked, _CDNA2),
    (32, 32, 8, Core.BFloat16, Float32, Int16, "llvm.amdgcn.mfma.f32.32x32x8bf16.1k", 4, 16, :blk4, _CDNA2),
    # gfx942 a REMPLACÉ le i8 K=16 (gfx908/90a) par K=32 à opérandes i64 : le
    # `mfma.i32.16x16x16i8` existe dans LLVM mais finit en « Cannot select » ici.
    (16, 16, 32, Int8, Int32, Int64, "llvm.amdgcn.mfma.i32.16x16x32.i8", 8, 4, :blocked, _CDNA3),
    (32, 32, 16, Int8, Int32, Int64, "llvm.amdgcn.mfma.i32.32x32x16.i8", 8, 16, :blk4, _CDNA3),
)

# Génère les surcharges device pour chaque ligne (émetteur partagé).
for row in _MFMA_OPS
    _emit_mfma_row(@__MODULE__, row...)
end

# Architecture GCN du device courant, suffixes de features retirés
# ("gfx942:sramecc+:xnack-" → "gfx942").
_gfx() = first(split(AMDGPU.device().gcn_arch, ':'))

# Une ligne de table → un NamedTuple de forme, si l'archi courante la valide.
_mfma_shape_if_supported(gfx, row) =
    let (M, N, K, CT, AccT, ST, INTR, na, nc, acclay, ARCHS) = row
        (gfx in ARCHS && gfx in _MFMA_VALIDATED) ?
            (M = M, N = N, K = K, compute = CT, acc = AccT) : nothing
    end

# Énumération des configs hardware du device courant (cf. docstring dans src/mma.jl).
# Table de base + registre des extensions optionnelles (fp8, remplie à leur __init__).
function mma_shapes(::AMDGPU.ROCBackend)
    gfx = _gfx()
    out = Any[]
    for row in _MFMA_OPS
        s = _mfma_shape_if_supported(gfx, row)
        s === nothing || push!(out, s)
    end
    append!(out, _ext_shapes(AMDGPU.ROCBackend))   # formes des ext optionnelles (fp8)
    return Tuple(out)
end
