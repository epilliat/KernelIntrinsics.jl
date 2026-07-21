# Façade WMMA pour KernelIntrinsics.MMA (chemin hardware NVIDIA).
#
# Surcharge les stubs _load_a/_load_b/_load_c/_fill_c/_mma/_store_d! pour les
# couples (type calcul, type accumulation) supportés par les tensor cores ; toute
# autre config retombe sur le fallback portable défini dans src/mma.jl.
#
# Généré par `@eval` depuis la table ci-dessous (même style que ext/AMDGPU/mma.jl
# et ext/CUDA/shuffle_vote.jl). Ajouter un type = ajouter une ligne… puis le TESTER :
# le testset HW balaye toutes les formes que `mma_supported` annonce.
#
# Partie portée quasi-verbatim de GemmKernels.jl (BSD-3, © 2020 Thomas Faingnaert
# and contributors ; IEEE TPDS 33(9) 2022, doi:10.1109/TPDS.2021.3136457) — voir
# THIRD_PARTY_LICENSES.

import KernelIntrinsics.MMA: MMAConfig, RowMajor, ColMajor, MulAdd
import KernelIntrinsics.MMA: _load_a, _load_b, _load_c, _fill_c, _mma, _store_d!, mma_supported, mma_shapes, _ext_shapes
using CUDA: WMMA

@inline _wmma_layout(::Type{RowMajor}) = WMMA.RowMajor
@inline _wmma_layout(::Type{ColMajor}) = WMMA.ColMajor

# WMMA veut un POINTEUR ; l'API interne parle en (row, col) 1-based. Le ré-encodage
# col-major est une multiplication — pas la division que coûtait le décodage
# inverse dans l'autre sens.
@inline _lin(A, row, col) = (col - 1) * size(A, 1) + row

# (type calcul CT, type accumulation AccT, capability minimale, formes supportées).
# Les formes dépendent du TYPE, pas seulement du backend : CUDA.jl ne gère bf16
# qu'en 16×16×16 (ses tables de taille de fragment ne couvrent pas les formes
# asymétriques — un 8×32×16 bf16 meurt sur un `_tuple_error` à la compilation).
# `Core.BFloat16` est le type primitif de Base : le nommer ne coûte aucune
# dépendance (les conversions viennent de BFloat16s, tiré par CUDA.jl).
#
# La capability minimale est une VRAIE contrainte matérielle, pas une formalité :
# les tensor cores n'existent qu'à partir de Volta (sm_70) et le WMMA bf16 exige
# Ampere (sm_80). Sans ce gardien, `mma_supported` promettait du hardware sur une
# Pascal — et l'appelant se retrouvait avec une erreur de compilation.
const _WMMA_TYPES = (
    (Float16, Float32, v"7.0", ((16, 16, 16), (8, 32, 16), (32, 8, 16))),
    (Core.BFloat16, Float32, v"8.0", ((16, 16, 16),)),
)

# Les overrides device sont émis UNE FOIS PAR FORME, avec M,N,K liés
# littéralement (comme le fait ext/AMDGPU/mma.jl). Laisser M,N,K libres faisait
# capturer l'override par TOUTE forme du bon type : un 16×16×8 fp16 n'a pas de
# chemin WMMA, mais l'override l'attrapait quand même et mourait sur le
# `_tuple_error` de CUDA.jl au lieu de retomber sur le fallback portable.
# Lier les littéraux rend la dégradation gracieuse : forme inconnue ⇒ fallback.
for (CT, AccT, MINCAP, SHAPES) in _WMMA_TYPES, (M, N, K) in SHAPES
    @eval begin
        CUDA.@device_override @inline _load_a(::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}, A, row, col, ::Type{L}) where {L} =
            WMMA.load_a(pointer(A, _lin(A, row, col)), size(A, 1), _wmma_layout(L), WMMA.Config{$M,$N,$K,$AccT})

        CUDA.@device_override @inline _load_b(::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}, B, row, col, ::Type{L}) where {L} =
            WMMA.load_b(pointer(B, _lin(B, row, col)), size(B, 1), _wmma_layout(L), WMMA.Config{$M,$N,$K,$AccT})

        CUDA.@device_override @inline _load_c(::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}, C, row, col, ::Type{L}) where {L} =
            WMMA.load_c(pointer(C, _lin(C, row, col)), size(C, 1), _wmma_layout(L), WMMA.Config{$M,$N,$K,$AccT})

        CUDA.@device_override @inline _fill_c(::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}, v) =
            WMMA.fill_c($AccT(v), WMMA.Config{$M,$N,$K,$AccT})

        # Tags d'usage contraints (MatrixA/MatrixB/Accumulator), comme MFMA et le
        # fallback : inverser a/b devient une MethodError à la FRONTIÈRE de KI, au
        # lieu d'être rattrapé un cran plus bas dans `WMMA.mma`. Les trois chemins
        # offrent alors la même garantie d'ordre d'opérandes.
        CUDA.@device_override @inline _mma(::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd},
                                           a::WMMA.Fragment{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,WMMA.MatrixA},
                                           b::WMMA.Fragment{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,WMMA.MatrixB},
                                           c::WMMA.Fragment{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,WMMA.Accumulator}) =
            WMMA.mma(a, b, c, WMMA.Config{$M,$N,$K,$AccT})

        CUDA.@device_override @inline _store_d!(::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}, C, row, col,
                                                d::WMMA.Fragment, ::Type{L}) where {L} =
            WMMA.store_d(pointer(C, _lin(C, row, col)), d, size(C, 1), _wmma_layout(L), WMMA.Config{$M,$N,$K,$AccT})

        # Query hôte : gatée sur la capability RÉELLE du device (cf. le côté AMD,
        # gaté sur _gfx()). Une forme tabulée ne suffit pas — encore faut-il que
        # la carte ait les tensor cores correspondants.
        mma_supported(::MMAConfig{$M,$N,$K,$CT,$AccT,MulAdd}) =
            CUDA.capability(CUDA.device()) >= $MINCAP
    end
end

# Énumération des configs hardware du device courant (cf. le docstring dans
# src/mma.jl). Construite depuis la MÊME table que les overrides et gatée sur la
# même capability : impossible qu'elle dérive de ce que `mma_supported` répond.
function mma_shapes(::CUDA.CUDABackend)
    cap = CUDA.capability(CUDA.device())
    out = Any[]
    for (CT, AccT, MINCAP, SHAPES) in _WMMA_TYPES
        cap >= MINCAP || continue
        for (M, N, K) in SHAPES
            push!(out, (M = M, N = N, K = K, compute = CT, acc = AccT))
        end
    end
    append!(out, _ext_shapes(CUDA.CUDABackend))   # formes des ext optionnelles (fp8 mma.sync)
    return Tuple(out)
end
