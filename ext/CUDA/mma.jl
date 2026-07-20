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
import KernelIntrinsics.MMA: _load_a, _load_b, _load_c, _fill_c, _mma, _store_d!, mma_supported
using CUDA: WMMA

@inline _wmma_layout(::Type{RowMajor}) = WMMA.RowMajor
@inline _wmma_layout(::Type{ColMajor}) = WMMA.ColMajor

# (type calcul CT, type accumulation AccT, formes supportées).
# Les formes dépendent du TYPE, pas seulement du backend : CUDA.jl ne gère bf16
# qu'en 16×16×16 (ses tables de taille de fragment ne couvrent pas les formes
# asymétriques — un 8×32×16 bf16 meurt sur un `_tuple_error` à la compilation).
# `Core.BFloat16` est le type primitif de Base : le nommer ne coûte aucune
# dépendance (les conversions viennent de BFloat16s, tiré par CUDA.jl).
const _WMMA_TYPES = (
    (Float16, Float32, ((16, 16, 16), (8, 32, 16), (32, 8, 16))),
    (Core.BFloat16, Float32, ((16, 16, 16),)),
)

for (CT, AccT, SHAPES) in _WMMA_TYPES
    @eval begin
        CUDA.@device_override @inline _load_a(::MMAConfig{M,N,K,$CT,$AccT,MulAdd}, A, idx, ::Type{L}) where {M,N,K,L} =
            WMMA.load_a(pointer(A, idx), size(A, 1), _wmma_layout(L), WMMA.Config{M,N,K,$AccT})

        CUDA.@device_override @inline _load_b(::MMAConfig{M,N,K,$CT,$AccT,MulAdd}, B, idx, ::Type{L}) where {M,N,K,L} =
            WMMA.load_b(pointer(B, idx), size(B, 1), _wmma_layout(L), WMMA.Config{M,N,K,$AccT})

        CUDA.@device_override @inline _load_c(::MMAConfig{M,N,K,$CT,$AccT,MulAdd}, C, idx, ::Type{L}) where {M,N,K,L} =
            WMMA.load_c(pointer(C, idx), size(C, 1), _wmma_layout(L), WMMA.Config{M,N,K,$AccT})

        CUDA.@device_override @inline _fill_c(::MMAConfig{M,N,K,$CT,$AccT,MulAdd}, v) where {M,N,K} =
            WMMA.fill_c($AccT(v), WMMA.Config{M,N,K,$AccT})

        CUDA.@device_override @inline _mma(::MMAConfig{M,N,K,$CT,$AccT,MulAdd},
                                           a::WMMA.Fragment, b::WMMA.Fragment, c::WMMA.Fragment) where {M,N,K} =
            WMMA.mma(a, b, c, WMMA.Config{M,N,K,$AccT})

        CUDA.@device_override @inline _store_d!(::MMAConfig{M,N,K,$CT,$AccT,MulAdd}, C, idx,
                                                d::WMMA.Fragment, ::Type{L}) where {M,N,K,L} =
            WMMA.store_d(pointer(C, idx), d, size(C, 1), _wmma_layout(L), WMMA.Config{M,N,K,$AccT})

        # Query hôte (grossière : par type/forme, pas par device exact).
        mma_supported(::MMAConfig{M,N,K,$CT,$AccT,MulAdd}) where {M,N,K} =
            (M, N, K) in $SHAPES
    end
end
