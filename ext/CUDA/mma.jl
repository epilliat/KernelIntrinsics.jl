# Façade WMMA pour KernelIntrinsics.MMA (chemin hardware NVIDIA).
#
# Surcharge les stubs _load_a/_load_b/_load_c/_fill_c/_mma/_store_d! pour les
# configs supportées par les tensor cores (fp16→fp32, MulAdd, formes WMMA) ;
# toute autre config retombe sur le fallback portable défini dans src/mma.jl.
#
# Partie portée quasi-verbatim de GemmKernels.jl (BSD-3, © 2020 Thomas Faingnaert
# and contributors ; IEEE TPDS 33(9) 2022, doi:10.1109/TPDS.2021.3136457) — voir
# THIRD_PARTY_LICENSES.

import KernelIntrinsics.MMA: MMAConfig, RowMajor, ColMajor, MulAdd
import KernelIntrinsics.MMA: _load_a, _load_b, _load_c, _fill_c, _mma, _store_d!, mma_supported
using CUDA: WMMA

@inline _wmma_layout(::Type{RowMajor}) = WMMA.RowMajor
@inline _wmma_layout(::Type{ColMajor}) = WMMA.ColMajor

# Formes WMMA supportées en fp16 (Ada et +).
const _CUDA_MMA_SHAPES = ((16, 16, 16), (8, 32, 16), (32, 8, 16))

# Query hôte : chemin HW dispo ? (grossière : par type/forme, pas par device exact)
mma_supported(::MMAConfig{M,N,K,Float16,Float32,MulAdd}) where {M,N,K} =
    (M, N, K) in _CUDA_MMA_SHAPES

# ── Overrides device (uniquement fp16→fp32, MulAdd) ─────────────────────────
CUDA.@device_override @inline _load_a(::MMAConfig{M,N,K,Float16,Float32,MulAdd}, A, idx, ::Type{L}) where {M,N,K,L} =
    WMMA.load_a(pointer(A, idx), size(A, 1), _wmma_layout(L), WMMA.Config{M,N,K,Float32})

CUDA.@device_override @inline _load_b(::MMAConfig{M,N,K,Float16,Float32,MulAdd}, B, idx, ::Type{L}) where {M,N,K,L} =
    WMMA.load_b(pointer(B, idx), size(B, 1), _wmma_layout(L), WMMA.Config{M,N,K,Float32})

CUDA.@device_override @inline _load_c(::MMAConfig{M,N,K,Float16,Float32,MulAdd}, C, idx, ::Type{L}) where {M,N,K,L} =
    WMMA.load_c(pointer(C, idx), size(C, 1), _wmma_layout(L), WMMA.Config{M,N,K,Float32})

CUDA.@device_override @inline _fill_c(::MMAConfig{M,N,K,Float16,Float32,MulAdd}, v) where {M,N,K} =
    WMMA.fill_c(Float32(v), WMMA.Config{M,N,K,Float32})

CUDA.@device_override @inline _mma(::MMAConfig{M,N,K,Float16,Float32,MulAdd},
                                   a::WMMA.Fragment, b::WMMA.Fragment, c::WMMA.Fragment) where {M,N,K} =
    WMMA.mma(a, b, c, WMMA.Config{M,N,K,Float32})

CUDA.@device_override @inline _store_d!(::MMAConfig{M,N,K,Float16,Float32,MulAdd}, C, idx,
                                        d::WMMA.Fragment, ::Type{L}) where {M,N,K,L} =
    WMMA.store_d(pointer(C, idx), d, size(C, 1), _wmma_layout(L), WMMA.Config{M,N,K,Float32})
