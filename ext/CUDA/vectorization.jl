
import KernelIntrinsics: _vload_batch, _vstore_batch!, _vload_norebase, _vstore_norebase!


CUDA.@device_override @inline function _vload_batch(A::CuDeviceArray{T}, idx, ::Val{Nitem})::NTuple{Nitem,T} where {T,Nitem}
    sz = 1 << trailing_zeros(Nitem * sizeof(T))  # Int: avoid UInt8 overflow when Nitem*sizeof(T) >= 256 (would give align 0)
    ptr = reinterpret(Core.LLVMPtr{NTuple{Nitem,T},AS.Global}, pointer(A))
    return unsafe_load(ptr, idx, Val(sz))
end

CUDA.@device_override @inline function _vstore_batch!(A::CuDeviceArray{T}, idx, values::NTuple{Nitem,T}) where {T,Nitem}
    sz = 1 << trailing_zeros(Nitem * sizeof(T))  # Int: avoid UInt8 overflow when Nitem*sizeof(T) >= 256 (would give align 0)
    ptr = reinterpret(Core.LLVMPtr{NTuple{Nitem,T},AS.Global}, pointer(A))
    unsafe_store!(ptr, values, idx, Val(sz))
end

CUDA.@device_override @inline function _vload_norebase(A::CuDeviceArray{T}, idx, ::Val{Nitem})::NTuple{Nitem,T} where {T,Nitem}
    ptr = reinterpret(Core.LLVMPtr{NTuple{Nitem,T},AS.Global}, pointer(A) + (idx - 1) * sizeof(T))
    sz = 1 << trailing_zeros(Nitem * sizeof(T))  # Int: avoid UInt8 overflow when Nitem*sizeof(T) >= 256 (would give align 0)
    return unsafe_load(ptr, 1, Val(sz))
end

CUDA.@device_override @inline function _vstore_norebase!(A::CuDeviceArray{T}, idx, values::NTuple{Nitem,T}) where {T,Nitem}
    ptr = reinterpret(Core.LLVMPtr{NTuple{Nitem,T},AS.Global}, pointer(A) + (idx - 1) * sizeof(T))
    sz = 1 << trailing_zeros(Nitem * sizeof(T))  # Int: avoid UInt8 overflow when Nitem*sizeof(T) >= 256 (would give align 0)
    unsafe_store!(ptr, values, 1, Val(sz))
end