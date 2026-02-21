import KernelIntrinsics: _vload_batch, _vstore_batch!, _vload_norebase, _vstore_norebase!

Base.Experimental.@overlay AMDGPU.method_table @inline function _vload_batch(A::ROCDeviceArray{T}, idx, ::Val{Nitem})::NTuple{Nitem,T} where {T,Nitem}
    sz = 0x01 << trailing_zeros(Nitem * sizeof(T))
    ptr = reinterpret(Core.LLVMPtr{NTuple{Nitem,T},1}, pointer(A))
    return unsafe_load(ptr, idx, Val(sz))
end

Base.Experimental.@overlay AMDGPU.method_table @inline function _vstore_batch!(A::ROCDeviceArray{T}, idx, values::NTuple{Nitem,T}) where {T,Nitem}
    sz = 0x01 << trailing_zeros(Nitem * sizeof(T))
    ptr = reinterpret(Core.LLVMPtr{NTuple{Nitem,T},1}, pointer(A))
    unsafe_store!(ptr, values, idx, Val(sz))
end

Base.Experimental.@overlay AMDGPU.method_table @inline function _vload_norebase(A::ROCDeviceArray{T}, idx, ::Val{Nitem})::NTuple{Nitem,T} where {T,Nitem}
    ptr = reinterpret(Core.LLVMPtr{NTuple{Nitem,T},1}, pointer(A) + (idx - 1) * sizeof(T))
    sz = 0x01 << trailing_zeros(Nitem * sizeof(T))
    return unsafe_load(ptr, 1, Val(sz))
end

Base.Experimental.@overlay AMDGPU.method_table @inline function _vstore_norebase!(A::ROCDeviceArray{T}, idx, values::NTuple{Nitem,T}) where {T,Nitem}
    ptr = reinterpret(Core.LLVMPtr{NTuple{Nitem,T},1}, pointer(A) + (idx - 1) * sizeof(T))
    sz = 0x01 << trailing_zeros(Nitem * sizeof(T))
    unsafe_store!(ptr, values, 1, Val(sz))
end