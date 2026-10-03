import KernelIntrinsics: _vload_batch, _vstore_batch!, _vload_norebase, _vstore_norebase!, vload, vstore!, _llvm_barrier

# Metal's AIR backend doesn't support inline assembly, so we can't implement
# the same pure compiler barrier as LLVM (call void asm sideeffect "", "~{memory}"()).
#
# Implementation uses air.atomic.fence with conservative ordering:
# - flags=DEVICE (1): Device memory (most vectorized operations)
# - order=SeqCst (5): Sequentially consistent (strongest ordering)
# - scope=Thread (0): Thread scope (no inter-thread sync)
#
# SeqCst with thread scope provides both compiler and hardware ordering
# guarantees with minimal overhead (no inter-thread synchronization).
Base.Experimental.@overlay Metal.method_table @inline function _llvm_barrier()
    ccall("extern air.atomic.fence", llvmcall, Cvoid, (Cuint, Cuint, Cuint),
          UInt32(1), UInt32(5), UInt32(0))
end

# ============================================================================
# Metal GPU Vectorized Memory Access Primitives
# ============================================================================
# KEY DESIGN DECISIONS:
#
# 1. SIMD.Vec vs NTuple:
#    Metal's AIR (Apple IR) backend recognizes LLVM vector types <N x T> 
#    (used by SIMD.Vec) to emit vector load/store instructions. However, 
#    Julia's NTuple{N,T} lowers to LLVM array types [N x T], which Metal 
#    scalarizes into N separate scalar loads. 
#
# 2. Primitive vs Composite Type :
#    SIMD.Vec{N,T} requires T to be a primitive LLVM scalar type (Integers, 
#    Floats). User-defined structs, even isbits, cannot be used directly 
#    as SIMD.Vec element types. We therefore flatten structs to primitive 
#    chunks (UInt8/UInt16/Float32) for the vector operation, then bitcast 
#    back to the original struct layout. This two-step process is necessary 
#    because Metal lacks CUDA's ability to vectorize arbitrary aggregate types.
# ============================================================================


# Determine the optimal primitive chunk type for flattening a struct of size sz.
@generated _chunk_type(sz) = sz == 1 ? :(UInt8) : sz == 2 ? :(UInt16) : :(Float32)

Base.Experimental.@overlay Metal.method_table @inline @generated function _vload_norebase(
    A::MtlDeviceArray{T}, idx, ::Val{Nitem}
) where {T, Nitem}

    if isprimitivetype(T)
        # For primitive types, we can use SIMD.Vec directly since T is a valid
        # LLVM scalar. This generates a single vector load instruction without
        # intermediate representations.

        sz = 0x01 << trailing_zeros(Nitem * sizeof(T))
        quote
            offset_ptr = pointer(A) + (idx - 1) * sizeof($T)
            ptr = reinterpret(Core.LLVMPtr{SIMD.Vec{$Nitem, $T}, 1}, offset_ptr)
            return NTuple{$Nitem, $T}(unsafe_load(ptr, 1, Val($sz)))
        end
    else
        # User-defined structs cannot be SIMD.Vec elements. We decompose the
        # struct into primitive chunks matching the total byte size, perform
        # the vector load on those chunks, then reinterpret the loaded bytes
        # as the original struct layout using pointer bitcasting.

        @assert isbitstype(T)
        struct_sz = sizeof(T)
        ChunkT = _chunk_type(struct_sz)
        n_chunks = div(Nitem * struct_sz, sizeof(ChunkT))
        sz = 0x01 << trailing_zeros(Nitem * struct_sz)
        
        quote
            # Load as vector of primitive chunks (forces vector instruction emission)
            offset_ptr = pointer(A) + (idx - 1) * sizeof($T)
            ptr = reinterpret(Core.LLVMPtr{SIMD.Vec{$n_chunks, $ChunkT}, 1}, offset_ptr)
            raw = unsafe_load(ptr, 1, Val($sz))

            # Bitcast: reinterpret loaded chunk vector memory as tuple of structs
            tuptr = Base.unsafe_convert(Ptr{NTuple{$Nitem, $T}}, pointer_from_objref(Ref(raw)))
            unsafe_load(tuptr)
        end
    end
end

Base.Experimental.@overlay Metal.method_table @inline @generated function _vstore_norebase!(
    A::MtlDeviceArray{T}, idx, values::NTuple{Nitem, T}
) where {T, Nitem}
    
    if isprimitivetype(T)
        sz = 0x01 << trailing_zeros(Nitem * sizeof(T))
        quote
            offset_ptr = pointer(A) + (idx - 1) * sizeof($T)
            ptr = reinterpret(Core.LLVMPtr{SIMD.Vec{$Nitem, $T}, 1}, offset_ptr)
            vec = SIMD.Vec{$Nitem, $T}(values)
            unsafe_store!(ptr, vec, 1, Val($sz))
            return nothing
        end
    else
        @assert isbitstype(T)
        struct_sz = sizeof(T)
        ChunkT = _chunk_type(struct_sz)
        n_chunks = div(Nitem * struct_sz, sizeof(ChunkT))
        sz = 0x01 << trailing_zeros(Nitem * struct_sz)
        
        quote
            offset_ptr = pointer(A) + (idx - 1) * sizeof($T)
            ptr = reinterpret(Core.LLVMPtr{SIMD.Vec{$n_chunks, $ChunkT}, 1}, offset_ptr)
            raw_ptr = Base.unsafe_convert(Ptr{SIMD.Vec{$n_chunks, $ChunkT}}, pointer_from_objref(Ref(values)))
            raw = unsafe_load(raw_ptr)
            unsafe_store!(ptr, raw, 1, Val($sz))
            return nothing
        end
    end
end


# ============================================================================
# Metal-specific overrides for vload/vstore
# ============================================================================
# These are used to bypass vload_multi/vload_pattern/vstore_pattern!/vstore_multi!
# that are quite slow on Metal. Also Metal does not seem too penalized by unalignment

@generated _is_pow2_T(::Type{T}) where {T} = ispow2(sizeof(T))

@inline function vload_metal(A::DenseArray{T}, idx, ::Val{Nitem}, ::Val{Rebase}) where {T, Nitem,Rebase}
    if !_is_pow2_T(T) || sizeof(T) == 1
        if Rebase
            base = (idx - 1) * Nitem + 1
            return ntuple(i -> A[base+i-1], Val(Nitem))
        else
            return ntuple(i -> A[idx+i-1], Val(Nitem))
        end
    else
        if Rebase
            idx_rebased = (idx - 1) * Nitem + 1
            _llvm_barrier()
            result = _vload_norebase(A, idx_rebased, Val(Nitem))
            _llvm_barrier()
            return result
        else
            _llvm_barrier()
            result = _vload_norebase(A, idx, Val(Nitem))
            _llvm_barrier()
            return result
        end
    end
end

Base.Experimental.@overlay Metal.method_table @inline function vload(
    A::DenseArray{T}, idx, ::Val{Nitem}, ::Val{Rebase}, ::Val{Alignment}
)::NTuple{Nitem,T} where {Alignment,T,Nitem,Rebase}
    vload_metal(A, idx, Val(Nitem), Val(Rebase))
end

Base.Experimental.@overlay Metal.method_table @inline function vload(
    A::DenseArray{T}, idx, ::Val{Nitem}, ::Val{Rebase}=Val(true)
)::NTuple{Nitem,T} where {T,Nitem,Rebase}
    vload_metal(A, idx, Val(Nitem), Val(Rebase))
end

@inline function vstore_metal!(
    A::DenseArray{T}, idx, values::NTuple{Nitem,T}, ::Val{Rebase}
) where {T,Nitem,Rebase}
    if !_is_pow2_T(T) || sizeof(T) == 1
        if Rebase
            base = (idx - 1) * Nitem + 1
            for i in ntuple(identity, Val(Nitem))
                A[base+i-1] = values[i]
            end
        else
            for i in ntuple(identity, Val(Nitem))
                A[idx+i-1] = values[i]
            end
        end
        return nothing
    else
        if Rebase
            idx_rebased = (idx - 1) * length(values) + 1
            _llvm_barrier()
            _vstore_norebase!(A, idx_rebased, values)
            _llvm_barrier()
        else
            _llvm_barrier()
            _vstore_norebase!(A, idx, values)
            _llvm_barrier()
        end
        return nothing
    end
end

Base.Experimental.@overlay Metal.method_table @inline function vstore!(
    A::DenseArray{T}, idx, values::NTuple{Nitem,T}, ::Val{Rebase}, ::Val{Alignment}
) where {Alignment,T,Nitem,Rebase}
    vstore_metal!(A, idx, values, Val(Rebase))
    return nothing
end

Base.Experimental.@overlay Metal.method_table @inline function vstore!(
    A::DenseArray{T}, idx, values::NTuple{Nitem,T}, ::Val{Rebase}=Val(true)
) where {T,Nitem,Rebase}
    vstore_metal!(A, idx, values, Val(Rebase))
    return nothing
end
