# ext/AMDGPU/scopes_ordering.jl

import KernelIntrinsics: Scope, Ordering
import KernelIntrinsics: Workgroup, Device, System
import KernelIntrinsics: Acquire, Release, AcqRel, SeqCst, Weak, Volatile, Relaxed
import KernelIntrinsics: fence, atomic_load, atomic_store!

const SCOPE_TO_GCN = Dict{Type{<:Scope},String}(
    Workgroup => "workgroup",
    Device => "agent",
    System => "system",
)

const FENCE_ORDER_TO_GCN = Dict{Type{<:Ordering},LLVM.AtomicOrdering.T}(
    Acquire => LLVM.AtomicOrdering.Acquire,
    Release => LLVM.AtomicOrdering.Release,
    AcqRel => LLVM.AtomicOrdering.AcquireRelease,
    SeqCst => LLVM.AtomicOrdering.SequentiallyConsistent,
)

const LOAD_ORDER_TO_GCN = Dict{Type{<:Ordering},LLVM.AtomicOrdering.T}(
    Weak => LLVM.AtomicOrdering.Unordered,
    Relaxed => LLVM.AtomicOrdering.Monotonic,
    Volatile => LLVM.AtomicOrdering.Monotonic,
    Acquire => LLVM.AtomicOrdering.Acquire,
)

const STORE_ORDER_TO_GCN = Dict{Type{<:Ordering},LLVM.AtomicOrdering.T}(
    Weak => LLVM.AtomicOrdering.Unordered,
    Relaxed => LLVM.AtomicOrdering.Monotonic,
    Volatile => LLVM.AtomicOrdering.Monotonic,
    Release => LLVM.AtomicOrdering.Release,
)

# Scoped orderings (need syncscope), unscoped ones do not
const SCOPED_LOAD_ORDERINGS = [Relaxed, Acquire]
const SCOPED_STORE_ORDERINGS = [Relaxed, Release]

# Types supported by the atomic loads and stores, which are naturally aligned
const ATOMIC_TYPES = [Int8, UInt8, Int16, UInt16, Float16, Int32, UInt32, Float32, Int64, UInt64, Float64]

# LLVM's "system" scope is the default one, which the IR doesn't spell out
atomic_scope(::Type{S}, ::Type{O}, scoped_orderings) where {S,O} =
    O in scoped_orderings ? SCOPE_TO_GCN[S] : "system"

@llvmgenerated builder function _fence(::Type{S}, ::Type{O})::Nothing where {S,O}
    fence!(builder, FENCE_ORDER_TO_GCN[O]; scope=SCOPE_TO_GCN[S])
    return nothing
end

@llvmgenerated builder function _atomic_load(ptr::Core.LLVMPtr{T,1}, ::Type{S},
                                             ::Type{O})::T where {T,S,O}
    T_val = convert(LLVM.LLVMType, T)
    # with typed pointers (Julia 1.10, 1.11), a Core.LLVMPtr is an i8 pointer; with opaque
    # pointers, the cast does nothing
    ptr = bitcast!(builder, ptr, LLVM.PointerType(T_val, 1))
    load!(builder, T_val, ptr; ordering=LOAD_ORDER_TO_GCN[O],
          scope=atomic_scope(S, O, SCOPED_LOAD_ORDERINGS), align=sizeof(T))
end

@llvmgenerated builder function _atomic_store!(ptr::Core.LLVMPtr{T,1}, val::T, ::Type{S},
                                               ::Type{O})::Nothing where {T,S,O}
    T_val = convert(LLVM.LLVMType, T)
    ptr = bitcast!(builder, ptr, LLVM.PointerType(T_val, 1))
    store!(builder, val, ptr; ordering=STORE_ORDER_TO_GCN[O],
           scope=atomic_scope(S, O, SCOPED_STORE_ORDERINGS), align=sizeof(T))
    return nothing
end

# ── Fence ─────────────────────────────────────────────────────────────────────

for ScopeType in [Workgroup, Device, System]
    for OrderType in [Acquire, Release, AcqRel, SeqCst]
        @eval begin
            Base.Experimental.@overlay AMDGPU.method_table @inline function fence(
                ::Type{$ScopeType}, ::Type{$OrderType}
            )
                _fence($ScopeType, $OrderType)
            end
        end
    end
end

# ── Atomic Load ───────────────────────────────────────────────────────────────

for ScopeType in [Workgroup, Device, System]
    for OrderType in [Weak, Relaxed, Volatile, Acquire]
        for T in ATOMIC_TYPES
            @eval begin
                Base.Experimental.@overlay AMDGPU.method_table @inline function atomic_load(
                    data::ROCDeviceArray{$T,N,1},
                    index::Integer,
                    ::Type{$ScopeType},
                    ::Type{$OrderType},
                ) where {N}
                    ptr = pointer(data, index)
                    _atomic_load(ptr, $ScopeType, $OrderType)
                end
            end
        end
    end
end

# ── Atomic Store ──────────────────────────────────────────────────────────────

for ScopeType in [Workgroup, Device, System]
    for OrderType in [Weak, Relaxed, Volatile, Release]
        for T in ATOMIC_TYPES
            @eval begin
                Base.Experimental.@overlay AMDGPU.method_table @inline function atomic_store!(
                    data::ROCDeviceArray{$T,N,1},
                    index::Integer,
                    val::$T,
                    ::Type{$ScopeType},
                    ::Type{$OrderType},
                ) where {N}
                    ptr = pointer(data, index)
                    _atomic_store!(ptr, val, $ScopeType, $OrderType)
                end
            end
        end
    end
end
