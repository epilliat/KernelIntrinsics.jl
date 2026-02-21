# ext/AMDGPU/scopes_ordering.jl

import KernelIntrinsics: Scope, Ordering
import KernelIntrinsics: Workgroup, Device, System
import KernelIntrinsics: Acquire, Release, AcqRel, SeqCst, Weak, Volatile, Relaxed
import KernelIntrinsics: fence, atomic_load, atomic_store!

const SCOPE_TO_GCN = Dict{Type{<:Scope},String}(
    Workgroup => "workgroup",
    Device => "agent",
    System => "system"
)

const FENCE_ORDER_TO_GCN = Dict{Type{<:Ordering},String}(
    Acquire => "acquire",
    Release => "release",
    AcqRel => "acq_rel",
    SeqCst => "seq_cst"
)

const LOAD_ORDER_TO_GCN = Dict{Type{<:Ordering},String}(
    Weak => "unordered",
    Relaxed => "monotonic",
    Volatile => "monotonic",
    Acquire => "acquire",
)

const STORE_ORDER_TO_GCN = Dict{Type{<:Ordering},String}(
    Weak => "unordered",
    Relaxed => "monotonic",
    Volatile => "monotonic",
    Release => "release",
)

const TYPE_TO_GCN = Dict{DataType,Tuple{String,Int}}(
    Int8 => ("i8", 1),
    UInt8 => ("i8", 1),
    Int16 => ("i16", 2),
    UInt16 => ("i16", 2),
    Int32 => ("i32", 4),
    UInt32 => ("i32", 4),
    Float32 => ("float", 4),
    Int64 => ("i64", 8),
    UInt64 => ("i64", 8),
    Float64 => ("double", 8),
)

# ── Fence ─────────────────────────────────────────────────────────────────────

for ScopeType in [Workgroup, Device, System]
    for OrderType in [Acquire, Release, AcqRel, SeqCst]
        scope_str = SCOPE_TO_GCN[ScopeType]
        order_str = FENCE_ORDER_TO_GCN[OrderType]
        ir = """
            fence syncscope("$scope_str") $order_str
            ret void
        """
        @eval begin
            Base.Experimental.@overlay AMDGPU.method_table @inline function fence(::Type{$ScopeType}, ::Type{$OrderType})
                Base.llvmcall($ir, Nothing, Tuple{})
            end
        end
    end
end

# ── Atomic Load ───────────────────────────────────────────────────────────────

for ScopeType in [Workgroup, Device, System]
    for OrderType in [Weak, Relaxed, Volatile, Acquire]
        scope_str = SCOPE_TO_GCN[ScopeType]
        order_str = LOAD_ORDER_TO_GCN[OrderType]
        for (T, (gcn_type, align)) in TYPE_TO_GCN
            syncscope = OrderType in [Weak, Volatile] ? "" : " syncscope(\"$scope_str\")"
            ir = """
                %ptr = inttoptr i64 %0 to $gcn_type*
                %val = load atomic $gcn_type, $gcn_type* %ptr$syncscope $order_str, align $align
                ret $gcn_type %val
            """
            @eval begin
                Base.Experimental.@overlay AMDGPU.method_table @inline function atomic_load(
                    data::ROCDeviceArray{$T,N,1},
                    index::Integer,
                    ::Type{$ScopeType},
                    ::Type{$OrderType}
                ) where {N}
                    ptr = data.ptr + (index - 1) * sizeof($T)
                    Base.llvmcall($ir, $T, Tuple{Int64}, reinterpret(Int64, ptr))
                end
            end
        end
    end
end

# ── Atomic Store ──────────────────────────────────────────────────────────────

for ScopeType in [Workgroup, Device, System]
    for OrderType in [Weak, Relaxed, Volatile, Release]
        scope_str = SCOPE_TO_GCN[ScopeType]
        order_str = STORE_ORDER_TO_GCN[OrderType]
        for (T, (gcn_type, align)) in TYPE_TO_GCN
            syncscope = OrderType in [Weak, Volatile] ? "" : " syncscope(\"$scope_str\")"
            ir = """
                %ptr = inttoptr i64 %0 to $gcn_type*
                store atomic $gcn_type %1, $gcn_type* %ptr$syncscope $order_str, align $align
                ret void
            """
            @eval begin
                Base.Experimental.@overlay AMDGPU.method_table @inline function atomic_store!(
                    data::ROCDeviceArray{$T,N,1},
                    index::Integer,
                    val::$T,
                    ::Type{$ScopeType},
                    ::Type{$OrderType}
                ) where {N}
                    ptr = data.ptr + (index - 1) * sizeof($T)
                    Base.llvmcall($ir, Nothing, Tuple{Int64,$T}, reinterpret(Int64, ptr), val)
                end
            end
        end
    end
end