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

const FENCE_ORDER_TO_GCN = Dict{Type{<:Ordering},String}(
    Acquire => "acquire",
    Release => "release",
    AcqRel => "acq_rel",
    SeqCst => "seq_cst",
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

# Scoped orderings (need syncscope), unscoped ones do not
const SCOPED_LOAD_ORDERINGS = [Relaxed, Acquire]
const SCOPED_STORE_ORDERINGS = [Relaxed, Release]

const TYPE_TO_LLVM = Dict{DataType,Tuple{String,Int}}(
    Int8 => ("i8", 1),
    UInt8 => ("i8", 1),
    Int16 => ("i16", 2),
    UInt16 => ("i16", 2),
    Float16 => ("half", 2),
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
        # A WORKGROUP-scope release fence lowers to NO instruction on gfx9xx (intra-workgroup
        # ordering is implicit) — so it does NOT drain outstanding vector-memory stores to the
        # device-coherent point (L2). For a cross-BLOCK release built on cache-bypassing
        # (Device-scope) atomic stores — the decoupled-lookback message-passing pattern — the
        # producer's data stores MUST be drained to L2 before it stores the flag, or a
        # consumer block can observe the flag ahead of the data. We add an explicit
        # `s_waitcnt vmcnt(0)` (encoded simm16 0xf70 = vmcnt 0, expcnt/lgkmcnt max) after the
        # release fence. This is the DRAIN WITHOUT the L2 writeback/invalidate that a
        # Device-scope fence emits (`buffer_wbl2`+`buffer_inv`, ~2x slower) — matching
        # rocPRIM's `atomic_fence_release_vmem_order_only()` for gfx94x. Applied to the
        # RELEASE and ACQREL workgroup fences (the store-ordering side); pure ACQUIRE needs no
        # drain (it orders following reads only). Device/System fences already drain+flush.
        needs_drain = (ScopeType === Workgroup) && (OrderType === Release || OrderType === AcqRel)
        fbody = needs_drain ?
            quote
                Base.llvmcall($ir, Nothing, Tuple{})
                ccall("llvm.amdgcn.s.waitcnt", llvmcall, Cvoid, (Int32,), Int32(0xf70))
                nothing
            end :
            quote
                Base.llvmcall($ir, Nothing, Tuple{})
            end
        @eval begin
            Base.Experimental.@overlay AMDGPU.method_table @inline function fence(
                ::Type{$ScopeType}, ::Type{$OrderType}
            )
                $fbody
            end
        end
    end
end

# ── Atomic Load ───────────────────────────────────────────────────────────────

for ScopeType in [Workgroup, Device, System]
    for OrderType in [Weak, Relaxed, Volatile, Acquire]
        scope_str = SCOPE_TO_GCN[ScopeType]
        order_str = LOAD_ORDER_TO_GCN[OrderType]
        use_syncscope = OrderType in SCOPED_LOAD_ORDERINGS
        for (T, (llvm_type, align)) in TYPE_TO_LLVM
            syncscope = use_syncscope ? " syncscope(\"$scope_str\")" : ""
            ir = """
                %val = load atomic $llvm_type, $llvm_type addrspace(1)* %0$syncscope $order_str, align $align
                ret $llvm_type %val
            """
            @eval begin
                Base.Experimental.@overlay AMDGPU.method_table @inline function atomic_load(
                    data::ROCDeviceArray{$T,N,1},
                    index::Integer,
                    ::Type{$ScopeType},
                    ::Type{$OrderType},
                ) where {N}
                    ptr = pointer(data, index)
                    Base.llvmcall(
                        $ir, $T,
                        Tuple{Core.LLVMPtr{$T,1}},
                        ptr,
                    )
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
        use_syncscope = OrderType in SCOPED_STORE_ORDERINGS
        for (T, (llvm_type, align)) in TYPE_TO_LLVM
            syncscope = use_syncscope ? " syncscope(\"$scope_str\")" : ""
            ir = """
                store atomic $llvm_type %1, $llvm_type addrspace(1)* %0$syncscope $order_str, align $align
                ret void
            """
            @eval begin
                Base.Experimental.@overlay AMDGPU.method_table @inline function atomic_store!(
                    data::ROCDeviceArray{$T,N,1},
                    index::Integer,
                    val::$T,
                    ::Type{$ScopeType},
                    ::Type{$OrderType},
                ) where {N}
                    ptr = pointer(data, index)
                    Base.llvmcall(
                        $ir, Nothing,
                        Tuple{Core.LLVMPtr{$T,1},$T},
                        ptr, val,
                    )
                end
            end
        end
    end
end