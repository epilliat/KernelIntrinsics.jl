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

# ── 128-bit (16-byte) atomic load/store — Device/Relaxed, gfx94x (CDNA3) ──────
#
# LLVM has NO atomic-i128 lowering for AMDGPU (`load atomic i128` → InvalidIRError), so the 16-byte
# coherent+atomic access is emitted as the raw GCN instruction, exactly as rocPRIM does
# (rocprim/intrinsics/atomic.hpp, ROCPRIM_TARGET_CDNA3): a SINGLE
#     global_load_dwordx4 … sc1     /     global_store_dwordx4 … sc1
# followed by `s_waitcnt vmcnt(0)`. `sc1` makes the access coherent at the agent (Device) scope — it
# bypasses the stale L1 and reaches the coherent point — and being ONE instruction it is a torn-free
# 16-byte snapshot. That atomicity is the whole point: it lets a {status, value} descriptor be
# published and read as a single unit, with no flag→value dependent second load and no fences (the
# decoupled-lookback packed-descriptor pattern; cf. the packed UInt64 path for ≤4-byte payloads).
#
# Emitted via `@asmcall` (LLVM.jl's *builder*-based inline asm), NOT `Base.llvmcall` with an IR
# string: a hand-written llvmcall asm string mishandles the pointer/operand ABI here and faults at
# runtime, while the builder does the register/operand setup correctly (verified on gfx942).
#
# Addressing is SADDR+VOFFSET (`global_load_dwordx4 vdst, voffset, saddr`) — precisely what the
# compiler itself emits for `data[index]`: the array BASE is uniform (an SGPR pair) and the per-lane
# byte OFFSET lives in a VGPR. That maps exactly onto the (array, index) API.
#
# RESTRICTED TO 16-BYTE **PRIMITIVE** TYPES (UInt128 / Int128). A composite 16-byte aggregate
# (ComplexF64, NTuple{2,UInt64}, NTuple{4,UInt32}, a struct) CANNOT be carried: `reinterpret` of a
# composite does not GPU-codegen (the same limitation that restricts the ≤4-byte packed path to
# `isprimitivetype`), and an aggregate is rejected by the `=v`/`v` asm constraint as well — both
# verified on gfx942. Callers holding a composite aggregate must use the split (flag + value) protocol.
#
# ARCH: `sc1` is CDNA3 (gfx94x / MI300). Older CDNA (gfx90a MI200, gfx908 MI100) spell the same cache
# behaviour `glc dlc` / `glc`; rocPRIM dispatches on the target. Callers must gate to gfx94x until an
# arch dispatch is added here. Only Device/Relaxed is provided — the coherence comes from `sc1` and the
# atomicity from the single instruction; no other scope/ordering pair is implemented differently.

for T in (UInt128, Int128)
    @eval begin
        Base.Experimental.@overlay AMDGPU.method_table @inline function atomic_load(
            data::ROCDeviceArray{$T,N,1},
            index::Integer,
            ::Type{Device},
            ::Type{Relaxed},
        ) where {N}
            base = reinterpret(UInt64, pointer(data, 1))   # uniform array base → SGPR pair (saddr)
            off = UInt32((index - 1) * 16)                 # per-lane byte offset → VGPR (voffset)
            @asmcall(
                "global_load_dwordx4 \$0, \$1, \$2 sc1\ns_waitcnt vmcnt(0)",
                "=v,v,s", true, $T, Tuple{UInt32,UInt64}, off, base,
            )
        end
        Base.Experimental.@overlay AMDGPU.method_table @inline function atomic_store!(
            data::ROCDeviceArray{$T,N,1},
            index::Integer,
            val::$T,
            ::Type{Device},
            ::Type{Relaxed},
        ) where {N}
            base = reinterpret(UInt64, pointer(data, 1))
            off = UInt32((index - 1) * 16)
            @asmcall(
                "global_store_dwordx4 \$0, \$1, \$2 sc1\ns_waitcnt vmcnt(0)",
                "v,v,s", true, Nothing, Tuple{UInt32,$T,UInt64}, off, val, base,
            )
        end
    end
end