# ext/AMDGPUExt/warp.jl
import KernelIntrinsics: Up, Down, Xor, Idx
import KernelIntrinsics: All, AnyLane, Uni, Ballot
import KernelIntrinsics: _shfl, _vote

# ── Shuffle ───────────────────────────────────────────────────────────────────
# AMDGPU uses LLVM intrinsics via AMDGPU.Device.* (not top-level AMDGPU.*).
# Note: no mask argument on AMD (wavefront uses hardware exec register).
# delta/lane must be Int32; shfl_down additionally requires a Cuint width arg.

const ROC_SHFL_DISPATCH = Dict(
    Up => :shfl_up,
    Down => :shfl_down,
    Xor => :shfl_xor,
    Idx => :shfl,
)

for T in (Int32, UInt32, Float32)
    for (direction, roc_fname) in ROC_SHFL_DISPATCH
        if direction != Idx
            @eval begin
                Base.Experimental.@overlay AMDGPU.method_table @inline _shfl(::Type{$direction}, mask, val::$T, src) =
                    AMDGPU.Device.$roc_fname(val, src)
            end
        end
    end
    @eval begin
        Base.Experimental.@overlay AMDGPU.method_table @inline _shfl(::Type{Idx}, mask, val::$T, src) =
            AMDGPU.Device.shfl(val, src - Int32(1))
    end
end

# ── Vote ──────────────────────────────────────────────────────────────────────
# AMDGPU does not have a `Uni` primitive (uniform predicate vote) — would have
# to approximate it with `ballot(pred) == activemask()`, same as `All`.
# `mask` is ignored on AMDGPU (wavefront participation is governed by the
# hardware `exec` mask register, not a software mask argument).
# `AMDGPU.Device.ballot(pred)::UInt64` always returns UInt64, with a runtime
# branch on `wavefrontsize()` — UInt32 result on wave32 widened to UInt64.
#
# Only `Ballot` is enabled. The other three modes (All / AnyLane / Uni) are
# available via the cross-backend `_vote` polyfill chain or, if you want
# native paths, can be added here using `AMDGPU.Device.ballot` + comparison.
# Direct LLVM intrinsic — bypasses AMDGPU.Device.ballot, which has a runtime
# branch on `wavefrontsize() == 32`. On gfx9xx (CDNA, incl. MI300X) the dead
# wave32 arm — `ccall("llvm.amdgcn.ballot", llvmcall, UInt32, …)` — emits a
# SETCC the AMDGPU instruction selector can't lower, failing the entire
# kernel compile with `Cannot select: i32 = SETCC ...`.
#
# Note on the intrinsic name: `llvm.amdgcn.ballot.i64` is the canonical
# overload-mangled name; AMDGPU.jl's older `llvm.amdgcn.ballot.w64` spelling
# compiles on the current LLVM/GPUCompiler combo but generates a kernel that
# faults at runtime. Use `.i64`.
#
# Hard-codes wave64. If RDNA (wave32) support is added later, dispatch on the
# compile-time target arch (e.g. via @device_override or AMDGPU.Compiler.target)
# rather than re-introducing the runtime wavefrontsize() check.
Base.Experimental.@overlay AMDGPU.method_table @inline _vote(::Type{Ballot}, mask, pred) =
    ccall("llvm.amdgcn.ballot.i64", llvmcall, UInt64, (Bool,), pred)