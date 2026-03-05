# ext/AMDGPUExt/warp.jl
import KernelIntrinsics: Up, Down, Xor, Idx
import KernelIntrinsics: All, AnyLane, Uni, Ballot
import KernelIntrinsics: _shfl, _vote

# ── Shuffle ───────────────────────────────────────────────────────────────────
# AMDGPU uses LLVM intrinsics via AMDGPU.Device.* (not top-level AMDGPU.*).
# Note: no mask argument on AMD (wavefront uses hardware exec register).
# delta/lane must be Int32; shfl_down additionally requires a Cuint width arg.
for T in (Int32, UInt32, Float32)
    @eval begin
        Base.Experimental.@overlay AMDGPU.method_table @inline _shfl(::Type{Up}, mask, val::$T, delta, ::Val{ws}) where ws =
            AMDGPU.Device.shfl_up(val, delta)

        Base.Experimental.@overlay AMDGPU.method_table @inline _shfl(::Type{Down}, mask, val::$T, delta, ::Val{ws}) where ws =
            AMDGPU.Device.shfl_down(val, delta)

        Base.Experimental.@overlay AMDGPU.method_table @inline _shfl(::Type{Xor}, mask, val::$T, lane, ::Val{ws}) where ws =
            AMDGPU.Device.shfl_xor(val, lane)

        Base.Experimental.@overlay AMDGPU.method_table @inline _shfl(::Type{Idx}, mask, val::$T, lane, ::Val{ws}) where ws =
            AMDGPU.Device.shfl(val, lane - one(lane))
    end
end

# ── Vote ──────────────────────────────────────────────────────────────────────
# AMDGPU does not have Uni (uniform predicate vote) — approximated via all.
# mask is ignored (AMD uses the hardware exec mask implicitly).
# Note: AMDGPU.Device.ballot returns UInt64 (wavefront is 64 lanes on AMD).
Base.Experimental.@overlay AMDGPU.method_table @inline _vote(::Type{All}, mask, pred) = AMDGPU.Device.vote_all(pred)
Base.Experimental.@overlay AMDGPU.method_table @inline _vote(::Type{AnyLane}, mask, pred) = AMDGPU.Device.vote_any(pred)
Base.Experimental.@overlay AMDGPU.method_table @inline _vote(::Type{Uni}, mask, pred) = AMDGPU.Device.vote_all(pred)   # approximation
Base.Experimental.@overlay AMDGPU.method_table @inline _vote(::Type{Ballot}, mask, pred) = AMDGPU.Device.ballot(pred)    # returns UInt64, not UInt32