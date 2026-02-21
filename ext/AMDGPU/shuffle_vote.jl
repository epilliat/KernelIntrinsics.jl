# ext/AMDGPUExt/warp.jl

import KernelIntrinsics: Up, Down, Xor, Idx
import KernelIntrinsics: All, AnyLane, Uni, Ballot
import KernelIntrinsics: _shfl, _vote

# ── Shuffle ───────────────────────────────────────────────────────────────────
# AMDGPU uses LLVM intrinsics via AMDGPU.jl's built-in warp shuffle functions.
# Note: no mask argument on AMD (wavefront is always full or use exec register).

for T in (Int32, UInt32, Float32)
    @eval begin
        Base.Experimental.@overlay AMDGPU.method_table @inline _shfl(::Type{Up}, mask, val::$T, delta, ::Val{ws}) where ws =
            AMDGPU.shfl_up(val, delta)

        Base.Experimental.@overlay AMDGPU.method_table @inline _shfl(::Type{Down}, mask, val::$T, delta, ::Val{ws}) where ws =
            AMDGPU.shfl_down(val, delta)

        Base.Experimental.@overlay AMDGPU.method_table @inline _shfl(::Type{Xor}, mask, val::$T, lane, ::Val{ws}) where ws =
            AMDGPU.shfl_xor(val, lane)

        Base.Experimental.@overlay AMDGPU.method_table @inline _shfl(::Type{Idx}, mask, val::$T, lane, ::Val{ws}) where ws =
            AMDGPU.shfl(val, lane)
    end
end

# ── Vote ──────────────────────────────────────────────────────────────────────
# AMDGPU does not have Uni (uniform predicate vote) — approximated via all.
# mask is ignored (AMD uses the hardware exec mask implicitly).

Base.Experimental.@overlay AMDGPU.method_table @inline _vote(::Type{All}, mask, pred) = AMDGPU.vote_all(pred)
Base.Experimental.@overlay AMDGPU.method_table @inline _vote(::Type{AnyLane}, mask, pred) = AMDGPU.vote_any(pred)
Base.Experimental.@overlay AMDGPU.method_table @inline _vote(::Type{Uni}, mask, pred) = AMDGPU.vote_all(pred)  # approximation
Base.Experimental.@overlay AMDGPU.method_table @inline _vote(::Type{Ballot}, mask, pred) = AMDGPU.vote_ballot(pred)