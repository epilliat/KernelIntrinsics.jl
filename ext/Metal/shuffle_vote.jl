import KernelIntrinsics: Up, Down, Xor, Idx
import KernelIntrinsics: All, AnyLane, Uni, Ballot, MatchAny
import KernelIntrinsics: _shfl, _vote, _match

const SHFL_DISPATCH = Dict(
    Up => :(Metal.simd_shuffle_up),
    Down => :(Metal.simd_shuffle_down)
)

for T in (:Float32, :Int32, :UInt32)
    for (direction, metal_fname) in SHFL_DISPATCH
        @eval begin
            Base.Experimental.@overlay Metal.method_table @inline _shfl(::Type{$direction}, mask, val::$T, src) =
                $metal_fname(val, src)
        end
    end
end

# =======================================================================================================
# These intrinsics are now exposed as julia function as of https://github.com/JuliaGPU/Metal.jl/pull/744
# Will update when they are part of official release
# =======================================================================================================
simd_shuffle_map = ((Float32, "f32"),
                    (Int32,   "s.i32"),
                    (UInt32,  "u.i32"))

for (jltype, suffix) in simd_shuffle_map
    @eval begin
        Base.Experimental.@overlay Metal.method_table @inline _shfl(::Type{Idx}, mask, val::$jltype, src) =
            ccall($"extern air.simd_shuffle.$suffix",
                llvmcall, $jltype, ($jltype, Int16), val, src - 0x1)

        Base.Experimental.@overlay Metal.method_table @inline _shfl(::Type{Xor}, mask, val::$jltype, src) =
            ccall($"extern air.simd_shuffle_xor.$suffix",
                llvmcall, $jltype, ($jltype, Int16), val, src)
    end
end

Base.Experimental.@overlay Metal.method_table @inline function _vote(::Type{Ballot}, mask, pred)
    # Direct LLVM call to air.simd_ballot
    ccall("extern air.simd_ballot.i64", llvmcall, UInt64, (Bool,), pred)
end

Base.Experimental.@overlay Metal.method_table @inline function _vote(::Type{All}, mask, pred)
    ballot_bits = _vote(Ballot, mask, pred)
    # Direct LLVM call to air.simd_vote_all
    ccall("extern air.simd_vote_all.i64", llvmcall, Bool, (UInt64,), ballot_bits)
end

Base.Experimental.@overlay Metal.method_table @inline function _vote(::Type{AnyLane}, mask, pred)
    ballot_bits = _vote(Ballot, mask, pred)
    # Direct LLVM call to air.simd_vote_any
    ccall("extern air.simd_vote_any.i64", llvmcall, Bool, (UInt64,), ballot_bits)
end

Base.Experimental.@overlay Metal.method_table @inline function _vote(::Type{Uni}, mask, pred)
    bits = ccall("extern air.simd_ballot.i64", llvmcall, UInt64, (Bool,), pred)
    active = ccall("extern air.simd_ballot.i64", llvmcall, UInt64, (Bool,), true)
    # Uniform: all active lanes same value (all true OR all false)
    return (bits == active) || (bits == UInt64(0))
end

# The portable polyfill in src/warp.jl can't be used here: its `@generated`
# body calls `_vote(Ballot, …)`, but Base.Experimental.@overlay overrides
# don't propagate into the polyfill body during inference, leaving _match
# dynamically dispatched (same root cause as the @match failure on AMDGPU).
# Inline the polyfill body directly into Metal's overlay using the
# LLVM ballot intrinsic, so type inference stays inside the overlay context.

for T in (UInt8, UInt16, UInt32, UInt64)
    Nbits = 8 * sizeof(T)
    body = Expr(:block,
        :(active = ccall("extern air.simd_ballot.i64", llvmcall, UInt64, (Bool,), true)),
        :(result = active),
    )
    for b in 1:Nbits
        push!(body.args, quote
            let bit_b = ((value >> $(b - 1)) & one($T)) != zero($T)
                ballot_b = ccall("extern air.simd_ballot.i64", llvmcall, UInt64, (Bool,), bit_b)
                result &= bit_b ? ballot_b : (active & ~ballot_b)
            end
        end)
    end
    push!(body.args, :(return result))
    @eval Base.Experimental.@overlay Metal.method_table @inline _match(::Type{MatchAny}, mask, value::$T) = $body
end