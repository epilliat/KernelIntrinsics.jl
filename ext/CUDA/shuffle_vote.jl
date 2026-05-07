import KernelIntrinsics: Up, Down, Xor, Idx
import KernelIntrinsics: All, AnyLane, Uni, Ballot
import KernelIntrinsics: MatchAny
import KernelIntrinsics: _shfl, _vote, _match

const CUDA_SHFL_DISPATCH = Dict(
    Up => :shfl_up_sync,
    Down => :shfl_down_sync,
    Xor => :shfl_xor_sync,
    Idx => :shfl_sync
)

for T in (Int32, UInt32, Float32)
    for (direction, cuda_fname) in CUDA_SHFL_DISPATCH
        @eval begin
            CUDA.@device_override @inline _shfl(::Type{$direction}, mask, val::$T, src) =
                $cuda_fname(mask, val, src, _warpsize())
            #CUDA.@device_override @inline _shfl(::Type{$direction}, mask, val::$T, src, ::Val{ws}) where {ws} =
            #    $cuda_fname(mask, val, src, ws)
        end
    end
end

const CUDA_VOTE_DISPATCH = Dict(
    All => :vote_all_sync,
    AnyLane => :vote_any_sync,
    Uni => :vote_uni_sync,
    Ballot => :vote_ballot_sync
)

for (ModeType, cuda_fname) in CUDA_VOTE_DISPATCH
    @eval begin
        CUDA.@device_override @inline _vote(::Type{$ModeType}, mask, pred) = $cuda_fname(mask, pred)
    end
end


# ── Match (sm_70+) ────────────────────────────────────────────────────────────
# CUDA.jl doesn't expose match.any.sync; reach in via the LLVM intrinsic.
# Returns UInt32 (CUDA's lane mask width), matching @vote(Ballot, _) on CUDA.
# UInt8 / UInt16 / UInt32 share the .i32 PTX intrinsic (HW promotes narrower
# values); UInt64 has its own .i64 variant.

for T in (UInt8, UInt16, UInt32)
    @eval CUDA.@device_override @inline _match(::Type{MatchAny}, mask, value::$T) =
        @typed_ccall("llvm.nvvm.match.any.sync.i32", llvmcall, UInt32,
                     (UInt32, UInt32), mask, UInt32(value))
end

CUDA.@device_override @inline _match(::Type{MatchAny}, mask, value::UInt64) =
    @typed_ccall("llvm.nvvm.match.any.sync.i64", llvmcall, UInt32,
                 (UInt32, UInt64), mask, value)