import KernelIntrinsics: Up, Down, Xor, Idx
import KernelIntrinsics: All, AnyLane, Uni, Ballot

import KernelIntrinsics: _shfl, _vote


const SHFL_DISPATCH = Dict(
    Up => :shfl_up_sync,
    Down => :shfl_down_sync,
    Xor => :shfl_xor_sync,
    Idx => :shfl_sync
)

for T in (Int32, UInt32, Float32)
    for (direction, cuda_fname) in SHFL_DISPATCH
        @eval begin
            CUDA.@device_override @inline _shfl(::Type{$direction}, mask, val::$T, src, ::Val{ws}) where ws =
                $cuda_fname(mask, val, src, ws)
        end
    end
end

const VOTE_DISPATCH = Dict(
    All => :vote_all_sync,
    AnyLane => :vote_any_sync,
    Uni => :vote_uni_sync,
    Ballot => :vote_ballot_sync
)

for (ModeType, cuda_fname) in VOTE_DISPATCH
    @eval begin
        CUDA.@device_override @inline _vote(::Type{$ModeType}, mask, pred) = $cuda_fname(mask, pred)
    end
end