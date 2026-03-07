module KernelIntrinsics

using KernelAbstractions

export @warpsize, @laneid

export @fence, @access
export vload, vstore!

export @shfl, @warpreduce, @warpfold, @vote
#export atomic_store, atomic_load, fence
#export Workgroup, Device, System
#export Acquire, Release, AcqRel, SeqCst, Weak, Volatile, Relaxed

# ============================================================================
# Abstract type definitions for compile-time dispatch
# ============================================================================

function get_warpsize end # outside kernels
function _warpsize end # inside kernels
function _laneid end

include("macros.jl")
include("helper.jl")

include("device.jl")
include("scopes_orderings.jl")
include("vectorization.jl")
include("warp.jl")


end # module KernelIntrinsics
