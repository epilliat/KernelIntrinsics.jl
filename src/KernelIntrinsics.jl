module KernelIntrinsics

using KernelAbstractions

export @warpsize

export @fence, @access
export vload, vstore!
export vload_multi, vstore_multi!

export @shfl, @warpreduce, @warpfold, @vote
#export atomic_store, atomic_load, fence
#export Workgroup, Device, System
#export Acquire, Release, AcqRel, SeqCst, Weak, Volatile, Relaxed

# ============================================================================
# Abstract type definitions for compile-time dispatch
# ============================================================================

function _warpsize end


include("helper.jl")
include("scopes_orderings.jl")
include("vectorization.jl")
include("warp.jl")
include("macros.jl")


end # module KernelIntrinsics
