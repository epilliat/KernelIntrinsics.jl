module KernelIntrinsicsCUDAExt

using CUDA
using CUDA: LLVMPtr, AS, CUDABackend
using KernelAbstractions
const KA = KernelAbstractions
using LLVM
using LLVM.Interop: @asmcall, @typed_ccall


import KernelIntrinsics: _warpsize, _laneid
# Import parent module and types


CUDA.@device_override @inline function _warpsize() # used inside kernels
    return 32
end

Base.Experimental.@overlay CUDA.method_table @inline function _laneid()
    return CUDA.laneid()
end

include("CUDA/device.jl")
include("CUDA/scopes_ordering.jl")
include("CUDA/shuffle_vote.jl")
include("CUDA/vectorization.jl")
include("CUDA/sleep.jl")
include("CUDA/dynlocalmem.jl")

end # module KernelIntrinsicsCUDAExt
