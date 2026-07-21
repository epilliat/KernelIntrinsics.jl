module KernelIntrinsicsCUDAExt

using CUDA
using CUDA: LLVMPtr, AS, CUDABackend
using KernelAbstractions
const KA = KernelAbstractions
using LLVM
using LLVM.Interop: @asmcall, @typed_ccall


import KernelIntrinsics
import KernelIntrinsics: _warpsize, _laneid
import KernelIntrinsics.MMA: _mma_hw, NVIDIATC
# Import parent module and types


CUDA.@device_override @inline function _warpsize() # used inside kernels
    return 32
end

Base.Experimental.@overlay CUDA.method_table @inline function _laneid()
    return CUDA.laneid()
end

# Jeton matériel MMA — le SEUL overlay du chemin MMA (cf. src/mma.jl, section
# « Jeton matériel »). Il renvoie un singleton : aucune donnée, donc rien à quoi
# un `undef` puisse s'accrocher, contrairement aux fonctions qui produisaient des
# fragments. Device-only : côté hôte `_mma_hw()` reste `NoHW()` ⇒ fallback.
Base.Experimental.@overlay CUDA.method_table @inline function _mma_hw()
    return NVIDIATC()
end

include("CUDA/device.jl")
include("CUDA/scopes_ordering.jl")
include("CUDA/shuffle_vote.jl")
include("CUDA/vectorization.jl")
include("CUDA/sleep.jl")
include("CUDA/dynlocalmem.jl")
include("CUDA/mma.jl")

end # module KernelIntrinsicsCUDAExt
