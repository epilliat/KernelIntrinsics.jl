using Pkg
Pkg.activate("roc")
using Revise


using KernelAbstractions
using KernelIntrinsics
import KernelIntrinsics as KI
using AMDGPU


using AMDGPU

@kernel function trunc_test_kernel(dst, src)
    I = @index(Global, Linear)
    x = src[I]
    dst[I] = x % UInt8  # should fail compilation
end

# Force compilation to AMDGPU IR without a device
AMDGPU.code_llvm(trunc_test_kernel(ROCBackend()), Tuple{ROCArray{UInt8,1},ROCArray{UInt32,1}})