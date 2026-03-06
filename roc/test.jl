using Pkg
Pkg.activate("roc")
using Revise


using KernelAbstractions
using KernelIntrinsics
import KernelIntrinsics as KI
using AMDGPU

using BenchmarkTools

using AMDGPU

src = ROCArray{Float32}(undef, 10)
@btime KI.device(src)
@btime KI.get_warpsize(KI.device($src))

