using Pkg
Pkg.activate("roc")
using Revise


using KernelAbstractions
using KernelIntrinsics
import KernelIntrinsics as KI
using AMDGPU

using BenchmarkTools

using AMDGPU
# vote_wavefront.jl
using KernelAbstractions
using AMDGPU
using KernelAbstractions
using AMDGPU
import AMDGPU.Device: ballot, activemask

using KernelAbstractions
using AMDGPU
import AMDGPU.Device: ballot


using KernelAbstractions
using CUDA
using KernelAbstractions


@kernel function test_vote_all!(out)
    i = @index(Local, Linear)
    lane = @laneid
    #out[i] = KI._vote(KI.All, 0xffffffff, true)#KI._vote_all_from_shfl(true)#
    out[i] = @vote(All, true)#KI._vote_all_from_shfl(true)#
end

@kernel function test_vote_all_false!(out)
    i = @index(Local, Linear)
    out[i] = @vote(All, isodd(i))
end

@kernel function test_vote_any!(out)
    i = @index(Local, Linear)
    out[i] = @vote(AnyLane, false)
end

@kernel function test_vote_any_true!(out)
    i = @index(Local, Linear)
    out[i] = @vote(AnyLane, i == 1)
end

backend = CUDABackend()
out = CuArray{Bool}(undef, 64)

test_vote_all!(backend, 64)(out, ndrange=64)
CUDA.synchronize()
@show Array(out)[1]  # expected: true

test_vote_all_false!(backend, 64)(out, ndrange=64)
CUDA.synchronize()
@show Array(out)[1]  # expected: false

test_vote_any!(backend, 64)(out, ndrange=64)
AMDGPU.synchronize()
@show Array(out)[1]  # expected: false

test_vote_any_true!(backend, 64)(out, ndrange=64)
AMDGPU.synchronize()
@show Array(out)[1]  # expected: true