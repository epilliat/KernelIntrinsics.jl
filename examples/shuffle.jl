using Pkg
Pkg.activate("local")
using Revise
using KernelIntrinsics

using KernelAbstractions, CUDA

@kernel function shfl_kernel(dst, src)
    I = @index(Global, Linear)
    val = src[I]
    offset = 1
    shuffled_val = @shfl(Up, val, offset) # By default, warpsize=32 and we take full mask. @shfl(Up, val, offset, 32, 0xffffffff)
    dst[I] = shuffled_val
end

src = cu([i for i in (1:32)])
dst = cu([0 for i in (1:32)])

shfl_kernel(CUDABackend())(dst, src; ndrange=32)
dst # [1, 1, 2, 3, 4 ..., 31]


struct sub
    a::Float16
    b::UInt8
end
struct complex_type
    x::Int32
    y::sub
    z::Float64
end
src = cu([complex_type(i, sub(i, i), i) for i in (1:32)])
dst = cu([complex_type(0, sub(0, 0), 0) for i in (1:32)])
shfl_kernel(CUDABackend())(dst, src; ndrange=32)

#%%
@kernel function warpreduce_kernel(dst, src)
    I = @index(Global, Linear)
    val = src[I]
    lane = (I - 1) % 32 + 1
    @warpreduce(val, lane, +) # By default, warpsize=32 and we take full mask. @shfl(Up, val, offset, 32, 0xffffffff)
    dst[I] = val
end

src = cu([i for i in (1:32)])
dst = cu([0 for i in (1:32)])

warpreduce_kernel(CUDABackend())(dst, src; ndrange=32)
dst # 1, 3, 6, ..., 528

##%
@kernel function warpfold_kernel(dst, src)
    I = @index(Global, Linear)
    val = src[I]
    lane = (I - 1) % 32 + 1
    @warpfold(val, lane, +) # By default, warpsize=32 and we take full mask. @shfl(Up, val, offset, 32, 0xffffffff)
    dst[I] = val
end

src = cu([i for i in (1:32)])
dst = cu([0 for i in (1:32)])

warpfold_kernel(CUDABackend())(dst, src; ndrange=32)
dst # 528, garbage ...