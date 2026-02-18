using KernelIntrinsics

using KernelAbstractions, CUDA

@kernel function vectorized_kernel(dst, src)
    values = vectorized_load(src, 2, Val(4))
    vectorized_store!(dst, 2, values)
end

src = cu([Int32(i) for i in (1:32)])
dst = cu([Int32(0) for i in (1:32)])

vectorized_kernel(CUDABackend())(dst, src; ndrange=1)
@show dst # [0,0,0,0, 5,6,7,8, 0, 0...]

buf = IOBuffer()
@device_code_ptx io = buf vectorized_kernel(CUDABackend())(dst, src; ndrange=1)
asm = String(take!(copy(buf)))

occursin("st.global.v4", asm) #true, would be .v2 for Int or Float64 instead of Int32
occursin("ld.global.v4", asm) #true
