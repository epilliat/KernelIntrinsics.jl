using KernelIntrinsics

using KernelAbstractions, CUDA

@kernel function fence_kernel(X, Flag)
    X[1] = 10
    @fence #make sure that X[i]=10 is visible for all threads among device before next reads and writes.
    Flag[1] = 1
end

X = cu([1])
Flag = cu([0])

fence_kernel(CUDABackend())(X, Flag; ndrange=1)

buf = IOBuffer()
@device_code_ptx io = buf fence_kernel(CUDABackend())(X, Flag; ndrange=1)
asm = String(take!(copy(buf)))
occursin("fence.acq_rel.gpu", asm) #true


@kernel function access_kernel(X, Flag)

    if @index(Global, Linear) == 1
        X[1] = 10
        @access Flag[1] = 1 # Release store, equivalent to @access Acquire Device Flag[1]
    end

    # For example, other threads might wait for Flag[1] == 1:

    while (@access Acquire Flag[1]) != 1 # Acquire load, equivalent to @access Acquire Device Flag[1]
    end

    # Do things ...
end

X = cu([i for i in (1:1000)])
Flag = cu([0])

fence_kernel(CUDABackend())(X, Flag; ndrange=1000)

buf = IOBuffer()
@device_code_ptx io = buf access_kernel(CUDABackend())(X, Flag; ndrange=1000)
asm = String(take!(copy(buf)))
occursin("st.release.gpu", asm) #true
occursin("ld.acquire.gpu", asm) #true
