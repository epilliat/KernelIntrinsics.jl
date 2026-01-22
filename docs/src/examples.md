# Examples

## Memory Fences

Use `@fence` to ensure memory operations are visible across threads before proceeding:
```julia
using KernelIntrinsics
using KernelAbstractions, CUDA

@kernel function fence_kernel(X, Flag)
    X[1] = 10
    @fence  # Ensure X[1]=10 is visible to all threads before next operations
    Flag[1] = 1
end

X = cu([1])
Flag = cu([0])
fence_kernel(CUDABackend())(X, Flag; ndrange=1)
```

The `@fence` macro generates `fence.acq_rel.gpu` instructions in PTX assembly, ensuring proper memory ordering across the GPU.

## Ordered Memory Access

The `@access` macro provides acquire/release semantics for fine-grained memory ordering:
```julia
@kernel function access_kernel(X, Flag)
    if @index(Global, Linear) == 1
        X[1] = 10
        @access Flag[1] = 1  # Release store
    end
    
    # Other threads wait for Flag[1] == 1
    while (@access Acquire Flag[1]) != 1  # Acquire load
    end
    
    # Safely use X[1] here
end

X = cu([i for i in 1:1000])
Flag = cu([0])
access_kernel(CUDABackend())(X, Flag; ndrange=1000)
```

This generates `st.release.gpu` and `ld.acquire.gpu` instructions, providing lock-free synchronization patterns.

## Warp Operations

### Shuffle Operations

Exchange values between threads within a warp:
```julia
@kernel function shfl_kernel(dst, src)
    I = @index(Global, Linear)
    val = src[I]
    offset = 1
    shuffled_val = @shfl(Up, val, offset)  # Default: warpsize=32, full mask
    dst[I] = shuffled_val
end

src = cu([i for i in 1:32])
dst = cu([0 for i in 1:32])
shfl_kernel(CUDABackend())(dst, src; ndrange=32)
# dst = [1, 1, 2, 3, 4, ..., 31]
```

### Shuffle with Custom Types

Unlike CUDA.jl, KernelIntrinsics.jl supports shuffle operations on arbitrary user-defined bitstype structs, including nested and composite types:
```julia
struct Sub
    a::Float16
    b::UInt8
end

struct ComplexType
    x::Int32
    y::Sub
    z::Float64
end

@kernel function shfl_custom_kernel(dst, src)
    I = @index(Global, Linear)
    val = src[I]
    offset = 1
    shuffled_val = @shfl(Up, val, offset)
    dst[I] = shuffled_val
end

src = cu([ComplexType(i, Sub(i, i), i) for i in 1:32])
dst = cu([ComplexType(0, Sub(0, 0), 0) for i in 1:32])
shfl_custom_kernel(CUDABackend())(dst, src; ndrange=32)
```

### Warp Reduce (Inclusive Scan)

Perform inclusive prefix sum within a warp:
```julia
@kernel function warpreduce_kernel(dst, src)
    I = @index(Global, Linear)
    val = src[I]
    lane = (I - 1) % 32 + 1
    @warpreduce(val, lane, +)
    dst[I] = val
end

src = cu([i for i in 1:32])
dst = cu([0 for i in 1:32])
warpreduce_kernel(CUDABackend())(dst, src; ndrange=32)
# dst = [1, 3, 6, 10, ..., 528]  (cumulative sum)
```

### Warp Fold (Reduction)

Reduce all values in a warp to a single result:
```julia
@kernel function warpfold_kernel(dst, src)
    I = @index(Global, Linear)
    val = src[I]
    lane = (I - 1) % 32 + 1
    @warpfold(val, lane, +)
    dst[I] = val
end

src = cu([i for i in 1:32])
dst = cu([0 for i in 1:32])
warpfold_kernel(CUDABackend())(dst, src; ndrange=32)
# dst[1] = 528 (sum of 1:32), rest are undefined
```

## Vectorized Memory Access

Use vectorized loads and stores for improved memory bandwidth:
```julia
@kernel function vectorized_kernel(dst, src)
    values = vload(src, 2, Val(4))  # Load 4 elements starting at index 2
    vstore!(dst, 2, values)         # Store 4 elements starting at index 2
end

src = cu([Int32(i) for i in 1:32])
dst = cu([Int32(0) for i in 1:32])
vectorized_kernel(CUDABackend())(dst, src; ndrange=1)
# dst = [0, 0, 0, 0, 5, 6, 7, 8, 0, 0, ...]
```

This generates efficient `ld.global.v4` and `st.global.v4` PTX instructions. The vector width depends on element type (v4 for Int32/Float32, v2 for Int64/Float64).

## Inspecting Generated Code

You can verify the generated PTX assembly to confirm proper instruction generation:
```julia
buf = IOBuffer()
@device_code_ptx io = buf fence_kernel(CUDABackend())(X, Flag; ndrange=1)
asm = String(take!(copy(buf)))
occursin("fence.acq_rel.gpu", asm)  # true
```