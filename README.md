# MemoryAccess.jl

A Julia package providing low-level memory access primitives and warp-level operations for GPU programming with KernelAbstractions.jl. MemoryAccess.jl enables fine-grained control over memory ordering, synchronization, and vectorized operations for high-performance GPU kernels.

## Cross-Architecture Support

Currently, MemoryAccess.jl is implemented exclusively for CUDA GPUs via the `CUDABackend`. The package leverages the CUDA.jl backend as well as CUDA-specific PTX instructions for memory fences (`fence.acq_rel.gpu`), ordered memory access (`ld.acquire.gpu`, `st.release.gpu`), warp shuffle operations, and vectorized memory transactions. While the current implementation is CUDA-specific, the macro-based API is designed with portability in mind. Future releases may extend support to other GPU backends (AMD ROCm, Intel oneAPI, Apple Metal) by adapting the underlying code generation to emit appropriate backend-specific instructions while maintaining the same high-level interface. Contributions to enable cross-platform support are welcome.

## Features

- **Memory Fences**: Explicit memory synchronization with `@fence` macro
- **Ordered Memory Access**: Acquire/release semantics with `@access` macro
- **Warp Operations**: Efficient intra-warp communication and reduction primitives
  - `@shfl`: Warp shuffle operations (Up, Down, Xor, Idx modes)
  - `@warpreduce`: Inclusive scan within a warp
  - `@warpfold`: Warp-wide reduction to a single value
- **Vectorized Memory Operations**: Hardware-accelerated vector loads and stores

## Installation
```julia
using Pkg
Pkg.add("MemoryAccess")
```

## Usage

### Memory Fences

Use `@fence` to ensure memory operations are visible across threads before proceeding:
```julia
using MemoryAccess
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

### Ordered Memory Access

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

### Warp Operations

#### Shuffle Operations

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

Unlike CUDA.jl, MemoryAccess.jl supports shuffle operations on **arbitrary user-defined bitstype structs**, including nested and composite types:
```julia
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
```

#### Warp Reduce (Inclusive Scan)

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
# dst = [1, 3, 6, 10, ..., 528]  # Cumulative sum
```

#### Warp Fold (Reduction)

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

### Vectorized Memory Access

Use vectorized loads and stores for improved memory bandwidth:
```julia
@kernel function vectorized_kernel(dst, src)
    values = vectorized_load(src, 2, Val(4))  # Load 4 elements starting at index 2
    vectorized_store!(dst, 2, values)          # Store 4 elements starting at index 2
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

## Memory Ordering Semantics

- **@fence**: Full acquire-release fence across all device threads
- **@access Release**: Ensures prior writes are visible before the store
- **@access Acquire**: Ensures subsequent reads see prior writes
- **@access Acquire Device**: Explicitly specifies device scope (default)

## Performance Considerations

- Warp operations are most efficient when all threads in a warp participate
- Vectorized operations can significantly improve memory bandwidth utilization
- Use the minimum required memory ordering (acquire/release over fences when possible)
- Default warp size is 32; operations assume full warp participation with mask `0xffffffff`

## Requirements

- Julia 1.6+
- KernelAbstractions.jl
- CUDA.jl (for CUDA backend examples)

## License

MIT License