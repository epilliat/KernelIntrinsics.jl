# KernelIntrinsics.jl

> ⚠️ **Warning**: This package provides low-level GPU primitives intended for library developers, not end users. If you're looking for high-level GPU programming in Julia, use [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) or [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) directly. KernelIntrinsics.jl is similar in scope to [GPUArraysCore.jl](https://github.com/JuliaGPU/GPUArrays.jl) — a building block for other packages.

> ⚠️ **Current Limitations**:
> - No support for array views (`SubArray`) yet
> - No bounds checking — out-of-bounds access will cause undefined behavior
> - CUDA-only (other backends planned)

A Julia package providing low-level memory access primitives and warp-level operations for GPU programming with KernelAbstractions.jl. KernelIntrinsics.jl enables fine-grained control over memory ordering, synchronization, and vectorized operations for high-performance GPU kernels.

## Cross-Architecture Support

Currently, KernelIntrinsics.jl is implemented exclusively for CUDA GPUs via the `CUDABackend`. The package leverages the CUDA.jl backend as well as CUDA-specific PTX instructions for memory fences (`fence.acq_rel.gpu`), ordered memory access (`ld.acquire.gpu`, `st.release.gpu`), warp shuffle operations, and vectorized memory transactions. While the current implementation is CUDA-specific, the macro-based API is designed with portability in mind. Future releases may extend support to other GPU backends (AMD ROCm, Intel oneAPI, Apple Metal) by adapting the underlying code generation to emit appropriate backend-specific instructions while maintaining the same high-level interface. Contributions to enable cross-platform support are welcome.

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
Pkg.add(url="https://github.com/YOURUSERNAME/KernelIntrinsics.jl")
```

## Usage

### Memory Fences

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

src = cu(Int32.(1:32))
dst = cu(zeros(Int32, 32))
shfl_kernel(CUDABackend())(dst, src; ndrange=32)
# dst = [1, 1, 2, 3, 4, ..., 31]
```

Unlike CUDA.jl, KernelIntrinsics.jl supports shuffle operations on **arbitrary user-defined bitstype structs**, including nested and composite types, as well as `NTuple`s:
```julia
# Nested structs
struct Sub
    a::Float16
    b::UInt8
end

struct ComplexType
    x::Int32
    y::Sub
    z::Float64
end

src = cu([ComplexType(i, Sub(i, i), i) for i in 1:32])
dst = cu([ComplexType(0, Sub(0, 0), 0) for i in 1:32])
shfl_kernel(CUDABackend())(dst, src; ndrange=32)

# NTuples
src = cu([(Int32(i), Int32(i + 100)) for i in 1:32])
dst = cu([(Int32(0), Int32(0)) for _ in 1:32])
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

src = cu(Int32.(1:32))
dst = cu(zeros(Int32, 32))
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

src = cu(Int32.(1:32))
dst = cu(zeros(Int32, 32))
warpfold_kernel(CUDABackend())(dst, src; ndrange=32)
# dst[1] = 528 (sum of 1:32), rest are undefined
```

### Vectorized Memory Access

KernelIntrinsics.jl provides several vectorized load/store functions that generate efficient PTX instructions (`ld.global.v4`, `st.global.v4`, etc.).

#### Basic Vectorized Access

Use `vload` and `vstore!` for aligned vectorized operations:
```julia
@kernel function vectorized_kernel(dst, src, i)
    # Load 4 elements with rebase (i=2 → loads from index 5,6,7,8)
    values = vload(src, i, Val(4), Val(true))
    
    # Store 4 elements with rebase
    vstore!(dst, i, values, Val(true))
end

src = cu(Int32.(1:32))
dst = cu(zeros(Int32, 32))
vectorized_kernel(CUDABackend())(dst, src, 2; ndrange=1)
# dst[5:8] = [5, 6, 7, 8]
```

#### Dynamic Alignment with `vload_multi` / `vstore_multi!`

When the starting index is not known at compile time, alignment cannot be guaranteed. `vload_multi` and `vstore_multi!` handle this by:

1. Computing `mod = (i - 1) % N + 1` at runtime (where `N` is the vector width)
2. Using a switch table to dispatch to the appropriate statically-compiled function with `Val(mod)`
3. Emitting a mix of vectorized instructions to maximize throughput
```julia
@kernel function dynamic_load_kernel(dst, src, i, ::Val{N}) where {N}
    # i can be any runtime value — alignment handled automatically
    values = vload_multi(src, i, Val(N))
    for j in 1:N
        dst[j] = values[j]
    end
end

src = cu(Int32.(1:100))
dst = cu(zeros(Int32, 16))

# Works for any starting index
dynamic_load_kernel(CUDABackend())(dst, src, 7, Val(16); ndrange=1)
# dst = [7, 8, 9, ..., 22]
```

The generated PTX will contain a mix of `ld.global.v4`, `ld.global.v2`, and scalar loads depending on the runtime alignment, maximizing memory throughput while handling arbitrary offsets.
```julia
@kernel function dynamic_store_kernel(dst, i)
    values = (Int32(10), Int32(20), Int32(30), Int32(40))
    vstore_multi!(dst, i, values)
end

dst = cu(zeros(Int32, 100))
dynamic_store_kernel(CUDABackend())(dst, 3; ndrange=1)
# dst[3:6] = [10, 20, 30, 40]
```

#### Pattern-Based Access

For custom access patterns, use `vload_pattern` and `vstore_pattern!`:
```julia
@kernel function pattern_kernel(dst, src, i)
    # Pattern (1, 2, 1) means: load 1, then 2, then 1 element
    values = vload_pattern(src, i, Val((1, 2, 1)))
    vstore_pattern!(dst, i, values, Val((1, 2, 1)))
end
```

## Inspecting Generated Code

You can verify the generated PTX assembly to confirm proper instruction generation:
```julia
buf = IOBuffer()
CUDA.@device_code_ptx io = buf fence_kernel(CUDABackend())(X, Flag; ndrange=1)
asm = String(take!(buf))
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
- `vload_multi`/`vstore_multi!` have a small runtime overhead for the alignment switch, but this is typically negligible compared to memory latency

## Requirements

- Julia 1.10+
- KernelAbstractions.jl
- CUDA.jl (for CUDA backend)

## License

MIT License