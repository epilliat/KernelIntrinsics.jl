# KernelIntrinsics.jl

> ⚠️ **Warning**: This package provides low-level GPU primitives intended for library developers, not end users. If you're looking for high-level GPU programming in Julia, use [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) or [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) directly. KernelIntrinsics.jl is similar in scope to [GPUArraysCore.jl](https://github.com/JuliaGPU/GPUArrays.jl) — a building block for other packages.

> ⚠️ **Current Limitations**:
> - No support for array views (`SubArray`) yet
> - No bounds checking — out-of-bounds access will cause undefined behavior
> - CUDA-only (other backends planned)

A Julia package providing low-level memory access primitives and warp-level operations for GPU programming with KernelAbstractions.jl. KernelIntrinsics.jl enables fine-grained control over memory ordering, synchronization, and vectorized operations for high-performance GPU kernels.

## Features

- **Memory Fences**: Explicit memory synchronization with `@fence` macro
- **Ordered Memory Access**: Acquire/release semantics with `@access` macro
- **Warp Operations**: Efficient intra-warp communication and reduction primitives
  - `@shfl`: Warp shuffle operations (Up, Down, Xor, Idx modes)
  - `@warpreduce`: Inclusive scan within a warp
  - `@warpfold`: Warp-wide reduction to a single value
- **Vectorized Memory Operations**: Hardware-accelerated vector loads and stores with `vload`, `vstore!`, `vload_multi`, and `vstore_multi!`

## Cross-Architecture Support

Currently, KernelIntrinsics.jl is implemented exclusively for CUDA GPUs via the `CUDABackend`. The package leverages CUDA-specific PTX instructions for memory fences (`fence.acq_rel.gpu`), ordered memory access (`ld.acquire.gpu`, `st.release.gpu`), warp shuffle operations, and vectorized memory transactions.

While the current implementation is CUDA-specific, the macro-based API is designed with portability in mind. Future releases may extend support to other GPU backends (AMD ROCm, Intel oneAPI, Apple Metal). Contributions to enable cross-platform support are welcome.

## Installation
```julia
using Pkg
Pkg.add(url="https://github.com/YOURUSERNAME/KernelIntrinsics.jl")
```

## Quick Start
```julia
using KernelIntrinsics
using KernelAbstractions, CUDA

@kernel function example_kernel(X, Flag)
    X[1] = 10
    @fence  # Ensure X[1]=10 is visible to all threads
    @access Flag[1] = 1  # Release store
end

X = cu([1])
Flag = cu([0])
example_kernel(CUDABackend())(X, Flag; ndrange=1)
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

## Implementation Notes

### Memory Ordering and Scopes

The implementation of memory fences, orderings, and scopes is inspired by [UnsafeAtomics.jl](https://github.com/JuliaConcurrent/UnsafeAtomics.jl). This package includes tests demonstrating that these primitives work correctly on CUDA, generating the expected PTX instructions (`fence.acq_rel.gpu`, `ld.acquire.gpu`, `st.release.gpu`, etc.).

### Warp Shuffle Operations

The warp shuffle implementation builds upon [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)'s approach but generalizes it to support **any concrete bitstype struct**, including nested and composite types, as well as `NTuple`s. The backend only needs to implement 32-bit shuffle operations; larger types are automatically decomposed. Future backends that do not natively support 32-bit atomic operations may require additional handling.

### Vectorized Memory Access

Vectorized loads and stores (`vload`, `vstore!`, `vload_multi`, `vstore_multi!`) use LLVM intrinsic functions to generate efficient vector instructions (`ld.global.v4`, `st.global.v4`).

**Current limitations:**
- Arrays must be contiguous in memory
- Views (`SubArray`) are not supported
- No bounds checking
- Future versions may add fallback paths for non-contiguous access

## Requirements

- Julia 1.10+
- KernelAbstractions.jl
- CUDA.jl (for CUDA backend)

## Contents
```@contents
Pages = ["api.md", "examples.md"]
Depth = 2
```

## License

MIT License