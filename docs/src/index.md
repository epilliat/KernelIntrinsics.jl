# KernelIntrinsics.jl

!!! warning "Not affiliated with submodule KernelIntrinsics of KernelAbstractions.jl"
    Despite its name, this package is **not affiliated with or part of [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)**. The name collision is a known issue. A future goal is to upstream its functionality into KernelAbstractions.jl and [UnsafeAtomics.jl](https://github.com/JuliaConcurrent/UnsafeAtomics.jl).

!!! warning "Low-level library — not for end users"
    This package provides low-level GPU primitives intended for library developers, not end users. If you're looking for high-level GPU programming in Julia, use [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) or [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) directly. KernelIntrinsics.jl is similar in scope to [GPUArraysCore.jl](https://github.com/JuliaGPU/GPUArrays.jl) — a building block for other packages.

!!! danger "Current Limitations"
    - No bounds checking for `vload`/`vstore!` on dense arrays — out-of-bounds access will cause undefined behavior
    - CUDA, ROCm, and Metal backends supported; oneAPI planned

## Features

- **Memory Fences**: Explicit memory synchronization with `@fence` macro
- **Ordered Memory Access**: Acquire/release semantics with `@access` macro
- **Warp Operations**: Efficient intra-warp communication and reduction primitives
  - `@shfl`: Warp shuffle operations (Up, Down, Xor, Idx modes)
  - `@warpreduce`: Inclusive scan within a warp
  - `@warpfold`: Warp-wide reduction to a single value
- **Spin-loop backoff**: `@sleep n` emits a hardware sleep hint (`s_sleep` on AMD, `nanosleep` on CUDA sm_70+, no-op elsewhere) so busy-wait/spin loops back off instead of hammering the memory subsystem
- **Vectorized Memory Operations**: Hardware-accelerated vector loads and stores with `vload` and `vstore!`. Two indexing modes are available: *rebased* (default) where `idx` selects a contiguous block of `Nitem` elements (`(idx-1)*Nitem+1 … idx*Nitem`), and *direct* where `idx` is the literal starting index. Array views (`SubArray`) are supported and fall back to scalar tuple operations automatically.

## Cross-Architecture Support

KernelIntrinsics.jl currently supports three backends:

- **CUDA** via `CUDABackend` — PTX instructions for fences (`fence.acq_rel.gpu`), ordered access (`ld.acquire.gpu`, `st.release.gpu`), warp shuffles, and vectorized loads/stores.
- **ROCm** via `ROCBackend` — equivalent primitives using AMDGPU-specific intrinsics.
- **Metal** via `MetalBackend` — Apple GPU support contributed by [@WilliBee](https://github.com/WilliBee) (PR [#5](https://github.com/epilliat/KernelIntrinsics.jl/pull/5)). Maps to AIR intrinsics (`air.atomic.fence`, `air.simd_*`); subject to Metal's narrower atomic type set (`Int32`/`UInt32`/`Float32`).

The macro-based API is designed with portability in mind. Future releases may extend support to additional GPU backends (Intel oneAPI). Contributions are welcome.

## Installation
```julia
using Pkg
Pkg.add("KernelIntrinsics")
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
- Default warp size is 32 for CUDA and 64 for ROCm; operations assume full warp/wavefront participation
- When `vload`/`vstore!` cannot prove pointer alignment at compile time, they emit a small runtime alignment dispatch — negligible compared to memory latency, but providing the `Alignment` argument removes it
- `vload`/`vstore!` on array views fall back to scalar tuple operations; performance-critical code should prefer contiguous arrays for vectorized instructions

## Implementation Notes

### Memory Ordering and Scopes

The implementation of memory fences, orderings, and scopes is inspired by [UnsafeAtomics.jl](https://github.com/JuliaConcurrent/UnsafeAtomics.jl). This package includes tests demonstrating that these primitives work correctly on CUDA and ROCm, generating the expected backend-specific instructions.

### Warp Shuffle Operations

The warp shuffle implementation builds upon [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)'s approach but generalizes it to support **any concrete bitstype struct**, including nested and composite types, as well as `NTuple`s. The backend only needs to implement 32-bit shuffle operations; larger types are automatically decomposed. On ROCm, wavefront-level shuffle operations are used as the equivalent primitive.

### Vectorized Memory Access

Vectorized loads and stores (`vload`, `vstore!`) use LLVM intrinsic functions to generate efficient vector instructions (`ld.global.v4`, `st.global.v4` on CUDA, and their ROCm/Metal equivalents) on contiguous arrays. Two indexing modes are provided:

- **Rebased** (`Val(true)`, default): `idx` is a *block index* — `vload(A, i, Val(N))` loads `A[(i-1)*N+1 : i*N]`. This is the form to reach for when you tile data into fixed-size blocks per thread; when the array's base pointer is `N`-aligned, it lowers to a single aligned vector load.
- **Direct** (`Val(false)`): `idx` is the literal starting index — `vload(A, i, Val(N), Val(false))` loads `A[i : i+N-1]`. Use this when the start position is data-dependent and not a multiple of `N`.

Both modes handle unknown alignment at runtime via an internal pattern-based dispatch, so correctness is preserved without the user managing alignment manually. Array views (`SubArray`) are fully supported via an automatic fallback to scalar tuple operations, ensuring correctness at the cost of vectorization.

**Current limitations:**
- No bounds checking for `vload`/`vstore!` on dense arrays — out-of-bounds access will cause undefined behavior
- Future versions may add vectorized support for non-contiguous views

## Requirements

- Julia 1.10+
- KernelAbstractions.jl
- CUDA.jl (for CUDA backend)
- AMDGPU.jl (for ROCm backend)
- Metal.jl (for Metal backend)

## Contents
```@contents
Pages = ["api.md", "examples.md"]
Depth = 2
```

## Sponsors

KernelIntrinsics.jl is an open-source project maintained on personal time. If it
is useful to you — especially in a production or HPC setting — you can support
its development via [GitHub Sponsors](https://github.com/sponsors/epilliat).

## License

MIT License