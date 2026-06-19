# KernelIntrinsics.jl
> ⚠️ **Warning**: This package provides low-level GPU primitives intended for library developers, not end users. If you're looking for high-level GPU programming in Julia, use [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) or [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) directly.

A Julia package providing low-level memory access primitives and warp-level operations for GPU programming with KernelAbstractions.jl:

- **Memory Fences** and **ordered memory access** (`@fence`, `@access`) with acquire/release semantics
- **Warp operations**: shuffle (`@shfl`), inclusive scan (`@warpreduce`), reduction (`@warpfold`), vote (`@vote`), match (`@match`)
- **Spin-loop backoff** (`@sleep`): hardware sleep hint (`s_sleep` on AMD, `nanosleep` on CUDA, no-op fallback) for busy-wait loops
- **Vectorized memory operations** (`vload`, `vstore!`) generating `ld.global.v4`/`st.global.v4` PTX instructions

Currently supports CUDA, ROCm and Metal backends. Other backends planned.

## Installation

```julia
using Pkg
Pkg.add("KernelIntrinsics")
```

## Quick start

```julia
using KernelIntrinsics
using KernelAbstractions, CUDA   # or AMDGPU / Metal

@kernel function example_kernel(X, Flag)
    X[1] = 10
    @fence              # make X[1]=10 visible to all threads
    @access Flag[1] = 1 # release store
end

X    = cu([1])
Flag = cu([0])
example_kernel(CUDABackend())(X, Flag; ndrange=1)
```

## Documentation

Full documentation available at: https://epilliat.github.io/KernelIntrinsics.jl/stable/

## Sponsors

KernelIntrinsics.jl is an open-source project maintained in my personal time.
If this package is useful to you — especially in a production or HPC setting —
you can support its development and maintenance via
[GitHub Sponsors](https://github.com/sponsors/epilliat).

Corporate sponsors receive priority support on issues and an acknowledgment
in the documentation.

## License

MIT License
