# KernelIntrinsics.jl
> ⚠️ **Warning**: This package provides low-level GPU primitives intended for library developers, not end users. If you're looking for high-level GPU programming in Julia, use [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) or [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) directly.

A Julia package providing low-level memory access primitives and warp-level operations for GPU programming with KernelAbstractions.jl:

- **Memory Fences** and **ordered memory access** (`@fence`, `@access`) with acquire/release semantics
- **Warp operations**: shuffle (`@shfl`), inclusive scan (`@warpreduce`), reduction (`@warpfold`), vote (`@vote`)
- **Spin-loop backoff** (`@sleep`): hardware sleep hint (`s_sleep` on AMD, `nanosleep` on CUDA, no-op fallback) for busy-wait loops
- **Vectorized memory operations** (`vload`, `vstore!`, `vload_multi`, `vstore_multi!`) generating `ld.global.v4`/`st.global.v4` PTX instructions

Currently supports CUDA, ROCm and Metal backends. Other backends planned.

## Installation

```julia
using Pkg
Pkg.add("KernelIntrinsics.jl")
```

## Documentation

Full documentation available at: https://epilliat.github.io/KernelIntrinsics.jl/stable/

## License

MIT License
