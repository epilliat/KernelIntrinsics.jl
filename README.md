# KernelIntrinsics.jl

> ⚠️ **Warning**: This package provides low-level GPU primitives intended for library developers, not end users. If you're looking for high-level GPU programming in Julia, use [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) or [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) directly.

A Julia package providing low-level memory access primitives and warp-level operations for GPU programming with KernelAbstractions.jl:

- **Memory Fences** and **ordered memory access** (`@fence`, `@access`) with acquire/release semantics
- **Warp operations**: shuffle (`@shfl`), inclusive scan (`@warpreduce`), reduction (`@warpfold`), vote (`@vote`)
- **Vectorized memory operations** (`vload`, `vstore!`, `vload_multi`, `vstore_multi!`) generating `ld.global.v4`/`st.global.v4` PTX instructions

Currently CUDA-only. Other backends planned.

## Installation
```julia
using Pkg
Pkg.add(url="https://github.com/epilliat/KernelIntrinsics.jl")
```

## Documentation

Full documentation available at: https://epilliat.github.io/KernelIntrinsics.jl/stable/

## License

MIT License