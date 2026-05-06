# ─────────────────────────────────────────────────────────────────────────────
# Backend selection
# ─────────────────────────────────────────────────────────────────────────────
using Pkg

# Pre-env probes: must work before any backend Pkg is loaded. The runtime check
# (CUDA.functional() / AMDGPU.functional() / Metal.functional()) below is the
# authoritative one — these probes are only used to pick a default backend when
# TEST_BACKEND is unset.
_has_cuda()  = Sys.which("nvidia-smi") !== nothing
_has_roc()   = Sys.which("rocm-smi")   !== nothing
_has_metal() = Sys.isapple()

TEST_BACKEND = get(ENV, "TEST_BACKEND") do
    backend_str = _has_cuda() ? "cuda" : _has_roc() ? "roc" : _has_metal() ? "metal" : "unknown"
    @info "TEST_BACKEND not set, defaulting to $backend_str"
    backend_str
end

Pkg.activate(joinpath(@__DIR__, "envs", TEST_BACKEND))
Pkg.instantiate()


using KernelIntrinsics
import KernelIntrinsics as KI
using KernelAbstractions
import KernelAbstractions: synchronize, get_backend
using Test


if TEST_BACKEND == "cuda"
    using CUDA
    if !CUDA.functional()
        @warn "No CUDA device found — skipping tests"
        exit(0)
    end
    AT = CuArray
    backend = CUDABackend()
    include("general_routine.jl")
elseif TEST_BACKEND == "roc"
    using AMDGPU
    if !AMDGPU.functional()
        @warn "No AMDGPU device found — skipping tests"
        exit(0)
    end
    AT = ROCArray
    backend = ROCBackend()
    include("general_routine.jl")
elseif TEST_BACKEND == "metal"
    using Metal
    if !Metal.functional()
        @warn "No Metal device found — skipping tests"
        exit(0)
    end
    AT = MtlArray
    backend = MetalBackend()
    include("general_routine.jl")
else
    error("Unknown backend: $TEST_BACKEND")
end
