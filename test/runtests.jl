# ─────────────────────────────────────────────────────────────────────────────
# Backend selection
# ─────────────────────────────────────────────────────────────────────────────
using Pkg
include("meta_helpers.jl")

TEST_BACKEND = get(ENV, "TEST_BACKEND") do
    backend_str = has_cuda() ? "cuda" : has_roc() ? "roc" : "unknown"
    @info "TEST_BACKEND not set, defaulting to $backend_str"
    backend_str
end


#Pkg.activate("test/envs/$TEST_BACKEND")
Pkg.activate("envs/$TEST_BACKEND") # when running tests
Pkg.instantiate()


function count_substring(haystack::AbstractString, needle::AbstractString)
    count = 0
    start = 1
    while true
        r = findnext(needle, haystack, start)
        r === nothing && break
        count += 1
        start = last(r) + 1
    end
    return count
end


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
else
    error("Unknown backend: $TEST_BACKEND")
end