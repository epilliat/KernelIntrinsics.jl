using Test
using KernelAbstractions
using KernelIntrinsics
using CUDA

## Helper function to count number of vectorize loads in asm

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

## Tests 

@testset "CUDA" begin
    @testset "access and fence" begin
        include("cuda/access_fences.jl")
        include("cuda/shfl.jl")
        include("cuda/vectorization_test.jl")
    end
end