using Test
using KernelAbstractions
using MemoryAccess
using CUDA


@testset "CUDA" begin
    @testset "access and fence" begin
        include("cuda/access_fences.jl")
    end
end