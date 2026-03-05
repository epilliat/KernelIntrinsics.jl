using Test
using KernelIntrinsics
using KernelAbstractions, AMDGPU
import KernelAbstractions: synchronize

backend = ROCBackend()

@testset "warpsize" begin
    @kernel function test_warpsize(a)
        a[1] = @warpsize
    end

    a = AMDGPU.zeros(Int32, 1)
    test_warpsize(ROCBackend())(a; ndrange=1)
    synchronize(ROCBackend())
    @test AMDGPU.@allowscalar a[1] == Int32(64)  # MI300X wavefront size is 64
end


@kernel function shfl_up_kernel(dst, src, offset)
    I = @index(Global, Linear)
    val = src[I]
    shuffled_val = @shfl(Up, val, offset)
    dst[I] = shuffled_val
end

@kernel function shfl_down_kernel(dst, src, offset)
    I = @index(Global, Linear)
    val = src[I]
    shuffled_val = @shfl(Down, val, offset)
    dst[I] = shuffled_val
end

@kernel function shfl_xor_kernel(dst, src, lane_mask)
    I = @index(Global, Linear)
    val = src[I]
    shuffled_val = @shfl(Xor, val, lane_mask)
    dst[I] = shuffled_val
end

@kernel function shfl_idx_kernel(dst, src, lane)
    I = @index(Global, Linear)
    val = src[I]
    shuffled_val = @shfl(Idx, val, lane)
    dst[I] = shuffled_val
end

const W = 64  # wavefront size on MI300X

@testset "KernelIntrinsics @shfl Tests" begin

    @testset "Shuffle Up - Int32" begin
        src = ROCArray(Int32.(1:W))
        dst = AMDGPU.zeros(Int32, W)
        shfl_up_kernel(backend)(dst, src, 1; ndrange=W)
        synchronize(backend)
        result = Array(dst)
        expected = Int32[1; 1:W-1...]
        @test result == expected
    end

    @testset "Shuffle Up - offset=4" begin
        src = ROCArray(Int32.(1:W))
        dst = AMDGPU.zeros(Int32, W)
        shfl_up_kernel(backend)(dst, src, 4; ndrange=W)
        synchronize(backend)
        result = Array(dst)
        expected = Int32[1, 2, 3, 4, 1:W-4...]
        @test result == expected
    end

    @testset "Shuffle Down - Int32" begin
        src = ROCArray(Int32.(1:W))
        dst = AMDGPU.zeros(Int32, W)
        shfl_down_kernel(backend)(dst, src, 1; ndrange=W)
        synchronize(backend)
        result = Array(dst)
        expected = Int32[2:W..., W]
        @test result == expected
    end

    @testset "Shuffle Xor - Int32" begin
        src = ROCArray(Int32.(0:W-1))
        dst = AMDGPU.zeros(Int32, W)
        shfl_xor_kernel(backend)(dst, src, 1; ndrange=W)
        synchronize(backend)
        result = Array(dst)
        # XOR with 1 swaps adjacent pairs
        expected = Int32[xor(i, 1) for i in 0:W-1]
        @test result == expected
    end

    @testset "Shuffle Xor - butterfly pattern" begin
        src = ROCArray(Int32.(0:W-1))
        dst = AMDGPU.zeros(Int32, W)
        shfl_xor_kernel(backend)(dst, src, 32; ndrange=W)
        synchronize(backend)
        result = Array(dst)
        # XOR with 32 swaps first half with second half
        expected = Int32[32:63..., 0:31...]
        @test result == expected
    end

    @testset "Shuffle Idx - broadcast lane 1" begin
        src = ROCArray(Int32.(1:W))
        dst = AMDGPU.zeros(Int32, W)
        shfl_idx_kernel(backend)(dst, src, 1; ndrange=W)
        synchronize(backend)
        result = Array(dst)
        expected = fill(Int32(1), W)
        @test result == expected
    end

    @testset "Shuffle Up - Float32" begin
        src = ROCArray(Float32.(1:W))
        dst = AMDGPU.zeros(Float32, W)
        shfl_up_kernel(backend)(dst, src, 1; ndrange=W)
        synchronize(backend)
        result = Array(dst)
        expected = Float32[1.0; 1:W-1...]
        @test result == expected
    end

    @testset "Shuffle Up - Float64" begin
        src = ROCArray(Float64.(1:W))
        dst = AMDGPU.zeros(Float64, W)
        shfl_up_kernel(backend)(dst, src, 1; ndrange=W)
        synchronize(backend)
        result = Array(dst)
        expected = Float64[1.0; 1:W-1...]
        @test result == expected
    end

    # Complex struct tests
    struct SubType
        a::Float16
        b::UInt8
    end

    struct ComplexType
        x::Int32
        y::SubType
        z::Float64
    end

    @testset "Shuffle Up - Complex struct" begin
        src = ROCArray([ComplexType(i, SubType(Float16(i), UInt8(i)), Float64(i)) for i in 1:W])
        dst = ROCArray([ComplexType(0, SubType(Float16(0), UInt8(0)), 0.0) for _ in 1:W])
        shfl_up_kernel(backend)(dst, src, 1; ndrange=W)
        synchronize(backend)
        result = Array(dst)

        @test result[1].x == 1
        @test result[1].y.a == Float16(1)
        @test result[1].y.b == UInt8(1)
        @test result[1].z == 1.0

        for i in 2:W
            @test result[i].x == i - 1
            @test result[i].y.a == Float16(i - 1)
            @test result[i].y.b == UInt8(i - 1)
            @test result[i].z == Float64(i - 1)
        end
    end

    @testset "Shuffle Down - Complex struct" begin
        src = ROCArray([ComplexType(i, SubType(Float16(i), UInt8(i)), Float64(i)) for i in 1:W])
        dst = ROCArray([ComplexType(0, SubType(Float16(0), UInt8(0)), 0.0) for _ in 1:W])
        shfl_down_kernel(backend)(dst, src, 1; ndrange=W)
        synchronize(backend)
        result = Array(dst)

        for i in 1:W-1
            @test result[i].x == i + 1
            @test result[i].y.a == Float16(i + 1)
            @test result[i].y.b == UInt8(i + 1)
            @test result[i].z == Float64(i + 1)
        end
        @test result[W].x == W
    end

    # Multiple wavefronts test
    @testset "Shuffle Up - Multiple wavefronts" begin
        src = ROCArray(Int32.(1:2W))
        dst = AMDGPU.zeros(Int32, 2W)
        shfl_up_kernel(backend)(dst, src, 1; ndrange=2W)
        synchronize(backend)
        result = Array(dst)

        # First wavefront
        @test result[1:W] == Int32[1; 1:W-1...]
        # Second wavefront (independent shuffle)
        @test result[W+1:2W] == Int32[W + 1; W+1:2W-1...]
    end

    @testset "Shuffle Xor - reduction pattern" begin
        src = ROCArray(Int32.(1:W))
        dst = AMDGPU.zeros(Int32, W)

        shfl_xor_kernel(backend)(dst, src, 1; ndrange=W)
        synchronize(backend)
        result = Array(dst)

        for i in 0:W÷2-1
            @test result[2i+1] == 2i + 2
            @test result[2i+2] == 2i + 1
        end
    end

    @testset "Shuffle Up - offset=0 (identity)" begin
        src = ROCArray(Int32.(1:W))
        dst = AMDGPU.zeros(Int32, W)
        shfl_up_kernel(backend)(dst, src, 0; ndrange=W)
        synchronize(backend)
        result = Array(dst)
        @test result == Int32.(1:W)
    end

    @testset "Shuffle Down - offset=0 (identity)" begin
        src = ROCArray(Int32.(1:W))
        dst = AMDGPU.zeros(Int32, W)
        shfl_down_kernel(backend)(dst, src, 0; ndrange=W)
        synchronize(backend)
        result = Array(dst)
        @test result == Int32.(1:W)
    end

    @testset "Shuffle Xor - mask=0 (identity)" begin
        src = ROCArray(Int32.(1:W))
        dst = AMDGPU.zeros(Int32, W)
        shfl_xor_kernel(backend)(dst, src, 0; ndrange=W)
        synchronize(backend)
        result = Array(dst)
        @test result == Int32.(1:W)
    end

    @testset "Shuffle Up - Int16" begin
        src = ROCArray(Int16.(1:W))
        dst = AMDGPU.zeros(Int16, W)
        shfl_up_kernel(backend)(dst, src, 1; ndrange=W)
        synchronize(backend)
        result = Array(dst)
        expected = Int16[1; 1:W-1...]
        @test result == expected
    end

    @testset "Shuffle Up - Float16" begin
        src = ROCArray(Float16.(1:W))
        dst = AMDGPU.zeros(Float16, W)
        shfl_up_kernel(backend)(dst, src, 1; ndrange=W)
        synchronize(backend)
        result = Array(dst)
        expected = Float16[1.0; 1:W-1...]
        @test result == expected
    end

    @testset "Shuffle Up - NTuple{2, Int32}" begin
        @kernel function shfl_tuple_kernel(dst, src)
            I = @index(Global, Linear)
            val = src[I]
            shuffled_val = @shfl(Up, val, 1)
            dst[I] = shuffled_val
        end

        src = ROCArray([(Int32(i), Int32(i + 100)) for i in 1:W])
        dst = ROCArray([(Int32(0), Int32(0)) for _ in 1:W])
        shfl_tuple_kernel(backend)(dst, src; ndrange=W)
        synchronize(backend)
        result = Array(dst)

        @test result[1] == (Int32(1), Int32(101))
        for i in 2:W
            @test result[i] == (Int32(i - 1), Int32(i + 99))
        end
    end
end

@testset "warpreduce with defaults" begin
    @kernel function test_warpreduce_defaults(dst, src)
        I = @index(Global, Linear)
        val = src[I]
        lane = (I - 1) % W + 1

        @warpreduce(val, lane, +)

        dst[I] = val
    end

    src = ROCArray(Int32.(1:W))
    dst = AMDGPU.zeros(Int32, W)
    test_warpreduce_defaults(ROCBackend())(dst, src; ndrange=W)
    synchronize(ROCBackend())

    result = Array(dst)
    expected = accumulate(+, 1:W)
    @test result == Int32.(expected)
end