using Test
using KernelIntrinsics
using KernelAbstractions, CUDA
import KernelAbstractions: synchronize
# Test kernel for shuffle up

backend = CUDABackend()

@testset "warpsize" begin
    @kernel function test_warpsize(a)
        a[1] = @warpsize
    end

    a = CUDA.zeros(Int32, 1)
    test_warpsize(CUDABackend())(a; ndrange=1)
    synchronize(CUDABackend())
    @test CUDA.@allowscalar a[1] == Int32(32)
end


@kernel function shfl_up_kernel(dst, src, offset)
    I = @index(Global, Linear)
    val = src[I]
    shuffled_val = @shfl(Up, val, offset)
    dst[I] = shuffled_val
end

# Test kernel for shuffle down
@kernel function shfl_down_kernel(dst, src, offset)
    I = @index(Global, Linear)
    val = src[I]
    shuffled_val = @shfl(Down, val, offset)
    dst[I] = shuffled_val
end

# Test kernel for shuffle xor
@kernel function shfl_xor_kernel(dst, src, lane_mask)
    I = @index(Global, Linear)
    val = src[I]
    shuffled_val = @shfl(Xor, val, lane_mask)
    dst[I] = shuffled_val
end

# Test kernel for shuffle idx (direct lane access)
@kernel function shfl_idx_kernel(dst, src, lane)
    I = @index(Global, Linear)
    val = src[I]
    shuffled_val = @shfl(Idx, val, lane)
    dst[I] = shuffled_val
end

src = cu(Int32.(1:32))
dst = cu(zeros(Int32, 32))
shfl_up_kernel(backend)(dst, src, 1; ndrange=32)
synchronize(backend)
result = Array(dst)
# Lane 0 keeps its value, others get value from lane - offset
expected = Int32[1; 1:31...]
@test result == expected
@testset "KernelIntrinsics @shfl Tests" begin

    @testset "Shuffle Up - Int32" begin
        src = cu(Int32.(1:32))
        dst = cu(zeros(Int32, 32))
        shfl_up_kernel(backend)(dst, src, 1; ndrange=32)
        synchronize(backend)
        result = Array(dst)
        # Lane 0 keeps its value, others get value from lane - offset
        expected = Int32[1; 1:31...]
        @test result == expected
    end

    @testset "Shuffle Up - offset=4" begin
        src = cu(Int32.(1:32))
        dst = cu(zeros(Int32, 32))
        shfl_up_kernel(backend)(dst, src, 4; ndrange=32)
        synchronize(backend)
        result = Array(dst)
        # Lanes 0-3 keep their values, others get value from lane - 4
        expected = Int32[1, 2, 3, 4, 1:28...]
        @test result == expected
    end

    @testset "Shuffle Down - Int32" begin
        src = cu(Int32.(1:32))
        dst = cu(zeros(Int32, 32))
        shfl_down_kernel(backend)(dst, src, 1; ndrange=32)
        synchronize(backend)
        result = Array(dst)
        # Lane 31 keeps its value, others get value from lane + offset
        expected = Int32[2:32..., 32]
        @test result == expected
    end

    @testset "Shuffle Xor - Int32" begin
        src = cu(Int32.(0:31))  # Use 0-indexed values for clarity
        dst = cu(zeros(Int32, 32))
        shfl_xor_kernel(backend)(dst, src, 1; ndrange=32)
        synchronize(backend)
        result = Array(dst)
        # XOR with 1 swaps adjacent pairs: 0↔1, 2↔3, etc.
        expected = Int32[1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14,
            17, 16, 19, 18, 21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 31, 30]
        @test result == expected
    end

    @testset "Shuffle Xor - butterfly pattern" begin
        src = cu(Int32.(0:31))
        dst = cu(zeros(Int32, 32))
        shfl_xor_kernel(backend)(dst, src, 16; ndrange=32)
        synchronize(backend)
        result = Array(dst)
        # XOR with 16 swaps first half with second half
        expected = Int32[16:31..., 0:15...]
        @test result == expected
    end

    @testset "Shuffle Idx - broadcast lane 1" begin
        src = cu(Int32.(1:32))
        dst = cu(zeros(Int32, 32))
        shfl_idx_kernel(backend)(dst, src, 1; ndrange=32)
        synchronize(backend)
        result = Array(dst)
        # All lanes get the value from lane 0
        expected = fill(Int32(1), 32)
        @test result == expected
    end

    @testset "Shuffle Up - Float32" begin
        src = cu(Float32.(1:32))
        dst = cu(zeros(Float32, 32))
        shfl_up_kernel(backend)(dst, src, 1; ndrange=32)
        synchronize(backend)
        result = Array(dst)
        expected = Float32[1.0; 1:31...]
        @test result == expected
    end

    @testset "Shuffle Up - Float64" begin
        src = cu(Float64.(1:32))
        dst = cu(zeros(Float64, 32))
        shfl_up_kernel(backend)(dst, src, 1; ndrange=32)
        synchronize(backend)
        result = Array(dst)
        expected = Float64[1.0; 1:31...]
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
        src = cu([ComplexType(i, SubType(Float16(i), UInt8(i)), Float64(i)) for i in 1:32])
        dst = cu([ComplexType(0, SubType(Float16(0), UInt8(0)), 0.0) for _ in 1:32])
        shfl_up_kernel(backend)(dst, src, 1; ndrange=32)
        synchronize(backend)
        result = Array(dst)

        # Lane 0 keeps its value, others shift up
        @test result[1].x == 1
        @test result[1].y.a == Float16(1)
        @test result[1].y.b == UInt8(1)
        @test result[1].z == 1.0

        for i in 2:32
            @test result[i].x == i - 1
            @test result[i].y.a == Float16(i - 1)
            @test result[i].y.b == UInt8(i - 1)
            @test result[i].z == Float64(i - 1)
        end
    end

    @testset "Shuffle Down - Complex struct" begin
        src = cu([ComplexType(i, SubType(Float16(i), UInt8(i)), Float64(i)) for i in 1:32])
        dst = cu([ComplexType(0, SubType(Float16(0), UInt8(0)), 0.0) for _ in 1:32])
        shfl_down_kernel(backend)(dst, src, 1; ndrange=32)
        synchronize(backend)
        result = Array(dst)

        # Lane 31 keeps its value, others shift down
        for i in 1:31
            @test result[i].x == i + 1
            @test result[i].y.a == Float16(i + 1)
            @test result[i].y.b == UInt8(i + 1)
            @test result[i].z == Float64(i + 1)
        end
        @test result[32].x == 32
    end

    # Multiple warps test
    @testset "Shuffle Up - Multiple warps" begin
        src = cu(Int32.(1:64))
        dst = cu(zeros(Int32, 64))
        shfl_up_kernel(backend)(dst, src, 1; ndrange=64)
        synchronize(backend)
        result = Array(dst)

        # First warp
        @test result[1:32] == Int32[1; 1:31...]
        # Second warp (independent shuffle)
        @test result[33:64] == Int32[33; 33:63...]
    end

    @testset "Shuffle Xor - reduction pattern" begin
        # Simulate a warp-level reduction pattern
        src = cu(Int32.(1:32))
        dst = cu(zeros(Int32, 32))

        # XOR with 1: swap pairs
        shfl_xor_kernel(backend)(dst, src, 1; ndrange=32)
        synchronize(backend)
        result = Array(dst)

        # Verify the butterfly exchange happened
        for i in 0:15
            @test result[2i+1] == 2i + 2  # Even lanes get odd values
            @test result[2i+2] == 2i + 1  # Odd lanes get even values
        end
    end

    # Edge cases
    @testset "Shuffle Up - offset=0 (identity)" begin
        src = cu(Int32.(1:32))
        dst = cu(zeros(Int32, 32))
        shfl_up_kernel(backend)(dst, src, 0; ndrange=32)
        synchronize(backend)
        result = Array(dst)
        @test result == Int32.(1:32)
    end

    @testset "Shuffle Down - offset=0 (identity)" begin
        src = cu(Int32.(1:32))
        dst = cu(zeros(Int32, 32))
        shfl_down_kernel(backend)(dst, src, 0; ndrange=32)
        synchronize(backend)
        result = Array(dst)
        @test result == Int32.(1:32)
    end

    @testset "Shuffle Xor - mask=0 (identity)" begin
        src = cu(Int32.(1:32))
        dst = cu(zeros(Int32, 32))
        shfl_xor_kernel(backend)(dst, src, 0; ndrange=32)
        synchronize(backend)
        result = Array(dst)
        @test result == Int32.(1:32)
    end

    # 16-bit types
    @testset "Shuffle Up - Int16" begin
        src = cu(Int16.(1:32))
        dst = cu(zeros(Int16, 32))
        shfl_up_kernel(backend)(dst, src, 1; ndrange=32)
        synchronize(backend)
        result = Array(dst)
        expected = Int16[1; 1:31...]
        @test result == expected
    end

    @testset "Shuffle Up - Float16" begin
        src = cu(Float16.(1:32))
        dst = cu(zeros(Float16, 32))
        shfl_up_kernel(backend)(dst, src, 1; ndrange=32)
        synchronize(backend)
        result = Array(dst)
        expected = Float16[1.0; 1:31...]
        @test result == expected
    end

    # Tuple types
    @testset "Shuffle Up - NTuple{2, Int32}" begin
        @kernel function shfl_tuple_kernel(dst, src)
            I = @index(Global, Linear)
            val = src[I]
            shuffled_val = @shfl(Up, val, 1)
            dst[I] = shuffled_val
        end

        src = cu([(Int32(i), Int32(i + 100)) for i in 1:32])
        dst = cu([(Int32(0), Int32(0)) for _ in 1:32])
        shfl_tuple_kernel(backend)(dst, src; ndrange=32)
        synchronize(backend)
        result = Array(dst)

        @test result[1] == (Int32(1), Int32(101))
        for i in 2:32
            @test result[i] == (Int32(i - 1), Int32(i + 99))
        end
    end
end