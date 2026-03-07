
struct SubType
    a::Float16
    b::UInt8
end

struct ComplexType
    x::Int32
    y::SubType
    z::Float64
end

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

# Wrap host array into the target device array type
to_device(x) = AT(x)

# Retrieve results back to host
from_device(x) = Array(x)
warpsz = KI.get_warpsize(KI.device(backend))


function launch(kernel, args...; ndrange)
    kernel(backend)(args...; ndrange=ndrange)
    synchronize(backend)
end

# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

@testset "KernelIntrinsics" begin

    # ── @warpsize ─────────────────────────────────────────────────────────────
    @testset "@warpsize" begin
        @kernel function test_warpsize(a)
            a[1] = @warpsize
        end

        a = AT(zeros(Int32, 1))
        test_warpsize(backend)(a; ndrange=1)
        synchronize(backend)
        @test Array(a)[1] == warpsz
    end

    # ── @shfl ─────────────────────────────────────────────────────────────────
    @testset "@shfl" begin
        @kernel function shfl_up_kernel(dst, src, offset)
            I = @index(Global, Linear)
            dst[I] = @shfl(Up, src[I], offset)
        end

        @kernel function shfl_down_kernel(dst, src, offset)
            I = @index(Global, Linear)
            dst[I] = @shfl(Down, src[I], offset)
        end

        @kernel function shfl_xor_kernel(dst, src, lane_mask)
            I = @index(Global, Linear)
            dst[I] = @shfl(Xor, src[I], lane_mask)
        end

        @kernel function shfl_idx_kernel(dst, src, lane)
            I = @index(Global, Linear)
            dst[I] = @shfl(Idx, src[I], lane)
        end

        @testset "Up Int32" begin
            src = to_device(Int32.(1:warpsz))
            dst = to_device(zeros(Int32, warpsz))
            launch(shfl_up_kernel, dst, src, 1; ndrange=warpsz)
            @test from_device(dst) == Int32[1; 1:warpsz-1...]
        end

        @testset "Down Int32" begin
            src = to_device(Int32.(1:warpsz))
            dst = to_device(zeros(Int32, warpsz))
            launch(shfl_down_kernel, dst, src, 1; ndrange=warpsz)
            @test from_device(dst) == Int32[2:warpsz..., warpsz]
        end

        @testset "Idx Int32" begin
            src = to_device(Int32.(1:warpsz))
            dst = to_device(zeros(Int32, warpsz))
            launch(shfl_idx_kernel, dst, src, 1; ndrange=warpsz)
            @test from_device(dst) == fill(Int32(1), warpsz)
        end

        @testset "Up Float32" begin
            src = to_device(Float32.(1:warpsz))
            dst = to_device(zeros(Float32, warpsz))
            launch(shfl_up_kernel, dst, src, 1; ndrange=warpsz)
            @test from_device(dst) == Float32[1.0; 1:warpsz-1...]
        end

        @testset "Up Float64" begin
            src = to_device(Float64.(1:warpsz))
            dst = to_device(zeros(Float64, warpsz))
            launch(shfl_up_kernel, dst, src, 1; ndrange=warpsz)
            @test from_device(dst) == Float64[1.0; 1:warpsz-1...]
        end

        @testset "Up Float16" begin
            src = to_device(Float16.(1:warpsz))
            dst = to_device(zeros(Float16, warpsz))
            launch(shfl_up_kernel, dst, src, 1; ndrange=warpsz)
            @test from_device(dst) == Float16[1.0; 1:warpsz-1...]
        end

        @testset "Up NTuple{2,Int32}" begin
            @kernel function shfl_tuple_kernel(dst, src)
                I = @index(Global, Linear)
                dst[I] = @shfl(Up, src[I], 1)
            end
            src = to_device([(Int32(i), Int32(i + 100)) for i in 1:warpsz])
            dst = to_device([(Int32(0), Int32(0)) for _ in 1:warpsz])
            launch(shfl_tuple_kernel, dst, src; ndrange=warpsz)
            result = from_device(dst)
            @test result[1] == (Int32(1), Int32(101))
            for i in 2:warpsz
                @test result[i] == (Int32(i - 1), Int32(i + 99))
            end
        end

        @testset "Up ComplexType" begin
            src = to_device([ComplexType(i, SubType(Float16(i), UInt8(i)), Float64(i)) for i in 1:warpsz])
            dst = to_device([ComplexType(0, SubType(Float16(0), UInt8(0)), 0.0) for _ in 1:warpsz])
            launch(shfl_up_kernel, dst, src, 1; ndrange=warpsz)
            result = from_device(dst)
            @test result[1] == ComplexType(1, SubType(Float16(1), UInt8(1)), 1.0)
            for i in 2:warpsz
                @test result[i] == ComplexType(i - 1, SubType(Float16(i - 1), UInt8(i - 1)), Float64(i - 1))
            end
        end

    end  # @shfl

    # ── @warpreduce ───────────────────────────────────────────────────────────
    @testset "@warpreduce" begin

        @kernel function kernel_warpreduce_sum(dst, src)
            I = @index(Global, Linear)
            val = src[I]
            @warpreduce(val, +)
            dst[I] = val
        end

        @testset "Int32 prefix sum" begin
            src = to_device(Int32.(1:warpsz))
            dst = to_device(zeros(Int32, warpsz))
            launch(kernel_warpreduce_sum, dst, src; ndrange=warpsz)
            @test from_device(dst) == Int32.(cumsum(1:warpsz))
        end

        @testset "Float32 prefix sum" begin
            data = Float32.(1:warpsz)
            src = to_device(data)
            dst = to_device(zeros(Float32, warpsz))
            launch(kernel_warpreduce_sum, dst, src; ndrange=warpsz)
            @test from_device(dst) ≈ Float32.(cumsum(data))
        end

    end  # @warpreduce

    # ── @warpfold ─────────────────────────────────────────────────────────────
    @testset "@warpfold" begin

        @kernel function kernel_warpfold_sum(dst, src)
            I = @index(Global, Linear)
            val = src[I]
            @warpfold(val, +)
            dst[I] = val
        end

        @testset "Int32 sum — all lanes hold warp total" begin
            src = to_device(Int32.(1:warpsz))
            dst = to_device(zeros(Int32, warpsz))
            launch(kernel_warpfold_sum, dst, src; ndrange=warpsz)
            @test from_device(dst)[1] == Int32(sum(1:warpsz))
        end

    end  # @warpfold

    # ── @vote ─────────────────────────────────────────────────────────────────
    @testset "@vote" begin

        @kernel function kernel_vote_all(dst, src, threshold)
            I = @index(Global, Linear)
            dst[I] = @vote(All, src[I] > threshold)
        end

        @kernel function kernel_vote_any(dst, src, threshold)
            I = @index(Global, Linear)
            dst[I] = @vote(AnyLane, src[I] > threshold)
        end

        @kernel function kernel_vote_uni(dst, src, threshold)
            I = @index(Global, Linear)
            dst[I] = @vote(Uni, src[I] > threshold)
        end

        @kernel function kernel_vote_ballot(dst, src, threshold)
            I = @index(Global, Linear)
            dst[I] = @vote(Ballot, src[I] > threshold)
        end

        @testset "All — true when every lane satisfies predicate" begin
            src = to_device(fill(Int32(10), warpsz))
            dst = to_device(zeros(Bool, warpsz))
            launch(kernel_vote_all, dst, src, Int32(5); ndrange=warpsz)
            @test all(from_device(dst))
        end

        @testset "AnyLane — true when at least one lane satisfies predicate" begin
            data = fill(Int32(0), warpsz)
            data[end] = Int32(200)
            src = to_device(data)
            dst = to_device(zeros(Bool, warpsz))
            launch(kernel_vote_any, dst, src, Int32(100); ndrange=warpsz)
            @test all(from_device(dst))
        end

        @testset "Uni — true when predicate is uniform" begin
            src = to_device(fill(Int32(10), warpsz))
            dst = to_device(zeros(Bool, warpsz))
            launch(kernel_vote_uni, dst, src, Int32(5); ndrange=warpsz)
            @test all(from_device(dst))
        end

        @testset "Ballot — all bits set when all lanes satisfy predicate" begin
            src = to_device(fill(Int32(10), warpsz))
            dst = to_device(zeros(UInt64, warpsz))
            launch(kernel_vote_ballot, dst, src, Int32(5); ndrange=warpsz)
            @test all(from_device(dst) .== UInt64((1 << warpsz) - 1))
        end

    end  # @vote

end  # KernelIntrinsics