using Test
using KernelIntrinsics
using KernelAbstractions
using CUDA

function count_substring(s, sub)
    count = 0
    i = 1
    while (i = findnext(sub, s, i)) !== nothing
        count += 1
        i += 1
    end
    count
end

@testset "KernelIntrinsics Vectorized Load/Store" begin

    @testset "vload" begin
        @testset "vload with rebase" begin
            @kernel function test_vload_rebase(a, b, i)
                y = KernelIntrinsics.vload(a, i, Val(4))
                b[1] = sum(y)
            end

            a = CuArray{Int32}(1:16)
            b = CUDA.zeros(Int32, 4)
            test_vload_rebase(CUDABackend())(a, b, 2; ndrange=1)
            synchronize(CUDABackend())
            @test CUDA.@allowscalar b[1] == 26  # sum of elements 5,6,7,8

            buf = IOBuffer()
            CUDA.@device_code_ptx io = buf test_vload_rebase(CUDABackend())(a, b, 2; ndrange=1)
            asm = String(take!(buf))
            @test occursin("ld.global.v4", asm)
        end

        @testset "vload without rebase" begin
            @kernel function test_vload_norebase(a, b, i)
                x = vload(a, i + 1, Val(4), Val(false))
                b[1] = sum(x)
            end

            a = CuArray{Int32}(1:16)
            b = CUDA.zeros(Int32, 4)
            test_vload_norebase(CUDABackend())(a, b, 2; ndrange=1)
            synchronize(CUDABackend())
            @test CUDA.@allowscalar b[1] == 18  # sum of elements 3,4,5,6
        end

        @testset "vload_pattern" begin
            @kernel function test_vload_pattern(a, b, i)
                values = KernelIntrinsics.vload_pattern(a, i, Val((1, 2, 1)))
                b[1] = sum(sum(values))
            end

            a = CuArray{Int32}(1:16)
            b = CUDA.zeros(Int32, 1)
            test_vload_pattern(CUDABackend())(a, b, 2; ndrange=1)
            synchronize(CUDABackend())

            buf = IOBuffer()
            CUDA.@device_code_ptx io = buf test_vload_pattern(CUDABackend())(a, b, 2; ndrange=1)
            asm = String(take!(buf))
            @test occursin("ld.global.v2", asm)
        end

        @testset "vload_multi" begin
            @kernel function test_vload_multi(a, b, i, ::Val{N}) where {N}
                values = KernelIntrinsics.vload_multi(a, i, Val(N))
                for j in 1:N
                    b[j] = values[j]
                end
            end

            # Test PTX generation
            a = CuArray{Int32}(1:100)
            b = CUDA.zeros(Int32, 8)
            buf = IOBuffer()
            CUDA.@device_code_ptx io = buf test_vload_multi(CUDABackend())(a, b, 2, Val(4); ndrange=1)
            asm = String(take!(buf))
            @test occursin("ld.global.v4", asm)

            # Test correctness for various alignments
            N = 16
            for shift in 1:16
                a = CuArray{Int32}(1:100)
                b = CUDA.zeros(Int32, N)
                test_vload_multi(CUDABackend())(a, b, shift, Val(N); ndrange=1)
                synchronize(CUDABackend())
                @test Array(b) == Int32.(shift:shift+N-1)
            end
        end
    end

    @testset "vstore" begin
        @testset "vstore with rebase" begin
            @kernel function test_vstore_rebase(b, i)
                values = (Int32(10), Int32(20), Int32(30), Int32(40))
                KernelIntrinsics.vstore!(b, i, values, Val(true))
            end

            b = CUDA.zeros(Int32, 16)
            test_vstore_rebase(CUDABackend())(b, 2; ndrange=1)
            synchronize(CUDABackend())
            @test CUDA.@allowscalar (b[5], b[6], b[7], b[8]) == (10, 20, 30, 40)

            buf = IOBuffer()
            CUDA.@device_code_ptx io = buf test_vstore_rebase(CUDABackend())(b, 2; ndrange=1)
            asm = String(take!(buf))
            @test occursin("st.global.v4", asm)
        end

        @testset "vstore without rebase" begin
            @kernel function test_vstore_norebase(b, i)
                values = (Int32(100), Int32(200), Int32(300), Int32(400))
                vstore!(b, i + 1, values, Val(false))
            end

            b = CUDA.zeros(Int32, 16)
            test_vstore_norebase(CUDABackend())(b, 2; ndrange=1)
            synchronize(CUDABackend())
            @test CUDA.@allowscalar (b[3], b[4], b[5], b[6]) == (100, 200, 300, 400)
        end

        @testset "vstore_pattern" begin
            @kernel function test_vstore_pattern(b, i)
                values = (Int32(1), Int32(2), Int32(3), Int32(4))
                KernelIntrinsics.vstore_pattern!(b, i, values, Val((1, 2, 1)))
            end

            b = CUDA.zeros(Int32, 16)
            test_vstore_pattern(CUDABackend())(b, 2; ndrange=1)
            synchronize(CUDABackend())
            @test CUDA.@allowscalar (b[2], b[3], b[4], b[5]) == (1, 2, 3, 4)

            buf = IOBuffer()
            CUDA.@device_code_ptx io = buf test_vstore_pattern(CUDABackend())(b, 2; ndrange=1)
            asm = String(take!(buf))
            @test occursin("st.global.v2", asm)
        end

        @testset "vstore_multi" begin
            @kernel function test_vstore_multi(b, i)
                values = (Int32(10), Int32(20), Int32(30), Int32(40))
                KernelIntrinsics.vstore_multi!(b, i, values)
            end

            b = CUDA.zeros(Int32, 100)
            test_vstore_multi(CUDABackend())(b, 2; ndrange=1)
            synchronize(CUDABackend())
            @test CUDA.@allowscalar (b[2], b[3], b[4], b[5]) == (10, 20, 30, 40)

            buf = IOBuffer()
            CUDA.@device_code_ptx io = buf test_vstore_multi(CUDABackend())(b, 2; ndrange=1)
            asm = String(take!(buf))
            @test occursin("st.global.v4", asm)
        end
    end

    @testset "Round-trip vload/vstore" begin
        @kernel function test_roundtrip(a, b, i)
            values = KernelIntrinsics.vload(a, i, Val(4), Val(true))
            KernelIntrinsics.vstore!(b, i, values, Val(true))
        end

        a = CuArray{Int32}(1:16)
        b = CUDA.zeros(Int32, 16)
        test_roundtrip(CUDABackend())(a, b, 2; ndrange=1)
        synchronize(CUDABackend())
        @test CUDA.@allowscalar (b[5], b[6], b[7], b[8]) == (5, 6, 7, 8)
    end
end