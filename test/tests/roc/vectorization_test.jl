using Test
using KernelIntrinsics
using KernelAbstractions
using AMDGPU

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

            a = ROCArray{Int32}(1:16)
            b = AMDGPU.zeros(Int32, 4)
            test_vload_rebase(ROCBackend())(a, b, 2; ndrange=1)
            synchronize(ROCBackend())
            @test AMDGPU.@allowscalar b[1] == 26  # sum of elements 5,6,7,8

            buf = IOBuffer()
            AMDGPU.@device_code_gcn io = buf test_vload_rebase(ROCBackend())(a, b, 2; ndrange=1)
            asm = String(take!(buf))
            @test occursin("global_load_dwordx4", asm)
        end

        @testset "vload without rebase" begin
            @kernel function test_vload_norebase(a, b, i)
                x = vload(a, i + 1, Val(4), Val(false))
                b[1] = sum(x)
            end

            a = ROCArray{Int32}(1:16)
            b = AMDGPU.zeros(Int32, 4)
            test_vload_norebase(ROCBackend())(a, b, 2; ndrange=1)
            synchronize(ROCBackend())
            @test AMDGPU.@allowscalar b[1] == 18  # sum of elements 3,4,5,6
        end

        @testset "vload_pattern" begin
            @kernel function test_vload_pattern(a, b, i)
                values = KernelIntrinsics.vload_pattern(a, i, Val((1, 2, 1)))
                b[1] = sum(sum(values))
            end

            a = ROCArray{Int32}(1:16)
            b = AMDGPU.zeros(Int32, 1)
            test_vload_pattern(ROCBackend())(a, b, 2; ndrange=1)
            synchronize(ROCBackend())

            buf = IOBuffer()
            AMDGPU.@device_code_gcn io = buf test_vload_pattern(ROCBackend())(a, b, 2; ndrange=1)
            asm = String(take!(buf))
            @test occursin("global_load_dwordx2", asm)
        end

        @testset "vload_multi" begin
            @kernel function test_vload_multi(a, b, i, ::Val{N}) where {N}
                values = KernelIntrinsics.vload_multi(a, i, Val(N))
                for j in 1:N
                    b[j] = values[j]
                end
            end

            # Test GCN generation
            a = ROCArray{Int32}(1:100)
            b = AMDGPU.zeros(Int32, 8)
            buf = IOBuffer()
            AMDGPU.@device_code_gcn io = buf test_vload_multi(ROCBackend())(a, b, 2, Val(4); ndrange=1)
            asm = String(take!(buf))
            @test occursin("global_load_dwordx4", asm)

            # Test correctness for various alignments
            N = 16
            for shift in 1:16
                a = ROCArray{Int32}(1:100)
                b = AMDGPU.zeros(Int32, N)
                test_vload_multi(ROCBackend())(a, b, shift, Val(N); ndrange=1)
                synchronize(ROCBackend())
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

            b = AMDGPU.zeros(Int32, 16)
            test_vstore_rebase(ROCBackend())(b, 2; ndrange=1)
            synchronize(ROCBackend())
            @test AMDGPU.@allowscalar (b[5], b[6], b[7], b[8]) == (10, 20, 30, 40)

            buf = IOBuffer()
            AMDGPU.@device_code_gcn io = buf test_vstore_rebase(ROCBackend())(b, 2; ndrange=1)
            asm = String(take!(buf))
            @test occursin("global_store_dwordx4", asm)
        end

        @testset "vstore without rebase" begin
            @kernel function test_vstore_norebase(b, i)
                values = (Int32(100), Int32(200), Int32(300), Int32(400))
                vstore!(b, i + 1, values, Val(false))
            end

            b = AMDGPU.zeros(Int32, 16)
            test_vstore_norebase(ROCBackend())(b, 2; ndrange=1)
            synchronize(ROCBackend())
            @test AMDGPU.@allowscalar (b[3], b[4], b[5], b[6]) == (100, 200, 300, 400)
        end

        @testset "vstore_pattern" begin
            @kernel function test_vstore_pattern(b, i)
                values = (Int32(1), Int32(2), Int32(3), Int32(4))
                KernelIntrinsics.vstore_pattern!(b, i, values, Val((1, 2, 1)))
            end

            b = AMDGPU.zeros(Int32, 16)
            test_vstore_pattern(ROCBackend())(b, 2; ndrange=1)
            synchronize(ROCBackend())
            @test AMDGPU.@allowscalar (b[2], b[3], b[4], b[5]) == (1, 2, 3, 4)

            buf = IOBuffer()
            AMDGPU.@device_code_gcn io = buf test_vstore_pattern(ROCBackend())(b, 2; ndrange=1)
            asm = String(take!(buf))
            @test occursin("global_store_dwordx2", asm)
        end

        @testset "vstore_multi" begin
            @kernel function test_vstore_multi(b, i)
                values = (Int32(10), Int32(20), Int32(30), Int32(40))
                KernelIntrinsics.vstore_multi!(b, i, values)
            end

            b = AMDGPU.zeros(Int32, 100)
            test_vstore_multi(ROCBackend())(b, 2; ndrange=1)
            synchronize(ROCBackend())
            @test AMDGPU.@allowscalar (b[2], b[3], b[4], b[5]) == (10, 20, 30, 40)

            buf = IOBuffer()
            AMDGPU.@device_code_gcn io = buf test_vstore_multi(ROCBackend())(b, 2; ndrange=1)
            asm = String(take!(buf))
            @test occursin("global_store_dwordx4", asm)
        end
    end

    @testset "Round-trip vload/vstore" begin
        @kernel function test_roundtrip(a, b, i)
            values = KernelIntrinsics.vload(a, i, Val(4), Val(true))
            KernelIntrinsics.vstore!(b, i, values, Val(true))
        end

        a = ROCArray{Int32}(1:16)
        b = AMDGPU.zeros(Int32, 16)
        test_roundtrip(ROCBackend())(a, b, 2; ndrange=1)
        synchronize(ROCBackend())
        @test AMDGPU.@allowscalar (b[5], b[6], b[7], b[8]) == (5, 6, 7, 8)
    end
end



@testset "vload with contiguous view (norebase)" begin
    @kernel function test_vload_view_norebase(a, b)
        y = KernelIntrinsics.vload(a, 1, Val(4), Val(false))
        b[1] = sum(y)
    end

    a = ROCArray{Int32}(1:16)

    # view starting at 2
    v2 = view(a, 2:16)
    b = AMDGPU.zeros(Int32, 1)
    test_vload_view_norebase(ROCBackend())(v2, b; ndrange=1)
    synchronize(ROCBackend())
    @test AMDGPU.@allowscalar b[1] == 14  # sum of 2,3,4,5

    # view starting at 3
    v3 = view(a, 3:16)
    test_vload_view_norebase(ROCBackend())(v3, b; ndrange=1)
    synchronize(ROCBackend())
    @test AMDGPU.@allowscalar b[1] == 18  # sum of 3,4,5,6
end

@testset "vload with contiguous view (rebase)" begin
    @kernel function test_vload_view_rebase(a, b)
        y = KernelIntrinsics.vload(a, 1, Val(4))  # Val(true) default
        b[1] = sum(y)
    end

    a = ROCArray{Int32}(1:16)

    # view starting at 2
    v2 = view(a, 2:16)
    b = AMDGPU.zeros(Int32, 1)
    test_vload_view_rebase(ROCBackend())(v2, b; ndrange=1)
    synchronize(ROCBackend())
    @test AMDGPU.@allowscalar b[1] == 14  # idx=1 rebase → elements 1..4 of view = 2,3,4,5

    # view starting at 3
    v3 = view(a, 3:16)
    test_vload_view_rebase(ROCBackend())(v3, b; ndrange=1)
    synchronize(ROCBackend())
    @test AMDGPU.@allowscalar b[1] == 18  # elements 1..4 of view = 3,4,5,6
end

@testset "vstore with contiguous view (norebase)" begin
    @kernel function test_vstore_view_norebase(b)
        values = (Int32(10), Int32(20), Int32(30), Int32(40))
        KernelIntrinsics.vstore!(b, 1, values, Val(false))
    end

    # view starting at 2
    b = AMDGPU.zeros(Int32, 16)
    v2 = view(b, 2:16)
    test_vstore_view_norebase(ROCBackend())(v2; ndrange=1)
    synchronize(ROCBackend())
    @test AMDGPU.@allowscalar (b[2], b[3], b[4], b[5]) == (10, 20, 30, 40)

    # view starting at 3
    b = AMDGPU.zeros(Int32, 16)
    v3 = view(b, 3:16)
    test_vstore_view_norebase(ROCBackend())(v3; ndrange=1)
    synchronize(ROCBackend())
    @test AMDGPU.@allowscalar (b[3], b[4], b[5], b[6]) == (10, 20, 30, 40)
end

@testset "vstore with contiguous view (rebase)" begin
    @kernel function test_vstore_view_rebase(b)
        values = (Int32(10), Int32(20), Int32(30), Int32(40))
        KernelIntrinsics.vstore!(b, 1, values, Val(true))
    end

    # view starting at 2
    b = AMDGPU.zeros(Int32, 16)
    v2 = view(b, 2:16)
    test_vstore_view_rebase(ROCBackend())(v2; ndrange=1)
    synchronize(ROCBackend())
    @test AMDGPU.@allowscalar (b[2], b[3], b[4], b[5]) == (10, 20, 30, 40)

    # view starting at 3
    b = AMDGPU.zeros(Int32, 16)
    v3 = view(b, 3:16)
    test_vstore_view_rebase(ROCBackend())(v3; ndrange=1)
    synchronize(ROCBackend())
    @test AMDGPU.@allowscalar (b[3], b[4], b[5], b[6]) == (10, 20, 30, 40)
end

@testset "roundtrip with contiguous view" begin
    @kernel function test_roundtrip_view(a, b)
        vals = KernelIntrinsics.vload(a, 1, Val(4), Val(false))
        KernelIntrinsics.vstore!(b, 1, vals, Val(false))
    end

    a = ROCArray{Int32}(1:16)

    # view starting at 2
    va = view(a, 2:16)
    b = AMDGPU.zeros(Int32, 16)
    vb = view(b, 2:16)
    test_roundtrip_view(ROCBackend())(va, vb; ndrange=1)
    synchronize(ROCBackend())
    @test AMDGPU.@allowscalar (b[2], b[3], b[4], b[5]) == (2, 3, 4, 5)

    # view starting at 3
    va3 = view(a, 3:16)
    b = AMDGPU.zeros(Int32, 16)
    vb3 = view(b, 3:16)
    test_roundtrip_view(ROCBackend())(va3, vb3; ndrange=1)
    synchronize(ROCBackend())
    @test AMDGPU.@allowscalar (b[3], b[4], b[5], b[6]) == (3, 4, 5, 6)
end


@testset "vload/vstore with strided GPU views" begin
    @testset "strided view (step=3) load" begin
        @kernel function test_vload_strided(a, b)
            vals = KernelIntrinsics.vload(a, 1, Val(4), Val(false))
            for i in 1:4
                b[i] = vals[i]
            end
        end

        a = ROCArray{Int32}(1:30)
        v = view(a, 2:3:30)  # elements: 2, 5, 8, 11, ...
        b = AMDGPU.zeros(Int32, 4)
        test_vload_strided(ROCBackend())(v, b; ndrange=1)
        synchronize(ROCBackend())
        @test Array(b) == Int32[2, 5, 8, 11]
    end

    @testset "strided view (step=3) store" begin
        @kernel function test_vstore_strided(b)
            values = (Int32(10), Int32(20), Int32(30), Int32(40))
            KernelIntrinsics.vstore!(b, 1, values, Val(false))
        end

        b = AMDGPU.zeros(Int32, 30)
        v = view(b, 2:3:30)
        test_vstore_strided(ROCBackend())(v; ndrange=1)
        synchronize(ROCBackend())
        @test AMDGPU.@allowscalar b[2] == 10
        @test AMDGPU.@allowscalar b[5] == 20
        @test AMDGPU.@allowscalar b[8] == 30
        @test AMDGPU.@allowscalar b[11] == 40
    end

    @testset "strided view (step=3) rebase" begin
        @kernel function test_vload_strided_rebase(a, b)
            vals = KernelIntrinsics.vload(a, 2, Val(4))  # rebase: elements 5..8 of view
            for i in 1:4
                b[i] = vals[i]
            end
        end

        a = ROCArray{Int32}(1:30)
        v = view(a, 2:3:30)  # elements: 2,5,8,11,14,17,20,23,26,29
        b = AMDGPU.zeros(Int32, 4)
        test_vload_strided_rebase(ROCBackend())(v, b; ndrange=1)
        synchronize(ROCBackend())
        @test Array(b) == Int32[14, 17, 20, 23]
    end

    @testset "fancy indexing view load" begin
        @kernel function test_vload_fancy(a, b)
            vals = KernelIntrinsics.vload(a, 1, Val(4), Val(false))
            for i in 1:4
                b[i] = vals[i]
            end
        end

        a = ROCArray{Int32}(1:20)
        v = view(a, ROCArray([3, 7, 1, 12, 19, 5, 10, 8]))
        b = AMDGPU.zeros(Int32, 4)
        test_vload_fancy(ROCBackend())(v, b; ndrange=1)
        synchronize(ROCBackend())
        @test Array(b) == Int32[3, 7, 1, 12]
    end

    @testset "fancy indexing view store" begin
        @kernel function test_vstore_fancy(b)
            values = (Int32(100), Int32(200), Int32(300), Int32(400))
            KernelIntrinsics.vstore!(b, 1, values, Val(false))
        end

        b = AMDGPU.zeros(Int32, 20)
        v = view(b, ROCArray([3, 7, 1, 12]))
        test_vstore_fancy(ROCBackend())(v; ndrange=1)
        synchronize(ROCBackend())
        @test AMDGPU.@allowscalar b[3] == 100
        @test AMDGPU.@allowscalar b[7] == 200
        @test AMDGPU.@allowscalar b[1] == 300
        @test AMDGPU.@allowscalar b[12] == 400
    end

    @testset "roundtrip strided view" begin
        @kernel function test_roundtrip_strided(a, b)
            vals = KernelIntrinsics.vload(a, 1, Val(4), Val(false))
            KernelIntrinsics.vstore!(b, 1, vals, Val(false))
        end

        a = ROCArray{Int32}(1:30)
        b = AMDGPU.zeros(Int32, 30)
        va = view(a, 2:3:30)
        vb = view(b, 2:3:30)
        test_roundtrip_strided(ROCBackend())(va, vb; ndrange=1)
        synchronize(ROCBackend())
        @test AMDGPU.@allowscalar b[2] == 2
        @test AMDGPU.@allowscalar b[5] == 5
        @test AMDGPU.@allowscalar b[8] == 8
        @test AMDGPU.@allowscalar b[11] == 11
    end
end


@testset "vload/vstore UInt8" begin
    @kernel function test_vload_u8(a, b)
        vals = KernelIntrinsics.vload(a, 1, Val(4))
        for i in 1:4
            b[i] = vals[i]
        end
    end

    @kernel function test_vstore_u8(b)
        values = (UInt8(10), UInt8(20), UInt8(30), UInt8(40))
        KernelIntrinsics.vstore!(b, 1, values)
    end

    @testset "vload rebase" begin
        a = ROCArray{UInt8}(UInt8.(1:16))
        b = AMDGPU.zeros(UInt8, 4)
        test_vload_u8(ROCBackend())(a, b; ndrange=1)
        synchronize(ROCBackend())
        @test Array(b) == UInt8[1, 2, 3, 4]
    end

    @testset "vload norebase" begin
        @kernel function test_vload_u8_norebase(a, b)
            vals = KernelIntrinsics.vload(a, 3, Val(4), Val(false))
            for i in 1:4
                b[i] = vals[i]
            end
        end

        a = ROCArray{UInt8}(UInt8.(1:16))
        b = AMDGPU.zeros(UInt8, 4)
        test_vload_u8_norebase(ROCBackend())(a, b; ndrange=1)
        synchronize(ROCBackend())
        @test Array(b) == UInt8[3, 4, 5, 6]
    end

    @testset "vstore rebase" begin
        b = AMDGPU.zeros(UInt8, 16)
        test_vstore_u8(ROCBackend())(b; ndrange=1)
        synchronize(ROCBackend())
        @test Array(b)[1:4] == UInt8[10, 20, 30, 40]
    end

    @testset "vload rebase view offset 1" begin
        a = ROCArray{UInt8}(UInt8.(1:16))
        v = view(a, 2:16)
        b = AMDGPU.zeros(UInt8, 4)
        test_vload_u8(ROCBackend())(v, b; ndrange=1)
        synchronize(ROCBackend())
        @test Array(b) == UInt8[2, 3, 4, 5]
    end

    @testset "vload rebase view offset 3" begin
        a = ROCArray{UInt8}(UInt8.(1:16))
        v = view(a, 4:16)
        b = AMDGPU.zeros(UInt8, 4)
        test_vload_u8(ROCBackend())(v, b; ndrange=1)
        synchronize(ROCBackend())
        @test Array(b) == UInt8[4, 5, 6, 7]
    end

    @testset "vload norebase view offset 1" begin
        @kernel function test_vload_u8_norebase_v(a, b)
            vals = KernelIntrinsics.vload(a, 1, Val(4), Val(false))
            for i in 1:4
                b[i] = vals[i]
            end
        end

        a = ROCArray{UInt8}(UInt8.(1:16))
        v = view(a, 2:16)
        b = AMDGPU.zeros(UInt8, 4)
        test_vload_u8_norebase_v(ROCBackend())(v, b; ndrange=1)
        synchronize(ROCBackend())
        @test Array(b) == UInt8[2, 3, 4, 5]
    end

    @testset "vstore view offset 1" begin
        b = AMDGPU.zeros(UInt8, 16)
        v = view(b, 2:16)
        test_vstore_u8(ROCBackend())(v; ndrange=1)
        synchronize(ROCBackend())
        @test Array(b)[2:5] == UInt8[10, 20, 30, 40]
    end

    @testset "multiple threads rebase" begin
        @kernel function test_vload_u8_multi(a, b)
            I = @index(Global, Linear)
            vals = KernelIntrinsics.vload(a, I, Val(4))
            base = (I - 1) * 4
            for i in 1:4
                b[base+i] = vals[i]
            end
        end

        a = ROCArray{UInt8}(UInt8.(1:128))
        b = AMDGPU.zeros(UInt8, 128)
        test_vload_u8_multi(ROCBackend())(a, b; ndrange=32)
        synchronize(ROCBackend())
        @test Array(b) == UInt8.(1:128)
    end
end