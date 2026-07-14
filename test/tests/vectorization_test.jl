# Cross-backend vload / vstore! tests. Backend-specific bits (array
# construction, IR capture, IR substring patterns, scalar reads) are routed
# through the harness (`to_device`, `from_device`, `@allowscalar`, `@capture_ir`)
# and the `HOOKS` traits in test/backend_hooks.jl.

@testset "KernelIntrinsics Vectorized Load/Store" begin

    @testset "vload" begin
        @testset "vload with rebase" begin
            @kernel function test_vload_rebase(a, b, i)
                y = KernelIntrinsics.vload(a, i, Val(4))
                b[1] = sum(y)
            end

            a = to_device(Int32.(1:16))
            b = to_device(zeros(Int32, 4))
            test_vload_rebase(backend)(a, b, 2; ndrange=1)
            synchronize(backend)
            @test @allowscalar b[1] == 26  # sum of elements 5,6,7,8

            asm = @capture_ir test_vload_rebase(backend)(a, b, 2; ndrange=1)
            assert_ir(HOOKS.ir.vload_v4, asm)
        end

        @testset "vload without rebase" begin
            @kernel function test_vload_norebase(a, b, i)
                x = vload(a, i + 1, Val(4), Val(false))
                b[1] = sum(x)
            end

            a = to_device(Int32.(1:16))
            b = to_device(zeros(Int32, 4))
            test_vload_norebase(backend)(a, b, 2; ndrange=1)
            synchronize(backend)
            @test @allowscalar b[1] == 18  # sum of elements 3,4,5,6
        end

        @testset "vload_pattern" begin
            @kernel function test_vload_pattern(a, b, i)
                values = KernelIntrinsics.vload_pattern(a, i, Val((1, 2, 1)))
                b[1] = sum(sum(values))
            end

            a = to_device(Int32.(1:16))
            b = to_device(zeros(Int32, 1))
            test_vload_pattern(backend)(a, b, 2; ndrange=1)
            synchronize(backend)

            asm = @capture_ir test_vload_pattern(backend)(a, b, 2; ndrange=1)
            assert_ir(HOOKS.ir.vload_v2, asm)
        end

        @testset "vload_multi" begin
            @kernel function test_vload_multi(a, b, i, ::Val{N}) where {N}
                values = KernelIntrinsics.vload_multi(a, i, Val(N))
                for j in 1:N
                    b[j] = values[j]
                end
            end

            a = to_device(Int32.(1:100))
            b = to_device(zeros(Int32, 8))
            asm = @capture_ir test_vload_multi(backend)(a, b, 2, Val(4); ndrange=1)
            assert_ir(HOOKS.ir.vload_v4, asm)

            # Correctness for various alignments
            N = 16
            for shift in 1:16
                a = to_device(Int32.(1:100))
                b = to_device(zeros(Int32, N))
                test_vload_multi(backend)(a, b, shift, Val(N); ndrange=1)
                synchronize(backend)
                @test from_device(b) == Int32.(shift:shift+N-1)
            end
        end
    end

    @testset "vstore" begin
        @testset "vstore with rebase" begin
            @kernel function test_vstore_rebase(b, i)
                values = (Int32(10), Int32(20), Int32(30), Int32(40))
                KernelIntrinsics.vstore!(b, i, values, Val(true))
            end

            b = to_device(zeros(Int32, 16))
            test_vstore_rebase(backend)(b, 2; ndrange=1)
            synchronize(backend)
            @test @allowscalar (b[5], b[6], b[7], b[8]) == (10, 20, 30, 40)

            asm = @capture_ir test_vstore_rebase(backend)(b, 2; ndrange=1)
            assert_ir(HOOKS.ir.vstore_v4, asm)
        end

        @testset "vstore without rebase" begin
            @kernel function test_vstore_norebase(b, i)
                values = (Int32(100), Int32(200), Int32(300), Int32(400))
                vstore!(b, i + 1, values, Val(false))
            end

            b = to_device(zeros(Int32, 16))
            test_vstore_norebase(backend)(b, 2; ndrange=1)
            synchronize(backend)
            @test @allowscalar (b[3], b[4], b[5], b[6]) == (100, 200, 300, 400)
        end

        @testset "vstore_pattern" begin
            @kernel function test_vstore_pattern(b, i)
                values = (Int32(1), Int32(2), Int32(3), Int32(4))
                KernelIntrinsics.vstore_pattern!(b, i, values, Val((1, 2, 1)))
            end

            b = to_device(zeros(Int32, 16))
            test_vstore_pattern(backend)(b, 2; ndrange=1)
            synchronize(backend)
            @test @allowscalar (b[2], b[3], b[4], b[5]) == (1, 2, 3, 4)

            asm = @capture_ir test_vstore_pattern(backend)(b, 2; ndrange=1)
            assert_ir(HOOKS.ir.vstore_v2, asm)
        end

        @testset "vstore_multi" begin
            @kernel function test_vstore_multi(b, i)
                values = (Int32(10), Int32(20), Int32(30), Int32(40))
                KernelIntrinsics.vstore_multi!(b, i, values)
            end

            b = to_device(zeros(Int32, 100))
            test_vstore_multi(backend)(b, 2; ndrange=1)
            synchronize(backend)
            @test @allowscalar (b[2], b[3], b[4], b[5]) == (10, 20, 30, 40)

            asm = @capture_ir test_vstore_multi(backend)(b, 2; ndrange=1)
            assert_ir(HOOKS.ir.vstore_v4, asm)
        end
    end

    @testset "Round-trip vload/vstore" begin
        @kernel function test_roundtrip(a, b, i)
            values = KernelIntrinsics.vload(a, i, Val(4), Val(true))
            KernelIntrinsics.vstore!(b, i, values, Val(true))
        end

        a = to_device(Int32.(1:16))
        b = to_device(zeros(Int32, 16))
        test_roundtrip(backend)(a, b, 2; ndrange=1)
        synchronize(backend)
        @test @allowscalar (b[5], b[6], b[7], b[8]) == (5, 6, 7, 8)
    end

    # Regression: a block of Nitem*sizeof(T) >= 256 bytes (e.g. 32xFloat64 = 256 B)
    # used to mis-compute the load/store alignment (`0x01 << trailing_zeros(...)`
    # overflowed UInt8 to 0 → Val(0) → on-device compile failure).
    @testset "Round-trip large block (>=256 B)" begin
        @kernel function test_rt_big(a, b, i)
            v = KernelIntrinsics.vload(a, i, Val(32), Val(true))
            KernelIntrinsics.vstore!(b, i, v, Val(true))
        end
        if HOOKS.supported.float64
            for T in (Float64, Int64)
                a = to_device(T.(1:64)); b = to_device(zeros(T, 64))
                test_rt_big(backend)(a, b, 2; ndrange=1)  # block 2 = elements 33:64
                synchronize(backend)
                @test from_device(b)[33:64] == T.(33:64)
            end
        end
    end
end


@testset "vload with contiguous view (norebase)" begin
    @kernel function test_vload_view_norebase(a, b)
        y = KernelIntrinsics.vload(a, 1, Val(4), Val(false))
        b[1] = sum(y)
    end

    a = to_device(Int32.(1:16))

    v2 = view(a, 2:16)
    b = to_device(zeros(Int32, 1))
    test_vload_view_norebase(backend)(v2, b; ndrange=1)
    synchronize(backend)
    @test @allowscalar b[1] == 14  # sum of 2,3,4,5

    v3 = view(a, 3:16)
    test_vload_view_norebase(backend)(v3, b; ndrange=1)
    synchronize(backend)
    @test @allowscalar b[1] == 18  # sum of 3,4,5,6
end

@testset "vload with contiguous view (rebase)" begin
    @kernel function test_vload_view_rebase(a, b)
        y = KernelIntrinsics.vload(a, 1, Val(4))  # Val(true) default
        b[1] = sum(y)
    end

    a = to_device(Int32.(1:16))

    v2 = view(a, 2:16)
    b = to_device(zeros(Int32, 1))
    test_vload_view_rebase(backend)(v2, b; ndrange=1)
    synchronize(backend)
    @test @allowscalar b[1] == 14  # idx=1 rebase → elements 1..4 of view = 2,3,4,5

    v3 = view(a, 3:16)
    test_vload_view_rebase(backend)(v3, b; ndrange=1)
    synchronize(backend)
    @test @allowscalar b[1] == 18  # elements 1..4 of view = 3,4,5,6
end

@testset "vstore with contiguous view (norebase)" begin
    @kernel function test_vstore_view_norebase(b)
        values = (Int32(10), Int32(20), Int32(30), Int32(40))
        KernelIntrinsics.vstore!(b, 1, values, Val(false))
    end

    b = to_device(zeros(Int32, 16))
    v2 = view(b, 2:16)
    test_vstore_view_norebase(backend)(v2; ndrange=1)
    synchronize(backend)
    @test @allowscalar (b[2], b[3], b[4], b[5]) == (10, 20, 30, 40)

    b = to_device(zeros(Int32, 16))
    v3 = view(b, 3:16)
    test_vstore_view_norebase(backend)(v3; ndrange=1)
    synchronize(backend)
    @test @allowscalar (b[3], b[4], b[5], b[6]) == (10, 20, 30, 40)
end

@testset "vstore with contiguous view (rebase)" begin
    @kernel function test_vstore_view_rebase(b)
        values = (Int32(10), Int32(20), Int32(30), Int32(40))
        KernelIntrinsics.vstore!(b, 1, values, Val(true))
    end

    b = to_device(zeros(Int32, 16))
    v2 = view(b, 2:16)
    test_vstore_view_rebase(backend)(v2; ndrange=1)
    synchronize(backend)
    @test @allowscalar (b[2], b[3], b[4], b[5]) == (10, 20, 30, 40)

    b = to_device(zeros(Int32, 16))
    v3 = view(b, 3:16)
    test_vstore_view_rebase(backend)(v3; ndrange=1)
    synchronize(backend)
    @test @allowscalar (b[3], b[4], b[5], b[6]) == (10, 20, 30, 40)
end

@testset "roundtrip with contiguous view" begin
    @kernel function test_roundtrip_view(a, b)
        vals = KernelIntrinsics.vload(a, 1, Val(4), Val(false))
        KernelIntrinsics.vstore!(b, 1, vals, Val(false))
    end

    a = to_device(Int32.(1:16))

    va = view(a, 2:16)
    b = to_device(zeros(Int32, 16))
    vb = view(b, 2:16)
    test_roundtrip_view(backend)(va, vb; ndrange=1)
    synchronize(backend)
    @test @allowscalar (b[2], b[3], b[4], b[5]) == (2, 3, 4, 5)

    va3 = view(a, 3:16)
    b = to_device(zeros(Int32, 16))
    vb3 = view(b, 3:16)
    test_roundtrip_view(backend)(va3, vb3; ndrange=1)
    synchronize(backend)
    @test @allowscalar (b[3], b[4], b[5], b[6]) == (3, 4, 5, 6)
end


@testset "vload/vstore with strided GPU views" begin
    @testset "strided view (step=3) load" begin
        @kernel function test_vload_strided(a, b)
            vals = KernelIntrinsics.vload(a, 1, Val(4), Val(false))
            for i in 1:4
                b[i] = vals[i]
            end
        end

        a = to_device(Int32.(1:30))
        v = view(a, 2:3:30)  # elements: 2, 5, 8, 11, ...
        b = to_device(zeros(Int32, 4))
        test_vload_strided(backend)(v, b; ndrange=1)
        synchronize(backend)
        @test from_device(b) == Int32[2, 5, 8, 11]
    end

    @testset "strided view (step=3) store" begin
        @kernel function test_vstore_strided(b)
            values = (Int32(10), Int32(20), Int32(30), Int32(40))
            KernelIntrinsics.vstore!(b, 1, values, Val(false))
        end

        b = to_device(zeros(Int32, 30))
        v = view(b, 2:3:30)
        test_vstore_strided(backend)(v; ndrange=1)
        synchronize(backend)
        @test @allowscalar b[2] == 10
        @test @allowscalar b[5] == 20
        @test @allowscalar b[8] == 30
        @test @allowscalar b[11] == 40
    end

    @testset "strided view (step=3) rebase" begin
        @kernel function test_vload_strided_rebase(a, b)
            vals = KernelIntrinsics.vload(a, 2, Val(4))  # rebase: elements 5..8 of view
            for i in 1:4
                b[i] = vals[i]
            end
        end

        a = to_device(Int32.(1:30))
        v = view(a, 2:3:30)  # elements: 2,5,8,11,14,17,20,23,26,29
        b = to_device(zeros(Int32, 4))
        test_vload_strided_rebase(backend)(v, b; ndrange=1)
        synchronize(backend)
        @test from_device(b) == Int32[14, 17, 20, 23]
    end

    @testset "fancy indexing view load" begin
        @kernel function test_vload_fancy(a, b)
            vals = KernelIntrinsics.vload(a, 1, Val(4), Val(false))
            for i in 1:4
                b[i] = vals[i]
            end
        end

        a = to_device(Int32.(1:20))
        v = view(a, to_device([3, 7, 1, 12, 19, 5, 10, 8]))
        b = to_device(zeros(Int32, 4))
        test_vload_fancy(backend)(v, b; ndrange=1)
        synchronize(backend)
        @test from_device(b) == Int32[3, 7, 1, 12]
    end

    @testset "fancy indexing view store" begin
        @kernel function test_vstore_fancy(b)
            values = (Int32(100), Int32(200), Int32(300), Int32(400))
            KernelIntrinsics.vstore!(b, 1, values, Val(false))
        end

        b = to_device(zeros(Int32, 20))
        v = view(b, to_device([3, 7, 1, 12]))
        test_vstore_fancy(backend)(v; ndrange=1)
        synchronize(backend)
        @test @allowscalar b[3] == 100
        @test @allowscalar b[7] == 200
        @test @allowscalar b[1] == 300
        @test @allowscalar b[12] == 400
    end

    @testset "roundtrip strided view" begin
        @kernel function test_roundtrip_strided(a, b)
            vals = KernelIntrinsics.vload(a, 1, Val(4), Val(false))
            KernelIntrinsics.vstore!(b, 1, vals, Val(false))
        end

        a = to_device(Int32.(1:30))
        b = to_device(zeros(Int32, 30))
        va = view(a, 2:3:30)
        vb = view(b, 2:3:30)
        test_roundtrip_strided(backend)(va, vb; ndrange=1)
        synchronize(backend)
        @test @allowscalar b[2] == 2
        @test @allowscalar b[5] == 5
        @test @allowscalar b[8] == 8
        @test @allowscalar b[11] == 11
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
        a = to_device(UInt8.(1:16))
        b = to_device(zeros(UInt8, 4))
        test_vload_u8(backend)(a, b; ndrange=1)
        synchronize(backend)
        @test from_device(b) == UInt8[1, 2, 3, 4]
    end

    @testset "vload norebase" begin
        @kernel function test_vload_u8_norebase(a, b)
            vals = KernelIntrinsics.vload(a, 3, Val(4), Val(false))
            for i in 1:4
                b[i] = vals[i]
            end
        end

        a = to_device(UInt8.(1:16))
        b = to_device(zeros(UInt8, 4))
        test_vload_u8_norebase(backend)(a, b; ndrange=1)
        synchronize(backend)
        @test from_device(b) == UInt8[3, 4, 5, 6]
    end

    @testset "vstore rebase" begin
        b = to_device(zeros(UInt8, 16))
        test_vstore_u8(backend)(b; ndrange=1)
        synchronize(backend)
        @test from_device(b)[1:4] == UInt8[10, 20, 30, 40]
    end

    @testset "vload rebase view offset 1" begin
        a = to_device(UInt8.(1:16))
        v = view(a, 2:16)
        b = to_device(zeros(UInt8, 4))
        test_vload_u8(backend)(v, b; ndrange=1)
        synchronize(backend)
        @test from_device(b) == UInt8[2, 3, 4, 5]
    end

    @testset "vload rebase view offset 3" begin
        a = to_device(UInt8.(1:16))
        v = view(a, 4:16)
        b = to_device(zeros(UInt8, 4))
        test_vload_u8(backend)(v, b; ndrange=1)
        synchronize(backend)
        @test from_device(b) == UInt8[4, 5, 6, 7]
    end

    @testset "vload norebase view offset 1" begin
        @kernel function test_vload_u8_norebase_v(a, b)
            vals = KernelIntrinsics.vload(a, 1, Val(4), Val(false))
            for i in 1:4
                b[i] = vals[i]
            end
        end

        a = to_device(UInt8.(1:16))
        v = view(a, 2:16)
        b = to_device(zeros(UInt8, 4))
        test_vload_u8_norebase_v(backend)(v, b; ndrange=1)
        synchronize(backend)
        @test from_device(b) == UInt8[2, 3, 4, 5]
    end

    @testset "vstore view offset 1" begin
        b = to_device(zeros(UInt8, 16))
        v = view(b, 2:16)
        test_vstore_u8(backend)(v; ndrange=1)
        synchronize(backend)
        @test from_device(b)[2:5] == UInt8[10, 20, 30, 40]
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

        a = to_device(UInt8.(1:128))
        b = to_device(zeros(UInt8, 128))
        test_vload_u8_multi(backend)(a, b; ndrange=32)
        synchronize(backend)
        @test from_device(b) == UInt8.(1:128)
    end
end
