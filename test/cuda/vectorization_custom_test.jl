using Test, CUDA, KernelAbstractions, KernelIntrinsics

struct Vec3_1
    x::Float32
    y::Float32
    z::Float32
end
Base.:+(a::Vec3_1, b::Vec3_1) = Vec3_1(a.x + b.x, a.y + b.y, a.z + b.z)
Base.zero(::Type{Vec3_1}) = Vec3_1(0f0, 0f0, 0f0)

struct S6
    a::Float16
    b::Float16
    c::Float16
end
Base.:+(a::S6, b::S6) = S6(a.a + b.a, a.b + b.b, a.c + b.c)
Base.zero(::Type{S6}) = S6(Float16(0), Float16(0), Float16(0))

struct Vec3_2
    x::Tuple{UInt8,Float16,UInt8}
    y::Float32
    z::Float32
end
Base.:+(a::Vec3_2, b::Vec3_2) = Vec3_2((a.x[1] + b.x[1], a.x[2] + b.x[2], a.x[3] + b.x[3]), a.y + b.y, a.z + b.z)
Base.zero(::Type{Vec3_2}) = Vec3_2((UInt8(0), Float16(0), UInt8(0)), 0f0, 0f0)

struct TupleStruct
    x::Float16
    y::UInt8
end
Base.:+(a::TupleStruct, b::TupleStruct) = TupleStruct(a.x + b.x, a.y + b.y)
Base.zero(::Type{TupleStruct}) = TupleStruct(Float16(0), UInt8(0))

@testset "Custom struct vload/vstore" begin

    @testset "Vec3_1 ($(sizeof(Vec3_1)) bytes, scalar fallback)" begin
        @test sizeof(Vec3_1) == 12
        @test !ispow2(sizeof(Vec3_1))

        @kernel function test_vload_vec3_1(a, b)
            vals = KernelIntrinsics.vload(a, 1, Val(4), Val(false))
            for i in 1:4
                b[i] = vals[i]
            end
        end

        @kernel function test_vstore_vec3_1(b)
            values = (Vec3_1(1f0, 2f0, 3f0), Vec3_1(4f0, 5f0, 6f0),
                Vec3_1(7f0, 8f0, 9f0), Vec3_1(10f0, 11f0, 12f0))
            KernelIntrinsics.vstore!(b, 1, values, Val(false))
        end

        src = CuArray([Vec3_1(Float32(i), Float32(i + 10), Float32(i + 20)) for i in 1:16])
        dst = CuArray(fill(zero(Vec3_1), 4))
        test_vload_vec3_1(CUDABackend())(src, dst; ndrange=1)
        synchronize(CUDABackend())
        result = Array(dst)
        @test result[1] == Vec3_1(1f0, 11f0, 21f0)
        @test result[2] == Vec3_1(2f0, 12f0, 22f0)
        @test result[3] == Vec3_1(3f0, 13f0, 23f0)
        @test result[4] == Vec3_1(4f0, 14f0, 24f0)

        dst2 = CuArray(fill(zero(Vec3_1), 8))
        test_vstore_vec3_1(CUDABackend())(dst2; ndrange=1)
        synchronize(CUDABackend())
        result2 = Array(dst2)
        @test result2[1] == Vec3_1(1f0, 2f0, 3f0)
        @test result2[2] == Vec3_1(4f0, 5f0, 6f0)
        @test result2[3] == Vec3_1(7f0, 8f0, 9f0)
        @test result2[4] == Vec3_1(10f0, 11f0, 12f0)

        # View
        v_src = view(src, 3:16)
        dst3 = CuArray(fill(zero(Vec3_1), 4))
        test_vload_vec3_1(CUDABackend())(v_src, dst3; ndrange=1)
        synchronize(CUDABackend())
        result3 = Array(dst3)
        @test result3[1] == Vec3_1(3f0, 13f0, 23f0)
        @test result3[2] == Vec3_1(4f0, 14f0, 24f0)
        @test result3[3] == Vec3_1(5f0, 15f0, 25f0)
        @test result3[4] == Vec3_1(6f0, 16f0, 26f0)
    end

    @testset "S6 ($(sizeof(S6)) bytes, scalar fallback)" begin
        @test sizeof(S6) == 6
        @test !ispow2(sizeof(S6))

        @kernel function test_vload_s6(a, b)
            vals = KernelIntrinsics.vload(a, 1, Val(4), Val(false))
            for i in 1:4
                b[i] = vals[i]
            end
        end

        @kernel function test_vstore_s6(b)
            values = (S6(Float16(1), Float16(2), Float16(3)),
                S6(Float16(4), Float16(5), Float16(6)),
                S6(Float16(7), Float16(8), Float16(9)),
                S6(Float16(10), Float16(11), Float16(12)))
            KernelIntrinsics.vstore!(b, 1, values, Val(false))
        end

        src = CuArray([S6(Float16(i), Float16(i + 10), Float16(i + 20)) for i in 1:16])
        dst = CuArray(fill(zero(S6), 4))
        test_vload_s6(CUDABackend())(src, dst; ndrange=1)
        synchronize(CUDABackend())
        result = Array(dst)
        @test result[1] == S6(Float16(1), Float16(11), Float16(21))
        @test result[2] == S6(Float16(2), Float16(12), Float16(22))
        @test result[3] == S6(Float16(3), Float16(13), Float16(23))
        @test result[4] == S6(Float16(4), Float16(14), Float16(24))

        dst2 = CuArray(fill(zero(S6), 8))
        test_vstore_s6(CUDABackend())(dst2; ndrange=1)
        synchronize(CUDABackend())
        result2 = Array(dst2)
        @test result2[1] == S6(Float16(1), Float16(2), Float16(3))
        @test result2[2] == S6(Float16(4), Float16(5), Float16(6))
        @test result2[3] == S6(Float16(7), Float16(8), Float16(9))
        @test result2[4] == S6(Float16(10), Float16(11), Float16(12))

        # View
        v_src = view(src, 2:16)
        dst3 = CuArray(fill(zero(S6), 4))
        test_vload_s6(CUDABackend())(v_src, dst3; ndrange=1)
        synchronize(CUDABackend())
        result3 = Array(dst3)
        @test result3[1] == S6(Float16(2), Float16(12), Float16(22))
        @test result3[2] == S6(Float16(3), Float16(13), Float16(23))
        @test result3[3] == S6(Float16(4), Float16(14), Float16(24))
        @test result3[4] == S6(Float16(5), Float16(15), Float16(25))
    end

    @testset "Vec3_2 ($(sizeof(Vec3_2)) bytes, vectorized path)" begin
        @test sizeof(Vec3_2) == 16
        @test ispow2(sizeof(Vec3_2))

        @kernel function test_vload_vec3_2(a, b)
            vals = KernelIntrinsics.vload(a, 1, Val(4), Val(false))
            for i in 1:4
                b[i] = vals[i]
            end
        end

        @kernel function test_vstore_vec3_2(b)
            values = (Vec3_2((UInt8(1), Float16(2), UInt8(3)), 4f0, 5f0),
                Vec3_2((UInt8(6), Float16(7), UInt8(8)), 9f0, 10f0),
                Vec3_2((UInt8(11), Float16(12), UInt8(13)), 14f0, 15f0),
                Vec3_2((UInt8(16), Float16(17), UInt8(18)), 19f0, 20f0))
            KernelIntrinsics.vstore!(b, 1, values, Val(false))
        end

        src = CuArray([Vec3_2((UInt8(i), Float16(i + 10), UInt8(i + 20)), Float32(i + 30), Float32(i + 40)) for i in 1:16])
        dst = CuArray(fill(zero(Vec3_2), 4))
        test_vload_vec3_2(CUDABackend())(src, dst; ndrange=1)
        synchronize(CUDABackend())
        result = Array(dst)
        @test result[1] == Vec3_2((UInt8(1), Float16(11), UInt8(21)), 31f0, 41f0)
        @test result[2] == Vec3_2((UInt8(2), Float16(12), UInt8(22)), 32f0, 42f0)
        @test result[3] == Vec3_2((UInt8(3), Float16(13), UInt8(23)), 33f0, 43f0)
        @test result[4] == Vec3_2((UInt8(4), Float16(14), UInt8(24)), 34f0, 44f0)

        dst2 = CuArray(fill(zero(Vec3_2), 8))
        test_vstore_vec3_2(CUDABackend())(dst2; ndrange=1)
        synchronize(CUDABackend())
        result2 = Array(dst2)
        @test result2[1] == Vec3_2((UInt8(1), Float16(2), UInt8(3)), 4f0, 5f0)
        @test result2[2] == Vec3_2((UInt8(6), Float16(7), UInt8(8)), 9f0, 10f0)
        @test result2[3] == Vec3_2((UInt8(11), Float16(12), UInt8(13)), 14f0, 15f0)
        @test result2[4] == Vec3_2((UInt8(16), Float16(17), UInt8(18)), 19f0, 20f0)

        # View
        v_src = view(src, 2:16)
        dst3 = CuArray(fill(zero(Vec3_2), 4))
        test_vload_vec3_2(CUDABackend())(v_src, dst3; ndrange=1)
        synchronize(CUDABackend())
        result3 = Array(dst3)
        @test result3[1] == Vec3_2((UInt8(2), Float16(12), UInt8(22)), 32f0, 42f0)
        @test result3[2] == Vec3_2((UInt8(3), Float16(13), UInt8(23)), 33f0, 43f0)
        @test result3[3] == Vec3_2((UInt8(4), Float16(14), UInt8(24)), 34f0, 44f0)
        @test result3[4] == Vec3_2((UInt8(5), Float16(15), UInt8(25)), 35f0, 45f0)
    end

    @testset "TupleStruct ($(sizeof(TupleStruct)) bytes)" begin
        @kernel function test_vload_ts(a, b)
            vals = KernelIntrinsics.vload(a, 1, Val(4), Val(false))
            for i in 1:4
                b[i] = vals[i]
            end
        end

        @kernel function test_vstore_ts(b)
            values = (TupleStruct(Float16(1), UInt8(2)),
                TupleStruct(Float16(3), UInt8(4)),
                TupleStruct(Float16(5), UInt8(6)),
                TupleStruct(Float16(7), UInt8(8)))
            KernelIntrinsics.vstore!(b, 1, values, Val(false))
        end

        src = CuArray([TupleStruct(Float16(i), UInt8(i + 10)) for i in 1:16])
        dst = CuArray(fill(zero(TupleStruct), 4))
        test_vload_ts(CUDABackend())(src, dst; ndrange=1)
        synchronize(CUDABackend())
        result = Array(dst)
        @test result[1] == TupleStruct(Float16(1), UInt8(11))
        @test result[2] == TupleStruct(Float16(2), UInt8(12))
        @test result[3] == TupleStruct(Float16(3), UInt8(13))
        @test result[4] == TupleStruct(Float16(4), UInt8(14))

        dst2 = CuArray(fill(zero(TupleStruct), 8))
        test_vstore_ts(CUDABackend())(dst2; ndrange=1)
        synchronize(CUDABackend())
        result2 = Array(dst2)
        @test result2[1] == TupleStruct(Float16(1), UInt8(2))
        @test result2[2] == TupleStruct(Float16(3), UInt8(4))
        @test result2[3] == TupleStruct(Float16(5), UInt8(6))
        @test result2[4] == TupleStruct(Float16(7), UInt8(8))

        # View
        v_src = view(src, 2:16)
        dst3 = CuArray(fill(zero(TupleStruct), 4))
        test_vload_ts(CUDABackend())(v_src, dst3; ndrange=1)
        synchronize(CUDABackend())
        result3 = Array(dst3)
        @test result3[1] == TupleStruct(Float16(2), UInt8(12))
        @test result3[2] == TupleStruct(Float16(3), UInt8(13))
        @test result3[3] == TupleStruct(Float16(4), UInt8(14))
        @test result3[4] == TupleStruct(Float16(5), UInt8(15))
    end
end