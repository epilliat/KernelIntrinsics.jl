# @sleep smoke test — there is no observable result semantics (it's a backoff
# hint), so we only assert the kernel compiles + runs on the active backend and
# does not perturb the work done alongside it.
@testset "@sleep" begin
    @kernel function sleep_kernel(dst, src)
        I = @index(Global, Linear)
        # constant and runtime-valued backoff, plus a small backoff ladder like
        # the scan lookback loop uses
        @sleep 4
        b = UInt32(1)
        for _ in 1:3
            @sleep b
            b = min(b << 1, UInt32(64))
        end
        dst[I] = src[I] + one(eltype(dst))
    end

    n = 4 * warpsz
    src = to_device(Int32.(1:n))
    dst = to_device(zeros(Int32, n))
    launch(sleep_kernel, dst, src; ndrange = n)
    @test from_device(dst) == Int32.(2:n+1)
end
