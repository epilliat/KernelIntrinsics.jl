# @dynlocalmem / launch!(; shmem) / max_dynamic_localmem
#
# The point of dynamic workgroup memory is the size: a static `@localmem` is
# capped at 48 KB by ptxas on every NVIDIA architecture, and no amount of
# re-parameterizing the allocation buys past it. So the tests that matter are
# the ones that go OVER that cap, and the one that goes over the *device* cap
# and has to fail cleanly rather than hang.

const DYN_WG = 256

# ONE layout function, called from the host (to size the launch) and from the
# device (to place each array). Deriving the two sides independently is the one
# way to corrupt a dynamic-shared kernel silently.
dyn_layout(::Type{T}, nh, ns) where {T} =
    (0, nh * sizeof(UInt32), nh * sizeof(UInt32) + ns * sizeof(T))

# `ns` (staged elements per block) is deliberately decoupled from -- and much
# larger than -- the workgroup size: the tile a block can stage is precisely what
# dynamic memory unlocks.
@kernel inbounds = true unsafe_indices = true function dyn_kernel!(
        dst, @Const(src), ::Val{nh}, ::Val{ns}, ::Type{T}
    ) where {nh, ns, T}
    lid = @index(Local, Linear)
    gid = @index(Group, Linear)
    o_h, o_s, _ = dyn_layout(T, nh, ns)

    hist = KI.@dynlocalmem UInt32 (nh,) o_h
    stage = KI.@dynlocalmem T (ns,) o_s

    if lid <= nh
        hist[lid] = UInt32(lid)
    end
    @synchronize

    base = (gid - 1) * ns
    # stage the tile reversed -- if the two arrays aliased, `hist` would be
    # clobbered and the addend below would come out wrong
    for p in lid:DYN_WG:ns
        stage[ns - p + 1] = src[base + p]
    end
    @synchronize

    for p in lid:DYN_WG:ns
        dst[base + p] = stage[p] + T(hist[(p - 1) % nh + 1])
    end
end

function run_dyn(::Type{T}, nh, ns, ngroups) where {T}
    N = ns * ngroups
    src = to_device(T.(1:N))
    dst = to_device(zeros(T, N))
    shmem = dyn_layout(T, nh, ns)[3]
    KI.launch!(
        dyn_kernel!(backend, DYN_WG), dst, src, Val(nh), Val(ns), T;
        ndrange = DYN_WG * ngroups, shmem = shmem
    )
    synchronize(backend)
    return from_device(dst), shmem
end

function dyn_expected(::Type{T}, nh, ns, ngroups) where {T}
    e = zeros(T, ns * ngroups)
    for g in 1:ngroups, p in 1:ns
        base = (g - 1) * ns
        e[base + p] = T(base + (ns - p + 1)) + T((p - 1) % nh + 1)
    end
    return e
end

@testset "@dynlocalmem" begin
    cap = KI.max_dynamic_localmem(KI.device(backend, 1))
    @test cap >= 48 * 1024

    @testset "offsets carve one blob without aliasing" begin
        got, shmem = run_dyn(Float32, 256, 1024, 4)
        @test shmem == 5120
        @test got == dyn_expected(Float32, 256, 1024, 4)
    end

    @testset "past the 48 KB static cap" begin
        # Size this from the DEVICE cap, not from a constant. A hardcoded 64 KB of staging asks
        # for 65 KB once the histogram is added — fine on an A100 (164 KB of shared) but 1 KB over
        # the 64 KB LDS of a CDNA3 part, where the launch guard correctly refuses it. Aim halfway
        # between the 48 KB static cap and whatever this device actually allows: that is past the
        # static cap everywhere, and within reach everywhere.
        target = (48 * 1024 + cap) ÷ 2
        ns = (target - 256 * sizeof(UInt32)) ÷ sizeof(Float32)
        got, shmem = run_dyn(Float32, 256, ns, 3)
        @test 48 * 1024 < shmem <= cap
        @test got == dyn_expected(Float32, 256, ns, 3)
    end

    @testset "at the device cap" begin
        ns = (cap - 256 * sizeof(UInt32)) ÷ sizeof(Float32)
        got, shmem = run_dyn(Float32, 256, ns, 2)
        @test shmem == cap
        @test got == dyn_expected(Float32, 256, ns, 2)
    end

    @testset "over the device cap throws, and is not sticky" begin
        ns = (cap ÷ sizeof(Float32)) + 4096
        @test_throws ArgumentError run_dyn(Float32, 256, ns, 1)
        # The process must stay usable after the rejection: this is what lets an
        # autotune sweep skip an impossible config and carry on in-process,
        # instead of needing one subprocess per candidate.
        got, _ = run_dyn(Float32, 256, 1024, 2)
        @test got == dyn_expected(Float32, 256, 1024, 2)
    end
end
