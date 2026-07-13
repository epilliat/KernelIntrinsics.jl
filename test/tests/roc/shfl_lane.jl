# The AMD shuffle must address the PHYSICAL lane.
#
# AMDGPU.Device.shfl_* derives the source lane from `activelane()` = `__ockl_activelane_u32` =
# `mbcnt(EXEC)` = the rank among the *ACTIVE* lanes. HIP's `__shfl` uses `__lane_id()` = mbcnt
# over an all-ones mask = the *physical* lane. They agree only when the whole wavefront is
# active. Under divergence `activelane()` compacts the index, so the shuffle silently reads the
# wrong lane — and, because it depends on EXEC, LLVM can neither hoist nor CSE it, so the lane
# index is recomputed at every shuffle. `ext/AMDGPU/shuffle_vote.jl` therefore bypasses
# AMDGPU.Device.shfl_* entirely. These tests pin that down.
#
# The divergent kernels below activate only ODD lanes and shuffle by 2, so every SOURCE lane is
# also odd, i.e. also active — the result is well defined regardless of what a `ds_bpermute`
# returns for an inactive lane. With the old activelane-based indexing an active lane `l` would
# read physical lane `(l-1)÷2 + 2` instead of `l + 2`: wrong for every l ≥ 3.

@testset "@shfl addresses the physical lane" begin
    n = warpsz                       # exactly one wavefront

    @testset "divergent Down" begin
        @kernel function shfl_down_divergent(dst, src, ::Val{ws}) where {ws}
            I = @index(Global, Linear)
            lane = (I - 1) % ws + 1  # 1-based
            v = src[I]
            r = v
            if isodd(lane)           # half the wave masked off -> EXEC is partial
                r = @shfl(Down, v, 2)
            end
            dst[I] = r
        end

        src = collect(Int32(1):Int32(n))
        d_src = to_device(src); d_dst = to_device(zeros(Int32, n))
        shfl_down_divergent(backend)(d_dst, d_src, Val(warpsz); ndrange=n)
        synchronize(backend)
        got = from_device(d_dst)

        # odd lanes read lane+2 (out of range -> own value); even lanes are untouched
        want = [isodd(l) ? (l + 2 <= n ? Int32(l + 2) : Int32(l)) : Int32(l) for l in 1:n]
        @test got == want
    end

    @testset "divergent Idx" begin
        @kernel function shfl_idx_divergent(dst, src, ::Val{ws}) where {ws}
            I = @index(Global, Linear)
            lane = (I - 1) % ws + 1
            v = src[I]
            r = v
            if isodd(lane)
                r = @shfl(Idx, v, 3)  # every active lane reads physical lane 3 (also odd)
            end
            dst[I] = r
        end

        src = collect(Int32(1):Int32(n))
        d_src = to_device(src); d_dst = to_device(zeros(Int32, n))
        shfl_idx_divergent(backend)(d_dst, d_src, Val(warpsz); ndrange=n)
        synchronize(backend)
        @test from_device(d_dst) == [isodd(l) ? Int32(3) : Int32(l) for l in 1:n]
    end

    # A fully unrolled shuffle sequence used to produce a GPU memory-access fault, because the
    # compiler restructured EXEC around the unrolled body and activelane() then returned an index
    # the code did not expect. With the physical lane it is correct by construction.
    @testset "unrolled reduce" begin
        @kernel function unrolled_warpsum(dst, src, ::Val{ws}) where {ws}
            I = @index(Global, Linear)
            v = src[I]
            Base.Cartesian.@nexprs 6 k -> begin      # 2^6 == 64 == wave64
                v += @shfl(Down, v, 1 << (k - 1))
            end
            dst[I] = v
        end

        src = collect(Int32(1):Int32(n))
        d_src = to_device(src); d_dst = to_device(zeros(Int32, n))
        unrolled_warpsum(backend)(d_dst, d_src, Val(warpsz); ndrange=n)
        synchronize(backend)
        @test from_device(d_dst)[1] == Int32(sum(1:n))   # lane 1 holds the wave-wide sum
    end

    @testset "ISA" begin
        @kernel function shfl_down_plain(dst, src)
            I = @index(Global, Linear)
            dst[I] = @shfl(Down, src[I], 1)
        end

        d_src = to_device(collect(Int32(1):Int32(n))); d_dst = to_device(zeros(Int32, n))
        asm = @capture_ir shfl_down_plain(backend)(d_dst, d_src; ndrange=n)

        @test occursin("ds_bpermute_b32", asm)          # the shuffle itself
        @test !occursin("activelane", asm)              # NOT the rank among active lanes
        # __lane_id(): mbcnt over an ALL-ONES mask -> EXEC-independent, hoistable, CSE-able
        @test occursin(r"v_mbcnt_lo_u32_b32\s+\S+,\s*-1", asm)
    end
end
