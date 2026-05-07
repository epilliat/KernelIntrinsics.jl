using KernelIntrinsics: MatchAny

# `@match(MatchAny, value)` returns, for each lane, the bitmask of all lanes
# whose value equals the caller's. Bit `lane - 1` is set iff lane `lane`
# matched. Result type follows `@vote(Ballot, _)` — `UInt32` on 32-lane warps,
# `UInt64` on 64-lane wavefronts.
#
# Helpers `to_device`, `from_device`, `launch`, and `warpsz` come from
# test/harness.jl.

@testset "@match" begin

    # ── all lanes equal → full active mask ────────────────────────────────────
    @testset "all-equal MatchAny ($T)" for T in (UInt8, UInt16, UInt32, UInt64)
        @kernel function match_all_kernel(dst, val::T) where {T}
            I = @index(Global, Linear)
            dst[I] = @match(MatchAny, val)
        end

        dst = to_device(zeros(UInt64, warpsz))
        launch(match_all_kernel, dst, T(7); ndrange=warpsz)

        # Active mask = lower `warpsz` bits set.
        active = warpsz == 64 ? typemax(UInt64) :
                                (UInt64(1) << warpsz) - UInt64(1)
        @test all(UInt64.(from_device(dst)) .== active)
    end

    # ── all distinct → only own bit set ───────────────────────────────────────
    @testset "all-distinct MatchAny" begin
        @kernel function match_distinct_kernel(dst, vals)
            I = @index(Global, Linear)
            dst[I] = @match(MatchAny, vals[I])
        end

        # Lane L (1-indexed) gets value L. With warpsz <= 32, distinct UInt8 fits.
        # For warpsz == 64 we'd need UInt16; clamp to UInt32 to be safe everywhere.
        vals = to_device(UInt32.(1:warpsz))
        dst  = to_device(zeros(UInt64, warpsz))
        launch(match_distinct_kernel, dst, vals; ndrange=warpsz)

        host = UInt64.(from_device(dst))
        @test all(host[L] == (UInt64(1) << (L - 1)) for L in 1:warpsz)
    end

    # ── two halves match each other ───────────────────────────────────────────
    @testset "two-halves MatchAny" begin
        @kernel function match_halves_kernel(dst, vals)
            I = @index(Global, Linear)
            dst[I] = @match(MatchAny, vals[I])
        end

        half = warpsz ÷ 2
        host_vals = vcat(zeros(UInt32, half), ones(UInt32, warpsz - half))
        vals = to_device(host_vals)
        dst  = to_device(zeros(UInt64, warpsz))
        launch(match_halves_kernel, dst, vals; ndrange=warpsz)

        host = UInt64.(from_device(dst))
        lower = (UInt64(1) << half) - UInt64(1)
        upper = warpsz == 64 ? (typemax(UInt64) - lower) :
                ((UInt64(1) << warpsz) - UInt64(1)) - lower
        @test all(host[L] == lower for L in 1:half)
        @test all(host[L] == upper for L in (half + 1):warpsz)
    end

    # ── popcount of the peer mask = group size ────────────────────────────────
    @testset "popcount(peer_mask) is group size" begin
        @kernel function match_popc_kernel(dst, vals)
            I = @index(Global, Linear)
            peer = @match(MatchAny, vals[I])
            dst[I] = UInt32(count_ones(peer))
        end

        # Group sizes: 5 lanes with value 0, 3 with value 1, the rest unique.
        # Test only meaningful for warpsz >= 8.
        host_vals = UInt32.(collect(1:warpsz))
        host_vals[1:5]  .= UInt32(100)   # group of 5
        host_vals[6:8]  .= UInt32(200)   # group of 3
        # Lanes 9..warpsz remain distinct (values 9..warpsz).
        vals = to_device(host_vals)
        dst  = to_device(zeros(UInt32, warpsz))
        launch(match_popc_kernel, dst, vals; ndrange=warpsz)

        host = from_device(dst)
        @test all(host[L] == 5 for L in 1:5)
        @test all(host[L] == 3 for L in 6:8)
        @test all(host[L] == 1 for L in 9:warpsz)
    end

    # ── trailing_zeros gives the leader lane (1-indexed) ──────────────────────
    @testset "trailing_zeros(peer_mask) → leader lane" begin
        @kernel function match_leader_kernel(dst, vals)
            I = @index(Global, Linear)
            peer = @match(MatchAny, vals[I])
            dst[I] = UInt32(trailing_zeros(peer)) + UInt32(1)
        end

        # Lanes 3..7 share a sentinel value; the rest are unique. Their leader
        # (lowest set bit in peer mask) is lane 3. Value must be > warpsz so it
        # doesn't collide with any lane index in `collect(1:warpsz)` — using 42
        # collides with lane 42 on wave64 (MI300X), pulling lane 42 into the
        # group and making its leader 3 instead of 42.
        host_vals = UInt32.(collect(1:warpsz))
        host_vals[3:7] .= UInt32(100)
        vals = to_device(host_vals)
        dst  = to_device(zeros(UInt32, warpsz))
        launch(match_leader_kernel, dst, vals; ndrange=warpsz)

        host = from_device(dst)
        @test all(host[L] == 3 for L in 3:7)
        # Other lanes are their own leader.
        @test all(host[L] == L for L in vcat([1, 2], collect(8:warpsz)))
    end
end
