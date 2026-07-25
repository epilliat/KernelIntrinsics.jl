# async_copy! / async_commit / async_wait / async_copy_supported
#
# Round-trip: stage a global buffer into shared/LDS with `async_copy!`, commit,
# wait, barrier, then copy shared→global out and assert the output equals the
# input. The SAME kernel drives both paths:
#   • the hardware path (cp.async on sm_80+, global_load_lds on gfx90a/94x),
#     exercised at widths 4/8/16 only when `async_copy_supported(backend)`;
#   • the register FALLBACK, exercised UNCONDITIONALLY at a 12-byte width that no
#     backend special-cases, so it degrades on every backend — this is what an
#     unsupported card (sm_<80, an untested arch, Metal) would run.
# So both paths are covered on any backend and the file is self-gating.

const AC_WG = 64
# Each thread owns a 16-byte (4-dword) shared slot. 16-byte alignment is what the
# cp.async.16 path needs and has no runtime fixup; a static @localmem base is
# ≥32-byte aligned on CUDA, so slot (lid-1) at byte 16*(lid-1) is 16-aligned.
const AC_SLOT = 4

# ND = dwords copied per thread → BYTES = 4*ND. Global is contiguous (stride ND),
# so thread lid's read starts at byte 4*ND*(lid-1) — 16/8/4-aligned for ND=4/2/1,
# which is exactly what cp.async.{16,8,4} require. The round trip is the identity.
@kernel inbounds = true unsafe_indices = true function ac_kernel!(
        dst, src, ::Val{ND}
    ) where {ND}
    lid = @index(Local, Linear)
    gid = @index(Group, Linear)
    smem = @localmem UInt32 (AC_WG * AC_SLOT,)

    gstart = (gid - 1) * AC_WG * ND + (lid - 1) * ND + 1
    sstart = (lid - 1) * AC_SLOT + 1

    KI.async_copy!(pointer(smem, sstart), pointer(src, gstart), Val(4 * ND))
    KI.async_commit()
    KI.async_wait(Val(0))
    @synchronize

    for j in 0:(ND - 1)
        dst[gstart + j] = smem[sstart + j]
    end
end

function run_ac(nd, ngroups)
    N = AC_WG * nd * ngroups
    src = to_device(UInt32.(1:N))
    dst = to_device(zeros(UInt32, N))
    # Workgroup size MUST be AC_WG — the per-thread slot math assumes it.
    ac_kernel!(backend, AC_WG)(dst, src, Val(nd); ndrange = AC_WG * ngroups)
    synchronize(backend)
    return from_device(dst), UInt32.(1:N)
end

@testset "async_copy" begin
    # FALLBACK — 12 bytes is not special-cased by any backend, so this is the
    # register load+store path even on hardware that has cp.async/global_load_lds.
    @testset "register fallback (Val(12))" begin
        got, exp = run_ac(3, 4)
        @test got == exp
    end

    if KI.async_copy_supported(backend)
        @testset "hardware path" begin
            for nd in (1, 2, 4)     # 4, 8, 16 bytes
                got, exp = run_ac(nd, 4)
                @test got == exp
            end
        end

        # Trust the IR, not just the numbers: a numeric pass alone cannot tell a
        # real DMA from a fallback that silently replaced it. Assert the actual
        # COPY instruction is emitted for the 16-byte path — not merely `cp.async`,
        # which the commit/wait scaffolding (`cp.async.commit_group`/`wait_group`)
        # emits on CUDA even when the copy itself has degraded to the register
        # fallback. The data-moving mnemonic is what proves the hardware path.
        @testset "emits the hardware DMA" begin
            dst = to_device(zeros(UInt32, AC_WG * 4 * 2))
            src = to_device(UInt32.(1:(AC_WG * 4 * 2)))
            asm = @capture_ir ac_kernel!(backend, AC_WG)(dst, src, Val(4); ndrange = AC_WG * 2)
            if TEST_BACKEND == "cuda"
                @test occursin("cp.async.ca.shared.global", asm)
            elseif TEST_BACKEND == "roc"
                @test occursin("global_load_lds", asm)
            end
        end
    else
        @info "async_copy_supported=false on this device — hardware path skipped, fallback covered above"
    end
end
