@kernel function test_amd_stores(data::ROCDeviceArray)
    @access Relaxed data[4] = 40
    @access Release data[5] = 50
    @access Device Release data[8] = 80
    @access Workgroup Release data[9] = 90
end

@kernel function test_amd_loads(data::ROCDeviceArray, dst::ROCDeviceArray)
    a = @access Relaxed data[4]
    b = @access Acquire data[5]
    c = @access Device Acquire data[8]
    d = @access Workgroup Acquire data[9]
    #e = @access System data[10]
    @access dst[1] = a
    @access dst[2] = b
    @access dst[3] = c
    @access dst[4] = d
    #@access Release dst[5] = e
end

@kernel function test_amd_fences(data::ROCDeviceArray)
    @fence
    @fence Device
    @fence Workgroup AcqRel
    #@fence System SeqCst
    @fence AcqRel Device
end

@kernel function test_amd_multidim(data::ROCDeviceArray)
    @access data[2, 2] = 0x20
end

# ── Run and capture GCN ISA ───────────────────────────────────────────────────

data = ROCArray{Int}(zeros(256))
dst = ROCArray{Int}(zeros(256))
buf = IOBuffer()

@device_code_gcn io = buf test_amd_stores(ROCBackend(), 256)(data; ndrange=256)
asm_stores = String(take!(copy(buf)))

@device_code_gcn io = buf test_amd_loads(ROCBackend(), 256)(data, dst; ndrange=256)
asm_loads = String(take!(copy(buf)))

@device_code_gcn io = buf test_amd_fences(ROCBackend(), 256)(data; ndrange=256)
asm_fences = String(take!(copy(buf)))

# ── Store tests ───────────────────────────────────────────────────────────────
@test occursin("sc1", asm_stores)   # device scope
@test occursin("sc0", asm_stores)   # workgroup scope

# ── Load tests ────────────────────────────────────────────────────────────────
@test occursin("s_waitcnt vmcnt(0)", asm_loads)   # acquire barrier
@test occursin("sc1", asm_loads)                  # device scope
@test occursin("sc0", asm_loads)                  # workgroup scope

# ── Fence tests ───────────────────────────────────────────────────────────────
@test occursin("s_waitcnt", asm_fences)
@test occursin("buffer_wbl2", asm_fences) || occursin("global_wb", asm_fences)
@test occursin("buffer_inv", asm_fences) || occursin("global_inv", asm_fences)

# ── Multidim ──────────────────────────────────────────────────────────────────
T = UInt8
data2 = ROCMatrix{T}(zeros(50, 50))
AMDGPU.@sync test_amd_multidim(ROCBackend(), 256)(data2; ndrange=256)
@test AMDGPU.@allowscalar data2[2, 2] == 0x20
# ── 128-bit atomics (UInt128 / Int128, Device/Relaxed) — gfx94x ───────────────
#
# One coherent+atomic `global_{load,store}_dwordx4 … sc1` + `s_waitcnt vmcnt(0)`. The SINGLE
# instruction is the contract: it makes the 16 bytes a torn-free snapshot, which is what lets a
# {status,value} descriptor be published/read as one unit (decoupled-lookback packed descriptor).
# A split into two dwordx2 would still be coherent but NOT atomic — it could tear — so the ISA test
# below asserts the load is NOT split.
#
# RESTRICTED TO 16-BYTE PRIMITIVE TYPES. A composite 16-byte aggregate (ComplexF64, NTuple{2,UInt64},
# NTuple{4,UInt32}, structs — incl. nested / 4-byte-aligned fields) CANNOT be carried: `reinterpret`
# of a composite does not GPU-codegen, and an aggregate is rejected by the `=v`/`v` asm constraint.
# Both were verified on gfx942; such payloads must use the split (flag + value) protocol instead.

@kernel function test_atomic128_store!(dst, src)
    i = Int(@index(Global))
    KI.atomic_store!(dst, i, src[i], KI.Device, KI.Relaxed)
end

@kernel function test_atomic128_load!(dst, src)
    i = Int(@index(Global))
    dst[i] = KI.atomic_load(src, i, KI.Device, KI.Relaxed)
end

for T in (UInt128, Int128)
    n = 64
    # values whose HIGH and LOW 64-bit halves both differ per element: a half-load / torn load
    # (e.g. a split into 2× dwordx2 picking up mismatched halves) cannot pass this.
    host = T[(T(i) << 64) | T(0x5a5a5a5a00000000 + i) for i in 1:n]

    # atomic_store! roundtrip: GPU atomically stores src → dst, host compares
    src = ROCArray(host)
    dst = ROCArray(zeros(T, n))
    AMDGPU.@sync test_atomic128_store!(ROCBackend(), 64)(dst, src; ndrange=n)
    @test Array(dst) == host

    # atomic_load roundtrip: GPU atomically loads src → dst, host compares
    src2 = ROCArray(host)
    dst2 = ROCArray(zeros(T, n))
    AMDGPU.@sync test_atomic128_load!(ROCBackend(), 64)(dst2, src2; ndrange=n)
    @test Array(dst2) == host
end

# ── ISA: one dwordx4 + sc1 + drain, and NOT split into two dwordx2 ────────────
let n = 64, src = ROCArray(zeros(UInt128, n)), dst = ROCArray(zeros(UInt128, n))
    buf128 = IOBuffer()
    @device_code_gcn io = buf128 test_atomic128_load!(ROCBackend(), 64)(dst, src; ndrange=n)
    asm_ld128 = String(take!(copy(buf128)))
    @test occursin("global_load_dwordx4", asm_ld128)      # single 16-byte load
    @test !occursin("global_load_dwordx2", asm_ld128)     # NOT split → stays atomic (no tearing)
    @test occursin("sc1", asm_ld128)                      # coherent at agent/Device scope
    @test occursin("s_waitcnt vmcnt(0)", asm_ld128)

    buf128s = IOBuffer()
    @device_code_gcn io = buf128s test_atomic128_store!(ROCBackend(), 64)(dst, src; ndrange=n)
    asm_st128 = String(take!(copy(buf128s)))
    @test occursin("global_store_dwordx4", asm_st128)
    @test !occursin("global_store_dwordx2", asm_st128)
    @test occursin("sc1", asm_st128)
end
