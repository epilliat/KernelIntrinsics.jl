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