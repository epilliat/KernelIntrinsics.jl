@kernel function test_access_memory_orderings(data::CuDeviceArray)
    # Default access
    @access data[1] = 10
    # Scope-less operations
    @access Volatile data[2] = 20
    @access Weak data[3] = 30
    # Stores with different orderings
    @access Relaxed data[4] = 40
    @access Release data[5] = 50
    # Stores with explicit scopes
    @access Device Release data[8] = 80
    @access Workgroup Release data[9] = 90
    # Loads with different orderings
    a = 0
    @access a = data[1] #Acquire
    @access Volatile b = data[2]
    @access Weak c = data[3]
    @access Relaxed d = data[4]
    @access Acquire e = data[5]
    #@access AcqRel f = data[6]
    # Loads with explicit scopes
    @access System j = data[10]
    @access Device Acquire h = data[8]
    @access Workgroup Acquire i = data[9]
    # Reversed argument order
    @access Release Device data[11] = 110
    @access Acquire Workgroup k = data[11]
    # Fences
    @fence # defaults to Device AcqRel
    @fence Device
    @fence Workgroup AcqRel
    @fence System SeqCst
    @fence AcqRel Device
end
data = CuArray{Int}([0 for i in (1:256)])
buf = IOBuffer()
@device_code_ptx io = buf test_access_memory_orderings(CUDABackend(), 256)(data; ndrange=256)
asm = String(take!(copy(buf)))
@test occursin("volatile", asm)
@test occursin("acquire", asm)
@test occursin("release", asm)
@test occursin("weak", asm)
@test occursin("fence.acq_rel.gpu", asm)
@test occursin("fence.acq_rel.cta", asm)
@test occursin("fence.sc.sys", asm)

#%%
T = UInt8
data = CuMatrix{T}(zeros(50, 50))
@kernel function test_access_multidim_indexing(data::CuDeviceArray)
    @access data[2, 2] = 0x20
end
CUDA.@sync test_access_multidim_indexing(CUDABackend(), 256)(data; ndrange=256)
@test CUDA.@allowscalar data[2, 2] == 0x20