# AMD lowering for `@dynlocalmem` / `launch!` / `max_dynamic_localmem`.
import KernelIntrinsics: _dynlocalmem, launch!, max_dynamic_localmem

# --- device side ------------------------------------------------------------
# A zero-length external LDS global is the dynamic-LDS handle: the backend maps
# every such global to the base of the region provisioned by `shmem` at launch,
# which is exactly the flat-blob semantics `@dynlocalmem` promises. This is the
# same primitive `AMDGPU.@ROCDynamicLocalArray` expands to.
#
# `zeroinit` is deliberately `false`: AMDGPU cannot zero dynamic LDS at the
# compiler level, so `zeroinit = true` open-codes a zeroing loop AND a
# `sync_workgroup()` — a hidden barrier inside what looks like an allocation,
# which would deadlock any kernel that allocates under divergence.
@amdgpu_overlay @inline function _dynlocalmem(::Type{T}, dims, offset) where {T}
    ptr = AMDGPU.Device.alloc_local(:KI_dynlocalmem, T, 0, false)
    return AMDGPU.Device.ROCDeviceArray(_as_dims(dims), ptr + offset)
end

@inline _as_dims(dims::Tuple) = dims
@inline _as_dims(dim::Integer) = (dim,)

# --- host side --------------------------------------------------------------
max_dynamic_localmem(dev::AMDGPU.HIPDevice) =
    Int(AMDGPU.HIP.properties(dev).sharedMemPerBlock)
max_dynamic_localmem(::ROCBackend) = max_dynamic_localmem(AMDGPU.device())

# Mirrors AMDGPU.jl's own `(::KA.Kernel{ROCBackend})(args...)` (src/ROCKernels.jl),
# with the dynamic-LDS request threaded through. Unlike CUDA there is no 48 KB
# opt-in step: the whole LDS is addressable, so `shmem` just has to fit.
function launch!(
        obj::KA.Kernel{ROCBackend}, args...;
        ndrange = nothing, workgroupsize = nothing, shmem::Integer = 0
    )
    if shmem > 0
        cap = max_dynamic_localmem(AMDGPU.device())
        shmem > cap && throw(
            ArgumentError(
                "requested $shmem B of dynamic workgroup memory, but this device " *
                    "allows at most $cap B of LDS per block"
            )
        )
    end

    ndrange, new_workgroupsize, iterspace, dynamic = KA.launch_config(obj, ndrange, workgroupsize)
    ctx = KA.mkcontext(obj, ndrange, iterspace)
    kernel = AMDGPU.@roc launch = false obj.f(ctx, args...)

    if KA.workgroupsize(obj) <: KA.DynamicSize && workgroupsize === nothing
        (; groupsize) = AMDGPU.launch_configuration(kernel; shmem = Int(shmem))
        new_workgroupsize = AMDGPU.threads_to_workgroupsize(groupsize, ndrange)
        iterspace, dynamic = KA.partition(obj, ndrange, new_workgroupsize)
        ctx = KA.mkcontext(obj, ndrange, iterspace)
    end

    nblocks = length(KA.blocks(iterspace))
    nthreads = length(KA.workitems(iterspace))
    nblocks == 0 && return nothing

    kernel(ctx, args...; groupsize = nthreads, gridsize = nblocks, shmem = Int(shmem))
    return nothing
end
