# CUDA lowering for `@dynlocalmem` / `launch!` / `max_dynamic_localmem`.
import KernelIntrinsics: _dynlocalmem, launch!, max_dynamic_localmem

# --- device side ------------------------------------------------------------
# `CuDynamicSharedArray` bounds-checks `offset + sizeof` against the dynamic
# region actually provisioned at launch, so an under-sized `shmem` surfaces as a
# BoundsError rather than as silent corruption. Keep that check reachable.
CUDA.@device_override Base.@propagate_inbounds function _dynlocalmem(
        ::Type{T}, dims, offset
    ) where {T}
    return CUDA.CuDynamicSharedArray(T, dims, offset)
end

# --- host side --------------------------------------------------------------
max_dynamic_localmem(dev::CUDA.CuDevice) =
    Int(CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN))
max_dynamic_localmem(::CUDABackend) = max_dynamic_localmem(CUDA.device())

# Anything at or below this is available to every kernel without asking; beyond
# it, the function must opt in explicitly (and the opt-in is per compiled
# function, so it has to happen after compilation and before the launch).
const _CUDA_STATIC_SHMEM_CAP = 48 * 1024

# Mirrors CUDA.jl's own `(::KA.Kernel{CUDABackend})(args...)` (src/CUDAKernels.jl),
# with the dynamic-shared-memory request threaded through. Fold this back into
# `KernelAbstractions` if it ever grows a `shmem` keyword.
function launch!(
        obj::KA.Kernel{CUDABackend}, args...;
        ndrange = nothing, workgroupsize = nothing, shmem::Integer = 0
    )
    backend = KA.backend(obj)

    if shmem > 0
        cap = max_dynamic_localmem(CUDA.device())
        shmem > cap && throw(
            ArgumentError(
                "requested $shmem B of dynamic workgroup memory, but " *
                    "$(CUDA.name(CUDA.device())) allows at most $cap B per block"
            )
        )
    end

    ndrange, workgroupsize, iterspace, dynamic = KA.launch_config(obj, ndrange, workgroupsize)
    ctx = KA.mkcontext(obj, ndrange, iterspace)

    # A static workgroup size lets the compiler bound the register budget, which
    # is not optional above 256 threads: regs x threads <= 65536, so a kernel
    # compiled for an unbounded 256-reg budget cannot launch at 512 or 1024.
    maxthreads = if KA.workgroupsize(obj) <: KA.StaticSize
        prod(KA.get(KA.workgroupsize(obj)))
    else
        nothing
    end

    kernel = @cuda launch = false always_inline = backend.always_inline maxthreads = maxthreads obj.f(ctx, args...)

    if shmem > _CUDA_STATIC_SHMEM_CAP
        CUDA.attributes(kernel.fun)[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = shmem
    end

    if KA.workgroupsize(obj) <: KA.DynamicSize && workgroupsize === nothing
        config = CUDA.launch_configuration(kernel.fun; shmem = Int(shmem), max_threads = prod(ndrange))
        threads = if backend.prefer_blocks
            t = min(prod(ndrange), config.threads)
            cu_blocks = max(cld(prod(ndrange), t), config.blocks)
            cld(prod(ndrange), cu_blocks)
        else
            config.threads
        end
        workgroupsize = CUDA.threads_to_workgroupsize(threads, ndrange)
        iterspace, dynamic = KA.partition(obj, ndrange, workgroupsize)
        ctx = KA.mkcontext(obj, ndrange, iterspace)
    end

    blocks = length(KA.blocks(iterspace))
    threads = length(KA.workitems(iterspace))
    blocks == 0 && return nothing

    kernel(ctx, args...; threads, blocks, shmem = Int(shmem))
    return nothing
end
