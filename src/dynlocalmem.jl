# ============================================================================
# Dynamic workgroup memory (`shared` on CUDA, `LDS` on AMD)
# ============================================================================
#
# `KernelAbstractions.@localmem` allocates workgroup memory *statically*, and
# ptxas hard-caps a static allocation at 48 KB on every NVIDIA architecture —
# a cap the size of the allocation cannot buy its way out of. The dynamic
# allocation path is the only way to reach the real per-block limit (163 KB on
# A100, 99 KB on Ada, 64 KB of LDS on gfx942), and it is opt-in: the amount is
# a *launch* parameter, so the device-side allocation and the host-side launch
# must agree.
#
# Unlike `@localmem` — which hands out N independent arrays — the dynamic
# region is ONE flat blob that the caller carves up with byte offsets. Compute
# the layout in a single pure function and call it from both the host (to size
# the launch) and the device (to place each array); deriving the two sides
# separately is the one way to corrupt this silently.

"""
    _dynlocalmem(T, dims, offset)

Backend dispatch target for [`@dynlocalmem`](@ref). GPU backends override this;
there is no host/CPU implementation (dynamic workgroup memory is a GPU concept).
"""
@inline function _dynlocalmem(::Type{T}, dims, offset) where {T}
    return error(
        "@dynlocalmem requires a GPU backend (CUDA or AMDGPU) to be loaded, " *
            "and is only valid inside a kernel."
    )
end

"""
    @dynlocalmem T dims [offset]

Allocate an array of `dims` elements of type `T` inside the workgroup's
**dynamic** memory region, starting `offset` **bytes** into that region.

The region is a single flat blob whose total size is fixed at launch by the
`shmem` keyword of [`launch!`](@ref). Every `@dynlocalmem` in a kernel views
that same blob — they are distinguished only by `offset`, so overlapping
offsets alias. `offset` is in bytes and must be aligned for `T`.

Unlike `@localmem`, this is not capped at 48 KB: query the device limit with
[`max_dynamic_localmem`](@ref) and pass a matching `shmem` at launch.

```julia
@kernel inbounds = true unsafe_indices = true function k!(dst, src, ::Val{off})
    hist   = @dynlocalmem UInt32 (256,)          # bytes [0, 1024)
    staged = @dynlocalmem Float32 (1024,) off    # bytes [off, off + 4096)
    ...
end

# host side — the SAME layout function feeds both the offsets and the total
KI.launch!(k!(backend, 256), dst, src, Val(1024); ndrange = n, shmem = 1024 + 4096)
```

See also: [`launch!`](@ref), [`max_dynamic_localmem`](@ref).
"""
macro dynlocalmem(T, dims, offset = 0)
    return quote
        $_dynlocalmem($(esc(T)), $(esc(dims)), $(esc(offset)))
    end
end

"""
    max_dynamic_localmem(dev) -> Int
    max_dynamic_localmem(backend) -> Int

Largest `shmem` (in bytes) a single workgroup may request on `dev`.

This is a **runtime query**, never an architecture table: CUDA reports
`MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` (163 KB on A100, 99 KB on Ada, 48 KB on
pre-Volta), AMD reports the LDS size (64 KB on gfx942).

See also: [`@dynlocalmem`](@ref), [`launch!`](@ref).
"""
function max_dynamic_localmem end

"""
    launch!(kernel, args...; ndrange, workgroupsize = nothing, shmem = 0)

Launch a `KernelAbstractions.Kernel` with `shmem` bytes of dynamic workgroup
memory, reachable from the kernel body via [`@dynlocalmem`](@ref).

Equivalent to calling the kernel directly (`kernel(args...; ndrange)`) except
for the `shmem` region, which `KernelAbstractions` does not expose. On CUDA,
`shmem` above 48 KB additionally requires a per-function opt-in; `launch!`
performs it.

Requesting more than [`max_dynamic_localmem`](@ref) throws — it never hangs, and
the error is not sticky, so a search over launch configurations can catch it and
carry on in the same process.
"""
function launch! end
