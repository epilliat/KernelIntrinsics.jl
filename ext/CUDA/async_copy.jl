# CUDA lowering for `async_copy!` / `async_commit` / `async_wait` /
# `async_copy_supported` — the `cp.async` (LDGSTS) path, sm_80+.
#
# The primitives live in the `CUDA.CG` submodule (cooperative groups):
#   CG.pipeline_memcpy_async(dst::LLVMPtr{T,Shared}, src::LLVMPtr{T,Global})
#       → cp.async.ca.shared.global.$(sizeof(T)),  sizeof(T) ∈ (4, 8, 16)
#   CG.pipeline_commit()                → cp.async.commit_group
#   CG.pipeline_wait_prior(Int32(n))    → cp.async.wait_group n
# Verified on RTX1000 (Ada) in KernelForge xp/gemm/cpasync_probe.jl.
#
# These are SIDE-EFFECTING void ops (a copy, a commit, a wait) — no returned
# fragment, no loop-carried phi — so `@device_override` is safe here (this is
# NOT the undef-accumulator overlay hazard that forced the MMA backend token).
import KernelIntrinsics: async_copy!, async_commit, async_wait, async_copy_supported
import KernelIntrinsics: _AS_GLOBAL, _AS_SHARED

# --- device side ------------------------------------------------------------
# `pipeline_memcpy_async` selects the instruction by `sizeof(T)`, so we only need
# to reinterpret the caller's pointers to a carrier of the requested byte width
# and issue ONE copy. Width is fixed by the `Val` — a distinct method per width
# so that an unsupported `BYTES` falls through to the generic register fallback
# rather than erroring. 16 bytes is the useful one; 4 and 8 are provided too.
@inline function _cu_async_copy!(dst::Core.LLVMPtr, src::Core.LLVMPtr, ::Type{C}) where {C}
    d = reinterpret(Core.LLVMPtr{C,AS.Shared}, dst)
    s = reinterpret(Core.LLVMPtr{C,AS.Global}, src)
    CUDA.CG.pipeline_memcpy_async(d, s)
    return nothing
end

CUDA.@device_override @inline async_copy!(
    dst::Core.LLVMPtr{T,_AS_SHARED}, src::Core.LLVMPtr{T,_AS_GLOBAL}, ::Val{16}
) where {T} = _cu_async_copy!(dst, src, NTuple{8,Float16})   # 16 B → cp.async.*.16

CUDA.@device_override @inline async_copy!(
    dst::Core.LLVMPtr{T,_AS_SHARED}, src::Core.LLVMPtr{T,_AS_GLOBAL}, ::Val{8}
) where {T} = _cu_async_copy!(dst, src, Float64)             # 8 B → cp.async.*.8

CUDA.@device_override @inline async_copy!(
    dst::Core.LLVMPtr{T,_AS_SHARED}, src::Core.LLVMPtr{T,_AS_GLOBAL}, ::Val{4}
) where {T} = _cu_async_copy!(dst, src, Float32)             # 4 B → cp.async.*.4

CUDA.@device_override @inline async_commit() = CUDA.CG.pipeline_commit()

CUDA.@device_override @inline async_wait(::Val{KEEP}) where {KEEP} =
    CUDA.CG.pipeline_wait_prior(Int32(KEEP))

# --- host side --------------------------------------------------------------
# cp.async is Ampere (sm_80) and up; on older cards the override methods would
# emit an instruction ptxas rejects, so the CALLER must consult this and fall
# back. A real device query, not an arch table.
async_copy_supported(::CUDABackend) = CUDA.capability(CUDA.device()) >= v"8.0"
