# AMD lowering for `async_copy!` / `async_wait` / `async_copy_supported` — the
# `global_load_lds` direct global→LDS DMA path (CDNA gfx90a/gfx94x).
#
# AMDGPU.jl does not wrap this, so we emit the amdgcn intrinsic by hand exactly
# as verified on MI300A (gfx942) in KernelForge xp/gemm/amd_lds_probe.jl, which
# COMPILED, RAN CORRECTLY, and whose GCN ISA showed the real `global_load_lds_dword`:
#
#   void @llvm.amdgcn.global.load.lds(ptr addrspace(1) g, ptr addrspace(3) l,
#                                     i32 size, i32 offset, i32 aux)     size ∈ {1,2,4}
#   void @llvm.amdgcn.s.waitcnt(i32 0)                 # drain all vmcnt
#
# The hardware moves ONE dword (4 bytes) per lane per instruction — NOT 16 like
# `cp.async`. So a `Val{BYTES}` copy issues `BYTES ÷ 4` dword DMAs at 4-byte
# strides (this is copy-width asymmetry #1 in src/async_copy.jl). There is no
# commit-group model, so `async_commit()` keeps the generic no-op and
# `async_wait` drains everything with `s_waitcnt vmcnt(0)`, ignoring KEEP
# (asymmetry #2). A `@synchronize` after the wait is still the caller's job.
#
# These are void side-effecting ops, so `@amdgpu_overlay` is safe (same rationale
# as the CUDA `@device_override`; NOT the MMA undef-fragment hazard).
import KernelIntrinsics: async_copy!, async_wait, async_copy_supported
import KernelIntrinsics: _AS_GLOBAL, _AS_SHARED

# --- device side ------------------------------------------------------------
# One dword (4 B) global→LDS DMA. `offset`/`aux` are 0; we advance the pointers
# themselves (byte arithmetic on LLVMPtr) instead of using the intrinsic's
# immediate offset, which keeps every issue a plain size-4 load.
@inline function _amd_load_lds_dword(g::Core.LLVMPtr{UInt32,_AS_GLOBAL}, l::Core.LLVMPtr{UInt32,_AS_SHARED})
    ccall(
        "llvm.amdgcn.global.load.lds", llvmcall, Cvoid,
        (Core.LLVMPtr{UInt32,_AS_GLOBAL}, Core.LLVMPtr{UInt32,_AS_SHARED}, Int32, Int32, Int32),
        g, l, Int32(4), Int32(0), Int32(0),
    )
    return nothing
end

# BYTES ÷ 4 dword DMAs, unrolled. Distinct method per width so a non-multiple-of-4
# `BYTES` falls through to the generic register fallback instead of issuing a
# partial dword.
@inline function _amd_async_copy!(dst::Core.LLVMPtr, src::Core.LLVMPtr, ::Val{NDWORD}) where {NDWORD}
    g = reinterpret(Core.LLVMPtr{UInt32,_AS_GLOBAL}, src)
    l = reinterpret(Core.LLVMPtr{UInt32,_AS_SHARED}, dst)
    ntuple(Val(NDWORD)) do w
        _amd_load_lds_dword(g + 4 * (w - 1), l + 4 * (w - 1))
    end
    return nothing
end

@amdgpu_overlay @inline async_copy!(
    dst::Core.LLVMPtr{T,_AS_SHARED}, src::Core.LLVMPtr{T,_AS_GLOBAL}, ::Val{16}
) where {T} = _amd_async_copy!(dst, src, Val(4))

@amdgpu_overlay @inline async_copy!(
    dst::Core.LLVMPtr{T,_AS_SHARED}, src::Core.LLVMPtr{T,_AS_GLOBAL}, ::Val{8}
) where {T} = _amd_async_copy!(dst, src, Val(2))

@amdgpu_overlay @inline async_copy!(
    dst::Core.LLVMPtr{T,_AS_SHARED}, src::Core.LLVMPtr{T,_AS_GLOBAL}, ::Val{4}
) where {T} = _amd_async_copy!(dst, src, Val(1))

# `s_waitcnt vmcnt(0)` drains the outstanding LDS DMAs (and every other vector
# memory op) before the staged data is read. KEEP is a NVIDIA commit-group notion
# with no AMD equivalent, so it is ignored — documented in src/async_copy.jl.
@amdgpu_overlay @inline async_wait(::Val{KEEP}) where {KEEP} =
    ccall("llvm.amdgcn.s.waitcnt", llvmcall, Cvoid, (Int32,), Int32(0))

# --- host side --------------------------------------------------------------
# `global_load_lds` is known-good on gfx90a and the gfx94x / gfx950 CDNA parts;
# gfx942 (MI300A/X) is the one hardware-verified here. Be conservative — return
# true only on architectures where the DMA is known correct, so an unlisted arch
# takes the register fallback rather than a wrong or non-selectable instruction.
const _ASYNC_LDS_ARCHS = ("gfx90a", "gfx940", "gfx941", "gfx942", "gfx950")

function async_copy_supported(::ROCBackend)
    gfx = first(split(AMDGPU.device().gcn_arch, ':'))   # "gfx942:sramecc+:xnack-" → "gfx942"
    return gfx in _ASYNC_LDS_ARCHS
end
