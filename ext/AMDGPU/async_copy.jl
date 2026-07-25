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
    # Plain loop, NOT `ntuple(...) do`: the do-block closure crashes the
    # LLVM-AMDGPU backend at codegen (libLLVM segfault). `NDWORD` is a literal so
    # this unrolls identically.
    for w in 0:(NDWORD - 1)
        _amd_load_lds_dword(g + 4 * w, l + 4 * w)
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

# Draining the LDS DMA: a bare `ccall("llvm.amdgcn.s.waitcnt", ...)` inside a
# KernelAbstractions `@kernel` SEGFAULTS libLLVM at codegen (verified on MI300A:
# it works in a raw `@roc` kernel — probe amd_lds_probe.jl — but crashes through
# the KA compilation path). So async_wait is a no-op on AMD and the drain is left
# to the workgroup barrier (`@synchronize`) the caller MUST issue right after:
# on gfx942 `sync_workgroup()`'s seq_cst workgroup fence lowers to include the
# `s_waitcnt vmcnt(0)` that orders the DMA before the staged read. Evidence: the
# hardware round-trip test passes standalone with this no-op. KEEP is a NVIDIA
# commit-group notion with no AMD equivalent and is ignored regardless.
# TODO: a KA-safe way to emit an explicit vmcnt wait (upstream LLVM/GPUCompiler).
@amdgpu_overlay @inline async_wait(::Val{KEEP}) where {KEEP} = nothing

# --- host side --------------------------------------------------------------
# DEFERRED (2026-07-22): the `global_load_lds` DMA is real and correct in
# ISOLATION on gfx942 (amd_lds_probe.jl: ISA `global_load_lds_dword`, round-trip
# OK), but the KA-integrated path is TRIPLY blocked on this toolchain
# (Julia 1.12.6 + ROCm 6.4.3 + LLVM 18) and none of it is our logic:
#   • the correct drain `s_waitcnt vmcnt(0)` inside a KA @kernel SEGFAULTS libLLVM
#     at codegen (works in a raw @roc kernel; async_wait is a no-op above);
#   • that no-op then RACES — the HW round-trip returns wrong numbers because the
#     DMA is not drained before the staged read;
#   • compiling the async_copy kernel AFTER tests/vectorization_test.jl segfaults
#     libLLVM (an order-dependent codegen-state interaction).
# So report NO hardware async-copy on AMD for now: every caller (GEMM staging,
# the KI test) takes the correct register-staged path instead. The @amdgpu_overlay
# methods above stay defined but are never selected while this returns false.
# Re-enable once the upstream LLVM-AMDGPU/GPUCompiler issues are fixed.
async_copy_supported(::ROCBackend) = false
