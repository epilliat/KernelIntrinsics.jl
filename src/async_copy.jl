# ============================================================================
# Asynchronous global → workgroup-memory copy (`cp.async` / `global_load_lds`)
# ============================================================================
#
# The one primitive a GEMM inner loop needs to overlap DRAM latency with tensor-
# core math: move a tile from global memory into shared/LDS WITHOUT parking it in
# a register first. On NVIDIA (sm_80+) this is `cp.async` (LDGSTS); on CDNA
# (gfx90a/gfx94x) it is `global_load_lds`. Where the hardware path is absent the
# call still WORKS — it degrades to a plain register-staged load+store — so a
# kernel written once against this API runs everywhere and is merely accelerated
# where the hardware allows. That is the difference from `@dynlocalmem`, which
# errors off-backend: an async copy has an exact, correct fallback, so it must
# never fail.
#
# TWO hardware asymmetries are absorbed here and each is documented at its site:
#
#   1. COPY WIDTH.  NVIDIA `cp.async` moves 4, 8 or 16 bytes per instruction (16
#      is the useful one); AMD `global_load_lds` moves ONE dword (4 bytes) per
#      instruction. So `async_copy!(dst, src, Val(16))` is a single instruction
#      on NVIDIA but FOUR dword issues on AMD. The generic `Val{BYTES}` names the
#      transfer size; each backend emits the right number of hardware ops.
#
#   2. SYNC MODEL.  NVIDIA has a commit-group / wait-group pipeline: issued copies
#      are batched by `async_commit()` into a group, and `async_wait(Val(KEEP))`
#      blocks until all but `KEEP` of the most-recent groups have landed (the
#      knob that lets a double-buffered loop keep the next tile in flight). AMD
#      has no commit-group model: `async_commit()` is a no-op and
#      `async_wait(_)` drains ALL outstanding vector-memory (`s_waitcnt vmcnt(0)`)
#      — KEEP is a NVIDIA-only feature and is IGNORED on AMD and on the fallback.
#      In every case the caller still needs a workgroup barrier (`@synchronize`)
#      after the wait before reading the staged tile.

# Numeric address spaces, identical on both hardware backends this targets:
# global = 1, shared/LDS = 3 (NVPTX `AS.Shared` and AMDGCN `AS.Local` are both 3).
# Using the literal AS in the generic signature lets the backend `@device_override`
# / `@amdgpu_overlay` methods share exactly this signature and shadow it inside a
# kernel; on hosts without a GPU package these methods are simply never reached.
const _AS_GLOBAL = 1
const _AS_SHARED = 3

"""
    async_copy!(dst::Core.LLVMPtr{T,3}, src::Core.LLVMPtr{T,1}, ::Val{BYTES})

Copy `BYTES` bytes from global memory (`src`, address space 1) into
workgroup/shared memory (`dst`, address space 3) **asynchronously**, without
staging the data through a register.

On NVIDIA sm_80+ this issues `cp.async` (a single LDGSTS instruction for
`BYTES ∈ (4, 8, 16)`); on CDNA gfx90a/gfx94x it issues `global_load_lds` (one
4-byte dword per instruction, so `BYTES` must be a multiple of 4 and a 16-byte
copy becomes four issues). On any other backend — or for a `BYTES` the hardware
path does not special-case — it degrades to a correct register-staged
load+store. The pointers are usually obtained with `pointer(shared_array, i)` and
`pointer(global_array, j)` inside a kernel.

A copy issued here is not observable in `dst` until an [`async_wait`](@ref)
(and, on NVIDIA, an intervening [`async_commit`](@ref)) followed by a workgroup
barrier (`@synchronize`).

For 16-byte NVIDIA copies the addresses must be 16-byte aligned (the instruction
has no runtime alignment fixup).

See also: [`async_commit`](@ref), [`async_wait`](@ref),
[`async_copy_supported`](@ref).
"""
@inline function async_copy!(
        dst::Core.LLVMPtr{T,_AS_SHARED}, src::Core.LLVMPtr{T,_AS_GLOBAL}, ::Val{BYTES}
    ) where {T,BYTES}
    # FALLBACK: a plain register-staged copy. `cp.async`/`global_load_lds` are the
    # accelerated paths installed by the backend extensions; with neither loaded
    # (or for an unsupported width) this runs, and it is genuinely correct — the
    # data simply passes through a register, one aligned chunk at a time. Copy in
    # 4-byte dwords when the width allows (natural 4-byte alignment, one scalar
    # load each) and byte-wise otherwise. The two branches delegate to helpers so
    # the reinterpreted pointers are HELPER ARGUMENTS, not if-branch locals — a
    # closure (`ntuple`) capturing an if-branch local boxes it and forces dynamic
    # dispatch, which is invalid GPU IR.
    if BYTES % 4 == 0
        _async_copy_fallback_words(dst, src, Val(BYTES ÷ 4))
    else
        _async_copy_fallback_bytes(dst, src, Val(BYTES))
    end
    return nothing
end

@inline function _async_copy_fallback_words(
        dst::Core.LLVMPtr, src::Core.LLVMPtr, ::Val{NW}
    ) where {NW}
    gp = reinterpret(Core.LLVMPtr{UInt32,_AS_GLOBAL}, src)
    lp = reinterpret(Core.LLVMPtr{UInt32,_AS_SHARED}, dst)
    ntuple(Val(NW)) do w
        unsafe_store!(lp + 4 * (w - 1), unsafe_load(gp + 4 * (w - 1)))
    end
    return nothing
end

@inline function _async_copy_fallback_bytes(
        dst::Core.LLVMPtr, src::Core.LLVMPtr, ::Val{NB}
    ) where {NB}
    gp = reinterpret(Core.LLVMPtr{UInt8,_AS_GLOBAL}, src)
    lp = reinterpret(Core.LLVMPtr{UInt8,_AS_SHARED}, dst)
    ntuple(Val(NB)) do w
        unsafe_store!(lp + (w - 1), unsafe_load(gp + (w - 1)))
    end
    return nothing
end

"""
    async_commit()

Close the group of [`async_copy!`](@ref) operations issued since the previous
commit (NVIDIA `cp.async.commit_group`). A subsequent [`async_wait`](@ref)
counts groups, not individual copies, so this is what delimits one batch.

No-op on AMD (no commit-group model) and on the register fallback.
"""
@inline async_commit() = nothing

"""
    async_wait(::Val{KEEP})

Block until all but the `KEEP` most-recently-committed [`async_commit`](@ref)
groups have completed (NVIDIA `cp.async.wait_group KEEP`). `async_wait(Val(0))`
waits for everything; a double-buffered loop uses `Val(1)` to keep the next
tile's copies in flight.

`KEEP` is a NVIDIA-only feature: on AMD this drains ALL outstanding vector memory
(`s_waitcnt vmcnt(0)`) and on the register fallback it is a no-op — in both cases
`KEEP` is ignored. A workgroup barrier (`@synchronize`) is still required after
the wait before the staged data may be read.
"""
@inline async_wait(::Val{KEEP}) where {KEEP} = nothing

"""
    async_copy_supported(backend) -> Bool

Whether a genuine **hardware** async-copy path exists on the current device of
`backend` (the KernelAbstractions backend). `false` means [`async_copy!`](@ref)
still works but goes through the register fallback — so a caller can pick a
copy strategy but never has to guard the call itself.

Host-side capability query (it inspects the device); not usable inside a kernel.
Takes the backend as an argument, like `mma_supported`, so the extensions
specialize it rather than overwrite a zero-argument default.

See also: [`async_copy!`](@ref).
"""
async_copy_supported(::Any) = false
