# ext/AMDGPUExt/warp.jl
import KernelIntrinsics: Up, Down, Xor, Idx
import KernelIntrinsics: All, AnyLane, Uni, Ballot
import KernelIntrinsics: MatchAny
import KernelIntrinsics: _shfl, _shfl_recurse, shfl_at, _vote, _match
using Base.Cartesian: @nexprs

# ── Shuffle ───────────────────────────────────────────────────────────────────
# Note: no mask argument on AMD (wavefront participation is the hardware exec register).
# delta/lane must be Int32.
#
# We do NOT use AMDGPU.Device.shfl_*. Those derive the source lane from
# `activelane()` = `__ockl_activelane_u32` = `mbcnt(EXEC)` = the rank among the *ACTIVE*
# lanes. HIP's `__shfl` uses `__lane_id()` = `mbcnt` with an ALL-ONES mask = the *physical*
# lane. The two differ the moment the wavefront diverges, and that costs twice:
#
#   * CORRECTNESS: under divergence `activelane()` returns a COMPACTED index, so the
#     shuffle reads the wrong lane. Observed as a GPU memory-access fault whenever the
#     compiler fully unrolled a shuffle sequence (a divergent `if` around the reduce was
#     accidentally *preventing* the unroll and hiding it).
#   * PERFORMANCE: `activelane()` depends on EXEC, so LLVM can neither hoist nor CSE it and
#     recomputes the lane index at EVERY shuffle. Measured on KernelForge's MI300A F64 scan
#     lookback: 9 `activelane()` (9 `v_mbcnt_lo` + 9 `v_mbcnt_hi`) for 8 `ds_bpermute`, plus
#     the width arithmetic and guard around each. Switching to the physical lane took that
#     scan from 40.2% to 44.8% of peak (−10%), dropped `v_readlane_b32` 23 → 0 and VGPR
#     spills 43 → 20, and made the unrolled variants pass.
#
# So: physical lane + a direct `ds_bpermute` at a precomputed index. No activelane, no
# `wavefrontsize()`, no width arithmetic.
#
# Hard-codes wave64, exactly like `_vote(Ballot, …)` below and for the same reason. If RDNA
# (wave32) support is added, dispatch on the compile-time target arch, NOT on a runtime
# `wavefrontsize()` check.
const ROC_WAVE = Int32(64)

# HIP's __lane_id(): mbcnt over an all-ones mask, so the result is the PHYSICAL lane and is
# independent of EXEC — hence hoistable and CSE-able, unlike activelane().
@inline _lane_id() = ccall("llvm.amdgcn.mbcnt.hi", llvmcall, UInt32, (UInt32, UInt32),
    0xffffffff, ccall("llvm.amdgcn.mbcnt.lo", llvmcall, UInt32, (UInt32, UInt32),
                      0xffffffff, UInt32(0)))
@inline _self() = reinterpret(Int32, _lane_id())

# `ds_bpermute` addresses lanes by BYTE offset (lane << 2) and operates on 32 bits, so each
# base type is bitcast through Int32. Wider types reach us via `_shfl_recurse` (src/warp.jl).
@inline _bperm(idx::Int32, v::Int32) = AMDGPU.Device.bpermute(idx << 0x2, v)

# Out-of-range source lane returns the lane's own value — the semantics of AMD's shfl_up /
# shfl_down (and of CUDA's shfl with a full-width segment).
@inline _shfl_idx_i32(v::Int32, lane::Int32)  = _bperm(lane & (ROC_WAVE - Int32(1)), v)
@inline function _shfl_up_i32(v::Int32, δ::Int32)
    self = _self(); idx = self - δ
    _bperm(ifelse(idx < Int32(0), self, idx), v)
end
@inline function _shfl_down_i32(v::Int32, δ::Int32)
    self = _self(); idx = self + δ
    _bperm(ifelse(idx >= ROC_WAVE, self, idx), v)
end
@inline _shfl_xor_i32(v::Int32, m::Int32) = _bperm(_self() ⊻ m, v)

const ROC_SHFL_IMPL = Dict(
    Up => :_shfl_up_i32,
    Down => :_shfl_down_i32,
    Xor => :_shfl_xor_i32,
)

for (direction, impl) in ROC_SHFL_IMPL
    @eval begin
        Base.Experimental.@overlay AMDGPU.method_table @inline _shfl(::Type{$direction}, mask, val::Int32, src) =
            $impl(val, Int32(src))
        Base.Experimental.@overlay AMDGPU.method_table @inline _shfl(::Type{$direction}, mask, val::UInt32, src) =
            reinterpret(UInt32, $impl(reinterpret(Int32, val), Int32(src)))
        Base.Experimental.@overlay AMDGPU.method_table @inline _shfl(::Type{$direction}, mask, val::Float32, src) =
            reinterpret(Float32, $impl(reinterpret(Int32, val), Int32(src)))
    end
end

# ── shfl_at: the caller hands us its PHYSICAL lane ───────────────────────────
# Same permute; we just skip rederiving the lane. `@warpreduce` cannot use this — its `lane` may
# be segment-local (see the shfl_at docstring) — so it is an explicit opt-in for callers that
# know their lane is the hardware one. On KernelForge's F64 scan that removed a 12-VGPR scratch
# spill and ~3.5%: without it the kernel holds its own lane AND the mbcnt-derived one.
#
# Written out method by method rather than generated in a loop: `Base.Experimental.@overlay` has
# to see a literal function definition, and it rejects one that reaches it through @eval with a
# `where` clause ("@overlay requires a function definition").
@inline function _idx_up_at(δ::Int32, lane::Int32)
    self = lane - Int32(1); idx = self - δ
    ifelse(idx < Int32(0), self, idx)
end
@inline function _idx_down_at(δ::Int32, lane::Int32, ws::Int32)
    self = lane - Int32(1); idx = self + δ
    ifelse(idx >= ws, self, idx)
end

Base.Experimental.@overlay AMDGPU.method_table @inline function shfl_at(::Type{Up}, val::Int32, src, lane, ::Val{ws}) where {ws}
    _bperm(_idx_up_at(Int32(src), Int32(lane)), val)
end
Base.Experimental.@overlay AMDGPU.method_table @inline function shfl_at(::Type{Up}, val::UInt32, src, lane, ::Val{ws}) where {ws}
    reinterpret(UInt32, _bperm(_idx_up_at(Int32(src), Int32(lane)), reinterpret(Int32, val)))
end
Base.Experimental.@overlay AMDGPU.method_table @inline function shfl_at(::Type{Up}, val::Float32, src, lane, ::Val{ws}) where {ws}
    reinterpret(Float32, _bperm(_idx_up_at(Int32(src), Int32(lane)), reinterpret(Int32, val)))
end
# Wider types (Float64/Int64/composites) split into 32-bit halves. This MUST recurse through
# `shfl_at`, not `_shfl`: the generic fallback drops the lane, and a Float64 warp scan is
# precisely what the lane was added for.
Base.Experimental.@overlay AMDGPU.method_table @inline function shfl_at(::Type{Up}, val, src, lane, ::Val{ws}) where {ws}
    _shfl_recurse(x -> shfl_at(Up, x, src, lane, Val(ws)), val)
end

Base.Experimental.@overlay AMDGPU.method_table @inline function shfl_at(::Type{Down}, val::Int32, src, lane, ::Val{ws}) where {ws}
    _bperm(_idx_down_at(Int32(src), Int32(lane), Int32(ws)), val)
end
Base.Experimental.@overlay AMDGPU.method_table @inline function shfl_at(::Type{Down}, val::UInt32, src, lane, ::Val{ws}) where {ws}
    reinterpret(UInt32, _bperm(_idx_down_at(Int32(src), Int32(lane), Int32(ws)), reinterpret(Int32, val)))
end
Base.Experimental.@overlay AMDGPU.method_table @inline function shfl_at(::Type{Down}, val::Float32, src, lane, ::Val{ws}) where {ws}
    reinterpret(Float32, _bperm(_idx_down_at(Int32(src), Int32(lane), Int32(ws)), reinterpret(Int32, val)))
end
Base.Experimental.@overlay AMDGPU.method_table @inline function shfl_at(::Type{Down}, val, src, lane, ::Val{ws}) where {ws}
    _shfl_recurse(x -> shfl_at(Down, x, src, lane, Val(ws)), val)
end

# `Idx` takes a 1-based source lane (KernelIntrinsics convention).
Base.Experimental.@overlay AMDGPU.method_table @inline _shfl(::Type{Idx}, mask, val::Int32, src) =
    _shfl_idx_i32(val, Int32(src) - Int32(1))
Base.Experimental.@overlay AMDGPU.method_table @inline _shfl(::Type{Idx}, mask, val::UInt32, src) =
    reinterpret(UInt32, _shfl_idx_i32(reinterpret(Int32, val), Int32(src) - Int32(1)))
Base.Experimental.@overlay AMDGPU.method_table @inline _shfl(::Type{Idx}, mask, val::Float32, src) =
    reinterpret(Float32, _shfl_idx_i32(reinterpret(Int32, val), Int32(src) - Int32(1)))

# ── Vote ──────────────────────────────────────────────────────────────────────
# AMDGPU does not have a `Uni` primitive (uniform predicate vote) — would have
# to approximate it with `ballot(pred) == activemask()`, same as `All`.
# `mask` is ignored on AMDGPU (wavefront participation is governed by the
# hardware `exec` mask register, not a software mask argument).
# `AMDGPU.Device.ballot(pred)::UInt64` always returns UInt64, with a runtime
# branch on `wavefrontsize()` — UInt32 result on wave32 widened to UInt64.
#
# Only `Ballot` is enabled. The other three modes (All / AnyLane / Uni) are
# available via the cross-backend `_vote` polyfill chain or, if you want
# native paths, can be added here using `AMDGPU.Device.ballot` + comparison.
# Direct LLVM intrinsic — bypasses AMDGPU.Device.ballot, which has a runtime
# branch on `wavefrontsize() == 32`. On gfx9xx (CDNA, incl. MI300X) the dead
# wave32 arm — `ccall("llvm.amdgcn.ballot", llvmcall, UInt32, …)` — emits a
# SETCC the AMDGPU instruction selector can't lower, failing the entire
# kernel compile with `Cannot select: i32 = SETCC ...`.
#
# Note on the intrinsic name: `llvm.amdgcn.ballot.i64` is the canonical
# overload-mangled name; AMDGPU.jl's older `llvm.amdgcn.ballot.w64` spelling
# compiles on the current LLVM/GPUCompiler combo but generates a kernel that
# faults at runtime. Use `.i64`.
#
# Hard-codes wave64. If RDNA (wave32) support is added later, dispatch on the
# compile-time target arch (e.g. via @device_override or AMDGPU.Compiler.target)
# rather than re-introducing the runtime wavefrontsize() check.
Base.Experimental.@overlay AMDGPU.method_table @inline _vote(::Type{Ballot}, mask, pred) =
    ccall("llvm.amdgcn.ballot.i64", llvmcall, UInt64, (Bool,), pred)


# ── Match ─────────────────────────────────────────────────────────────────────
# The portable polyfill in src/warp.jl can't be used here: its `@generated`
# body calls `_vote(Ballot, …)`, but Base.Experimental.@overlay overrides
# don't propagate into the polyfill body during inference, leaving _match
# dynamically dispatched (same root cause as the original @match failure on
# AMDGPU). Inline the polyfill body directly into AMDGPU's overlay using the
# LLVM ballot intrinsic, so type inference stays inside the overlay context.

# Per-type unrolled match.any polyfill, generated at @eval time. Each
# specialization is a flat sequence of ballot calls + AND-tree updates,
# avoiding any @generated indirection (which we observed gets dynamically
# dispatched when called from within an @overlay-installed method).
for T in (UInt8, UInt16, UInt32, UInt64)
    Nbits = 8 * sizeof(T)
    body = Expr(:block,
        :(active = ccall("llvm.amdgcn.ballot.i64", llvmcall, UInt64, (Bool,), true)),
        :(result = active),
    )
    for b in 1:Nbits
        push!(body.args, quote
            let bit_b = ((value >> $(b - 1)) & one($T)) != zero($T)
                ballot_b = ccall("llvm.amdgcn.ballot.i64", llvmcall, UInt64, (Bool,), bit_b)
                result &= bit_b ? ballot_b : (active & ~ballot_b)
            end
        end)
    end
    push!(body.args, :(return result))
    @eval Base.Experimental.@overlay AMDGPU.method_table @inline _match(::Type{MatchAny}, mask, value::$T) = $body
end