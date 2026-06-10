# AMDGPU lowering for `@sleep` → `s_sleep` (llvm.amdgcn.s.sleep).
# s_sleep takes an IMMEDIATE operand (units of 64 cycles), so the value must be
# a compile-time constant. `_ssleep(Val{K})` bakes K into the call; the runtime
# `n` is mapped to the nearest power-of-two immediate in 1:64 by a branch ladder
# (each branch selects a concrete `Val`, so the immarg constraint is satisfied).
import KernelIntrinsics: _sleep

@inline _ssleep(::Val{K}) where {K} =
    ccall("llvm.amdgcn.s.sleep", llvmcall, Cvoid, (Int32,), Int32(K))

Base.Experimental.@overlay AMDGPU.method_table @inline function _sleep(n::Integer)
    un = n % UInt32
    un >= 0x40 ? _ssleep(Val(64)) :
    un >= 0x20 ? _ssleep(Val(32)) :
    un >= 0x10 ? _ssleep(Val(16)) :
    un >= 0x08 ? _ssleep(Val(8)) :
    un >= 0x04 ? _ssleep(Val(4)) :
    un >= 0x02 ? _ssleep(Val(2)) :
    _ssleep(Val(1))
    return nothing
end
