# CUDA lowering for `@sleep` → PTX `nanosleep.u32` (sm_70+).
# All CUDA GPUs this package targets (Ampere/Ada) are ≥ sm_70.
import KernelIntrinsics: _sleep

CUDA.@device_override @inline function _sleep(n::Integer)
    LLVM.Interop.@asmcall(
        "nanosleep.u32 \$0;",
        "r,~{memory}",
        true,
        Nothing,
        Tuple{UInt32},
        n % UInt32,
    )
    return nothing
end
