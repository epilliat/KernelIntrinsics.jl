# Test harness shared across all backend test files.
#
# Depends on globals set up by runtests.jl: AT, backend, KI.
# Must be included AFTER runtests.jl has selected and imported the backend.

# Wrap a host array into the active backend's array type.
to_device(x) = AT(x)

# Retrieve a device array back to the host.
from_device(x) = Array(x)

# Launch a KernelAbstractions kernel on the active backend and synchronize.
function launch(kernel, args...; ndrange)
    kernel(backend)(args...; ndrange=ndrange)
    synchronize(backend)
end

# Warp/wavefront size on the active device (32 on CUDA, 64 on ROCm, 32 on Metal SIMD-group).
const warpsz = KI.get_warpsize(KI.device(backend, 1))
