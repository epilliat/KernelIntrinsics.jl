import KernelIntrinsics: select_device!, get_device, get_warpsize, list_devices

select_device!(::ROCBackend, i::Integer) = AMDGPU.device!(AMDGPU.devices()[i])
get_device(::ROCBackend) = AMDGPU.device()
get_warpsize(::ROCBackend) = AMDGPU.wavefrontsize(AMDGPU.device())
list_devices(::ROCBackend) = AMDGPU.devices()