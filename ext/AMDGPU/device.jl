import KernelIntrinsics: select_device!, get_device, get_warpsize, list_devices, name, device, deviceid

select_device!(::ROCBackend, i::Integer) = AMDGPU.device!(AMDGPU.devices()[i])
list_devices(::ROCBackend) = AMDGPU.devices()

name(dev::AMDGPU.HIPDevice) = AMDGPU.HIP.name(dev)

device(src::AMDGPU.ROCArray) = AMDGPU.device(src)
device(::ROCBackend) = AMDGPU.device()

deviceid(dev::AMDGPU.HIPDevice) = AMDGPU.device_id(dev)  # 1-based
deviceid(src::AMDGPU.ROCArray) = AMDGPU.device_id(device(src))  # 1-based

get_warpsize(dev::AMDGPU.HIPDevice) = dev.wavefrontsize