import KernelIntrinsics: select_device!, get_warpsize, devices, name, device, deviceid

select_device!(::ROCBackend, i::Integer) = AMDGPU.device!(AMDGPU.devices()[i])
devices(::ROCBackend) = AMDGPU.devices()

name(dev::AMDGPU.HIPDevice) = AMDGPU.HIP.name(dev)

device(src::AMDGPU.ROCArray) = AMDGPU.device(src)
device(::ROCBackend) = AMDGPU.device()

deviceid(dev::AMDGPU.HIPDevice) = AMDGPU.device_id(dev)  # 1-based

get_warpsize(dev::AMDGPU.HIPDevice) = dev.wavefrontsize
