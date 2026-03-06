import KernelIntrinsics: select_device!, get_warpsize, list_devices, device, name, deviceid


select_device!(::CUDABackend, i::Integer) = CUDA.device!(i - 1)
list_devices(::CUDABackend) = collect(CUDA.devices())
name(dev::CUDA.CuDevice) = CUDA.name(dev)
device(src::CUDA.CuArray) = CUDA.device(src)
device(::CUDABackend) = CUDA.device()
deviceid(dev::CUDA.CuDevice) = CUDA.deviceid(dev) + 1  # 1-based
deviceid(src::CUDA.CuArray) = CUDA.deviceid(CUDA.device(src)) + 1  # 1-based
get_warpsize(dev::CUDA.CuDevice) = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_WARP_SIZE)
get_warpsize(src::CUDA.CuArray) = get_warpsize(device(src))