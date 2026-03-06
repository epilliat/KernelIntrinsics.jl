import KernelIntrinsics: select_device!, get_warpsize, devices, device, name, deviceid


select_device!(::CUDABackend, i::Integer) = CUDA.device!(i - 1)
devices(::CUDABackend) = collect(CUDA.devices())
name(dev::CUDA.CuDevice) = CUDA.name(dev)


device(src::CUDA.CuArray) = CUDA.device(src)
device(::CUDABackend) = CUDA.device()

deviceid(dev::CUDA.CuDevice) = CUDA.deviceid(dev) + 1  # 1-based
get_warpsize(dev::CUDA.CuDevice) = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_WARP_SIZE)