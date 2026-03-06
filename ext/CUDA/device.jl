import KernelIntrinsics: select_device!, get_device, get_warpsize, list_devices, device, name


select_device!(::CUDABackend, i::Integer) = CUDA.device!(i - 1)
get_device(::CUDABackend) = CUDA.device()
#get_warpsize(::CUDABackend) = 32
list_devices(::CUDABackend) = collect(CUDA.devices())
name(dev::CUDA.CuDevice) = CUDA.name(dev)
device(src::CUDA.CuArray) = CUDA.device(src)
device() = CUDA.device()
