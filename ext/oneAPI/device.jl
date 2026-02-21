import KernelIntrinsics: select_device!, get_device, get_warpsize, list_devices

select_device!(::oneAPIBackend, i::Integer) = oneAPI.device!(oneAPI.devices()[i])
get_device(::oneAPIBackend) = oneAPI.device()
get_warpsize(::oneAPIBackend) = oneAPI.properties(oneAPI.device()).maxSubGroupSize
list_devices(::oneAPIBackend) = oneAPI.devices()