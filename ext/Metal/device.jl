import KernelIntrinsics: select_device!, get_warpsize, devices, name, device, deviceid

# Compatibility for Metal.jl API changes (MTLDeviceInstance -> MTLDevice)
const MTLDeviceType = isdefined(Metal.MTL, :MTLDeviceInstance) ? Metal.MTL.MTLDeviceInstance : Metal.MTL.MTLDevice

select_device!(::MetalBackend, i::Integer) = (i == 1 || @warn "Metal exposes a single device; ignoring index $i")
devices(::MetalBackend) = [Metal.device()]

name(dev::MTLDeviceType) = String(dev.name)

device(src::Metal.MtlArray) = Metal.device(src)
device(::MetalBackend) = Metal.device()

deviceid(dev::MTLDeviceType) = 1  # Metal only has one device

function get_warpsize(dev::MTLDeviceType = Metal.device())
    kernel = @metal launch=false (() -> nothing)()  # dummy kernel
    pipeline = kernel.pipeline
    Int(pipeline.threadExecutionWidth)
end