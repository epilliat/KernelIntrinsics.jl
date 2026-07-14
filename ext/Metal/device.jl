import KernelIntrinsics: select_device!, get_warpsize, devices, name, device, deviceid

select_device!(::MetalBackend, i::Integer) = (i == 1 || @warn "Metal exposes a single device; ignoring index $i")
devices(::MetalBackend) = [Metal.device()]

name(dev::Metal.MTL.MTLDevice) = String(dev.name)

device(src::Metal.MtlArray) = Metal.device(src)
device(::MetalBackend) = Metal.device()

deviceid(dev::Metal.MTL.MTLDevice) = 1  # Metal only has one device

function get_warpsize(dev::Metal.MTL.MTLDevice = Metal.device())
    kernel = @metal launch=false (() -> nothing)()  # dummy kernel
    pipeline = kernel.pipeline
    Int(pipeline.threadExecutionWidth)
end