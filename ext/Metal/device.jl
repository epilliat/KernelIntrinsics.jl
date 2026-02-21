
select_device!(::MetalBackend, i::Integer) = (i == 1 || @warn "Metal exposes a single device; ignoring index $i")
get_device(::MetalBackend) = Metal.device()
get_warpsize(::MetalBackend) = 32
list_devices(::MetalBackend) = [Metal.device()]