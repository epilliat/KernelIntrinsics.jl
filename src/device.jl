function select_device! end
function get_warpsize end


function devices end
function device end
function deviceid end

function name end

device(backend::Backend, i::Integer) = devices(backend)[i]
device(x::SubArray) = device(parent(x))

deviceid(src::AbstractArray) = deviceid(device(src))  # 1-based
deviceid(x::SubArray) = deviceid(parent(x))

get_warpsize(src::AbstractArray) = get_warpsize(device(src))