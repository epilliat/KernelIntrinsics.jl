function select_device! end
function get_device end
function warpsize end
function list_devices end

function name end
function device end
function deviceid end


device(x::SubArray) = device(parent(x))
deviceid(x::SubArray) = deviceid(parent(x))