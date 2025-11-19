# Mainly based on CUDA.jl




abstract type Direction end
struct Up <: Direction end
struct Down <: Direction end
struct Xor <: Direction end
struct Idx <: Direction end  # for regular shfl_sync (index-based)

abstract type Mode end
struct All <: Mode end
struct Any <: Mode end
struct Uni <: Mode end
struct Ballot <: Mode end



function _shfl end
function _vote end

for DirectType in (Up, Down, Xor, Idx)
    @eval begin
        @inline _shfl(::Type{$DirectType}, mask, val, src, ::Val{ws}) where ws =
            _shfl_recurse(x -> _shfl($DirectType, mask, x, src, Val(ws)), val)
    end
end

# unsigned integers
_shfl_recurse(op, x::UInt8) = op(UInt32(x)) % UInt8
_shfl_recurse(op, x::UInt16) = op(UInt32(x)) % UInt16
_shfl_recurse(op, x::UInt64) = (UInt64(op((x >>> 32) % UInt32)) << 32) | op((x & typemax(UInt32)) % UInt32)
_shfl_recurse(op, x::UInt128) = (UInt128(op((x >>> 64) % UInt64)) << 64) | op((x & typemax(UInt64)) % UInt64)

# signed integers
_shfl_recurse(op, x::Int8) = reinterpret(Int8, _shfl_recurse(op, reinterpret(UInt8, x)))
_shfl_recurse(op, x::Int16) = reinterpret(Int16, _shfl_recurse(op, reinterpret(UInt16, x)))
_shfl_recurse(op, x::Int64) = reinterpret(Int64, _shfl_recurse(op, reinterpret(UInt64, x)))
_shfl_recurse(op, x::Int128) = reinterpret(Int128, _shfl_recurse(op, reinterpret(UInt128, x)))

# floating point numbers
_shfl_recurse(op, x::Float16) = reinterpret(Float16, _shfl_recurse(op, reinterpret(UInt16, x)))
_shfl_recurse(op, x::Float64) = reinterpret(Float64, _shfl_recurse(op, reinterpret(UInt64, x)))

_shfl_recurse(op, x::Bool) = op(UInt32(x)) % Bool


function _shfl_recurse(op, val::T) where {T}
    if isprimitivetype(T)
        return op(val)
    else
        return T(ntuple(Val(fieldcount(T))) do i
            _shfl_recurse(op, getfield(val, i))
        end...)
    end
end




# Macro to dispatch on DirectType type
macro shfl(DirectType, val, src, ws=32, mask=0xffffffff)
    return quote
        _shfl($DirectType, $(esc(mask)), $(esc(val)), $(esc(src)), Val($(esc(ws))))
    end
end


macro warpreduce(val, lane, op=:+, ws=32, mask=0xffffffff)
    quote
        local offset = 1
        while offset < $(esc(ws))
            shuffled = @shfl(Up, $(esc(val)), offset, $(esc(ws)), $(esc(mask)))
            if $(esc(lane)) > offset
                $(esc(val)) = $(esc(op))(shuffled, $(esc(val)))
            end
            offset <<= 1
        end
    end
end

macro vote(ModeType, pred, mask=0xffffffff)
    return quote
        _vote($ModeType, $(esc(mask)), $(esc(pred)))
    end
end

