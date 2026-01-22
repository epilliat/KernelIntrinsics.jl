# Mainly based on CUDA.jl




"""
    Direction

Abstract type representing warp shuffle directions.

Subtypes:
- [`Up`](@ref): Shuffle values from lower lane indices
- [`Down`](@ref): Shuffle values from higher lane indices  
- [`Xor`](@ref): Shuffle values using XOR of lane indices
- [`Idx`](@ref): Shuffle values from a specific lane index
"""
abstract type Direction end

"""
    Up <: Direction

Shuffle direction where each lane receives a value from a lane with a lower index.

`@shfl(Up, val, offset)`: Lane `i` receives the value from lane `i - offset`.
Lanes where `i < offset` keep their original value.

Result for `offset=1`: `[1, 1, 2, 3, 4, ..., 31]`
"""
struct Up <: Direction end

"""
    Down <: Direction

Shuffle direction where each lane receives a value from a lane with a higher index.

`@shfl(Down, val, offset)`: Lane `i` receives the value from lane `i + offset`.
Lanes where `i + offset >= warpsize` keep their original value.

Result for `offset=1`: `[2, 3, 4, ..., 31, 32, 32]`
"""
struct Down <: Direction end

"""
    Xor <: Direction

Shuffle direction where each lane exchanges values based on XOR of lane indices.

`@shfl(Xor, val, mask)`: Lane `i` receives the value from lane `i ⊻ mask`.

Common patterns:
- `mask=1`: Swap adjacent pairs (0↔1, 2↔3, ...)
- `mask=16`: Swap first and second half of warp
"""
struct Xor <: Direction end

"""
    Idx <: Direction

Shuffle direction where all lanes receive a value from a specific lane index.

`@shfl(Idx, val, lane)`: All lanes receive the value from lane `lane`.

Useful for broadcasting a value from one lane to all others.
"""
struct Idx <: Direction end

"""
    Mode

Abstract type representing warp vote modes.

Subtypes:
- [`All`](@ref): True if predicate is true for all lanes
- [`Any`](@ref): True if predicate is true for any lane
- [`Uni`](@ref): True if predicate is uniform across all lanes
- [`Ballot`](@ref): Returns a bitmask of predicate values
"""
abstract type Mode end

"""
    All <: Mode

Vote mode that returns true if the predicate is true for all participating lanes.
"""
struct All <: Mode end

"""
    Any <: Mode

Vote mode that returns true if the predicate is true for any participating lane.
"""
struct Any <: Mode end

"""
    Uni <: Mode

Vote mode that returns true if the predicate has the same value across all participating lanes.
"""
struct Uni <: Mode end

"""
    Ballot <: Mode

Vote mode that returns a UInt32 bitmask where bit `i` is set if lane `i`'s predicate is true.
"""
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


#function _shfl_recurse(op, val::T)::T where {T}
#    if isprimitivetype(T)
#        return op(val)
#    else
#        return T(ntuple(i -> _shfl_recurse(op, getfield(val, i)), Val(fieldcount(T)))...)
#    end
#end
@generated function _shfl_recurse(op, val::T) where {T}
    if isprimitivetype(T)
        return :(op(val))
    else
        # Generate code for each field at compile time
        field_exprs = [:(_shfl_recurse(op, getfield(val, $i)))
                       for i in 1:fieldcount(T)]

        return quote
            T($(field_exprs...))
        end
    end
end

@generated function _shfl_recurse(op, val::NTuple{N,E}) where {N,E}
    if isprimitivetype(E)
        # Direct shuffle of each element
        exprs = [:(op(val[$i])) for i in 1:N]
        return :(tuple($(exprs...)))
    else
        # Recursive shuffle for complex element types
        exprs = [:(_shfl_recurse(op, val[$i])) for i in 1:N]
        return :(tuple($(exprs...)))
    end
end




"""
    @shfl(direction, val, src, [warpsize=32], [mask=0xffffffff])

Perform a warp shuffle operation, exchanging values between lanes within a warp.

# Arguments
- `direction`: Shuffle direction ([`Up`](@ref), [`Down`](@ref), [`Xor`](@ref), or [`Idx`](@ref))
- `val`: Value to shuffle (supports primitives, structs, and NTuples)
- `src`: Offset (for Up/Down), XOR mask (for Xor), or source lane (for Idx)
- `warpsize`: Warp size (default: 32)
- `mask`: Lane participation mask (default: 0xffffffff for all lanes)

# Example
```julia
@kernel function shfl_kernel(dst, src)
    I = @index(Global, Linear)
    val = src[I]
    
    shuffled = @shfl(Up, val, 1)      # Get value from lane below
    shuffled = @shfl(Down, val, 1)    # Get value from lane above
    shuffled = @shfl(Xor, val, 1)     # Swap with adjacent lane
    shuffled = @shfl(Idx, val, 0)     # Broadcast lane 0 to all
    
    dst[I] = shuffled
end
```

See also: [`@warpreduce`](@ref), [`@warpfold`](@ref)
"""
macro shfl(DirectType, val, src, ws=32, mask=0xffffffff)
    return quote
        _shfl($DirectType, $(esc(mask)), $(esc(val)), $(esc(src)), Val($(esc(ws))))
    end
end

"""
    @warpreduce(val, lane, [op=+], [warpsize=32], [mask=0xffffffff])

Perform an inclusive prefix scan (reduction) within a warp.

Each lane `i` accumulates values from lanes `1` to `i` using the specified operator.
Uses shuffle-up operations internally.

# Arguments
- `val`: Value to reduce (modified in-place)
- `lane`: Current lane index (1-based)
- `op`: Binary reduction operator (default: `+`)
- `warpsize`: Warp size (default: 32)
- `mask`: Lane participation mask (default: 0xffffffff)

# Example
```julia
@kernel function scan_kernel(dst, src)
    I = @index(Global, Linear)
    val = src[I]
    lane = (I - 1) % 32 + 1
    
    @warpreduce(val, lane, +)
    
    dst[I] = val  # Contains prefix sum
end

# Input:  [1, 2, 3, 4, ..., 32]
# Output: [1, 3, 6, 10, ..., 528]
```

See also: [`@warpfold`](@ref), [`@shfl`](@ref)
"""
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



"""
    @warpfold(val, lane, [op=+], [warpsize=32], [mask=0xffffffff])

Perform a warp-wide reduction, folding all values to a single result in lane 1.

Combines all values across the warp using the specified operator.
Uses shuffle-down operations internally.

# Arguments
- `val`: Value to reduce (modified in-place)
- `lane`: Current lane index (1-based, unused but kept for API consistency)
- `op`: Binary reduction operator (default: `+`)
- `warpsize`: Warp size (default: 32)
- `mask`: Lane participation mask (default: 0xffffffff)

# Example
```julia
@kernel function reduce_kernel(dst, src)
    I = @index(Global, Linear)
    val = src[I]
    lane = (I - 1) % 32 + 1
    
    @warpfold(val, lane, +)
    
    if lane == 1
        dst[1] = val  # Contains sum of all 32 values
    end
end

# Input:  [1, 2, 3, ..., 32]
# Output: dst[1] = 528
```

See also: [`@warpreduce`](@ref), [`@shfl`](@ref)
"""
macro warpfold(val, lane, op=:+, ws=32, mask=0xffffffff)
    quote
        local offset = 1
        while offset < $(esc(ws))
            shuffled = @shfl(Down, $(esc(val)), offset, $(esc(ws)), $(esc(mask)))
            $(esc(val)) = $(esc(op))(shuffled, $(esc(val)))
            offset <<= 1
        end
    end
end


"""
    @vote(mode, predicate, [mask=0xffffffff])

Perform a warp vote operation, evaluating a predicate across all lanes.

# Arguments
- `mode`: Vote mode ([`All`](@ref), [`Any`](@ref), [`Uni`](@ref), or [`Ballot`](@ref))
- `predicate`: Boolean predicate to evaluate
- `mask`: Lane participation mask (default: 0xffffffff)

# Example
```julia
@kernel function vote_kernel(dst, src, threshold)
    I = @index(Global, Linear)
    val = src[I]
    
    all_above = @vote(All, val > threshold)   # True if all lanes above threshold
    any_above = @vote(Any, val > threshold)   # True if any lane above threshold
    uniform = @vote(Uni, val > threshold)     # True if all lanes have same result
    mask = @vote(Ballot, val > threshold)     # Bitmask of which lanes are above
    
    dst[I] = mask
end
```

See also: [`@shfl`](@ref)
"""
macro vote(ModeType, pred, mask=0xffffffff)
    return quote
        _vote($ModeType, $(esc(mask)), $(esc(pred)))
    end
end

