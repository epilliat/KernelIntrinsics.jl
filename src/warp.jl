# Mainly based on CUDA.jl

# --- Types ---

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

Result for `offset=1` (warpsize=32): `[1, 1, 2, 3, 4, ..., 31]`
"""
struct Up <: Direction end

"""
    Down <: Direction

Shuffle direction where each lane receives a value from a lane with a higher index.

`@shfl(Down, val, offset)`: Lane `i` receives the value from lane `i + offset`.
Lanes where `i + offset >= warpsize` keep their original value.

Result for `offset=1` (warpsize=32): `[2, 3, 4, ..., 31, 32, 32]`
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

`@shfl(Idx, val, lane)`: All lanes receive the value from lane `lane` (0-based).

Useful for broadcasting a value from one lane to all others.
"""
struct Idx <: Direction end

"""
    Mode

Abstract type representing warp vote modes.

Subtypes:
- [`All`](@ref): True if predicate is true for all lanes
- [`AnyLane`](@ref): True if predicate is true for any lane
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
    AnyLane <: Mode

Vote mode that returns true if the predicate is true for any participating lane.
"""
struct AnyLane <: Mode end

"""
    Uni <: Mode

Vote mode that returns true if the predicate has the same value across all participating lanes.
"""
struct Uni <: Mode end

"""
    Ballot <: Mode

Vote mode that returns a `UInt32` bitmask where bit `i` (0-based) is set if lane `i`'s
predicate is true.
"""
struct Ballot <: Mode end


# --- Implementation ---

function _shfl end
function _vote end

for DirectType in (Up, Down, Xor, Idx)
    @eval begin
        @inline _shfl(::Type{$DirectType}, mask, val, src) =
            _shfl_recurse(x -> _shfl($DirectType, mask, x, src), val)
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

# floating point
_shfl_recurse(op, x::Float16) = reinterpret(Float16, _shfl_recurse(op, reinterpret(UInt16, x)))
_shfl_recurse(op, x::Float64) = reinterpret(Float64, _shfl_recurse(op, reinterpret(UInt64, x)))

_shfl_recurse(op, x::Bool) = op(UInt32(x)) % Bool

@generated function _shfl_recurse(op, val::T) where {T}
    if isprimitivetype(T)
        return :(op(val))
    else
        field_exprs = [:(_shfl_recurse(op, getfield(val, $i))) for i in 1:fieldcount(T)]
        return quote
            T($(field_exprs...))
        end
    end
end

@generated function _shfl_recurse(op, val::NTuple{N,E}) where {N,E}
    if isprimitivetype(E)
        exprs = [:(op(val[$i])) for i in 1:N]
        return :(tuple($(exprs...)))
    else
        exprs = [:(_shfl_recurse(op, val[$i])) for i in 1:N]
        return :(tuple($(exprs...)))
    end
end

"""
    @shfl(direction, val, src, [warpsize=@warpsize()], [mask=0xffffffff])

Perform a warp shuffle operation, exchanging values between lanes within a warp.

The default warpsize is retrieved at runtime via `@warpsize()`, which queries the backend
(32 on CUDA, but may differ on future backends).

# Arguments
- `direction`: Shuffle direction ([`Up`](@ref), [`Down`](@ref), [`Xor`](@ref), or [`Idx`](@ref))
- `val`: Value to shuffle (supports primitives, structs, and NTuples)
- `src`: Offset (for `Up`/`Down`), XOR mask (for `Xor`), or source lane 0-based index (for `Idx`)
- `warpsize`: Warp size (default: `@warpsize()`)
- `mask`: Lane participation mask (default: `0xffffffff` for all lanes)

# Example
```julia
@kernel function shfl_kernel(dst, src)
    I = @index(Global, Linear)
    val = src[I]

    shuffled = @shfl(Up, val, 1)    # Lane i receives from lane i-1; lane 0 keeps its value
    shuffled = @shfl(Down, val, 1)  # Lane i receives from lane i+1; last lane keeps its value
    shuffled = @shfl(Xor, val, 1)   # Swap adjacent pairs (lane 0↔1, 2↔3, ...)
    shuffled = @shfl(Idx, val, 0)   # Broadcast lane 0 to all lanes

    dst[I] = shuffled
end
```

See also: [`@warpreduce`](@ref), [`@warpfold`](@ref)
"""
macro shfl(DirectType, val, src, mask=0xffffffff)
    return quote
        _shfl($DirectType, $(esc(mask)), $(esc(val)), $(esc(src)))
    end
end

"""
    @warpreduce(val, op, [lane=@laneid()], [warpsize=@warpsize()], [mask=0xffffffff])

Perform an inclusive prefix scan within a warp using shuffle-up operations.

After this macro, lane `i` (1-based) holds the result of applying `op` to the values
of lanes `1` through `i`. The result in the last lane is the warp-wide reduction.

The default warpsize is retrieved at runtime via `@warpsize()`, which queries the backend
(32 on CUDA, but may differ on future backends).

# Arguments
- `val`: Value to scan (modified in-place)
- `op`: Binary associative operator (default: `+`)
- `lane`: Current lane index (1-based; default: `@laneid()`)
- `warpsize`: Warp size (default: `@warpsize()`)
- `mask`: Lane participation mask (default: `0xffffffff`)

# Example
```julia
@kernel function scan_kernel(dst, src)
    I = @index(Global, Linear)
    val = src[I]

    @warpreduce(val, +)

    dst[I] = val
end

# Input:  [1, 2, 3, 4, ..., 32]
# Output: [1, 3, 6, 10, ..., 528]
```

See also: [`@warpfold`](@ref), [`@shfl`](@ref)
"""
macro warpreduce(val, op, lane=:(KernelIntrinsics._laneid()), ws=:(KernelIntrinsics._warpsize()), mask=0xffffffff)
    quote
        local offset = 1
        while offset < $(esc(ws))
            shuffled = @shfl(Up, $(esc(val)), offset, $(esc(mask)))
            if $(esc(lane)) > offset
                $(esc(val)) = $(esc(op))(shuffled, $(esc(val)))
            end
            offset <<= 1
        end
    end
end

"""
    @warpfold(val, op, [lane=@laneid()], [warpsize=@warpsize()], [mask=0xffffffff])

Perform a warp-wide reduction, combining all lane values using the specified operator.
Uses shuffle-down operations internally. After this macro, all lanes hold the
warp-wide result.

The default warpsize is retrieved at runtime via `@warpsize()`, which queries the backend
(32 on CUDA, but may differ on future backends).

# Arguments
- `val`: Value to reduce (modified in-place)
- `op`: Binary associative operator (default: `+`)
- `lane`: Current lane index (1-based; accepted for API consistency but unused; default: `@laneid()`)
- `warpsize`: Warp size (default: `@warpsize()`)
- `mask`: Lane participation mask (default: `0xffffffff`)

# Example
```julia
@kernel function reduce_kernel(dst, src)
    I = @index(Global, Linear)
    val = src[I]
    lane = (I - 1) % @warpsize() + 1

    @warpfold(val, +)

    if lane == 1
        dst[1] = val  # Contains sum of all warp values
    end
end

# Input:  [1, 2, 3, ..., 32]
# Output: dst[1] = 528
```

See also: [`@warpreduce`](@ref), [`@shfl`](@ref)
"""
macro warpfold(val, op, lane=:(KernelIntrinsics._laneid()), ws=:(KernelIntrinsics._warpsize()), mask=0xffffffff)
    quote
        local offset = 1
        while offset < $(esc(ws))
            shuffled = @shfl(Down, $(esc(val)), offset)
            $(esc(val)) = $(esc(op))(shuffled, $(esc(val)))
            offset <<= 1
        end
    end
end

"""
    @vote(mode, predicate, [mask=0xffffffff])

Perform a warp vote operation, evaluating a predicate across all participating lanes.

# Arguments
- `mode`: Vote mode ([`All`](@ref), [`AnyLane`](@ref), [`Uni`](@ref), or [`Ballot`](@ref))
- `predicate`: Boolean predicate to evaluate
- `mask`: Lane participation mask (default: `0xffffffff` for all lanes)

# Example
```julia
@kernel function vote_kernel(dst, src, threshold)
    I = @index(Global, Linear)
    val = src[I]

    all_above = @vote(All,     val > threshold)  # true if all lanes satisfy predicate
    any_above = @vote(AnyLane, val > threshold)  # true if any lane satisfies predicate
    uniform   = @vote(Uni,     val > threshold)  # true if all lanes have the same result
    bits      = @vote(Ballot,  val > threshold)  # UInt32 bitmask: bit i (0-based) set if lane i satisfies predicate

    dst[I] = bits
end
```

See also: [`@shfl`](@ref)
"""
macro vote(ModeType, pred, mask=0xffffffff)
    return quote
        _vote($ModeType, $(esc(mask)), $(esc(pred)))
    end
end


# Simple fallbacks to shuffle operations if backends do not provide them:

# Fallback implementations using shuffle operations
@inline _vote(::Type{Ballot}, mask, pred) = _ballot_from_shfl(pred)
@inline _vote(::Type{All}, mask, pred) = _vote_all_from_shfl(pred)
@inline _vote(::Type{AnyLane}, mask, pred) = _vote_any_from_shfl(pred)
@inline _vote(::Type{Uni}, mask, pred) = _vote_all_from_shfl(pred) | _vote_all_from_shfl(!pred)


@inline function _ballot_from_shfl(pred)::UInt64
    lane = @laneid # 1-based
    ws = @warpsize

    # use UInt64 from the start so it works for both warpsize 32 and 64
    val = UInt64(pred) << (lane - 1)  # bit at lane position

    @warpreduce(val, |)   # lane ws holds full OR in lower ws bits

    # broadcast from last lane (0-based index = ws - 1)
    result = @shfl(Idx, val, ws)

    return result  # lower 32 bits used for wave32, all 64 for wave64
end

@inline function _vote_all_from_shfl(pred)
    lane = @laneid()
    ws = @warpsize()

    @warpfold(pred, &)  # all lanes AND together

    result = @shfl(Idx, pred, 1)
    return result
end

@inline function _vote_any_from_shfl(pred)
    lane = @laneid()
    ws = @warpsize()

    @warpfold(pred, |)  # any lane OR together

    result = @shfl(Idx, pred, 1)
    return result != UInt32(0)
end