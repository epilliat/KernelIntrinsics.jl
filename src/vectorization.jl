
"""
    vload_multi(A::AbstractArray{T}, i, ::Val{Nitem}) -> NTuple{Nitem,T}

Load `Nitem` elements from array `A` starting at index `i`, automatically handling arbitrary alignment.

Computes alignment at runtime (`mod = (pointer_offset + i - 1) % Nitem + 1`) and dispatches 
to a statically-compiled load pattern via a switch table. This generates a mix of 
`ld.global.v4`, `ld.global.v2`, and scalar loads to maximize throughput.

# Example
```julia
src = cu(Int32.(1:100))

# Works for any starting index — alignment handled automatically
values = vload_multi(src, 7, Val(8))  # loads elements 7:14
```

See also: [`vload`](@ref), [`vstore_multi!`](@ref)
"""
function vload_multi end

"""
    vstore_multi!(A::AbstractArray{T}, i, values::NTuple{Nitem,T}) -> Nothing

Store `Nitem` elements to array `A` starting at index `i`, automatically handling arbitrary alignment.

Computes alignment at runtime (`mod = (pointer_offset + i - 1) % Nitem + 1`) and dispatches 
to a statically-compiled store pattern via a switch table. This generates a mix of 
`st.global.v4`, `st.global.v2`, and scalar stores to maximize throughput.

# Example
```julia
dst = cu(zeros(Int32, 100))

# Works for any starting index — alignment handled automatically
vstore_multi!(dst, 7, (Int32(1), Int32(2), Int32(3), Int32(4)))
```

See also: [`vstore!`](@ref), [`vload_multi`](@ref)
"""
function vstore_multi! end


"""
    vload(A::AbstractArray{T}, idx, ::Val{Nitem}, ::Val{Rebase}=Val(true)) -> NTuple{Nitem,T}

Load `Nitem` elements from array `A` as a tuple, using vectorized memory operations on GPU.

# Arguments
- `A`: Source array
- `idx`: Starting index
- `Nitem`: Number of elements to load (must be a power of 2)
- `Rebase`: Indexing mode (default: `Val(true)`)

# Indexing Modes
- `Val(true)` (rebased): Index is multiplied by `Nitem`, so `idx=2` loads elements `[5,6,7,8]` for `Nitem=4`. This mode generates optimal aligned vector loads (`ld.global.v4`).
- `Val(false)` (direct): Loads starting directly at `idx`, so `idx=2` loads elements `[2,3,4,5]`. Handles misaligned access by decomposing into smaller aligned loads.

# Example
```julia
a = CuArray{Int32}(1:16)

# Rebased indexing (default): idx=2 → loads elements 5,6,7,8
values = vload(a, 2, Val(4))  # returns (5, 6, 7, 8)

# Direct indexing: idx=2 → loads elements 2,3,4,5
values = vload(a, 2, Val(4), Val(false))  # returns (2, 3, 4, 5)
```

See also: [`vstore!`](@ref)
"""
@inline function vload(A::AbstractArray{T}, idx, ::Val{Nitem}, ::Val{Rebase}=Val(true))::NTuple{Nitem,T} where {T,Nitem,Rebase}
    if Rebase
        _llvm_barrier()  # Barrier before load
        result = _vload_batch(A, idx, Val(Nitem))::NTuple{Nitem,T}
        _llvm_barrier()  # Barrier before load
    else
        mod = (idx - 1) % Nitem + 1
        result = vload_multi(A, idx, mod, Val(Nitem))::NTuple{Nitem,T}
        _llvm_barrier()  # Barrier before load
    end
    return result
end


"""
    vstore!(A::AbstractArray{T}, idx, values::NTuple{Nitem,T}, ::Val{Rebase}=Val(true)) -> Nothing

Store `Nitem` elements from a tuple to array `A`, using vectorized memory operations on GPU.

# Arguments
- `A`: Destination array
- `idx`: Starting index
- `values`: Tuple of `Nitem` elements to store
- `Rebase`: Indexing mode (default: `Val(true)`)

# Indexing Modes
- `Val(true)` (rebased): Index is multiplied by `Nitem`, so `idx=2` stores to elements `[5,6,7,8]` for `Nitem=4`. This mode generates optimal aligned vector stores (`st.global.v4`).
- `Val(false)` (direct): Stores starting directly at `idx`, so `idx=2` stores to elements `[2,3,4,5]`. Handles misaligned access by decomposing into smaller aligned stores.

# Example
```julia
b = CUDA.zeros(Int32, 16)

# Rebased indexing (default): idx=2 → stores to elements 5,6,7,8
vstore!(b, 2, (Int32(10), Int32(20), Int32(30), Int32(40)))

# Direct indexing: idx=2 → stores to elements 2,3,4,5
vstore!(b, 2, (Int32(10), Int32(20), Int32(30), Int32(40)), Val(false))
```

See also: [`vload`](@ref)
"""
@inline function vstore!(A::AbstractArray{T}, idx, values::NTuple{Nitem,T}, ::Val{Rebase}=Val(true)) where {T,Nitem,Rebase}
    if Rebase
        _llvm_barrier()  # Barrier before store
        _vstore_batch!(A, idx, values)
        _llvm_barrier()  # Barrier after store
    else
        mod = (idx - 1) % Nitem + 1
        vstore_multi!(A, idx, mod, values)
        _llvm_barrier()  # Barrier before load
    end
    return nothing
end



function generate_partN(N::Int)
    # Check that N is a power of 2
    @assert N > 0 && ispow2(N) "N must be a positive power of 2"
    result = []  # Use untyped array

    # Add the aligned case (N,) at the beginning
    push!(result, (N,))

    for r in 1:(N-1)
        pattern = Int[]
        remaining = N
        pos = r  # current position modulo N

        while remaining > 0
            # Find the largest power of 2 we can load at current alignment
            # The alignment constraint is: load size must divide current position
            # or current position must be 0 mod load_size
            max_load = N ÷ 2  # Never load more than N/2 when misaligned

            while max_load > 0
                if pos % max_load == 0 && max_load <= remaining
                    break
                end
                max_load ÷= 2
            end

            if max_load == 0
                max_load = 1  # Always can load 1
            end

            push!(pattern, max_load)
            remaining -= max_load
            pos = (pos + max_load) % N
        end

        push!(result, tuple(pattern...))
    end

    return tuple(result...)
end


@inline @generated function vload_pattern(a, i, ::Val{pattern}) where {pattern}
    P = length(pattern)

    load_exprs = []
    offset = 0

    # Add initial barrier
    push!(load_exprs, :(_llvm_barrier()))

    for (idx, size_val) in enumerate(pattern)
        push!(load_exprs, :($(Symbol(:t_, idx)) = _vload_norebase(a, i + $offset, Val($size_val))))
        # Add barrier after each load
        push!(load_exprs, :(_llvm_barrier()))
        offset += size_val
    end

    tuple_syms = [:($(Symbol(:t_, idx))...) for idx in 1:P]

    quote
        $(load_exprs...)
        ($(tuple_syms...),)
    end
end

@inline @generated function vload_multi(a::AbstractArray{T}, i, mod, ::Val{Nitem})::NTuple{Nitem,T} where {Nitem,T}
    parts = generate_partN(Nitem)

    exprs = [:(mod === $n && return vload_pattern(a, i, $(Val(pattern))))
             for (n, pattern) in enumerate(parts)]

    quote
        $(exprs...)
    end
end
@inline function vload_multi(a::AbstractArray{T}, i, ::Val{Nitem})::NTuple{Nitem,T} where {Nitem,T}
    mod = ((Int(pointer(a)) + i - 1) % Nitem) + 1
    values = vload_multi(a, i, mod, Val(Nitem))::NTuple{Nitem,T}
    return values
end
@inline @generated function vstore_pattern!(a, i, values::NTuple{Nitem,T}, ::Val{pattern}) where {Nitem,T,pattern}
    P = length(pattern)
    store_exprs = []
    offset = 0
    val_offset = 0

    # Add initial barrier
    push!(store_exprs, :(_llvm_barrier()))

    for (idx, size_val) in enumerate(pattern)
        # Extract the slice of values for this store
        indices = (val_offset+1):(val_offset+size_val)
        slice_expr = :(values[$(indices[1]):$(indices[end])])

        # Create tuple from the slice
        tuple_expr = Expr(:tuple, [:(values[$j]) for j in indices]...)

        push!(store_exprs, :(_vstore_norebase!(a, i + $offset, $tuple_expr)))
        # Add barrier after each store
        push!(store_exprs, :(_llvm_barrier()))

        offset += size_val
        val_offset += size_val
    end

    quote
        $(store_exprs...)
        return nothing
    end
end

@inline @generated function vstore_multi!(a::AbstractArray{T}, i, mod, values::NTuple{Nitem,T}) where {Nitem,T}
    parts = generate_partN(Nitem)
    exprs = [:(mod === $n && return vstore_pattern!(a, i, values, $(Val(pattern))))
             for (n, pattern) in enumerate(parts)]
    quote
        $(exprs...)
    end
end
@inline function vstore_multi!(a::AbstractArray{T}, i, values::NTuple{Nitem,T}) where {Nitem,T}
    mod = ((Int(pointer(a)) + i - 1) % Nitem) + 1
    vstore_multi!(a, i, mod, values)
end

## CPU Fallbacks

@inline function _vload_batch(A::AbstractArray{T}, idx, ::Val{Nitem})::NTuple{Nitem,T} where {T,Nitem}
    base = (idx - 1) * Nitem + 1
    return ntuple(i -> A[base+i-1], Val(Nitem))
end

@inline function _vload_norebase(A::AbstractArray{T}, idx, ::Val{Nitem})::NTuple{Nitem,T} where {T,Nitem}
    return ntuple(i -> A[idx+i-1], Val(Nitem))
end

@inline function _vstore_batch!(A::AbstractArray{T}, idx, values::NTuple{Nitem,T}) where {T,Nitem}
    base = (idx - 1) * Nitem + 1
    for i in 1:Nitem
        A[base+i-1] = values[i]
    end
    return
end

@inline function _vstore_norebase!(A::AbstractArray{T}, idx, values::NTuple{Nitem,T}) where {T,Nitem}
    for i in 1:Nitem
        A[idx+i-1] = values[i]
    end
    return
end