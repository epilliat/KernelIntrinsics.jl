"""
    vload_multi(A::AbstractArray{T}, i, ::Val{Nitem}) -> NTuple{Nitem,T}

Load `Nitem` elements from array `A` starting at index `i`, automatically handling arbitrary alignment.
`Nitem` must be a positive power of 2.

Computes alignment at runtime as `mod = (base_ptr_in_elements + (i - 1)) % Nitem + 1`, where
`base_ptr_in_elements = pointer(A) ÷ sizeof(T)`, and dispatches to a statically-compiled load
pattern via a switch table. This generates a mix of `ld.global.v4`, `ld.global.v2`, and scalar
loads to maximize throughput.

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
`Nitem` must be a positive power of 2.

Computes alignment at runtime as `mod = (base_ptr_in_elements + (i - 1)) % Nitem + 1`, where
`base_ptr_in_elements = pointer(A) ÷ sizeof(T)`, and dispatches to a statically-compiled store
pattern via a switch table. This generates a mix of `st.global.v4`, `st.global.v2`, and scalar
stores to maximize throughput.

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
    vload(A::AbstractArray{T}, idx, ::Val{Nitem}, ::Val{Rebase}=Val(true), ::Val{Alignment}=Val(-1)) -> NTuple{Nitem,T}

Load `Nitem` elements from array `A` as a tuple, using vectorized memory operations on GPU.
`Nitem` must be a positive power of 2.

# Arguments
- `A`: Source array
- `idx`: Starting index
- `Nitem`: Number of elements to load (must be a positive power of 2)
- `Rebase`: Indexing mode (default: `Val(true)`)
- `Alignment`: Known pointer alignment (default: `Val(-1)` = unknown)
  - `Val(1)`: pointer is `Nitem`-aligned → single `ld.global.vN`, no runtime check
  - `Val(2..Nitem)`: known misalignment offset → static load pattern, no runtime dispatch
  - `Val(-1)`: unknown → runtime pointer alignment check (default)
  Only meaningful for `Rebase=true`; ignored for `Rebase=false`.

# Indexing Modes
- `Val(true)` (rebased): Uses 1-based block indexing — `idx` selects the `idx`-th contiguous
  block of `Nitem` elements, i.e. loads from `(idx-1)*Nitem + 1` to `idx*Nitem`. For example,
  `idx=2` loads elements `[5,6,7,8]` for `Nitem=4`. When the array base pointer is
  `Nitem`-aligned, this generates optimal aligned vector loads (`ld.global.v4`); otherwise
  falls back to [`vload_multi`](@ref).
- `Val(false)` (direct): Loads starting directly at `idx`, so `idx=2` loads elements
  `[2,3,4,5]`. Always uses [`vload_multi`](@ref) to handle potential misalignment.

# Example
```julia
a = CuArray{Int32}(1:16)

# Rebased indexing (default): idx=2 → loads block 2, i.e. elements 5,6,7,8
values = vload(a, 2, Val(4))  # returns (5, 6, 7, 8)

# Direct indexing: idx=2 → loads elements 2,3,4,5
values = vload(a, 2, Val(4), Val(false))  # returns (2, 3, 4, 5)

# Known alignment: view offset by 1 element → Alignment=2
v = view(a, 2:16)
values = vload(v, 1, Val(4), Val(true), Val(2))  # static (1,2,1) pattern, no runtime branch
```

See also: [`vstore!`](@ref)
"""
function vload end

"""
    vstore!(A::AbstractArray{T}, idx, values::NTuple{Nitem,T}, ::Val{Rebase}=Val(true), ::Val{Alignment}=Val(-1)) -> Nothing

Store `Nitem` elements from a tuple to array `A`, using vectorized memory operations on GPU.
`Nitem` must be a positive power of 2.

# Arguments
- `A`: Destination array
- `idx`: Starting index
- `values`: Tuple of `Nitem` elements to store
- `Rebase`: Indexing mode (default: `Val(true)`)
- `Alignment`: Known pointer alignment (default: `Val(-1)` = unknown)
  Only meaningful for `Rebase=true`; ignored for `Rebase=false`.

# Indexing Modes
- `Val(true)` (rebased): Uses 1-based block indexing — `idx` selects the `idx`-th contiguous
  block of `Nitem` elements, i.e. stores to `(idx-1)*Nitem + 1` through `idx*Nitem`. For
  example, `idx=2` stores to elements `[5,6,7,8]` for `Nitem=4`. When the array base pointer
  is `Nitem`-aligned, this generates optimal aligned vector stores (`st.global.v4`); otherwise
  falls back to [`vstore_multi!`](@ref).
- `Val(false)` (direct): Stores starting directly at `idx`, so `idx=2` stores to elements
  `[2,3,4,5]`. Always uses [`vstore_multi!`](@ref) to handle potential misalignment.

# Example
```julia
b = CUDA.zeros(Int32, 16)

# Rebased indexing (default): idx=2 → stores to block 2, i.e. elements 5,6,7,8
vstore!(b, 2, (Int32(10), Int32(20), Int32(30), Int32(40)))

# Direct indexing: idx=2 → stores to elements 2,3,4,5
vstore!(b, 2, (Int32(10), Int32(20), Int32(30), Int32(40)), Val(false))
```

See also: [`vload`](@ref)
"""
function vstore! end

# ---------------------------------------------------------------------------
# vload — DenseArray (CuArray, CuDeviceArray, etc.)
# ---------------------------------------------------------------------------

# Outer @inline handles non-pow2 types before reaching @generated
@inline function vload(A::DenseArray{T}, idx, ::Val{Nitem}, ::Val{Rebase}, ::Val{Alignment})::NTuple{Nitem,T} where {Alignment,T,Nitem,Rebase}
    if !ispow2(sizeof(T))
        if Rebase
            base = (idx - 1) * Nitem + 1
            return ntuple(i -> A[base+i-1], Val(Nitem))
        else
            return ntuple(i -> A[idx+i-1], Val(Nitem))
        end
    end
    return _vload_pow2(A, idx, Val(Nitem), Val(Rebase), Val(Alignment))
end

@generated function _vload_pow2(A::DenseArray{T}, idx, ::Val{Nitem}, ::Val{Rebase}, ::Val{Alignment})::NTuple{Nitem,T} where {Alignment,T,Nitem,Rebase}
    # sizeof(T) is guaranteed pow2 here — all branches resolved at compile time
    if Rebase
        if Alignment == 1  # aligned: single vN load
            return quote
                @inline
                _llvm_barrier()
                result = _vload_batch(A, idx, Val($Nitem))::NTuple{$Nitem,$T}
                _llvm_barrier()
                return result
            end
        elseif Alignment > 1  # known misalignment: static pattern, zero runtime dispatch
            pattern = generate_partN(Nitem)[Alignment]
            return quote
                @inline
                _llvm_barrier()
                linear_idx = (idx - 1) * $Nitem + 1
                result = vload_pattern(A, linear_idx, $(Val(pattern)))::NTuple{$Nitem,$T}
                _llvm_barrier()
                return result
            end
        else  # Alignment == -1: unknown, runtime pointer alignment check
            return quote
                @inline
                _llvm_barrier()
                elem_offset = Int(pointer(A)) ÷ sizeof($T)
                if elem_offset % $Nitem == 0
                    result = _vload_batch(A, idx, Val($Nitem))::NTuple{$Nitem,$T}
                else
                    linear_idx = (idx - 1) * $Nitem + 1
                    result = vload_multi(A, linear_idx, Val($Nitem))::NTuple{$Nitem,$T}
                end
                _llvm_barrier()
                return result
            end
        end
    else  # Rebase=false: effective alignment depends on pointer AND idx → always runtime
        return quote
            @inline
            _llvm_barrier()
            result = vload_multi(A, idx, Val($Nitem))::NTuple{$Nitem,$T}
            _llvm_barrier()
            return result
        end
    end
end

# Default wrappers for DenseArray
@inline vload(A::DenseArray{T}, idx, v::Val{Nitem}, r::Val{Rebase}) where {T,Nitem,Rebase} =
    vload(A, idx, v, r, Val(-1))

@inline vload(A::DenseArray{T}, idx, v::Val{Nitem}) where {T,Nitem} =
    vload(A, idx, v, Val(true), Val(-1))

# ---------------------------------------------------------------------------
# vload — AbstractArray fallback (strided views, fancy indexing, etc.)
# ---------------------------------------------------------------------------

@inline function vload(A::AbstractArray{T}, idx, ::Val{Nitem}, ::Val{Rebase}=Val(true))::NTuple{Nitem,T} where {T,Nitem,Rebase}
    if Rebase
        return _vload_batch(A, idx, Val(Nitem))::NTuple{Nitem,T}
    else
        return _vload_norebase(A, idx, Val(Nitem))::NTuple{Nitem,T}
    end
end

# ---------------------------------------------------------------------------
# vstore! — DenseArray (CuArray, CuDeviceArray, etc.)
# ---------------------------------------------------------------------------

# Outer @inline handles non-pow2 types before reaching @generated
@inline function vstore!(A::DenseArray{T}, idx, values::NTuple{Nitem,T}, ::Val{Rebase}, ::Val{Alignment}) where {Alignment,T,Nitem,Rebase}
    if !ispow2(sizeof(T))
        if Rebase
            base = (idx - 1) * Nitem + 1
            for i in 1:Nitem
                A[base+i-1] = values[i]
            end
        else
            for i in 1:Nitem
                A[idx+i-1] = values[i]
            end
        end
        return nothing
    end
    return _vstore_pow2!(A, idx, values, Val(Rebase), Val(Alignment))
end

@generated function _vstore_pow2!(A::DenseArray{T}, idx, values::NTuple{Nitem,T}, ::Val{Rebase}, ::Val{Alignment}) where {Alignment,T,Nitem,Rebase}
    # sizeof(T) is guaranteed pow2 here — all branches resolved at compile time
    if Rebase
        if Alignment == 1  # aligned: single vN store
            return quote
                @inline
                _llvm_barrier()
                _vstore_batch!(A, idx, values)
                _llvm_barrier()
                return nothing
            end
        elseif Alignment > 1  # known misalignment: static pattern, zero runtime dispatch
            pattern = generate_partN(Nitem)[Alignment]
            return quote
                @inline
                _llvm_barrier()
                linear_idx = (idx - 1) * $Nitem + 1
                vstore_pattern!(A, linear_idx, values, $(Val(pattern)))
                _llvm_barrier()
                return nothing
            end
        else  # Alignment == -1: unknown, runtime pointer alignment check
            return quote
                @inline
                _llvm_barrier()
                elem_offset = Int(pointer(A)) ÷ sizeof($T)
                if elem_offset % $Nitem == 0
                    _vstore_batch!(A, idx, values)
                else
                    linear_idx = (idx - 1) * $Nitem + 1
                    vstore_multi!(A, linear_idx, values)
                end
                _llvm_barrier()
                return nothing
            end
        end
    else  # Rebase=false: effective alignment depends on pointer AND idx → always runtime
        return quote
            @inline
            _llvm_barrier()
            vstore_multi!(A, idx, values)
            _llvm_barrier()
            return nothing
        end
    end
end

# Default wrappers for DenseArray
@inline vstore!(A::DenseArray{T}, idx, values::NTuple{Nitem,T}, r::Val{Rebase}) where {T,Nitem,Rebase} =
    vstore!(A, idx, values, r, Val(-1))

@inline vstore!(A::DenseArray{T}, idx, values::NTuple{Nitem,T}) where {T,Nitem} =
    vstore!(A, idx, values, Val(true), Val(-1))

# ---------------------------------------------------------------------------
# vstore! — AbstractArray fallback (strided views, fancy indexing, etc.)
# ---------------------------------------------------------------------------

@inline function vstore!(A::AbstractArray{T}, idx, values::NTuple{Nitem,T}, ::Val{Rebase}=Val(true)) where {T,Nitem,Rebase}
    if Rebase
        _vstore_batch!(A, idx, values)
    else
        _vstore_norebase!(A, idx, values)
    end
    return nothing
end

# ---------------------------------------------------------------------------
# generate_partN — compile-time load/store pattern generation
# ---------------------------------------------------------------------------

function generate_partN(N::Int)
    @assert N > 0 && ispow2(N) "N must be a positive power of 2"
    result = []

    # Index 1: aligned case
    push!(result, (N,))

    for r in 1:(N-1)
        pattern = Int[]
        remaining = N
        pos = r  # current position modulo N

        while remaining > 0
            max_load = N ÷ 2  # Never load more than N/2 when misaligned
            while max_load > 0
                if pos % max_load == 0 && max_load <= remaining
                    break
                end
                max_load ÷= 2
            end
            if max_load == 0
                max_load = 1
            end
            push!(pattern, max_load)
            remaining -= max_load
            pos = (pos + max_load) % N
        end

        push!(result, tuple(pattern...))
    end

    return tuple(result...)
end

# ---------------------------------------------------------------------------
# vload_pattern / vload_multi
# ---------------------------------------------------------------------------

@inline @generated function vload_pattern(a, i, ::Val{pattern}) where {pattern}
    P = length(pattern)
    load_exprs = []
    offset = 0

    push!(load_exprs, :(_llvm_barrier()))

    for (idx, size_val) in enumerate(pattern)
        push!(load_exprs, :($(Symbol(:t_, idx)) = _vload_norebase(a, i + $offset, Val($size_val))))
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
    elem_offset = Int(pointer(a)) ÷ sizeof(T) + (i - 1)
    mod = elem_offset % Nitem + 1
    values = vload_multi(a, i, mod, Val(Nitem))::NTuple{Nitem,T}
    return values
end

# ---------------------------------------------------------------------------
# vstore_pattern! / vstore_multi!
# ---------------------------------------------------------------------------

@inline @generated function vstore_pattern!(a, i, values::NTuple{Nitem,T}, ::Val{pattern}) where {Nitem,T,pattern}
    P = length(pattern)
    store_exprs = []
    offset = 0
    val_offset = 0

    push!(store_exprs, :(_llvm_barrier()))

    for (idx, size_val) in enumerate(pattern)
        tuple_expr = Expr(:tuple, [:(values[$j]) for j in (val_offset+1):(val_offset+size_val)]...)
        push!(store_exprs, :(_vstore_norebase!(a, i + $offset, $tuple_expr)))
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
    elem_offset = Int(pointer(a)) ÷ sizeof(T) + (i - 1)
    mod = elem_offset % Nitem + 1
    vstore_multi!(a, i, mod, values)
end

# ---------------------------------------------------------------------------
# Scalar fallbacks
# ---------------------------------------------------------------------------

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