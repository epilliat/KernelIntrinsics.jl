@inline function vectorized_load(A::AbstractArray{T}, idx, ::Val{Nitem})::NTuple{Nitem,T} where {T,Nitem}
    id_base = (idx - 1) * Nitem
    return ntuple(i -> A[id_base+i], Val(Nitem))
end
@inline function vectorized_cached_load(A::AbstractArray{T}, idx, ::Val{Nitem})::NTuple{Nitem,T} where {T,Nitem}
    id_base = (idx - 1) * Nitem
    return ntuple(i -> A[id_base+i], Val(Nitem))
end


@inline function vectorized_store!(A::AbstractArray{T}, idx, values::NTuple{Nitem,T}) where {T,Nitem}
    id_base = (idx - 1) * Nitem
    for i in (1:Nitem)
        A[id_base+i] = values[i]
    end
    return
end