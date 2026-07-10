"""
    @fence [Scope] [Ordering]

Insert a memory fence with specified scope and ordering.

A memory fence ensures that memory operations before the fence are visible to other threads
before operations after the fence. This is essential for correct synchronization in parallel GPU code.

# Arguments
- `Scope` (optional): Visibility scope, one of `Device` (default, maps to `.gpu` in PTX),
  `Workgroup` (maps to `.cta`), or `System` (maps to `.sys`).
- `Ordering` (optional): Memory ordering, one of `Acquire`, `Release`, `AcqRel` (default),
  or `SeqCst`. `Weak`, `Volatile`, and `Relaxed` are not valid for fences.

Arguments can be specified in any order.

# Generated PTX
- `@fence` → `fence.acq_rel.gpu`
- `@fence Workgroup` → `fence.acq_rel.cta`
- `@fence System SeqCst` → `fence.sc.sys`

# Example
```julia
@kernel function synchronized_kernel(X, Flag)
    X[1] = 10
    @fence  # Ensure X[1]=10 is visible to other threads before continuing
    Flag[1] = 1
end

# Explicit scope and ordering
@fence Device AcqRel
@fence Workgroup Release
@fence System SeqCst
@fence SeqCst Device  # Order doesn't matter
```

See also: [`@access`](@ref)
"""
macro fence end

"""
    @access [Scope] [Ordering] expr

Perform a memory load or store with specified scope and ordering semantics.

This macro provides fine-grained control over memory ordering for lock-free synchronization
patterns on GPU. It generates appropriate `ld.acquire` or `st.release` PTX instructions.

# Arguments
- `Scope` (optional): Visibility scope, one of `Device` (default), `Workgroup`, or `System`.
  Cannot be specified with `Volatile` or `Weak` orderings, as those are scope-less.
- `Ordering` (optional): Memory ordering (see below).
- `expr`: A load or store expression (see Syntax Forms).

Arguments can be specified in any order.

# Orderings
**For loads** (default: `Acquire`):
- `Acquire`: Subsequent reads see all writes before the corresponding release.
- `Relaxed`: No ordering guarantees.
- `Volatile`: Volatile load — bypasses cache, scope-less.
- `Weak`: Weak load — scope-less.

**For stores** (default: `Release`):
- `Release`: Prior writes are visible to other threads before this store.
- `Relaxed`: No ordering guarantees.
- `Volatile`: Volatile store — bypasses cache, scope-less.
- `Weak`: Weak store — scope-less.

`AcqRel` and `SeqCst` are not valid for individual loads/stores; use [`@fence`](@ref) instead.

# Syntax Forms
```julia
@access array[idx] = value                  # Release store (default)
@access var = array[idx]                    # Acquire load, result bound to var (default)
@access array[idx]                          # Acquire load, result returned directly

@access Release array[idx] = value          # Explicit ordering
@access Acquire var = array[idx]            # Explicit ordering
@access Device Release array[idx] = value   # Explicit scope and ordering
@access SeqCst Device  # Order doesn't matter
```

# Example
```julia
@kernel function producer_consumer(X, Flag)
    if @index(Global, Linear) == 1
        X[1] = 42
        @access Flag[1] = 1  # Release store: X[1]=42 visible before Flag[1]=1
    end

    # Other threads spin-wait using standalone load form
    while (@access Acquire Flag[1]) != 1
    end
    # Now X[1] is guaranteed to be 42
end
```

See also: [`@fence`](@ref)
"""
macro access end

"""
    @warpsize()

Return the warp size of the current backend as an `Int`.
Queries the backend at runtime — 32 on CUDA, 64 on ROCm.

See also: [`@laneid`](@ref), [`@shfl`](@ref), [`@warpreduce`](@ref), [`@warpfold`](@ref)
"""
macro warpsize()
    quote
        _warpsize()
    end
end

"""
    @laneid()

Return the 1-based lane index of the current thread within its warp/wavefront.

See also: [`@warpsize`](@ref), [`@warpreduce`](@ref), [`@warpfold`](@ref)
"""
macro laneid()
    quote
        _laneid()
    end
end

# Classify a macro argument as a Scope or Ordering by its SYMBOL NAME — NOT by `eval`ing it.
# `eval(arg)` runs in the KernelIntrinsics module at macro-expansion time; when another package
# expands `@fence`/`@access` with an explicit scope/ordering during ITS precompilation, that is an
# "Evaluation into the closed module KernelIntrinsics" and breaks incremental precompilation. Names
# match the Scope/Ordering singleton type names; a qualified argument (e.g. `KI.Device`) is accepted
# by taking its leaf symbol. Behaviour is otherwise identical: the original arg expr is returned
# unchanged (it resolves to the type at the call site, exactly as before).
const _SCOPE_SYMS = (:Device, :Workgroup, :System)
const _ORDERING_SYMS = (:Acquire, :Release, :AcqRel, :SeqCst, :Relaxed, :Weak, :Volatile)
_arg_leaf(a::Symbol) = a
_arg_leaf(a::QuoteNode) = a.value isa Symbol ? a.value : :_notaname_
_arg_leaf(a::Expr) = a.head === :. ? _arg_leaf(a.args[end]) : :_notaname_
_arg_leaf(::Any) = :_notaname_
_is_scope_arg(a) = _arg_leaf(a) in _SCOPE_SYMS
_is_ordering_arg(a) = _arg_leaf(a) in _ORDERING_SYMS

function scope_ordering(args...)
    scope = nothing
    ordering = nothing

    if length(args) > 2
        throw(ArgumentError(
            "Too many arguments: expected 0-2, got $(length(args)). " *
            "Usage: @fence [Scope] [Ordering]"
        ))
    end

    valid_scopes = "Device, Workgroup, System"
    valid_orderings = "Acquire, Release, AcqRel, SeqCst, Relaxed, Weak, Volatile"

    if length(args) == 0
        # Use defaults (both nothing)

    elseif length(args) == 1
        if _is_scope_arg(args[1])
            scope = args[1]
        elseif _is_ordering_arg(args[1])
            ordering = args[1]
        else
            throw(ArgumentError(
                "'$(args[1])' is neither a Scope nor an Ordering.\n" *
                "Valid scopes: $valid_scopes\n" *
                "Valid orderings: $valid_orderings"
            ))
        end

    elseif length(args) == 2
        if _is_scope_arg(args[1]) && _is_ordering_arg(args[2])
            scope = args[1]
            ordering = args[2]
        elseif _is_ordering_arg(args[1]) && _is_scope_arg(args[2])
            ordering = args[1]
            scope = args[2]
        else
            throw(ArgumentError(
                "Arguments must be one Scope and one Ordering (in any order).\n" *
                "Got: $(args[1]) and $(args[2])\n" *
                "Valid scopes: $valid_scopes\n" *
                "Valid orderings: $valid_orderings"
            ))
        end
    end

    return scope, ordering
end

macro fence(args...)
    #No arguments - fallback to AcqRel and Device
    scope, ordering = scope_ordering(args...)
    if _arg_leaf(ordering) in (:Weak, :Volatile, :Relaxed)
        throw(ArgumentError(
            "Fences allows synchronizing orderings: Acquire, Release, AcqRel, or SeqCst."
        ))
    end
    scope = isnothing(scope) ? Device : scope
    ordering = isnothing(ordering) ? AcqRel : ordering
    return quote
        $(fence)($scope, $ordering)
    end
end

macro access(args...)
    expr = args[end]
    scope, ordering = scope_ordering(args[begin:end-1]...)

    # Validate the ordering by symbol name (no `eval` — see scope_ordering note)
    if !isnothing(ordering)
        oname = _arg_leaf(ordering)
        if oname in (:AcqRel, :SeqCst)
            throw(ArgumentError(
                "AcqRel and SeqCst are not valid orderings for loads and stores; use @fence instead."
            ))
        end
        if !isnothing(scope) && oname in (:Weak, :Volatile)
            throw(ArgumentError(
                "Cannot specify a scope with $(oname) ordering. " *
                "$(oname) operations are scope-less."
            ))
        end
    end

    scope = isnothing(scope) ? Device : scope

    # Handle standalone array access (returns atomic_load value)
    if isa(expr, Expr) && expr.head == :ref
        ordering = isnothing(ordering) ? Acquire : ordering
        array = expr.args[1]
        idxs = [esc(i) for i in expr.args[2:end]]
        V = esc(array)
        return quote
            $atomic_load($(V), $LinearIndices($(V))[$(idxs...)], $scope, $ordering)
        end
    end

    # Check expr is valid
    if !isa(expr, Expr) || expr.head != :(=)
        throw(ArgumentError(
            "Invalid @access syntax. Expected: @access [Scope] [Ordering] array[index] = value " *
            "or @access [Scope] [Ordering] variable = array[index]"
        ))
    end

    lhs = expr.args[end-1]
    rhs = expr.args[end]

    if isa(lhs, Expr) && lhs.head == :ref #STORE
        ordering = isnothing(ordering) ? Release : ordering
        array = lhs.args[1]
        idxs = [esc(i) for i in lhs.args[2:end]]
        V = esc(array)
        return quote
            $atomic_store!($(V), $LinearIndices($(V))[$(idxs...)], $(esc(rhs)), $scope, $ordering)
        end

    elseif isa(rhs, Expr) && rhs.head == :ref #LOAD
        ordering = isnothing(ordering) ? Acquire : ordering
        array = rhs.args[1]
        idxs = [esc(i) for i in rhs.args[2:end]]
        V = esc(array)
        return quote
            $(esc(lhs)) = $atomic_load($(V), $LinearIndices($(V))[$(idxs...)], $scope, $ordering)
        end
    end

    throw(ArgumentError(
        "Invalid @access syntax. Expected: @access [Scope] [Ordering] array[index] = value " *
        "or @access [Scope] [Ordering] variable = array[index]"
    ))
end