macro warpsize()
    quote
        _warpsize()
    end
end


function scope_ordering(args...)
    scope = nothing
    ordering = nothing

    # Check number of arguments
    if length(args) > 2
        throw(ArgumentError(
            "Too many arguments: expected 0-2, got $(length(args)). " *
            "Usage: @fence [Scope] [Ordering]"
        ))
    end

    # Valid types (hardcoded to avoid InteractiveUtils dependency)
    valid_scopes = "Device, Workgroup, System"
    valid_orderings = "Acquire, Release, AcqRel, SeqCst, Relaxed, Weak, Volatile"

    # Validate all arguments are defined
    for arg in args
        if !isdefined(KernelIntrinsics, arg)
            throw(ArgumentError(
                "'$arg' is not defined.\n" *
                "Valid scopes: $valid_scopes\n" *
                "Valid orderings: $valid_orderings"
            ))
        end
    end

    # Parse arguments based on length
    if length(args) == 0
        # Use defaults (both nothing)

    elseif length(args) == 1
        val = eval(args[1])
        if val <: Scope
            scope = args[1]
        elseif val <: Ordering
            ordering = args[1]
        else
            throw(ArgumentError(
                "'$(nameof(val))' is neither a Scope nor an Ordering.\n" *
                "Valid scopes: $valid_scopes\n" *
                "Valid orderings: $valid_orderings"
            ))
        end

    elseif length(args) == 2
        val1 = eval(args[1])
        val2 = eval(args[2])

        if val1 <: Scope && val2 <: Ordering
            scope = args[1]
            ordering = args[2]
        elseif val1 <: Ordering && val2 <: Scope
            ordering = args[1]
            scope = args[2]
        else
            throw(ArgumentError(
                "Arguments must be one Scope and one Ordering (in any order).\n" *
                "Got: $(nameof(val1)) and $(nameof(val2))\n" *
                "Valid scopes: $valid_scopes\n" *
                "Valid orderings: $valid_orderings"
            ))
        end
    end

    return scope, ordering
end

"""
    @fence [Scope] [Ordering]

Insert a memory fence with specified scope and ordering.

A memory fence ensures that memory operations before the fence are visible to other threads
before operations after the fence. This is essential for correct synchronization in parallel GPU code.

# Arguments
- `Scope` (optional): Visibility scope, one of `Device` (default), `Workgroup`, or `System`
- `Ordering` (optional): Memory ordering, one of `Acquire`, `Release`, `AcqRel` (default), or `SeqCst`

Arguments can be specified in any order. `Weak`, `Volatile`, and `Relaxed` orderings are not valid for fences.

# Generated PTX
- `@fence` → `fence.acq_rel.gpu`
- `@fence Workgroup` → `fence.acq_rel.cta`
- `@fence System SeqCst` → `fence.sc.sys`

# Example
```julia
@kernel function synchronized_kernel(X, Flag)
    X[1] = 10
    @fence  # Ensure X[1]=10 is visible before continuing
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
macro fence(args...)
    #No arguments - fallback to AcqRel and Device
    scope, ordering = scope_ordering(args...)
    scope = isnothing(scope) ? Device : scope
    ordering = isnothing(ordering) ? AcqRel : ordering
    if eval(ordering) in [Weak, Volatile, Relaxed]
        throw(ArgumentError(
            "Fences allows synchronizing orderings: Acquire, Release, AcqRel, or SeqCst."
        ))
    end
    return quote
        $(fence)($scope, $ordering)
    end
end

"""
    @access [Scope] [Ordering] expr

Perform a memory load or store with specified scope and ordering semantics.

This macro provides fine-grained control over memory ordering for lock-free synchronization
patterns on GPU. It generates appropriate `ld.acquire` or `st.release` PTX instructions.

# Arguments
- `Scope` (optional): Visibility scope, one of `Device` (default), `Workgroup`, or `System`
- `Ordering` (optional): Memory ordering (see below)
- `expr`: Either a load (`var = array[idx]`) or store (`array[idx] = value`) expression

# Orderings
**For loads** (default: `Acquire`):
- `Acquire`: Subsequent reads see all writes before the corresponding release
- `Relaxed`: No ordering guarantees
- `Volatile`: Volatile load (scope-less)
- `Weak`: Weak load (scope-less)

**For stores** (default: `Release`):
- `Release`: Prior writes are visible before this store
- `Relaxed`: No ordering guarantees
- `Volatile`: Volatile store (scope-less)
- `Weak`: Weak store (scope-less)

`AcqRel` and `SeqCst` are not valid for individual loads/stores (use `@fence` instead).
`Volatile` and `Weak` cannot have an explicit scope.

# Syntax Forms
```julia
@access array[idx] = value          # Release store (default)
@access var = array[idx]            # Acquire load (default)
@access array[idx]                  # Acquire load, returns value directly

@access Release array[idx] = value  # Explicit ordering
@access Acquire var = array[idx]    # Explicit ordering
@access Device Release array[idx] = value  # Explicit scope and ordering
```

# Example
```julia
@kernel function producer_consumer(X, Flag)
    if @index(Global, Linear) == 1
        X[1] = 42
        @access Flag[1] = 1  # Release store: X[1]=42 visible before Flag[1]=1
    end

    # Other threads wait
    while (@access Acquire Flag[1]) != 1
    end
    # Now X[1] is guaranteed to be 42
end
```

See also: [`@fence`](@ref)
"""
macro access(args...)
    expr = args[end]
    scope, ordering = scope_ordering(args[begin:end-1]...)

    # Check ordering exists and evaluate it
    if !isnothing(ordering)
        ordering_val = eval(ordering)
        if ordering_val in [AcqRel, SeqCst]
            throw(ArgumentError(
                "SeqCst or AcqRel are not a valid ordering for loads and stores"
            ))
        end
        if !isnothing(scope)
            if ordering_val in [Weak, Volatile]
                throw(ArgumentError(
                    "Cannot specify a scope with $(nameof(ordering_val)) ordering. " *
                    "$(nameof(ordering_val)) operations are scope-less."
                ))
            end
        end
    end

    scope = isnothing(scope) ? Device : scope

    # NEW: Handle standalone array access (returns atomic_load value)
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