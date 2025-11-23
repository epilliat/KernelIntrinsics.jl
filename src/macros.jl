using InteractiveUtils

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

    # Validate all arguments are defined
    for arg in args
        if !isdefined(MemoryAccess, arg)
            valid_scopes = join(nameof.(subtypes(Scope)), ", ")
            valid_orderings = join(nameof.(subtypes(Ordering)), ", ")
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
            valid_scopes = join(nameof.(subtypes(Scope)), ", ")
            valid_orderings = join(nameof.(subtypes(Ordering)), ", ")
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
            valid_scopes = join(nameof.(subtypes(Scope)), ", ")
            valid_orderings = join(nameof.(subtypes(Ordering)), ", ")
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