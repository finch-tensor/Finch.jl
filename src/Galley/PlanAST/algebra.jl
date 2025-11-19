# This file manages the logic of algebraic properties. For the moment, we just spoof Finch's
# properties, but in the future we can extend these or write our own.

function isassociative(f)
    if typeof(f) == PlanNode
        throw(error("Can't check associativity of plan nodes!"))
    end
    if f == choose(false)
        return true
    end
    return Finch.isassociative(Finch.DefaultAlgebra(), f)
end

function iscommutative(f)
    if typeof(f) == PlanNode
        throw(error("Can't check commutativity of plan nodes!"))
    end
    if f == choose(false)
        return true
    end
    return Finch.iscommutative(Finch.DefaultAlgebra(), f)
end

function isunarynull(f)
    if typeof(f) == PlanNode
        throw(error("Can't check commutativity of plan nodes!"))
    end
    return f in (max, min, *, +)
end

function isdistributive(f, g)
    if typeof(f) == PlanNode && typeof(g) == PlanNode
        throw(error("Can't check distributivity of plan nodes!"))
    end
    return Finch.isdistributive(Finch.DefaultAlgebra(), f, g)
end

function cansplitpush(f, g)
    if typeof(f) == PlanNode && typeof(g) == PlanNode
        throw(error("Can't check splitpush of plan nodes!"))
    end
    if !ismissing(repeat_operator(f)) && f == g && iscommutative(f) && isassociative(f)
        return true
    elseif (f == choose(false)) && (g == |)
        return true
    end
    return false
end

# If there exists a g such that f(x_,...{n times},...x) = g(x, n),
# then return g.
function repeat_operator(f)
    if typeof(f) == PlanNode
        throw(error("Can't check repeat operator of plan nodes!"))
    end
    if isidempotent(f)
        return nothing
    elseif f == +
        return *
    elseif f == *
        return exp
    else
        return missing
    end
end

function isidentity(f, x)
    if typeof(f) == PlanNode && typeof(x) == PlanNode
        throw(error("Can't check identity of plan nodes!"))
    end
    return Finch.isidentity(Finch.DefaultAlgebra(), f, x)
end

function isannihilator(f, x)
    if typeof(f) == PlanNode && typeof(x) == PlanNode
        throw(error("Can't check annihilator of plan nodes!"))
    end
    return Finch.isannihilator(Finch.DefaultAlgebra(), f, x)
end

function isidempotent(f)
    if typeof(f) == PlanNode
        throw(error("Can't check idempotency of plan nodes!"))
    end
    return Finch.isidempotent(Finch.DefaultAlgebra(), f)
end

function is_binary(f)
    return (f == |) || (f == &)
end
