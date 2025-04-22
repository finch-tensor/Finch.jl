abstract type AbstractAlgebra end
struct DefaultAlgebra <: AbstractAlgebra end

isassociative(alg) = (f) -> isassociative(alg, f)
isassociative(alg, f::FinchNode) = f.kind === literal && isassociative(alg, f.val)
"""
    isassociative(algebra, f)

Return true when `f(a..., f(b...), c...) = f(a..., b..., c...)` in `algebra`.
"""
isassociative(::Any, f) = false
isassociative(::AbstractAlgebra, ::typeof(or)) = true
isassociative(::AbstractAlgebra, ::typeof(and)) = true
isassociative(::AbstractAlgebra, ::typeof(|)) = true
isassociative(::AbstractAlgebra, ::typeof(&)) = true
isassociative(::AbstractAlgebra, ::typeof(coalesce)) = true
isassociative(::AbstractAlgebra, ::typeof(something)) = true
isassociative(::AbstractAlgebra, ::typeof(+)) = true
isassociative(::AbstractAlgebra, ::typeof(*)) = true
isassociative(::AbstractAlgebra, ::typeof(min)) = true
isassociative(::AbstractAlgebra, ::typeof(max)) = true
isassociative(::AbstractAlgebra, ::typeof(minby)) = true
isassociative(::AbstractAlgebra, ::typeof(maxby)) = true
isassociative(::AbstractAlgebra, ::Chooser) = true

iscommutative(alg) = (f) -> iscommutative(alg, f)
iscommutative(alg, f::FinchNode) = f.kind === literal && iscommutative(alg, f.val)
"""
    iscommutative(algebra, f)

Return true when for all permutations p, `f(a...) = f(a[p]...)` in `algebra`.
"""
iscommutative(::Any, f) = false
iscommutative(::AbstractAlgebra, ::typeof(or)) = true
iscommutative(::AbstractAlgebra, ::typeof(and)) = true
iscommutative(::AbstractAlgebra, ::typeof(|)) = true
iscommutative(::AbstractAlgebra, ::typeof(&)) = true
iscommutative(::AbstractAlgebra, ::typeof(+)) = true
iscommutative(::AbstractAlgebra, ::typeof(*)) = true
iscommutative(::AbstractAlgebra, ::typeof(min)) = true
iscommutative(::AbstractAlgebra, ::typeof(max)) = true

isabelian(alg) = (f) -> isabelian(alg, f)
isabelian(alg, f) = isassociative(alg, f) && iscommutative(alg, f)

isdistributive(alg) = (f, g) -> isdistributive(alg, f, g)
function isdistributive(alg, f::FinchNode, x::FinchNode)
    isliteral(f) && isliteral(x) && isdistributive(alg, f.val, x.val)
end
"""
    isdistributive(algebra, f, g)

Return true when `f(a, g(b, c)) = g(f(a, b), f(a, c))` in `algebra`.
"""
isdistributive(::Any, f, g) = false
isdistributive(::AbstractAlgebra, ::typeof(+), ::typeof(*)) = true

isidempotent(alg) = (f) -> isidempotent(alg, f)
isidempotent(alg, f::FinchNode) = f.kind === literal && isidempotent(alg, f.val)
"""
    isidempotent(algebra, f)

Return true when `f(a, b) = f(f(a, b), b)` in `algebra`.
"""
isidempotent(::Any, f) = false
isidempotent(::AbstractAlgebra, ::typeof(overwrite)) = true
isidempotent(::AbstractAlgebra, ::typeof(|)) = true
isidempotent(::AbstractAlgebra, ::typeof(&)) = true
isidempotent(::AbstractAlgebra, ::typeof(min)) = true
isidempotent(::AbstractAlgebra, ::typeof(max)) = true
isidempotent(::AbstractAlgebra, ::typeof(minby)) = true
isidempotent(::AbstractAlgebra, ::typeof(maxby)) = true
isidempotent(::AbstractAlgebra, ::Chooser) = true

"""
    isidentity(algebra, f, x)

Return true when `f(a..., x, b...) = f(a..., b...)` in `algebra`.
"""
isidentity(alg) = (f, x) -> isidentity(alg, f, x)
function isidentity(alg, f::FinchNode, x::FinchNode)
    isliteral(f) && isidentity_by_fn(alg, f.val, x)
end
isidentity_by_fn(alg, f, x::FinchNode) = isliteral(x) && isidentity(alg, f, x.val)
isidentity(::Any, f, x) = false
isidentity(::AbstractAlgebra, ::typeof(or), x) = x === false
isidentity(::AbstractAlgebra, ::typeof(and), x) = x === true
isidentity(::AbstractAlgebra, ::typeof(coalesce), x) = ismissing(x)
isidentity(::AbstractAlgebra, ::typeof(something), x) = !ismissing(x) && isnothing(x)
isidentity(::AbstractAlgebra, ::typeof(+), x) = !ismissing(x) && iszero(x)
isidentity(::AbstractAlgebra, ::typeof(*), x) = !ismissing(x) && isone(x)
isidentity(::AbstractAlgebra, ::typeof(|), x) = !ismissing(x) && iszero(x)
isidentity(::AbstractAlgebra, ::typeof(&), x) = !ismissing(x) && x == ~(zero(x))
isidentity(::AbstractAlgebra, ::typeof(min), x) = !ismissing(x) && isinf(x) && x > 0
isidentity(::AbstractAlgebra, ::typeof(max), x) = !ismissing(x) && isinf(x) && x < 0
function isidentity(::Finch.AbstractAlgebra, ::InitMax{D}, x) where {D}
    isequal(x, D) || isequal(x == D, true)
end
function isidentity(::Finch.AbstractAlgebra, ::InitMin{D}, x) where {D}
    isequal(x, D) || isequal(x == D, true)
end

function isidentity_by_fn(alg::AbstractAlgebra, ::typeof(minby), x::FinchNode)
    if @capture x call(tuple, ~a::isliteral, ~b)
        return isidentity(alg, min, a.val)
    elseif isliteral(x)
        return isidentity(alg, min, first(x.val))
    end
    return false
end
function isidentity_by_fn(alg::AbstractAlgebra, ::typeof(maxby), x::FinchNode)
    if @capture x call(tuple, ~a::isliteral, ~b)
        return isidentity(alg, max, a.val)
    elseif isliteral(x)
        return isidentity(alg, max, first(x.val))
    end
    return false
end
function isidentity(::AbstractAlgebra, ::Chooser{Vf}, x) where {Vf}
    isequal(x, Vf) || isequal(x == Vf, true)
end
function isidentity(::AbstractAlgebra, ::InitWriter{Vf}, x) where {Vf}
    isequal(x, Vf) || isequal(x == Vf, true)
end

isannihilator(alg) = (f, x) -> isannihilator(alg, f, x)
function isannihilator(alg, f::FinchNode, x::FinchNode)
    isliteral(f) && isannihilator_by_fn(alg, f.val, x)
end
isannihilator_by_fn(alg, f, x::FinchNode) = isliteral(x) && isannihilator(alg, f, x.val)
"""
    isannihilator(algebra, f, x)

Return true when `f(a..., x, b...) = x` in `algebra`.
"""
isannihilator(::Any, f, x) = false
isannihilator(::AbstractAlgebra, ::typeof(+), x) = ismissing(x) || isinf(x)
isannihilator(::AbstractAlgebra, ::typeof(*), x) = ismissing(x) || iszero(x)
isannihilator(::AbstractAlgebra, ::typeof(min), x) = ismissing(x) || isinf(x) && x < 0
isannihilator(::AbstractAlgebra, ::typeof(max), x) = ismissing(x) || isinf(x) && x > 0
isannihilator(::AbstractAlgebra, ::typeof(or), x) = ismissing(x) || x === true
isannihilator(::AbstractAlgebra, ::typeof(and), x) = ismissing(x) || x === false
isannihilator(::AbstractAlgebra, ::typeof(|), x) = !ismissing(x) && x == ~(zero(x))
isannihilator(::AbstractAlgebra, ::typeof(&), x) = !ismissing(x) && iszero(x)
function isannihilator_by_fn(alg::AbstractAlgebra, ::typeof(minby), x::FinchNode)
    if @capture x call(tuple, ~a::isliteral, ~b)
        return isannihilator(alg, min, a.val)
    elseif isliteral(x)
        return isannihilator(alg, min, first(x.val))
    end
    return false
end
function isannihilator_by_fn(alg::AbstractAlgebra, ::typeof(maxby), x::FinchNode)
    if @capture x call(tuple, ~a::isliteral, ~b)
        isannihilator(alg, max, a.val)
    elseif isliteral(x)
        isannihilator(alg, max, first(x.val))
    end
    return false
end
function isannihilator(::AbstractAlgebra, ::Chooser{Vf}, x) where {Vf}
    !(isequal(x, Vf) || isequal(x == Vf, true))
end
#isannihilator(::AbstractAlgebra, ::InitWriter{Vf}, x) where {Vf} = !(isequal(x, Vf) || isequal(x == Vf, true))

isinverse(alg) = (f, g) -> isinverse(alg, f, g)
function isinverse(alg, f::FinchNode, g::FinchNode)
    isliteral(f) && isliteral(g) && isinverse(alg, f.val, g.val)
end
"""
    isinverse(algebra, f, g)

Return true when `f(a, g(a))` is the identity under `f` in `algebra`.
"""
isinverse(::Any, f, g) = false
isinverse(::AbstractAlgebra, ::typeof(-), ::typeof(+)) = true
isinverse(::AbstractAlgebra, ::typeof(inv), ::typeof(*)) = true

isinvolution(alg) = (f) -> isinvolution(alg, f)
isinvolution(alg, f::FinchNode) = isliteral(f) && isinvolution(alg, f.val)
"""
    isinvolution(algebra, f)

Return true when `f(f(a)) = a` in `algebra`.
"""
isinvolution(::Any, f) = false
isinvolution(::AbstractAlgebra, ::typeof(-)) = true
isinvolution(::AbstractAlgebra, ::typeof(inv)) = true

return_type(alg) = (f, args...) -> return_type(alg, f, args...)
"""
    return_type(algebra, f, arg_types...)

Give the return type of `f` when applied to arguments of types `arg_types...` in `algebra`.
Used to determine output types of functions in the high-level interface.
This function falls back to [`Base.promote_op`](https://github.com/JuliaLang/julia/blob/46933b8a708453f9caecb8abce88770c4bff675d/base/promotion.jl#L508-L619).
"""
return_type(alg, f, arg_types...) = Base.promote_op(f, arg_types...)

collapsed(alg, idx, ext, lhs, f::FinchNode, rhs) = collapsed(alg, idx, ext, lhs, f.val, rhs)
"""
    collapsed(algebra, f, idx, ext, node)

Return collapsed expression with respect to f.
"""
function collapsed(alg, idx, ext, lhs, f::Any, rhs)
    if isidempotent(alg, f)
        sieve(call(>=, measure(ext), get_smallest_measure(ext)), assign(lhs, f, rhs)) # Hmm.. Why do we need sieve for  only idempotent?
    else
        nothing # Hmm.. Why do we need sieve for  only idempotent?
    end # Hmm.. Why do we need sieve for  only idempotent?
end # Hmm.. Why do we need sieve for  only idempotent?

function collapsed(alg, idx, ext, lhs, f::typeof(-), rhs)
    assign(lhs, f, call(*, measure(ext), rhs))
end
function collapsed(alg, idx, ext, lhs, f::typeof(*), rhs)
    assign(lhs, f, call(^, rhs, measure(ext)))
end
function collapsed(alg, idx, ext::VirtualExtent, lhs, f::typeof(+), rhs)
    assign(lhs, f, call(*, measure(ext), rhs))
end
function collapsed(alg, idx, ext::VirtualContinuousExtent, lhs, f::typeof(+), rhs)
    if (@capture rhs call(*, ~a1..., call(d, ~i1..., idx, ~i2...), ~a2...)) # Lebesgue
        if prove(FinchCompiler(), call(==, measure(ext), 0))
            assign(lhs, f, literal(0))
        else
            assign(
                lhs,
                f,
                call(*, call(drop_eps, measure(ext)), a1..., a2..., call(d, i1..., i2...)),
            )
        end
    else # Counting
        if prove(FinchCompiler(), call(==, measure(ext), 0))
            assign(lhs, f, rhs)
        else
            sieve(call(==, measure(ext), 0), assign(lhs, f, rhs)) # Undefined if measure != 0
            #block(sieve(call(==, measure(ext), 0), assign(lhs, f, rhs)),
            #      sieve(call(!=, measure(ext), 0), assign(lhs, f, Inf))) #TODO : add "else" in sieve
        end
    end
end

getvars(arr::AbstractArray) = mapreduce(getvars, vcat, arr; init=[])
getvars(arr) = getroot(arr) === nothing ? [] : [getroot(arr)]
getroot(arr) = nothing
function getvars(node::FinchNode)
    if node.kind == variable
        return [node]
    elseif node.kind == virtual
        return getvars(node.val)
    elseif istree(node)
        return mapreduce(getvars, vcat, arguments(node); init=[])
    else
        return []
    end
end

ortho(var, stmt) = !(var in getvars(stmt))

"""
    StaticHash

A hash function which is static, i.e. the hashes are the same when objects are hashed in the same order.
The hash is used to memoize the results of simplification and proof rules.
"""
struct StaticHash
    counts::Dict{Tuple{Any,DataType},UInt}
end
StaticHash() = StaticHash(Dict{Tuple{Any,DataType},UInt}())

(h::StaticHash)(x) = get!(h.counts, (x, typeof(x)), UInt(length(h.counts)))

"""
    SymbolicContext

A compiler context for symbolic computation, defined on an algebra.
"""
@kwdef struct SymbolicContext
    algebra
    shash = StaticHash()
    simplify_rules = get_simplify_rules(algebra, shash)
    prove_rules = get_prove_rules(algebra, shash)
    prove_cache = Dict()
end

"""
    get_algebra(ctx)

get the algebra used in the current context
"""
get_algebra(ctx::SymbolicContext) = ctx.algebra

"""
    get_static_hash(ctx)

Return an object which can be called as a hash function. The hashes
are the same when objects are hashed in the same order.
"""
get_static_hash(ctx::SymbolicContext) = ctx.shash

include("simplify.jl")
include("prove.jl")
