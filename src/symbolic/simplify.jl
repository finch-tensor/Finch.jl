"""
    getunbound(stmt)

Return an iterator over the indices in a Finch program that have yet to be bound.
```julia
julia> getunbound(@finch_program for i=_; :a[i, j] += 2 end)
[j]
julia> getunbound(@finch_program i + j * 2 * i)
[i, j]
```
"""
getunbound(ex) = istree(ex) ? mapreduce(getunbound, union, arguments(ex); init=[]) : []

function getunbound(ex::FinchNode)
    if ex.kind === index
        return [ex]
    elseif @capture ex call(d, ~idx...)
        return []
    elseif ex.kind === loop
        return setdiff(union(getunbound(ex.body), getunbound(ex.ext)), getunbound(ex.idx))
    elseif istree(ex)
        return mapreduce(Finch.getunbound, union, arguments(ex); init=[])
    else
        return []
    end
end

"""
    get_simplify_rules(alg, shash)

Return the program rule set for Finch. One can dispatch on the `alg` trait to
specialize the rule set for different algebras. Defaults to a collection of
straightforward rules that use the algebra to check properties of functions
like associativity, commutativity, etc. `shash` is an object that can be called
to return a static hash value. This rule set simplifies, normalizes, and
propagates constants, and is the basis for how Finch understands sparsity.
"""
function get_simplify_rules(alg, shash)
    return [
        (@rule call(~f::isliteral, ~a::(All(isliteral))...) =>
            literal(getval(f)(getval.(a)...))),
        (@rule call(~f::isassociative(alg), ~a..., call(~f, ~b...), ~c...) =>
            call(f, a..., b..., c...)),
        (@rule call(~f::iscommutative(alg), ~a...) => if !(issorted(a; by=shash))
            call(f, sort(a; by=shash)...)
        end),
        (@rule call(~f::isidempotent(alg), ~a...) => if !allunique(a)
            call(f, unique(a)...)
        end),
        (@rule call(~f::isassociative(alg), ~a..., ~b::isliteral, ~c::isliteral, ~d...) =>
            call(f, a..., f.val(b.val, c.val), d...)),
        (@rule call(
            ~f::isabelian(alg), ~a..., ~b::isliteral, ~c..., ~d::isliteral, ~e...
        ) => call(f, a..., f.val(b.val, d.val), c..., e...)),
        (@rule call(~f, ~a..., ~b, ~c...) => if isannihilator(alg, f, b)
            b
        end),
        (@rule call(~f, ~a..., ~b, ~c, ~d...) => if isidentity(alg, f, b)
            call(f, a..., c, d...)
        end),
        (@rule call(~f, ~a..., ~b, ~c, ~d...) => if isidentity(alg, f, c)
            call(f, a..., b, d...)
        end),
        (@rule call(~f, ~a) => if isassociative(alg, f)
            a
        end), #TODO
        (@rule call(>=, ~a, ~b) => call(<=, b, a)),
        (@rule call(>, ~a, ~b) => call(<, b, a)),
        (@rule call(<, Inf, ~a) => literal(false)),
        (@rule call(<, ~a, -Inf) => literal(false)),
        (@rule call(>, ~a, Inf) => literal(false)),
        (@rule call(>, -Inf, ~a) => literal(false)),
        (@rule call(<=, ~a, call(max, ~b...)) => call(or, map(x -> call(<=, a, x), b)...)),
        (@rule call(<, ~a, call(max, ~b...)) => call(or, map(x -> call(<, a, x), b)...)),
        (@rule call(<=, call(max, ~a...), ~b) => call(and, map(x -> call(<=, x, b), a)...)),
        (@rule call(<, call(max, ~a...), ~b) => call(and, map(x -> call(<, x, b), a)...)),
        (@rule call(<=, ~a, call(min, ~b...)) => call(and, map(x -> call(<=, a, x), b)...)),
        (@rule call(<, ~a, call(min, ~b...)) => call(and, map(x -> call(<, a, x), b)...)),
        (@rule call(<=, call(min, ~a...), ~b) => call(or, map(x -> call(<=, x, b), a)...)),
        (@rule call(<, call(min, ~a...), ~b) => call(or, map(x -> call(<, x, b), a)...)),
        (@rule call(==, ~a, ~a) => literal(true)),
        (@rule call(<=, ~a, ~a) => literal(true)),
        (@rule call(<, ~a, ~a) => literal(false)),
        (@rule assign(access(~a, updater(~g), ~i...), ~f, ~b) => if isidentity(alg, f, b)
            block()
        end), #updater(auto)
        (@rule assign(access(~a, ~m, ~i...), $(literal(missing))) => block()),
        (@rule assign(access(~a, ~m, ~i..., $(literal(missing)), ~j...), ~b) => block()),
        (@rule call(coalesce, ~a..., ~b, ~c...) =>
            if isvalue(b) && !(Missing <: b.type) || isliteral(b) && !ismissing(b.val)
                call(coalesce, a..., b)
            end),
        (@rule call(something, ~a..., ~b, ~c...) =>
            if isvalue(b) && !(Nothing <: b.type) || isliteral(b) && b != literal(nothing)
                call(something, a..., b)
            end),
        (@rule call(~f, ~a..., cached(~b, ~c::isliteral), ~d...) =>
            cached(call(f, a..., b, d...), literal(call(f, a..., c.val, d...)))),
        (@rule cached(cached(~a, ~b), ~c) => cached(a, c)), (@rule call(identity, ~a) => a),
        (@rule call(overwrite, ~a, ~b) => b),
        (@rule call(~f::isliteral, ~a, ~b) => if f.val isa InitWriter
            b
        end),
        (@rule assign(~a::isliteral, ~op, ~b) => if a.val === Null()
            block()
        end),
        (@rule call(~f::isliteral, true, ~b) => if f.val isa FilterOp
            b
        end),
        (@rule call(~f::isliteral, false, ~b) => if f.val isa FilterOp
            f.val(false, nothing)
        end),
        (@rule call(~f::isliteral, ~b, ~c::isliteral) => if f.val isa FilterOp{c.val}
            c
        end),
        (@rule call(ifelse, true, ~a, ~b) => a),
        (@rule call(ifelse, false, ~a, ~b) => b),
        (@rule call(ifelse, ~a, ~b, ~b) => b),
        (@rule call(norm, ~x::isliteral, ~y) => if iszero(x.val)
            x
        end),
        (@rule block(~a1..., sieve(~c, ~b1), sieve(~c, ~b2), ~a2...) =>
            block(a1..., sieve(~c, block(b1, b2)), a2...)
    ),
        (@rule call(~f, call(~g, ~a, ~b...)) =>
            if isinverse(alg, f, g) && isassociative(alg, g)
                call(g, call(f, a), map(c -> call(f, call(g, c)), b)...)
            end),

        #TODO should put a zero here, but we need types
        #=
        (@rule call(~g, ~a..., ~b, ~c..., call(~f, ~b), ~d...) => if isinverse(alg, f, g) && isassociative(alg, g)
            call(g, a..., c..., d...)
        end),
        (@rule call(~g, ~a..., call(~f, ~b), ~c..., ~b, ~d...) => if isinverse(alg, f, g) && isassociative(alg, g)
            call(g, a..., c..., d...)
        end),
        =#
        (@rule call(+, ~a..., ~b, ~c..., call(-, ~b), ~d...) =>
            if isinverse(alg, -, +) && isassociative(alg, +)
                call(+, false, a..., c..., d...)
            end),
        (@rule call(+, ~a..., call(-, ~b), ~c..., ~b, ~d...) =>
            if isinverse(alg, -, +) && isassociative(alg, +)
                call(+, false, a..., c..., d...)
            end), (@rule call(-, ~a, ~b) => call(+, a, call(-, b))),
        (@rule call(/, ~a, ~b) => call(*, a, call(inv, b))),
        (@rule call(~f::isinvolution(alg), call(~f, ~a)) => a),
        (@rule call(~f, ~a..., call(~g, ~b), ~c...) => if isdistributive(alg, g, f)
            call(g, call(f, a..., b, c...))
        end), (@rule call(/, ~a) => call(inv, a)), (@rule sieve(true, ~a) => a),
        (@rule sieve(false, ~a) => block()), #TODO should add back skipvisitor

        # Top-down reduction
        (@rule loop(~idx, ~ext::isvirtual, ~body) => begin
            body_contain_idx = idx ∈ getunbound(body)
            if !body_contain_idx
                decl_in_scope = filter(
                    !isnothing,
                    map(node -> if @capture(node, declare(~tns, ~init, ~op))
                            tns
                        elseif @capture(node, define(~var, ~val, ~body_2))
                            var
                        end, PostOrderDFS(body)),
                )
                Postwalk(
                    @rule assign(access(~lhs, updater(~g), ~j...), ~f, ~rhs) => begin #updater(auto)
                        access_in_rhs = filter(
                            !isnothing,
                            map(
                                node ->
                                    if @capture(node, access(~tns, reader(), ~k...))
                                        tns # TODO add getroot here?
                                    elseif @capture(node, ~var::isvariable)
                                        var
                                    end, PostOrderDFS(rhs)),
                        )
                        if !(lhs in decl_in_scope) &&
                            isempty(intersect(access_in_rhs, decl_in_scope))
                            collapsed(alg, idx, ext.val, access(lhs, updater(f), j...), f, rhs)
                        end
                    end
                )(
                    body
                )
            end
        end),

        # Lifting sieve
        (@rule loop(~idx, ~ext::isvirtual, sieve(~cond, ~body)) => begin
            if idx ∉ getunbound(cond)
                sieve(cond, loop(idx, ext, body))
            end
        end),

        # Bottom-up reduction1
        (@rule loop(
            ~idx, ~ext::isvirtual, assign(access(~lhs, updater(~g), ~j...), ~f, ~rhs)
        ) => begin #updater(auto)
            if idx ∉ j && idx ∉ getunbound(rhs)
                collapsed(alg, idx, ext.val, access(lhs, updater(f), j...), f, rhs)
            end
        end),

        ## Bottom-up reduction2
        (@rule loop(
            ~idx,
            ~ext::isvirtual,
            block(~s1..., assign(access(~lhs, updater(~g), ~j...), ~f, ~rhs), ~s2...),
        ) => begin #updater(auto)
            if ortho(getroot(lhs), s1) && ortho(getroot(lhs), s2)
                if idx ∉ j && idx ∉ getunbound(rhs)
                    body = block(s1..., assign(access(lhs, updater(f), j...), f, rhs), s2...)
                    decl_in_scope = filter(
                        !isnothing,
                        map(node -> if @capture(node, declare(~tns, ~init, ~op))
                                tns
                            elseif @capture(node, define(~var, ~val, ~body_2))
                                var
                            end, PostOrderDFS(body)),
                    )

                    access_in_rhs = filter(
                        !isnothing,
                        map(node -> if @capture(node, access(~tns, reader(), ~k...))
                                tns
                            elseif @capture(node, ~var::isvariable)
                                var
                            end, PostOrderDFS(rhs)),
                    )

                    if !(lhs in decl_in_scope) &&
                        isempty(intersect(access_in_rhs, decl_in_scope))
                        collapsed_body = collapsed(
                            alg, idx, ext.val, access(lhs, updater(f), j...), f, rhs
                        )
                        block(collapsed_body, loop(idx, ext, block(s1..., s2...)))
                    end
                end
            end
        end),
        (@rule block(~s1..., thaw(~a::isvariable, ~f), ~s2..., freeze(~a, ~f), ~s3...) =>
            if ortho(a, s2)
                block(s1..., s2..., s3...)
            end),
        (@rule block(~s1..., freeze(~a::isvariable, ~f), ~s2..., thaw(~a, ~f), ~s3...) =>
            if ortho(a, s2)
                block(s1..., s2..., s3...)
            end),
    ]
end

@kwdef mutable struct Simplify
    body
end

struct SimplifyStyle end

get_style(ctx, ::Simplify, root) = SimplifyStyle()
combine_style(a::SimplifyStyle, b::SimplifyStyle) = a

function lower(ctx::AbstractCompiler, root, ::SimplifyStyle)
    root = Rewrite(Prewalk((x) -> if x.kind === virtual
        visit_simplify(x.val)
    end))(root)
    root = simplify(ctx, root)
    ctx(root)
end

FinchNotation.finch_leaf(x::Simplify) = virtual(x)

visit_simplify(node) = node
visit_simplify(node::Simplify) = node.body
function visit_simplify(node::FinchNode)
    if node.kind === access && node.tns.kind === virtual
        visit_simplify_access(node, node.tns.val)
    elseif node.kind === virtual
        visit_simplify(node.val)
    else
        node
    end
end

"""
   simplify(ctx, node)

simplify the program `node` using the rules in `ctx`
"""
function simplify(ctx::SymbolicContext, node)
    Rewrite(
        Fixpoint(
            Chain([
                Prewalk(Fixpoint(Chain(ctx.simplify_rules))),
                #these rules are non-customizeable:
                Prewalk(
                    Chain([
                        (@rule loop(~i, ~a, block()) => block()),
                        (@rule sieve(~cond, block()) => block()),
                        (@rule block(~a..., block(~b...), ~c...) =>
                            block(a..., b..., c...)),
                        (@rule define(~a::isvariable, ~v::isconstant, ~body) => begin
                            body_2 = Postwalk(@rule a => v)(body)
                            if body_2 !== nothing
                                #We cannot remove the definition because we aren't sure if the variable gets referenced from a virtual.
                                define(a, v, body_2)
                            end
                        end),
                        (@rule assign(~a, ~op, cached(~b, ~c)) => assign(a, op, b)),
                    ]),
                ),
            ]),
        ),
    )(
        node
    )
end
