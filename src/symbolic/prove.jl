"""
    get_prove_rules(alg, shash)

Return the bound rule set for Finch. One can dispatch on the `alg` trait to
specialize the rule set for different algebras. `shash` is an object that can be
called to return a static hash value. This rule set is used to analyze loop
bounds in Finch.
"""
function get_prove_rules(alg, shash)
    return [
        (@rule call(~f::isliteral, ~a::isliteral, ~b::(All(isliteral))...) =>
            literal(getval(f)(getval(a), getval.(b)...))),
        (@rule call(~f::isassociative(alg), ~a..., call(~f, ~b...), ~c...) =>
            call(f, a..., b..., c...)),
        (@rule call(~f::iscommutative(alg), ~a...) =>
            if !(issorted(a; by=x -> (!isliteral(x), shash(x))))
                call(f, sort(a; by=x -> (!isliteral(x), shash(x)))...)
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
        (@rule(cached(~a, ~b::isliteral) => b.val)),
        (@rule cached(cached(~a, ~b), ~c) => cached(a, c)),
        (@rule call(==, ~a, ~a) => literal(true)),
        (@rule call(>=, ~a, ~b) => call(==, a, call(max, a, b))),
        (@rule call(<=, ~a, ~b) => call(==, call(max, a, b), b)),
        (@rule call(identity, ~a) => a),
        (@rule call(overwrite, ~a, ~b) => b),
        (@rule call(ifelse, true, ~a, ~b) => a),
        (@rule call(ifelse, false, ~a, ~b) => b),
        (@rule call(ifelse, ~a, ~b, ~b) => b),
        (@rule $(literal(-0.0)) => literal(0.0)),
        (@rule call(-, ~a, ~b) => call(+, a, call(-, b))),
        (@rule call(-, call(+, ~a...)) =>
            call(+, map(x -> call(-, x), a)...)),
        (@rule call(+, ~a..., ~b, ~c..., call(-, ~b), ~d...) =>
            call(+, 0, a..., c..., d...)),
        (@rule call(+, ~a..., call(-, ~b), ~c..., ~b, ~d...) =>
            call(+, 0, a..., c..., d...)),
        (@rule call(+, ~a..., call(min, ~b...), ~c...) =>
            call(min, map(x -> call(+, a..., x, c...), b)...)),
        (@rule call(+, ~a..., call(max, ~b...), ~c...) =>
            call(max, map(x -> call(+, a..., x, c...), b)...)),
        (@rule call(max, ~a1..., call(min, ~a2..., call(max, ~a3...), ~a4...), ~a5...) =>
            if !(isdisjoint(a3, a1) && isdisjoint(a3, a5))
                call(
                    max,
                    a1...,
                    call(min, a2..., call(max, setdiff(setdiff(a3, a1), a5)...), a4...),
                    a5...,
                )
            end),
        (@rule call(min, ~a1..., call(max, ~a2..., call(min, ~a3...), ~a4...), ~a5...) =>
            if !(isdisjoint(a3, a1) && isdisjoint(a3, a5))
                call(
                    min,
                    a1...,
                    call(max, a2..., call(min, setdiff(setdiff(a3, a1), a5)...), a4...),
                    a5...,
                )
            end),
        (@rule call(max, ~a1..., call(min, ~a2...), ~a3...) =>
            if !(isdisjoint(a1, a2) && isdisjoint(a2, a3))
                call(max, a1..., a3...)
            end),
        (@rule call(min, ~a1..., call(max, ~a2...), ~a3...) =>
            if !(isdisjoint(a1, a2) && isdisjoint(a2, a3))
                call(min, a1..., a3...)
            end),
        (@rule call(max, ~a1..., call(min, ~a2...), ~a3..., call(min, ~a4...), ~a5...) =>
            if !(isdisjoint(a2, a4))
                call(
                    max,
                    a1...,
                    call(
                        min,
                        intersect(a2, a4)...,
                        call(
                            max,
                            call(min, setdiff(a2, a4)...),
                            call(min, setdiff(a4, a2)...),
                        ),
                    ),
                    a3...,
                    a5...,
                )
            end),
        (@rule call(min, ~a1..., call(max, ~a2...), ~a3..., call(max, ~a4...), ~a5...) =>
            if !(isdisjoint(a2, a4))
                call(
                    min,
                    a1...,
                    call(
                        max,
                        intersect(a2, a4)...,
                        call(
                            min,
                            call(max, setdiff(a2, a4)...),
                            call(max, setdiff(a4, a2)...),
                        ),
                    ),
                    a3...,
                    a5...,
                )
            end), (@rule call(min, ~a1..., call(max), ~a2...) => call(min, a1..., a2...)),
        (@rule call(max, ~a1..., call(min), ~a2...) => call(min, a1..., a2...)),
        (@rule call(
            min,
            ~a1...,
            call(+, ~b::isliteral, ~a2...),
            ~a3...,
            call(+, ~c::isliteral, ~a2...),
            ~a4...,
        ) =>
            call(min, a1..., call(+, min(b.val, c.val), ~a2...), ~a3..., ~a4...)),
        (@rule call(
            min, ~a1..., call(+, ~a2...), ~a3..., call(+, ~c::isliteral, ~a2...), ~a4...
        ) =>
            call(min, a1..., call(+, min(0, c.val), ~a2...), ~a3..., ~a4...)),
        (@rule call(min, ~a1..., ~a2, ~a3..., call(+, ~c::isliteral, ~a2), ~a4...) =>
            call(min, a1..., call(+, min(0, c.val), ~a2), ~a3..., ~a4...)),
        (@rule call(
            min, ~a1..., call(+, ~b::isliteral, ~a2...), ~a3..., call(+, ~a2...), ~a4...
        ) =>
            call(min, a1..., call(+, min(b.val, 0), ~a2...), ~a3..., ~a4...)),
        (@rule call(min, ~a1..., call(+, ~b::isliteral, ~a2), ~a3..., ~a2, ~a4...) =>
            call(min, a1..., call(+, min(b.val, 0), ~a2), ~a3..., ~a4...)),
        (@rule call(
            max,
            ~a1...,
            call(+, ~b::isliteral, ~a2...),
            ~a3...,
            call(+, ~c::isliteral, ~a2...),
            ~a4...,
        ) =>
            call(max, a1..., call(+, max(b.val, c.val), ~a2...), ~a3..., ~a4...)),
        (@rule call(
            max, ~a1..., call(+, ~a2...), ~a3..., call(+, ~c::isliteral, ~a2...), ~a4...
        ) =>
            call(max, a1..., call(+, max(0, c.val), ~a2...), ~a3..., ~a4...)),
        (@rule call(max, ~a1..., ~a2, ~a3..., call(+, ~c::isliteral, ~a2), ~a4...) =>
            call(max, a1..., call(+, max(0, c.val), ~a2), ~a3..., ~a4...)),
        (@rule call(
            max, ~a1..., call(+, ~b::isliteral, ~a2...), ~a3..., call(+, ~a2...), ~a4...
        ) =>
            call(max, a1..., call(+, max(b.val, 0), ~a2...), ~a3..., ~a4...)),
        (@rule call(max, ~a1..., call(+, ~b::isliteral, ~a2), ~a3..., ~a2, ~a4...) =>
            call(max, a1..., call(+, max(b.val, 0), ~a2), ~a3..., ~a4...)),
        (@rule call(~f::isinvolution(alg), call(~f, ~a)) => a),

        # Clamping rules
        #=
        # this rule is great but too expensive
        (@rule call(max, ~a, call(min, ~b, ~c)) => begin
            if prove(FinchCompiler(), call(<=, a, b)) # a = low, b = high
              call(min, b, call(max, a, c))
            elseif prove(FinchCompiler(), call(<=, a, c)) # a = low, c = high
              call(min, c, call(max, b, a))
            end
          end),
        =#

        #(@rule call(~f, ~a..., call(~g, ~b), ~c...) => if isdistributive(alg, f, g)
        #    call(g, call(f, a..., b, c...))
        #end),
    ]
end

"""
    prove(ctx, root; verbose = false)

use the rules in `ctx` to attempt to prove that the program `root` is true.
Return false if the program cannot be shown to be true.
"""
function prove(ctx::SymbolicContext, root::FinchNode; verbose=false)
    root = Rewrite(Prewalk(Fixpoint(Chain([
        @rule(cached(~a, ~b::isliteral) => b.val)
    ]))))(
        root
    )
    names = Dict()
    function rename(node::FinchNode)
        if node.kind == virtual
            get!(names, node, value(Symbol(:virtual_, length(names) + 1)))
        elseif node.kind == index
            value(node.name)
        elseif isvalue(node)
            get!(names, node, value(Symbol(:value_, length(names) + 1)))
        end
    end
    root = Rewrite(Postwalk(rename))(root)
    res = Rewrite(
        Fixpoint(Prewalk(Memo(Fixpoint(Chain(ctx.prove_rules)), ctx.prove_cache)))
    )(
        root
    )
    if verbose
        @info "proving..." root res
    end
    if isliteral(res)
        return res.val
    else
        return false
    end
end
