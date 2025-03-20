"""
    evaluate_partial(ctx, root)

This pass evaluates tags, global variable definitions, and foldable functions
into the context bindings.
"""
function evaluate_partial(ctx, root)
    root = Rewrite(
        Fixpoint(
            Postwalk(
                Chain([
                    (@rule tag(~var, ~bind::isindex) => bind),
                    (@rule tag(~var, ~bind::isvariable) => bind),
                    (@rule tag(~var, ~bind::isliteral) => bind),
                    (@rule tag(~var, ~bind::isvalue) => bind),
                    (@rule tag(~var, ~bind::isvirtual) => begin
                        get_binding!(ctx, var, bind)
                        var
                    end
                ),
                ]),
            ),
        ),
    )(
        root
    )

    root = Rewrite(
        Fixpoint(
            Chain([
                Fixpoint(
                    @rule define(~a::isvariable, ~v::Or(isconstant, isvirtual), ~s) =>
                        begin
                            set_binding!(ctx, a, v)
                            s
                        end
                ),
                Postwalk(
                    Fixpoint(
                        Chain([
                            (@rule ~v::isvariable => if has_binding(ctx, v)
                                val = get_binding(ctx, v)
                                if isvariable(val) || isconstant(val)
                                    val
                                end
                            end),
                            (@rule call(~f::isliteral, ~a::(All(isliteral))...) =>
                                finch_leaf(getval(f)(getval.(a)...))),
                            (@rule define(~a::isvariable, ~v::isconstant, ~body) => begin
                                body_2 = Postwalk(@rule a => v)(body)
                                if body_2 !== nothing
                                    #We cannot remove the definition because we aren't sure if the variable gets referenced from a virtual.
                                    define(a, v, body_2)
                                end
                            end),
                            (@rule block(~a) => a),
                            (@rule block(~a1..., block(~b...), ~a2...) =>
                                block(a1..., b..., a2...)),
                            (@rule block(
                                ~a1..., define(~b, ~v, ~c), yieldbind(~d...), ~a2...
                            ) =>
                                block(
                                    a1..., define(b, v, block(c, yieldbind(d...))), a2...
                                )),
                        ]),
                    ),
                ),
            ]),
        ),
    )(
        root
    )
end
