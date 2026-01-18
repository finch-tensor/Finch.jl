function isfoldable(x)
    isconstant(x) || (x.kind === call && isliteral(x.op) && all(isfoldable, x.args))
end

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
                    (@rule tag(~var, ~bind::isvalue) => if has_binding(ctx, var)
                        get_binding(ctx, var)
                    else
                        bind
                    end),
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
                            (@rule call(
                                ~f::isliteral,
                                ~a::(All(Or(isvariable, isvirtual, isfoldable)))...,
                            ) => begin
                                x = virtual_call(ctx, f.val, a...)
                                if x !== nothing
                                    finch_leaf(x)
                                end
                            end),
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

"""
    virtual_type(ctx, algebra, arg)

Return the narrowest type constraint on the argument `arg` that is compatible with the algebra.
"""
virtual_type(ctx, alg, arg) = Any
virtual_type(ctx, arg) = virtual_type(ctx, get_algebra(ctx), arg)

function virtual_type(ctx, alg, arg::FinchNode)
    if arg.kind === literal
        return typeof(arg.val)
    elseif arg.kind === value
        return arg.type
    elseif arg.kind === variable || arg.kind === index
        if has_binding(ctx, arg)
            return virtual_type(ctx, alg, get_binding(ctx, arg))
        else
            return Any
        end
    elseif arg.kind === virtual
        return virtual_type(ctx, alg, arg.val)
    elseif @capture arg call(~f::isliteral, ~args...)
        arg_types = map(arg.args) do arg
            virtual_type(ctx, alg, arg)
        end
        T = return_type(alg, f.val, arg_types...)
        return return_type(alg, f.val, arg_types...)
    else
        return Any
    end
end

"""
    virtual_call(ctx, f, args...)

Given the virtual arguments `args...`, and a literal function `f`, return a virtual
object representing the result of the function call. If the function is not
foldable, return nothing. This function is used so that we can call e.g. tensor
wrapper constructors and dimension constructors in finch code. Implementations should overload
`virtual_call_def` to provide the actual implementation.
"""
function virtual_call(ctx, f, args...)
    virtual_call_def(
        ctx,
        get_algebra(ctx),
        f,
        Tuple{map(arg -> virtual_type(ctx, arg), args)...},
        args...)
end

virtual_call_def(ctx, alg, f, arg_types, args...) = nothing
