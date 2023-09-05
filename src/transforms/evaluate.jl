isfoldable(x) = isconstant(x) || (x.kind === call && isliteral(x.op) && all(isfoldable, x.args))

"""
    evaluate_partial(root, ctx)

This pass evaluates tags, global variable definitions, and foldable functions
into the context bindings.
"""
function evaluate_partial(root, ctx)
    root = Rewrite(Fixpoint(Postwalk(Chain([
        (@rule tag(~var, ~bind::isindex) => bind),
        (@rule tag(~var, ~bind::isvariable) => bind),
        (@rule tag(~var, ~bind::isliteral) => bind),
        (@rule tag(~var, ~bind::isvalue) => bind),
        (@rule tag(~var, ~bind::isvirtual) => begin
            get!(ctx.bindings, var, bind)
            var
        end
        )
    ]))))(root)
    root = Rewrite(Fixpoint(Chain([
        Fixpoint(@rule block(~s1..., block(~s2...), ~s3...)=> block(s1..., s2..., s3...)),
        Fixpoint(@rule block(define(~a::isvariable, ~v::Or(isconstant, isvirtual)), ~s...) => begin
            ctx.bindings[a] = v
            block(s...)
        end),
        Postwalk(Fixpoint(Chain([
            (@rule call(~f::isliteral, ~a::(All(Or(isvariable, isvirtual, isfoldable)))...) => virtual_call(f.val, ctx, a...)),
            (@rule call(~f::isliteral, ~a::(All(isliteral))...) => finch_leaf(getval(f)(getval.(a)...))),
            (@rule block(~s1..., define(~a::isvariable, ~v::isconstant), ~s2...) => begin
                s2_2 = Postwalk(@rule a => v)(block(s2...))
                if s2_2 !== nothing
                    #We cannot remove the definition because we aren't sure if the variable gets referenced from a virtual.
                    block(s1..., define(a, v), s2_2.bodies...)
                end
            end),
        ])))
    ])))(root)
end

virtual_call(f, ctx, a...) = nothing

function virtual_call(::typeof(default), ctx, a) 
    if haskey(ctx.bindings, getroot(a))
        return virtual_default(a, ctx)
    end
end


virtual_uncall(x) = nothing

function unevaluate_partial(root, ctx)
    tnss = unique(filter(!isnothing, map(node->if @capture(node, access(~A, ~m, ~i...)) getroot(A) end, PostOrderDFS(root))))
    for tns in tnss
        if haskey(ctx.bindings, tns)
            root = block(define(tns, ctx.bindings[tns]), root)
            delete!(ctx.bindings, tns)
        end
    end
    Rewrite(Fixpoint(Postwalk(@rule ~x::isvirtual => virtual_uncall(x.val))))(root)
end