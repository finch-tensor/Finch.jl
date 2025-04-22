struct DepthCalculator
    rec
end

depth_calculator(root) = DepthCalculator(DepthCalculatorVisitor()(root))

function (ctx::DepthCalculator)(node::FinchNode)
    maximum(node -> get(ctx.rec, node, 0), PostOrderDFS(node))
end

@kwdef struct DepthCalculatorVisitor
    depth = 1
    rec = Dict()
end

function (ctx::DepthCalculatorVisitor)(node::FinchNode)
    if node.kind === loop
        ctx(node.ext)
        ctx.rec[node.idx] = ctx.depth
        ctx_2 = DepthCalculatorVisitor(; depth=(ctx.depth + 1), rec=ctx.rec)
        ctx_2(node.body)
    elseif node.kind === define
        ctx.rec[node.lhs] = ctx.depth
        ctx(node.body)
    elseif istree(node)
        for child in children(node)
            ctx(child)
        end
    end
    return ctx.rec
end

struct ConcordizeVisitor
    ctx
    scope
end

freshen(ctx::ConcordizeVisitor, tags...) = freshen(ctx.ctx, tags...)

function (ctx::ConcordizeVisitor)(node::FinchNode)
    function isbound(x::FinchNode)
        if isconstant(x) || x in ctx.scope
            return true
        elseif @capture x access(~tns, ~mode, ~idxs...)
            return getroot(tns) in ctx.scope && all(isbound, idxs)
        elseif istree(x)
            return all(isbound, arguments(x))
        else
            return false
        end
    end
    isboundindex(x) = isindex(x) && isbound(x)
    isboundnotindex(x) = !isindex(x) && isbound(x)

    selects = []

    if node.kind === loop || node.kind === assign || node.kind === define ||
        node.kind === sieve
        node = Rewrite(
            Postwalk(
                Fixpoint(
                    @rule access(
                        ~tns, ~mode, ~i..., ~j::isboundnotindex, ~k::All(isboundindex)...
                    ) => begin
                        j_2 = index(freshen(ctx, :s))
                        push!(selects, j_2 => j)
                        push!(ctx.scope, j_2)
                        access(tns, mode, i..., j_2, k...)
                    end
                ),
            ),
        )(
            node
        )
    end

    if node.kind === loop
        ctx_2 = ConcordizeVisitor(ctx.ctx, union(ctx.scope, [node.idx]))
        node = loop(node.idx, node.ext, ctx_2(node.body))
    elseif node.kind === define
        ctx_2 = ConcordizeVisitor(ctx.ctx, union(ctx.scope, [node.lhs]))
        node = define(node.lhs, node.rhs, ctx_2(node.body))
    elseif node.kind === declare
        push!(ctx.scope, node.tns)
    elseif istree(node)
        node = similarterm(node, operation(node), map(ctx, children(node)))
    end

    for (select_idx, idx_ex) in reverse(selects)
        var = variable(freshen(ctx, :v))

        node = define(var, idx_ex, loop(select_idx, VirtualExtent(var, var), node))
    end

    node
end

"""
    concordize(ctx, root)

A raw index is an index expression consisting of a single index node (i.e.
`A[i]` as opposed to `A[i + 1]`). A Finch program is concordant when all indices
are raw and column major with respect to the program loop ordering.  The
`concordize` transformation ensures that tensor indices are concordant by
inserting loops and lifting index expressions or transposed indices into the
loop bounds.

For example,

```
@finch for i = :
    b[] += A[f(i)]
end
```
becomes
```
@finch for i = :
    t = f(i)
    for s = t:t
        b[] += A[s]
    end
end
```

and

```
@finch for i = :, j = :
    b[] += A[i, j]
end
```
becomes
```
@finch for i = :, j = :, s = i:i
    b[] += A[s, j]
end
```
"""
function concordize(ctx::AbstractCompiler, root)
    depth = depth_calculator(root)
    if issafe(get_mode_flag(ctx))
        for node in PostOrderDFS(root)
            if @capture node access(~tns, ~mode, ~i...)
                for n in 1:length(i)
                    if 1 <= depth(i[n]) < maximum(depth.(i[(n + 1):end]); init=0)
                        push_preamble!(
                            ctx,
                            quote
                                @warn(
                                    "Performance Warning: non-concordant traversal of $($(sprint(Finch.FinchNotation.display_expression, MIME"text/plain"(), node))) (hint: most arrays prefer column major or first index fast, run in fast mode to ignore this warning)"
                                )
                            end,
                        )
                    end
                end
            end
        end
    end
    root = Rewrite(
        Postwalk(
            Fixpoint(
                @rule access(~tns, ~mode, ~i..., ~j::isindex, ~k...) => begin
                    if depth(j) <= maximum(depth.(k); init=0)
                        access(~tns, ~mode, ~i..., call(identity, j), ~k...)
                    end
                end
            ),
        ),
    )(
        root
    )
    tnss = unique([
        getroot(node.tns) for node in PostOrderDFS(root) if node.kind === access
    ])
    ConcordizeVisitor(ctx, tnss)(root)
end
