struct DepthCalculator
    rec
end

depth_calculator(root) = DepthCalculator(DepthCalculatorVisitor()(root))

(ctx::DepthCalculator)(node::FinchNode) = maximum(node -> get(ctx.rec, node, 0), PostOrderDFS(node))

@kwdef struct DepthCalculatorVisitor
    depth = 1
    rec = Dict()
end

function (ctx::DepthCalculatorVisitor)(node::FinchNode)
    if node.kind === loop
        ctx(node.ext)
        ctx.rec[node.idx] = ctx.depth
        ctx_2 = DepthCalculatorVisitor(depth=ctx.depth+1, rec=ctx.rec)
        ctx_2(node.body)
    elseif node.kind === define
        ctx.rec[node.lhs] = ctx.depth
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

freshen(ctx::ConcordizeVisitor, tags...) = freshen(ctx.ctx.code, tags...)

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

    if node.kind === loop || node.kind === assign || node.kind === define || node.kind === sieve
        node = Rewrite(Postwalk(Fixpoint(
            @rule access(~tns, ~mode, ~i..., ~j::isboundnotindex, ~k::All(isboundindex)...) => begin
                j_2 = index(freshen(ctx, :s))
                push!(selects, j_2 => j)
                push!(ctx.scope, j_2)
                access(tns, mode, i..., j_2, k...)
            end
        )))(node)
    end

    if node.kind === loop
        ctx_2 = ConcordizeVisitor(ctx.ctx, union(ctx.scope, [node.idx]))
        node = loop(node.idx, node.ext, ctx_2(node.body))
    elseif node.kind === define
        push!(ctx.scope, node.lhs)
    elseif node.kind === declare
        push!(ctx.scope, node.tns)
    elseif istree(node)
        node = similarterm(node, operation(node), map(ctx, children(node)))
    end

    for (select_idx, idx_ex) in reverse(selects)
        var = variable(freshen(ctx, :v))

        node = block(
            define(var, idx_ex),
            loop(select_idx, Extent(var, var), node)
        )
    end

    node
end

"""
    concordize(root, ctx)

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
function concordize(root, ctx::AbstractCompiler)
    depth = depth_calculator(root)
    root = Rewrite(Postwalk(Fixpoint(@rule access(~tns, ~mode, ~i..., ~j::isindex, ~k...) => begin
        if depth(j) < maximum(depth.(k), init=0)
            access(~tns, ~mode, ~i..., call(identity, j), ~k...)
        end
    end)))(root)
    ConcordizeVisitor(ctx, collect(keys(ctx.bindings)))(root)
end