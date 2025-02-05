@kwdef mutable struct DeclareDimensions
    ctx
    dims = Dict()
    hints = Dict()
end

"""
    dimensionalize!(prgm, ctx)

A program traversal which coordinates dimensions based on shared indices. In
particular, loops and declaration statements have dimensions. Accessing a tensor
with a raw index `hints` that the loop should have a dimension corresponding to
the tensor axis. Accessing a tensor on the left hand side with a raw index also
`hints` that the tensor declaration should have a dimension corresponding to the
loop axis.  All hints inside a loop body are
used to evaluate loop dimensions, and all hints after a declaration until the
first freeze are used to evaluate declaration dimensions.
One may refer to the automatically determined dimension using a
variable named `_` or `:`. Index sharing is transitive, so `A[i] = B[i]` and `B[j]
= C[j]` will induce a gathering of the dimensions of `A`, `B`, and `C` into one.

The dimensions are semantically evaluated just before the corresponding loop or
declaration statement.  The program is assumed to be scoped, so that all loops
have unique index names.

See also: [`virtual_size`](@ref), [`virtual_resize!`](@ref), [`combinedim`](@ref)
"""
function dimensionalize!(prgm, ctx)
    prgm = DeclareDimensions(; ctx=ctx)(prgm)
    return prgm
end

struct FinchCompileError
    msg
end

function (ctx::DeclareDimensions)(node::FinchNode)
    if node.kind === access
        @assert @capture node access(~tns, ~mode, ~idxs...)
        if node.mode.kind !== reader && haskey(ctx.hints, getroot(tns))
            shape = map(suggest, virtual_size(ctx.ctx, tns))
            push!(ctx.hints[getroot(tns)], node)
        else
            shape = virtual_size(ctx.ctx, tns)
        end
        length(idxs) > length(shape) && throw(
            DimensionMismatch(
                "more indices than dimensions in $(sprint(show, MIME("text/plain"), node))"
            ),
        )
        length(idxs) < length(shape) && throw(
            DimensionMismatch(
                "less indices than dimensions in $(sprint(show, MIME("text/plain"), node))"
            ),
        )
        idxs = map(zip(shape, idxs)) do (dim, idx)
            if isindex(idx)
                ctx.dims[idx] = resultdim(ctx.ctx, dim, get(ctx.dims, idx, auto))
                idx
            else
                ctx(idx) #Probably not strictly necessary to preserve the result of this, since this expr can't contain a statement and so won't be modified
            end
        end
        access(tns, mode, idxs...)
    elseif node.kind === loop
        if node.ext.kind !== virtual
            error("could not evaluate $(node.ext) into a dimension")
        end
        ctx.dims[node.idx] = node.ext.val
        body = ctx(node.body)
        ctx.dims[node.idx] != auto ||
            throw(FinchCompileError("could not resolve dimension of index $(node.idx)"))
        return loop(
            node.idx,
            cache_dim!(ctx.ctx, getname(node.idx), resolvedim(ctx.dims[node.idx])),
            body,
        )
    elseif node.kind === block
        block(map(ctx, node.bodies)...)
    elseif node.kind === declare
        ctx.hints[node.tns] = []
        node
    elseif node.kind === freeze
        if haskey(ctx.hints, node.tns)
            shape = virtual_size(ctx.ctx, node.tns)
            shape = map(suggest, shape)
            for hint in ctx.hints[node.tns]
                @assert @capture hint access(~tns, updater(~f), ~idxs...)
                shape = map(zip(shape, idxs)) do (dim, idx)
                    if isindex(idx)
                        resultdim(ctx.ctx, dim, ctx.dims[idx])
                    else
                        resultdim(ctx.ctx, dim, auto) #TODO I can't think of a case where this doesn't equal `dim`
                    end
                end
            end
            #TODO tns ignored here
            shape = map(resolvedim, shape)
            tns = virtual_resize!(ctx.ctx, node.tns, shape...)
            delete!(ctx.hints, node.tns)
        end
        node
    elseif istree(node)
        return similarterm(node, operation(node), map(ctx, arguments(node)))
    else
        return node
    end
end
