struct ThunkStyle end

@kwdef struct Thunk
    preamble = quote end
    body
    epilogue = quote end
end
FinchNotation.finch_leaf(x::Thunk) = virtual(x)

Base.show(io::IO, ex::Thunk) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::Thunk)
    print(io, "Thunk()")
end

get_style(ctx, ::Thunk, root) = ThunkStyle()
combine_style(a::DefaultStyle, b::ThunkStyle) = ThunkStyle()
combine_style(a::ThunkStyle, b::ThunkStyle) = ThunkStyle()
combine_style(a::ThunkStyle, b::SimplifyStyle) = ThunkStyle()

function lower(ctx::AbstractCompiler, node, ::ThunkStyle)
    contain(ctx) do ctx_2
        node_2 = unwrap_thunk(ctx_2, node)
        contain(ctx_2) do ctx_3
            (ctx_3)(node_2)
        end
    end
end

function unwrap_thunk(ctx, node::FinchNode)
    if node.kind === virtual
        unwrap_thunk(ctx, node.val)
    elseif istree(node)
        similarterm(
            node, operation(node), map(arg -> unwrap_thunk(ctx, arg), arguments(node))
        )
    else
        node
    end
end

unwrap_thunk(ctx, node) = node

function unwrap_thunk(ctx, node::Thunk)
    push_preamble!(ctx, node.preamble)
    push_epilogue!(ctx, node.epilogue)
    node.body(ctx)
end
