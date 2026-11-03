@kwdef struct Spike
    body
    tail
end

Base.show(io::IO, ex::Spike) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::Spike)
    print(io, "Spike(body = ")
    print(io, ex.body)
    print(io, ")")
end

FinchNotation.finch_leaf(x::Spike) = virtual(x)

struct SpikeStyle end

get_style(ctx, ::Spike, root) = root.kind === loop ? SpikeStyle() : DefaultStyle()
combine_style(a::DefaultStyle, b::SpikeStyle) = SpikeStyle()
combine_style(a::LookupStyle, b::SpikeStyle) = SpikeStyle()
combine_style(a::RunStyle, b::SpikeStyle) = SpikeStyle()
combine_style(a::ThunkStyle, b::SpikeStyle) = ThunkStyle()
combine_style(a::SimplifyStyle, b::SpikeStyle) = a
combine_style(a::AcceptRunStyle, b::SpikeStyle) = SpikeStyle()
combine_style(a::SpikeStyle, b::SpikeStyle) = SpikeStyle()

function lower(ctx::AbstractCompiler, root::FinchNode, ::SpikeStyle)
    if root.kind === loop
        body_ext = similar_extent(
            root.ext, getstart(root.ext), call(-, getstop(root.ext), getunit(root.ext))
        )
        root_body = Rewrite(
            Postwalk(
                @rule access(~a::isvirtual, ~i...) =>
                    access(get_spike_body(ctx, a.val, root.ext, body_ext), ~i...)
            ),
        )(
            root.body
        )
        @assert isvirtual(root.ext)
        if prove(ctx, call(<=, measure(body_ext), 0))
            body_expr = quote end
        else
            #TODO check body nonempty
            body_expr = contain(ctx) do ctx_2
                (ctx_2)(loop(
                    root.idx,
                    body_ext,
                    root_body,
                ))
            end
        end

        tail_ext = similar_extent(root.ext, getstop(root.ext), getstop(root.ext))
        root_tail = Rewrite(
            Postwalk(
                @rule access(~a::isvirtual, ~i...) =>
                    access(get_spike_tail(ctx, a.val, root.ext, tail_ext), ~i...)
            ),
        )(
            root.body
        )
        tail_expr = contain(ctx) do ctx_2
            (ctx_2)(loop(
                root.idx,
                tail_ext,
                root_tail,
            ))
        end
        return Expr(:block, body_expr, tail_expr)
    else
        error("unimplemented")
    end
end

get_spike_body(ctx, node, ext, ext_2) = node
get_spike_body(ctx, node::Spike, ext, ext_2) = Run(node.body)

get_spike_tail(ctx, node, ext, ext_2) = node
get_spike_tail(ctx, node::Spike, ext, ext_2) = Run(node.tail)

function truncate(ctx, node::Spike, ext, ext_2)
    if prove(ctx, call(>=, call(-, getstop(ext), getunit(ext)), getstop(ext_2)))
        Run(node.body)
    elseif prove(ctx, call(==, getstop(ext), getstop(ext_2)))
        node
    else
        return Switch([
            value(:($(ctx(getstop(ext_2))) < $(ctx(getstop(ext))))) => Run(node.body),
            literal(true) => node,
        ])
    end
end
