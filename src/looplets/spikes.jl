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

(ctx::Stylize{LowerJulia})(node::Spike) = ctx.root.kind === chunk ? SpikeStyle() : DefaultStyle()
combine_style(a::DefaultStyle, b::SpikeStyle) = SpikeStyle()
combine_style(a::RunStyle, b::SpikeStyle) = SpikeStyle()
combine_style(a::ThunkStyle, b::SpikeStyle) = ThunkStyle()
combine_style(a::SimplifyStyle, b::SpikeStyle) = SimplifyStyle()
combine_style(a::AcceptRunStyle, b::SpikeStyle) = SpikeStyle()
combine_style(a::SpikeStyle, b::SpikeStyle) = SpikeStyle()

function (ctx::LowerJulia)(root::FinchNode, ::SpikeStyle)
    if root.kind === chunk
        body_ext = Extent(getstart(root.ext), call(-, getstop(root.ext), 1))
        root_body = Rewrite(Postwalk(
            @rule access(~a::isvirtual, ~i...) => access(get_spike_body(a.val, ctx, root.ext, body_ext), ~i...)
        ))(root.body)
        @assert isvirtual(root.ext)
        if measure(root.ext.val) == 1
            body_expr = quote end
        else
            #TODO check body nonempty
            body_expr = contain(ctx) do ctx_2
                (ctx_2)(chunk(
                    root.idx,
                    body_ext,
                    root_body,
                ))
            end
        end
        tail_ext = Extent(getstop(root.ext), getstop(root.ext))
        root_tail = Rewrite(Postwalk(
            @rule access(~a::isvirtual, ~i...) => access(get_spike_tail(a.val, ctx, root.ext, tail_ext), ~i...)
        ))(root.body)
        tail_expr = contain(ctx) do ctx_2
            (ctx_2)(chunk(
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

get_spike_body(node, ctx, ext, ext_2) = node
get_spike_body(node::Spike, ctx, ext, ext_2) = Run(node.body)
get_spike_body(node::Shift, ctx, ext, ext_2) = Shift(
    body = get_spike_body(node.body, ctx,
        shiftdim(ext, call(-, node.delta)),
        shiftdim(ext_2, call(-, node.delta))),
    delta = node.delta)

get_spike_tail(node, ctx, ext, ext_2) = node
get_spike_tail(node::Spike, ctx, ext, ext_2) = Run(node.tail)
get_spike_tail(node::Shift, ctx, ext, ext_2) = Shift(
    body = get_spike_tail(node.body, ctx,
        shiftdim(ext, call(-, node.delta)),
        shiftdim(ext_2, call(-, node.delta))),
    delta = node.delta)

supports_shift(::SpikeStyle) = true

function truncate(node::Spike, ctx, ext, ext_2)
    if query(call(==, getstop(ext), getstop(ext_2)), ctx)
        return node
    elseif query(call(>, getstop(ext), getstop(ext_2)), ctx)
        return Run(node.body)
    else
        return Switch([
            value(:($(ctx(getstop(ext_2))) < $(ctx(getstop(ext))))) => Run(node.body),
            literal(true) => node,
        ])
    end
end