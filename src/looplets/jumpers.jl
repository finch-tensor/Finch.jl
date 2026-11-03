struct JumperStyle end

@kwdef struct Jumper
    preamble = nothing
    stop = (ctx, ext) -> nothing
    chunk = nothing
    next = (ctx, ext) -> nothing
    body = (ctx, ext) -> chunk
    seek = (ctx, start) -> error("seek not implemented error")
end

Base.show(io::IO, ex::Jumper) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::Jumper)
    print(io, "Jumper(...)")
end

FinchNotation.finch_leaf(x::Jumper) = virtual(x)

get_style(ctx, ::Jumper, root) = root.kind === loop ? JumperStyle() : DefaultStyle()

combine_style(a::DefaultStyle, b::JumperStyle) = JumperStyle()
combine_style(a::LookupStyle, b::JumperStyle) = JumperStyle()
combine_style(a::JumperStyle, b::JumperStyle) = JumperStyle()
combine_style(a::JumperStyle, b::RunStyle) = RunStyle()
combine_style(a::SimplifyStyle, b::JumperStyle) = a
combine_style(a::JumperStyle, b::AcceptRunStyle) = JumperStyle()
combine_style(a::JumperStyle, b::SpikeStyle) = SpikeStyle()
combine_style(a::JumperStyle, b::SwitchStyle) = SwitchStyle()
combine_style(a::JumperStyle, b::SequenceStyle) = SequenceStyle()
combine_style(a::ThunkStyle, b::JumperStyle) = ThunkStyle()
combine_style(a::JumperStyle, b::PhaseStyle) = b

jumper_seek(ctx, node::Jumper, ext) = node.seek(ctx, ext)
jumper_seek(ctx, node, ext) = quote end

jumper_range(ctx, node, ext) = auto

function jumper_range(ctx, node::FinchNode, ext)
    if @capture node access(~tns::isvirtual, ~i...)
        jumper_range(ctx, tns.val, ext)
    else
        return auto
    end
end

function jumper_range(ctx, node::Jumper, ext)
    push_preamble!(ctx, node.seek !== nothing ? node.seek(ctx, ext) : quote end)
    push_preamble!(ctx, node.preamble !== nothing ? node.preamble : quote end)
    ext_2 = similar_extent(ext, getstart(ext), node.stop(ctx, ext))
    bound_measure_below!(ext_2, get_smallest_measure(ext))
end

jumper_body(ctx, node, ext, ext_2) = truncate(ctx, node, ext, ext_2)

function jumper_body(ctx, node::FinchNode, ext, ext_2)
    if @capture node access(~tns::isvirtual, ~m, ~i...)
        access(jumper_body(ctx, tns.val, ext, ext_2), m, i...)
    else
        return node
    end
end

function jumper_body(ctx, node::Jumper, ext, ext_2)
    next = node.next(ctx, ext_2)
    if next !== nothing
        Switch([
            value(:($(ctx(node.stop(ctx, ext))) == $(ctx(getstop(ext_2))))) => Thunk(;
                body=(ctx) -> truncate(
                    ctx,
                    node.chunk,
                    ext,
                    similar_extent(ext, getstart(ext_2), getstop(ext)),
                ),
                epilogue=next,
            ),
            literal(true) => Stepper(;
                preamble=node.preamble,
                stop=node.stop,
                chunk=node.chunk,
                next=node.next,
                seek=node.seek,
            ),
        ])
    else
        node.body(ctx, ext_2)
    end
end

function lower(ctx::AbstractCompiler, root::FinchNode, style::JumperStyle)
    root.kind === loop || error("unimplemented")

    i = getname(root.idx)
    i0 = freshen(ctx, i, :_start)
    push_preamble!(
        ctx,
        quote
            $i = $(ctx(getstart(root.ext)))
        end,
    )

    guard = :($i <= $(ctx(getstop(root.ext))))

    #foreach(filter(isvirtual, collect(PostOrderDFS(root.body)))) do node
    #    push_preamble!(ctx, jumper_seek(ctx, node.val, root.ext))
    #end

    body_2 = contain(ctx) do ctx_2
        push_preamble!(ctx_2, :($i0 = $i))
        i1 = freshen(ctx_2, i)

        ext_1 = bound_measure_below!(
            similar_extent(root.ext, value(i0), getstop(root.ext)),
            get_smallest_measure(root.ext),
        )
        ext_2 = mapreduce(
            (node) -> jumper_range(ctx_2, node, ext_1),
            (a, b) -> virtual_union(ctx_2, a, b),
            PostOrderDFS(root.body),
        )
        ext_3 = virtual_intersect(ctx_2, ext_1, ext_2)
        ext_4 = cache_dim!(ctx_2, :phase, ext_3)

        body = Rewrite(Postwalk(node -> jumper_body(ctx_2, node, ext_1, ext_4)))(root.body)
        body = quote
            $i1 = $i
            $(
                contain(ctx_2) do ctx_3
                    ctx_3(loop(root.idx, ext_4, body))
                end
            )

            $i = $(ctx_2(getstop(ext_4))) + $(ctx_2(getunit(ext_4)))
        end

        if prove(ctx_2, call(>=, measure(ext_4), 0))
            body
        else
            quote
                if $(ctx_2(getstop(ext_4))) >= $(ctx_2(getstart(ext_4)))
                    $body
                end
            end
        end
    end

    @assert isvirtual(root.ext)

    if prove(ctx, call(==, measure(root.ext.val), get_smallest_measure(root.ext.val)))
        body_2
    else
        return quote
            while $guard
                $body_2
            end
        end
    end
end
