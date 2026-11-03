truncate(ctx, node, ext, ext_2) = node

@kwdef struct Phase
    body
    start = (ctx, ext) -> nothing
    stop = (ctx, ext) -> nothing
    range =
        (ctx, ext) -> similar_extent(
            ext,
            something(start(ctx, ext), getstart(ext)),
            something(stop(ctx, ext), getstop(ext)),
        )
end
FinchNotation.finch_leaf(x::Phase) = virtual(x)

Base.show(io::IO, ex::Phase) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::Phase)
    print(io, "Phase()")
end

function phase_range(ctx, node::FinchNode, ext)
    if @capture node access(~tns::isvirtual, ~i...)
        phase_range(ctx, tns.val, ext)
    else
        return auto
    end
end

phase_range(ctx, node, ext) = auto
phase_range(ctx, node::Phase, ext) = node.range(ctx, ext)

function phase_body(ctx, node::FinchNode, ext, ext_2)
    if @capture node access(~tns::isvirtual, ~m, ~i...)
        access(phase_body(ctx, tns.val, ext, ext_2), m, i...)
    else
        return node
    end
end
phase_body(ctx, node::Phase, ext, ext_2) = node.body(ctx, ext_2)
phase_body(ctx, node, ext, ext_2) = truncate(ctx, node, ext, ext_2)

abstract type PhaseStyle end
struct SequencePhaseStyle <: PhaseStyle end
struct StepperPhaseStyle <: PhaseStyle end
struct JumperPhaseStyle <: PhaseStyle end

phase_op(::SequencePhaseStyle) = virtual_intersect
phase_op(::StepperPhaseStyle) = virtual_intersect
phase_op(::JumperPhaseStyle) = virtual_union

get_style(ctx, ::Phase, root) = root.kind === loop ? SequencePhaseStyle() : DefaultStyle()

combine_style(a::DefaultStyle, b::PhaseStyle) = b
combine_style(a::LookupStyle, b::PhaseStyle) = b
combine_style(a::T, b::T) where {T<:PhaseStyle} = b
combine_style(a::PhaseStyle, b::RunStyle) = a
combine_style(a::PhaseStyle, b::SpikeStyle) = a
combine_style(a::SimplifyStyle, b::PhaseStyle) = b
combine_style(a::AcceptRunStyle, b::PhaseStyle) = b
combine_style(a::SwitchStyle, b::PhaseStyle) = a
combine_style(a::ThunkStyle, b::PhaseStyle) = a

function lower(ctx::AbstractCompiler, root::FinchNode, style::PhaseStyle)
    if root.kind === loop
        i = getname(root.idx)
        i0 = freshen(ctx, i)

        body = root.body

        ext_2 = mapreduce(
            (node) -> phase_range(ctx, node, root.ext),
            (a, b) -> phase_op(style)(ctx, a, b),
            PostOrderDFS(body),
        )

        ext_3 = virtual_intersect(ctx, root.ext.val, ext_2)

        ext_4 = cache_dim!(ctx, :phase, ext_3)

        body = Rewrite(Postwalk(node -> phase_body(ctx, node, root.ext, ext_4)))(body)
        body = quote
            $i0 = $i
            $(
                contain(ctx) do ctx_4
                    (ctx_4)(loop(
                        root.idx,
                        ext_4,
                        body,
                    ))
                end
            )

            $i = $(ctx(getstop(ext_4))) + $(ctx(getunit(ext_4)))
        end

        if prove(ctx, call(>=, measure(ext_4), 0))
            return body
        else
            return quote
                if $(ctx(getstop(ext_4))) >= $(ctx(getstart(ext_4)))
                    $body
                end
            end
        end
    else
        error("unimplemented")
    end
end
