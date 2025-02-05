@kwdef struct Sequence
    phases
end

Base.show(io::IO, ex::Sequence) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::Sequence)
    print(io, "Sequence()")
end

FinchNotation.finch_leaf(x::Sequence) = virtual(x)

struct SequenceStyle end

get_style(ctx, ::Sequence, root) = root.kind === loop ? SequenceStyle() : DefaultStyle()
combine_style(a::DefaultStyle, b::SequenceStyle) = SequenceStyle()
combine_style(a::LookupStyle, b::SequenceStyle) = SequenceStyle()
combine_style(a::ThunkStyle, b::SequenceStyle) = ThunkStyle()
combine_style(a::RunStyle, b::SequenceStyle) = SequenceStyle()
combine_style(a::SimplifyStyle, b::SequenceStyle) = a
combine_style(a::AcceptRunStyle, b::SequenceStyle) = SequenceStyle()
combine_style(a::SequenceStyle, b::SequenceStyle) = SequenceStyle()
combine_style(a::SequenceStyle, b::SwitchStyle) = SwitchStyle()
combine_style(a::SpikeStyle, b::SequenceStyle) = SequenceStyle()
combine_style(a::SequenceStyle, b::PhaseStyle) = b

function lower(ctx::AbstractCompiler, root::FinchNode, ::SequenceStyle)
    if root.kind === loop
        phases = get_sequence_phases(ctx, root.body, root.ext)

        i = getname(root.idx)
        i0 = freshen(ctx, i, :_start)
        step = freshen(ctx, i, :_step)

        thunk = quote
            $i = $(ctx(getstart(root.ext)))
        end

        for (key, body) in phases
            push!(
                thunk.args,
                contain(ctx) do ctx_2
                    ctx_2(loop(root.idx, root.ext, body))
                end,
            )
        end

        return thunk
    else
        error("unimplemented")
    end
end

get_sequence_phases(ctx, node, ext) = [[] => node]
function get_sequence_phases(ctx, node::FinchNode, ext)
    if node.kind === virtual
        get_sequence_phases(ctx, node.val, ext)
    elseif istree(node)
        map(
            flatten((
                product(map(arg -> get_sequence_phases(ctx, arg, ext), arguments(node))...),
            )),
        ) do phases
            keys = map(first, phases)
            bodies = map(last, phases)
            return reduce(vcat, keys; init=[]) =>
                similarterm(node, operation(node), collect(bodies))
        end
    else
        [[] => node]
    end
end

function get_sequence_phases(ctx, node::Sequence, ext)
    new_phases = []

    prev_stop = call(-, getstart(ext), getunit(ext))
    for curr in node.phases
        curr_start = call(+, prev_stop, getunit(ext))
        curr_stop = getstop(phase_range(ctx, curr, ext))
        push!(
            new_phases,
            Phase(; body=curr.body, start=(ctx, ext_2) -> curr_start, stop=curr.stop),
        )
        prev_stop = curr_stop
    end

    return enumerate(new_phases)
end
