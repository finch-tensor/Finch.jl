@kwdef struct Unfurled <: AbstractVirtualCombinator
    arr
    ndims = 0
    body
    Unfurled(arr, ndims, body) = begin
        body = body isa Unfurled ? body.body : body
        ndims = arr isa Unfurled ? arr.ndims : ndims
        arr = arr isa Unfurled ? arr.arr : arr
        new(arr, ndims, body)
    end
    Unfurled(arr, body) = Unfurled(arr, 0, body)
end

function virtual_transfer(
    ctx::AbstractCompiler, tns::Unfurled, arch, style
)
    Unfurled(tns.arr, tns.ndims, virtual_transfer(ctx, tns.body, arch, style))
end

function reroot_set!(ctx::AbstractCompiler, tns::Unfurled, diff)
    reroot_set!(ctx, tns.body, diff)
end

function reroot_get(ctx::AbstractCompiler, tns::Unfurled, diff)
    Unfurled(
        tns.arr,
        tns.ndims,
        reroot_get(ctx, tns.body, diff),
    )
end

Base.show(io::IO, ex::Unfurled) = Base.show(io, MIME"text/plain"(), ex)

function Base.show(io::IO, mime::MIME"text/plain", ex::Unfurled)
    print(io, "Unfurled(")
    print(io, ex.arr)
    print(io, ", ")
    print(io, ex.ndims)
    print(io, ", ")
    print(io, ex.body)
    print(io, ")")
end

function Base.summary(io::IO, ex::Unfurled)
    print(io, "Unfurled($(summary(ex.arr)), $(ex.ndims), $(summary(ex.body)))")
end

FinchNotation.finch_leaf(x::Unfurled) = virtual(x)

virtual_size(ctx, tns::Unfurled) = virtual_size(ctx, tns.arr)[1:(end - tns.ndims)]
virtual_resize!(ctx, tns::Unfurled, dims...) = virtual_resize!(ctx, tns.arr, dims...) # TODO SHOULD NOT HAPPEN BREAKS LIFECYCLES
virtual_fill_value(ctx, tns::Unfurled) = virtual_fill_value(ctx, tns.arr)

function instantiate(ctx, tns::Unfurled, mode)
    Unfurled(tns.arr, tns.ndims, instantiate(ctx, tns.body, mode))
end

get_style(ctx, node::Unfurled, root) = get_style(ctx, node.body, root)

function popdim(node::Unfurled, ctx)
    #I think this is an equivalent form, but it doesn't pop the unfurled node
    #from scalars. I'm not sure if that's good or bad.
    @assert node.ndims + 1 <= length(virtual_size(ctx, node.arr))
    return Unfurled(node.arr, node.ndims + 1, node.body)
end

function truncate(ctx, node::Unfurled, ext, ext_2)
    Unfurled(node.arr, node.ndims, truncate(ctx, node.body, ext, ext_2))
end

function get_point_body(ctx, node::Unfurled, ext, idx)
    pass_nothing(get_point_body(ctx, node.body, ext, idx)) do body_2
        popdim(Unfurled(node.arr, node.ndims, body_2), ctx)
    end
end

function unwrap_thunk(ctx, node::Unfurled)
    Unfurled(node.arr, node.ndims, unwrap_thunk(ctx, node.body))
end

function get_run_body(ctx, node::Unfurled, ext)
    pass_nothing(get_run_body(ctx, node.body, ext)) do body_2
        popdim(Unfurled(node.arr, node.ndims, body_2), ctx)
    end
end

function get_acceptrun_body(ctx, node::Unfurled, ext)
    pass_nothing(get_acceptrun_body(ctx, node.body, ext)) do body_2
        popdim(Unfurled(node.arr, node.ndims, body_2), ctx)
    end
end

function get_sequence_phases(ctx, node::Unfurled, ext)
    map(get_sequence_phases(ctx, node.body, ext)) do (keys, body)
        return keys => Unfurled(node.arr, node.ndims, body)
    end
end

function phase_body(ctx, node::Unfurled, ext, ext_2)
    Unfurled(node.arr, node.ndims, phase_body(ctx, node.body, ext, ext_2))
end

phase_range(ctx, node::Unfurled, ext) = phase_range(ctx, node.body, ext)

function get_spike_body(ctx, node::Unfurled, ext, ext_2)
    Unfurled(node.arr, node.ndims, get_spike_body(ctx, node.body, ext, ext_2))
end

function get_spike_tail(ctx, node::Unfurled, ext, ext_2)
    Unfurled(node.arr, node.ndims, get_spike_tail(ctx, node.body, ext, ext_2))
end

visit_fill_leaf_leaf(node, tns::Unfurled) = visit_fill_leaf_leaf(node, tns.body)

visit_simplify(node::Unfurled) = Unfurled(node.arr, node.ndims, visit_simplify(node.body))

function get_switch_cases(ctx, node::Unfurled)
    map(get_switch_cases(ctx, node.body)) do (guard, body)
        guard => Unfurled(node.arr, node.ndims, body)
    end
end

function unfurl(ctx, tns::Unfurled, ext, mode, proto)
    Unfurled(tns.arr, tns.ndims, unfurl(ctx, tns.body, ext, mode, proto))
end

stepper_range(ctx, node::Unfurled, ext) = stepper_range(ctx, node.body, ext)
function stepper_body(ctx, node::Unfurled, ext, ext_2)
    Unfurled(node.arr, node.ndims, stepper_body(ctx, node.body, ext, ext_2))
end
stepper_seek(ctx, node::Unfurled, ext) = stepper_seek(ctx, node.body, ext)

jumper_range(ctx, node::Unfurled, ext) = jumper_range(ctx, node.body, ext)
function jumper_body(ctx, node::Unfurled, ext, ext_2)
    Unfurled(node.arr, node.ndims, jumper_body(ctx, node.body, ext, ext_2))
end
jumper_seek(ctx, node::Unfurled, ext) = jumper_seek(ctx, node.body, ext)

function short_circuit_cases(ctx, tns::Unfurled, op)
    map(short_circuit_cases(ctx, tns.body, op)) do (guard, body)
        guard => Unfurled(tns.arr, tns.ndims, body)
    end
end

lower(ctx::AbstractCompiler, node::Unfurled, ::DefaultStyle) = ctx(node.body)

getroot(tns::Unfurled) = getroot(tns.arr)

is_injective(ctx, lvl::Unfurled) = is_injective(ctx, lvl.arr)
is_atomic(ctx, lvl::Unfurled) = is_atomic(ctx, lvl.arr)
is_concurrent(ctx, lvl::Unfurled) = is_concurrent(ctx, lvl.arr)

function lower_access(ctx::AbstractCompiler, tns::Unfurled, mode)
    lower_access(ctx, tns.body, mode)
end

function lower_assign(ctx::AbstractCompiler, tns::Unfurled, mode, op, rhs)
    lower_assign(ctx, tns.body, mode, op, rhs)
end
