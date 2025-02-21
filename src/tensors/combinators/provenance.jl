@kwdef struct Provenance <: AbstractVirtualCombinator
    arr
    ndims = 0
    body
    Provenance(arr, ndims, body) = begin
        body = body isa Provenance ? body.body : body
        ndims = arr isa Provenance ? arr.ndims : ndims
        arr = arr isa Provenance ? arr.arr : arr
        new(arr, ndims, body)
    end
    Provenance(arr, body) = Provenance(arr, 0, body)
end

Base.show(io::IO, ex::Provenance) = Base.show(io, MIME"text/plain"(), ex)

function Base.show(io::IO, mime::MIME"text/plain", ex::Provenance)
    print(io, "Provenance(")
    print(io, ex.arr)
    print(io, ", ")
    print(io, ex.ndims)
    print(io, ", ")
    print(io, ex.body)
    print(io, ")")
end

function Base.summary(io::IO, ex::Provenance)
    print(io, "Provenance($(summary(ex.arr)), $(ex.ndims), $(summary(ex.body)))")
end

FinchNotation.finch_leaf(x::Provenance) = virtual(x)

virtual_size(ctx, tns::Provenance) = (error();virtual_size(ctx, tns.arr)[1:(end - tns.ndims)])
virtual_resize!(ctx, tns::Provenance, dims...) = virtual_resize!(ctx, tns.arr, dims...) # TODO SHOULD NOT HAPPEN BREAKS LIFECYCLES
virtual_fill_value(ctx, tns::Provenance) = virtual_fill_value(ctx, tns.arr)

function instantiate(ctx, tns::Provenance, mode)
    Provenance(tns.arr, tns.ndims, instantiate(ctx, tns.body, mode))
end

get_style(ctx, node::Provenance, root) = get_style(ctx, node.body, root)

function popdim(node::Provenance, ctx)
    #I think this is an equivalent form, but it doesn't pop the unfurled node
    #from scalars. I'm not sure if that's good or bad.
    #@assert node.ndims + 1 <= length(virtual_size(ctx, node.arr))
    return Provenance(node.arr, node.ndims + 1, node.body)
end

function truncate(ctx, node::Provenance, ext, ext_2)
    Provenance(node.arr, node.ndims, truncate(ctx, node.body, ext, ext_2))
end

function get_point_body(ctx, node::Provenance, ext, idx)
    pass_nothing(get_point_body(ctx, node.body, ext, idx)) do body_2
        popdim(Provenance(node.arr, node.ndims, body_2), ctx)
    end
end

function unwrap_thunk(ctx, node::Provenance)
    Provenance(node.arr, node.ndims, unwrap_thunk(ctx, node.body))
end

function get_run_body(ctx, node::Provenance, ext)
    pass_nothing(get_run_body(ctx, node.body, ext)) do body_2
        popdim(Provenance(node.arr, node.ndims, body_2), ctx)
    end
end

function get_acceptrun_body(ctx, node::Provenance, ext)
    pass_nothing(get_acceptrun_body(ctx, node.body, ext)) do body_2
        popdim(Provenance(node.arr, node.ndims, body_2), ctx)
    end
end

function get_sequence_phases(ctx, node::Provenance, ext)
    map(get_sequence_phases(ctx, node.body, ext)) do (keys, body)
        return keys => Provenance(node.arr, node.ndims, body)
    end
end

function phase_body(ctx, node::Provenance, ext, ext_2)
    Provenance(node.arr, node.ndims, phase_body(ctx, node.body, ext, ext_2))
end

phase_range(ctx, node::Provenance, ext) = phase_range(ctx, node.body, ext)

function get_spike_body(ctx, node::Provenance, ext, ext_2)
    Provenance(node.arr, node.ndims, get_spike_body(ctx, node.body, ext, ext_2))
end

function get_spike_tail(ctx, node::Provenance, ext, ext_2)
    Provenance(node.arr, node.ndims, get_spike_tail(ctx, node.body, ext, ext_2))
end

visit_fill_leaf_leaf(node, tns::Provenance) = visit_fill_leaf_leaf(node, tns.body)

visit_simplify(node::Provenance) = Provenance(node.arr, node.ndims, visit_simplify(node.body))

function get_switch_cases(ctx, node::Provenance)
    map(get_switch_cases(ctx, node.body)) do (guard, body)
        guard => Provenance(node.arr, node.ndims, body)
    end
end

function unfurl(ctx, tns::Provenance, ext, mode, proto)
    Provenance(tns.arr, tns.ndims, unfurl(ctx, tns.body, ext, mode, proto))
end

stepper_range(ctx, node::Provenance, ext) = stepper_range(ctx, node.body, ext)
function stepper_body(ctx, node::Provenance, ext, ext_2)
    Provenance(node.arr, node.ndims, stepper_body(ctx, node.body, ext, ext_2))
end
stepper_seek(ctx, node::Provenance, ext) = stepper_seek(ctx, node.body, ext)

jumper_range(ctx, node::Provenance, ext) = jumper_range(ctx, node.body, ext)
function jumper_body(ctx, node::Provenance, ext, ext_2)
    Provenance(node.arr, node.ndims, jumper_body(ctx, node.body, ext, ext_2))
end
jumper_seek(ctx, node::Provenance, ext) = jumper_seek(ctx, node.body, ext)

function short_circuit_cases(ctx, tns::Provenance, op)
    map(short_circuit_cases(ctx, tns.body, op)) do (guard, body)
        guard => Provenance(tns.arr, tns.ndims, body)
    end
end

lower(ctx::AbstractCompiler, node::Provenance, ::DefaultStyle) = ctx(node.body)

getroot(tns::Provenance) = getroot(tns.arr)

is_injective(ctx, lvl::Provenance) = is_injective(ctx, lvl.arr)
is_atomic(ctx, lvl::Provenance) = is_atomic(ctx, lvl.arr)
is_concurrent(ctx, lvl::Provenance) = is_concurrent(ctx, lvl.arr)

function lower_access(ctx::AbstractCompiler, tns::Provenance, mode)
    lower_access(ctx, tns.body, mode)
end

function lower_assign(ctx::AbstractCompiler, tns::Provenance, mode, op, rhs)
    lower_assign(ctx, tns.body, mode, op, rhs)
end
