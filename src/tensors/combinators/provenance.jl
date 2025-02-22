@kwdef struct Provenance <: AbstractVirtualCombinator
    path
    body
    Provenance(path, body) = begin
        while body isa Provenance
            path = repath(path, body.path)
            body = body.body
        end
        new(path, body)
    end
end

struct Parent end
struct SubLevelOf
    parent
end
struct SubTensorOf
    parent
end

repath(parent, child::Parent) = parent
repath(parent, child::SubLevelOf) = SubLevelOf(repath(parent, child.parent))
repath(parent, child::SubTensorOf) = SubTensorOf(repath(parent, child.parent))

Base.show(io::IO, ex::Provenance) = Base.show(io, MIME"text/plain"(), ex)

function Base.show(io::IO, mime::MIME"text/plain", ex::Provenance)
    print(io, "Provenance(")
    print(io, ex.path)
    print(io, ", ")
    print(io, ex.body)
    print(io, ")")
end

function Base.summary(io::IO, ex::Provenance)
    print(io, "Provenance($(summary(ex.path)), $(summary(ex.body)))")
end

FinchNotation.finch_leaf(x::Provenance) = virtual(x)

function virtual_size(ctx, tns::Provenance)
    (error(); virtual_size(ctx, tns.path)[1:(end - tns.ndims)])
end
virtual_resize!(ctx, tns::Provenance, dims...) = virtual_resize!(ctx, tns.path, dims...) # TODO SHOULD NOT HAPPEN BREAKS LIFECYCLES
virtual_fill_value(ctx, tns::Provenance) = virtual_fill_value(ctx, tns.path)

function instantiate(ctx, tns::Provenance, mode)
    Provenance(tns.path, instantiate(ctx, tns.body, mode))
end

get_style(ctx, node::Provenance, root) = get_style(ctx, node.body, root)

function popdim(node::Provenance, ctx)
    #I think this is an equivalent form, but it doesn't pop the unfurled node
    #from scalars. I'm not sure if that's good or bad.
    #@assert node.ndims + 1 <= length(virtual_size(ctx, node.path))
    #TODO this should probably absorb the child provenance or something?
    return Provenance(node.path, node.body)
end

function truncate(ctx, node::Provenance, ext, ext_2)
    Provenance(node.path, truncate(ctx, node.body, ext, ext_2))
end

function get_point_body(ctx, node::Provenance, ext, idx)
    pass_nothing(get_point_body(ctx, node.body, ext, idx)) do body_2
        popdim(Provenance(node.path, body_2), ctx)
    end
end

function unwrap_thunk(ctx, node::Provenance)
    Provenance(node.path, unwrap_thunk(ctx, node.body))
end

function get_run_body(ctx, node::Provenance, ext)
    pass_nothing(get_run_body(ctx, node.body, ext)) do body_2
        popdim(Provenance(node.path, body_2), ctx)
    end
end

function get_acceptrun_body(ctx, node::Provenance, ext)
    pass_nothing(get_acceptrun_body(ctx, node.body, ext)) do body_2
        popdim(Provenance(node.path, body_2), ctx)
    end
end

function get_sequence_phases(ctx, node::Provenance, ext)
    map(get_sequence_phases(ctx, node.body, ext)) do (keys, body)
        return keys => Provenance(node.path, body)
    end
end

function phase_body(ctx, node::Provenance, ext, ext_2)
    Provenance(node.path, phase_body(ctx, node.body, ext, ext_2))
end

phase_range(ctx, node::Provenance, ext) = phase_range(ctx, node.body, ext)

function get_spike_body(ctx, node::Provenance, ext, ext_2)
    Provenance(node.path, get_spike_body(ctx, node.body, ext, ext_2))
end

function get_spike_tail(ctx, node::Provenance, ext, ext_2)
    Provenance(node.path, get_spike_tail(ctx, node.body, ext, ext_2))
end

visit_fill_leaf_leaf(node, tns::Provenance) = visit_fill_leaf_leaf(node, tns.body)

visit_simplify(node::Provenance) = Provenance(node.path, visit_simplify(node.body))

function get_switch_cases(ctx, node::Provenance)
    map(get_switch_cases(ctx, node.body)) do (guard, body)
        guard => Provenance(node.path, body)
    end
end

function unfurl(ctx, tns::Provenance, ext, mode, proto)
    Provenance(tns.path, unfurl(ctx, tns.body, ext, mode, proto))
end

stepper_range(ctx, node::Provenance, ext) = stepper_range(ctx, node.body, ext)
function stepper_body(ctx, node::Provenance, ext, ext_2)
    Provenance(node.path, stepper_body(ctx, node.body, ext, ext_2))
end
stepper_seek(ctx, node::Provenance, ext) = stepper_seek(ctx, node.body, ext)

jumper_range(ctx, node::Provenance, ext) = jumper_range(ctx, node.body, ext)
function jumper_body(ctx, node::Provenance, ext, ext_2)
    Provenance(node.path, jumper_body(ctx, node.body, ext, ext_2))
end
jumper_seek(ctx, node::Provenance, ext) = jumper_seek(ctx, node.body, ext)

function short_circuit_cases(ctx, tns::Provenance, op)
    map(short_circuit_cases(ctx, tns.body, op)) do (guard, body)
        guard => Provenance(tns.path, body)
    end
end

lower(ctx::AbstractCompiler, node::Provenance, ::DefaultStyle) = ctx(node.body)

getroot(tns::Provenance) = getroot(tns.path)

is_injective(ctx, lvl::Provenance) = is_injective(ctx, lvl.path)
is_atomic(ctx, lvl::Provenance) = is_atomic(ctx, lvl.path)
is_concurrent(ctx, lvl::Provenance) = is_concurrent(ctx, lvl.path)

function lower_access(ctx::AbstractCompiler, tns::Provenance, mode)
    lower_access(ctx, tns.body, mode)
end

function lower_assign(ctx::AbstractCompiler, tns::Provenance, mode, op, rhs)
    lower_assign(ctx, tns.body, mode, op, rhs)
end
