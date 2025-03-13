struct OffsetArray{Delta<:Tuple,Body} <: AbstractCombinator
    body::Body
    delta::Delta
end

Base.show(io::IO, ex::OffsetArray) = print(io, "OffsetArray($(ex.body), $(ex.delta)")

function transfer(device, tns::OffsetArray{Ti}) where {Ti}
    body_2 = transfer(device, tns.body)
    return OffsetArray{Ti}(body_2, tns.delta)
end

function labelled_show(io::IO, ::OffsetArray)
    print(io, "OffsetArray [$(join(map(d -> ":+$d", ex.delta), ", "))]")
end

labelled_children(ex::OffsetArray) = [LabelledTree(ex.body)]

struct VirtualOffsetArray <: AbstractVirtualCombinator
    body
    delta
end

function distribute(
    ctx::AbstractCompiler, tns::VirtualOffsetArray, arch, style
)
    VirtualOffsetArray(distribute(ctx, tns.body, arch, style), tns.delta)
end

function reroot_set!(ctx::AbstractCompiler, tns::VirtualOffsetArray, diff)
    reroot_set!(ctx, tns.body, diff)
end

function reroot_get(ctx::AbstractCompiler, tns::VirtualOffsetArray, diff)
    VirtualOffsetArray(
        reroot_get(ctx, tns.body, diff),
        tns.delta,
    )
end

is_injective(ctx, lvl::VirtualOffsetArray) = is_injective(ctx, lvl.body)
is_atomic(ctx, lvl::VirtualOffsetArray) = is_atomic(ctx, lvl.body)
is_concurrent(ctx, lvl::VirtualOffsetArray) = is_concurrent(ctx, lvl.body)

Base.show(io::IO, ex::VirtualOffsetArray) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::VirtualOffsetArray)
    print(io, "VirtualOffsetArray($(ex.body), $(ex.delta))")
end

function Base.summary(io::IO, mime::MIME"text/plain", ex::VirtualOffsetArray)
    print(io, "VOffset($(summary(ex.body)), $(ex.delta))")
end

FinchNotation.finch_leaf(x::VirtualOffsetArray) = virtual(x)

function virtualize(ctx, ex, ::Type{OffsetArray{Delta,Body}}) where {Delta,Body}
    delta = map(enumerate(Delta.parameters)) do (n, param)
        virtualize(ctx, :($ex.delta[$n]), param)
    end
    VirtualOffsetArray(virtualize(ctx, :($ex.body), Body), delta)
end

"""
    offset(tns, delta...)

Create an `OffsetArray` such that `offset(tns, delta...)[i...] == tns[i .+ delta...]`.
The dimensions declared by an OffsetArray are shifted, so that `size(offset(tns, delta...)) == size(tns) .+ delta`.
"""
offset(body, delta...) = OffsetArray(body, delta)
virtual_call(ctx, ::typeof(offset), body, delta...) = VirtualOffsetArray(body, delta)

function unwrap(ctx, arr::VirtualOffsetArray, var)
    call(offset, unwrap(ctx, arr.body, var), arr.delta...)
end

function lower(ctx::AbstractCompiler, tns::VirtualOffsetArray, ::DefaultStyle)
    :(OffsetArray($(ctx(tns.body)), $(ctx(tns.delta))))
end

function virtual_size(ctx::AbstractCompiler, arr::VirtualOffsetArray)
    map(zip(virtual_size(ctx, arr.body), arr.delta)) do (dim, delta)
        shiftdim(dim, call(-, delta))
    end
end

function virtual_resize!(ctx::AbstractCompiler, arr::VirtualOffsetArray, dims...)
    dims_2 = map(zip(dims, arr.delta)) do (dim, delta)
        shiftdim(dim, delta)
    end
    virtual_resize!(ctx, arr.body, dims_2...)
end

function virtual_fill_value(ctx::AbstractCompiler, arr::VirtualOffsetArray)
    virtual_fill_value(ctx, arr.body)
end

function instantiate(ctx, arr::VirtualOffsetArray, mode)
    VirtualOffsetArray(instantiate(ctx, arr.body, mode), arr.delta)
end

get_style(ctx, node::VirtualOffsetArray, root) = get_style(ctx, node.body, root)

function popdim(node::VirtualOffsetArray)
    if length(node.delta) == 1
        return node.body
    else
        return VirtualOffsetArray(node.body, node.delta[1:(end - 1)])
    end
end

function truncate(ctx, node::VirtualOffsetArray, ext, ext_2)
    VirtualOffsetArray(
        truncate(
            ctx, node.body, shiftdim(ext, node.delta[end]), shiftdim(ext_2, node.delta[end])
        ),
        node.delta,
    )
end

function get_point_body(ctx, node::VirtualOffsetArray, ext, idx)
    pass_nothing(
        get_point_body(
            ctx, node.body, shiftdim(ext, node.delta[end]), call(+, idx, node.delta[end])
        ),
    ) do body_2
        popdim(VirtualOffsetArray(body_2, node.delta))
    end
end

function unwrap_thunk(ctx, node::VirtualOffsetArray)
    VirtualOffsetArray(unwrap_thunk(ctx, node.body), node.delta)
end

function get_run_body(ctx, node::VirtualOffsetArray, ext)
    pass_nothing(get_run_body(ctx, node.body, shiftdim(ext, node.delta[end]))) do body_2
        popdim(VirtualOffsetArray(body_2, node.delta))
    end
end

function get_acceptrun_body(ctx, node::VirtualOffsetArray, ext)
    pass_nothing(
        get_acceptrun_body(ctx, node.body, shiftdim(ext, node.delta[end]))
    ) do body_2
        popdim(VirtualOffsetArray(body_2, node.delta))
    end
end

function get_sequence_phases(ctx, node::VirtualOffsetArray, ext)
    map(get_sequence_phases(ctx, node.body, shiftdim(ext, node.delta[end]))) do (keys, body)
        return keys => VirtualOffsetArray(body, node.delta)
    end
end

function phase_body(ctx, node::VirtualOffsetArray, ext, ext_2)
    VirtualOffsetArray(
        phase_body(
            ctx, node.body, shiftdim(ext, node.delta[end]), shiftdim(ext_2, node.delta[end])
        ),
        node.delta,
    )
end
function phase_range(ctx, node::VirtualOffsetArray, ext)
    shiftdim(
        phase_range(ctx, node.body, shiftdim(ext, node.delta[end])),
        call(-, node.delta[end]),
    )
end

function get_spike_body(ctx, node::VirtualOffsetArray, ext, ext_2)
    VirtualOffsetArray(
        get_spike_body(
            ctx, node.body, shiftdim(ext, node.delta[end]), shiftdim(ext_2, node.delta[end])
        ),
        node.delta,
    )
end
function get_spike_tail(ctx, node::VirtualOffsetArray, ext, ext_2)
    VirtualOffsetArray(
        get_spike_tail(
            ctx, node.body, shiftdim(ext, node.delta[end]), shiftdim(ext_2, node.delta[end])
        ),
        node.delta,
    )
end

visit_fill_leaf_leaf(node, tns::VirtualOffsetArray) =
    visit_fill_leaf_leaf(node, tns.body)
function visit_simplify(node::VirtualOffsetArray)
    VirtualOffsetArray(visit_simplify(node.body), node.delta)
end

function get_switch_cases(ctx, node::VirtualOffsetArray)
    map(get_switch_cases(ctx, node.body)) do (guard, body)
        guard => VirtualOffsetArray(body, node.delta)
    end
end

function stepper_range(ctx, node::VirtualOffsetArray, ext)
    shiftdim(
        stepper_range(ctx, node.body, shiftdim(ext, node.delta[end])),
        call(-, node.delta[end]),
    )
end
function stepper_body(ctx, node::VirtualOffsetArray, ext, ext_2)
    VirtualOffsetArray(
        stepper_body(
            ctx, node.body, shiftdim(ext, node.delta[end]), shiftdim(ext_2, node.delta[end])
        ),
        node.delta,
    )
end
function stepper_seek(ctx, node::VirtualOffsetArray, ext)
    stepper_seek(ctx, node.body, shiftdim(ext, node.delta[end]))
end

function jumper_range(ctx, node::VirtualOffsetArray, ext)
    shiftdim(
        jumper_range(ctx, node.body, shiftdim(ext, node.delta[end])),
        call(-, node.delta[end]),
    )
end
function jumper_body(ctx, node::VirtualOffsetArray, ext, ext_2)
    VirtualOffsetArray(
        jumper_body(
            ctx, node.body, shiftdim(ext, node.delta[end]), shiftdim(ext_2, node.delta[end])
        ),
        node.delta,
    )
end
function jumper_seek(ctx, node::VirtualOffsetArray, ext)
    jumper_seek(ctx, node.body, shiftdim(ext, node.delta[end]))
end

function short_circuit_cases(ctx, node::VirtualOffsetArray, op)
    map(short_circuit_cases(ctx, node.body, op)) do (guard, body)
        guard => VirtualOffsetArray(body, node.delta)
    end
end

getroot(tns::VirtualOffsetArray) = getroot(tns.body)

function unfurl(ctx, tns::VirtualOffsetArray, ext, mode, proto)
    VirtualOffsetArray(
        unfurl(ctx, tns.body, shiftdim(ext, tns.delta[end]), mode, proto), tns.delta
    )
end

function lower_access(ctx::AbstractCompiler, tns::VirtualOffsetArray, mode)
    lower_access(ctx, tns.body, mode)
end

function lower_assign(ctx::AbstractCompiler, tns::VirtualOffsetArray, mode, op, rhs)
    lower_assign(ctx, tns.body, mode, op, rhs)
end
