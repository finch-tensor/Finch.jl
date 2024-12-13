struct PermissiveArray{dims, Body} <: AbstractCombinator
    body::Body
end

PermissiveArray(body, dims) = PermissiveArray{dims}(body)
PermissiveArray{dims}(body::Body) where {dims, Body} = PermissiveArray{dims, Body}(body)

Base.show(io::IO, ex::PermissiveArray{dims}) where {dims} = print(io, "PermissiveArray($(ex.body), $dims)")

labelled_show(io::IO, ::PermissiveArray{dims}) where {dims} =
    print(io, "PermissiveArray [$(join(map(d -> d ? "~:" : ":", dims), ", "))]")

labelled_children(ex::PermissiveArray) = [LabelledTree(ex.body)]

struct VirtualPermissiveArray <: AbstractVirtualCombinator
    body
    dims
end

is_injective(ctx, lvl::VirtualPermissiveArray) = is_injective(ctx, lvl.body)
is_atomic(ctx, lvl::VirtualPermissiveArray) = is_atomic(ctx, lvl.body)
is_concurrent(ctx, lvl::VirtualPermissiveArray) = is_concurrent(ctx, lvl.body)


Base.show(io::IO, ex::VirtualPermissiveArray) = Base.show(io, MIME"text/plain"(), ex)
Base.show(io::IO, mime::MIME"text/plain", ex::VirtualPermissiveArray) =
	print(io, "VirtualPermissiveArray($(ex.body), $(ex.dims))")

Base.summary(io::IO, ex::VirtualPermissiveArray) = print(io, "VPermissive($(summary(ex.body)), $(ex.dims))")

FinchNotation.finch_leaf(x::VirtualPermissiveArray) = virtual(x)

virtualize(ctx, ex, ::Type{PermissiveArray{dims, Body}}) where {dims, Body} =
    VirtualPermissiveArray(virtualize(ctx, :($ex.body), Body), dims)

"""
    permissive(tns, dims...)

Create an `PermissiveArray` where `permissive(tns, dims...)[i...]` is `missing`
if `i[n]` is not in the bounds of `tns` when `dims[n]` is `true`.  This wrapper
allows all permissive dimensions to be exempt from dimension checks, and is
useful when we need to access an array out of bounds, or for padding.
More formally,
```
    permissive(tns, dims...)[i...] =
        if any(n -> dims[n] && !(i[n] in axes(tns)[n]))
            missing
        else
            tns[i...]
        end
```
"""
permissive(body, dims...) = PermissiveArray(body, dims)
function virtual_call(ctx, ::typeof(permissive), body, dims...)
    @assert All(isliteral)(dims)
    VirtualPermissiveArray(body, map(dim -> dim.val, dims))
end

unwrap(ctx, arr::VirtualPermissiveArray, var) = call(permissive, unwrap(ctx, arr.body, var), arr.dims...)

lower(ctx::AbstractCompiler, tns::VirtualPermissiveArray, ::DefaultStyle) = :(PermissiveArray($(ctx(tns.body)), $(tns.dims)))

virtual_size(ctx::AbstractCompiler, arr::VirtualPermissiveArray) =
    ifelse.(arr.dims, (dimless,), virtual_size(ctx, arr.body))

virtual_resize!(ctx::AbstractCompiler, arr::VirtualPermissiveArray, dims...) =
    virtual_resize!(ctx, arr.body, ifelse.(arr.dims, virtual_size(ctx, arr.body), dim))

virtual_fill_value(ctx::AbstractCompiler, arr::VirtualPermissiveArray) = virtual_fill_value(ctx, arr.body)

instantiate(ctx, arr::VirtualPermissiveArray, mode) =
    VirtualPermissiveArray(instantiate(ctx, arr.body, mode), arr.dims)

get_style(ctx, node::VirtualPermissiveArray, root) = get_style(ctx, node.body, root)

function popdim(node::VirtualPermissiveArray)
    if length(node.dims) == 1
        return node.body
    else
        return VirtualPermissiveArray(node.body, node.dims[1:end-1])
    end
end

truncate(ctx, node::VirtualPermissiveArray, ext, ext_2) = VirtualPermissiveArray(truncate(ctx, node.body, ext, ext_2), node.dims)

get_point_body(ctx, node::VirtualPermissiveArray, ext, idx) =
    pass_nothing(get_point_body(ctx, node.body, ext, idx)) do body_2
        popdim(VirtualPermissiveArray(body_2, node.dims))
    end

unwrap_thunk(ctx, node::VirtualPermissiveArray) = VirtualPermissiveArray(unwrap_thunk(ctx, node.body), node.dims)

get_run_body(ctx, node::VirtualPermissiveArray, ext) =
    pass_nothing(get_run_body(ctx, node.body, ext)) do body_2
        popdim(VirtualPermissiveArray(body_2, node.dims))
    end

get_acceptrun_body(ctx, node::VirtualPermissiveArray, ext) =
    pass_nothing(get_acceptrun_body(ctx, node.body, ext)) do body_2
        popdim(VirtualPermissiveArray(body_2, node.dims))
    end

get_sequence_phases(ctx, node::VirtualPermissiveArray, ext) =
    map(get_sequence_phases(ctx, node.body, ext)) do (keys, body)
        return keys => VirtualPermissiveArray(body, node.dims)
    end

phase_body(ctx, node::VirtualPermissiveArray, ext, ext_2) = VirtualPermissiveArray(phase_body(ctx, node.body, ext, ext_2), node.dims)
phase_range(ctx, node::VirtualPermissiveArray, ext) = phase_range(ctx, node.body, ext)

get_spike_body(ctx, node::VirtualPermissiveArray, ext, ext_2) = VirtualPermissiveArray(get_spike_body(ctx, node.body, ext, ext_2), node.dims)
get_spike_tail(ctx, node::VirtualPermissiveArray, ext, ext_2) = VirtualPermissiveArray(get_spike_tail(ctx, node.body, ext, ext_2), node.dims)

visit_fill_leaf_leaf(node, tns::VirtualPermissiveArray) = visit_fill_leaf_leaf(node, tns.body)
visit_simplify(node::VirtualPermissiveArray) = VirtualPermissiveArray(visit_simplify(node.body), node.dims)

get_switch_cases(ctx, node::VirtualPermissiveArray) = map(get_switch_cases(ctx, node.body)) do (guard, body)
    guard => VirtualPermissiveArray(body, node.dims)
end

stepper_range(ctx, node::VirtualPermissiveArray, ext) = stepper_range(ctx, node.body, ext)
stepper_body(ctx, node::VirtualPermissiveArray, ext, ext_2) = VirtualPermissiveArray(stepper_body(ctx, node.body, ext, ext_2), node.dims)
stepper_seek(ctx, node::VirtualPermissiveArray, ext) = stepper_seek(ctx, node.body, ext)

jumper_range(ctx, node::VirtualPermissiveArray, ext) = jumper_range(ctx, node.body, ext)
jumper_body(ctx, node::VirtualPermissiveArray, ext, ext_2) = VirtualPermissiveArray(jumper_body(ctx, node.body, ext, ext_2), node.dims)
jumper_seek(ctx, node::VirtualPermissiveArray, ext) = jumper_seek(ctx, node.body, ext)

function short_circuit_cases(ctx, node::VirtualPermissiveArray, op)
    map(short_circuit_cases(ctx, node.body, op)) do (guard, body)
        guard => VirtualPermissiveArray(body, node.dims)
    end
end

getroot(tns::VirtualPermissiveArray) = getroot(tns.body)

function unfurl(ctx, tns::VirtualPermissiveArray, ext, mode, proto)
    tns_2 = unfurl(ctx, tns.body, ext, mode, proto)
    dims = virtual_size(ctx, tns.body)
    garb = (mode.kind === reader) ? FillLeaf(literal(missing)) : FillLeaf(Null())
    if tns.dims[end] && dims[end] != dimless
        VirtualPermissiveArray(
            Unfurled(
                tns,
                Sequence([
                    Phase(
                        stop = (ctx, ext_2) -> call(-, getstart(dims[end]), 1),
                        body = (ctx, ext) -> Run(garb),
                    ),
                    Phase(
                        stop = (ctx, ext_2) -> getstop(dims[end]),
                        body = (ctx, ext_2) -> truncate(ctx, tns_2, dims[end], ext_2)
                    ),
                    Phase(
                        body = (ctx, ext_2) -> Run(garb),
                    )
                ]),
            ),
            tns.dims
        )
    else
        VirtualPermissiveArray(tns_2, tns.dims)
    end
end

function lower_access(ctx::AbstractCompiler, tns::VirtualPermissiveArray, mode)
    lower_access(ctx, tns.body, mode)
end

function lower_assign(ctx::AbstractCompiler, tns::VirtualPermissiveArray, mode, op, rhs)
    lower_assign(ctx, tns.body, mode, op, rhs)
end
