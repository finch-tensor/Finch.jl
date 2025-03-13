struct PermissiveArray{dims,Body} <: AbstractCombinator
    body::Body
end

PermissiveArray(body, dims) = PermissiveArray{dims}(body)
PermissiveArray{dims}(body::Body) where {dims,Body} = PermissiveArray{dims,Body}(body)

function Base.show(io::IO, ex::PermissiveArray{dims}) where {dims}
    print(io, "PermissiveArray($(ex.body), $dims)")
end

function labelled_show(io::IO, ::PermissiveArray{dims}) where {dims}
    print(io, "PermissiveArray [$(join(map(d -> d ? "~:" : ":", dims), ", "))]")
end

labelled_children(ex::PermissiveArray) = [LabelledTree(ex.body)]

struct VirtualPermissiveArray <: AbstractVirtualCombinator
    body
    dims
end

function distribute(
    ctx::AbstractCompiler, tns::VirtualPermissiveArray, arch, style
)
    VirtualPermissiveArray(distribute(ctx, tns.body, arch, style), tns.dims)
end

function reroot_set!(ctx::AbstractCompiler, tns::VirtualPermissiveArray, diff)
    reroot_set!(ctx, tns.body, diff)
end

function reroot_get(ctx::AbstractCompiler, tns::VirtualPermissiveArray, diff)
    VirtualPermissiveArray(
        reroot_get(ctx, tns.body, diff),
        tns.dims,
    )
end

is_injective(ctx, lvl::VirtualPermissiveArray) = is_injective(ctx, lvl.body)
is_atomic(ctx, lvl::VirtualPermissiveArray) = is_atomic(ctx, lvl.body)
is_concurrent(ctx, lvl::VirtualPermissiveArray) = is_concurrent(ctx, lvl.body)

Base.show(io::IO, ex::VirtualPermissiveArray) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::VirtualPermissiveArray)
    print(io, "VirtualPermissiveArray($(ex.body), $(ex.dims))")
end

function Base.summary(io::IO, ex::VirtualPermissiveArray)
    print(io, "VPermissive($(summary(ex.body)), $(ex.dims))")
end

FinchNotation.finch_leaf(x::VirtualPermissiveArray) = virtual(x)

function virtualize(ctx, ex, ::Type{PermissiveArray{dims,Body}}) where {dims,Body}
    VirtualPermissiveArray(virtualize(ctx, :($ex.body), Body), dims)
end

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

function unwrap(ctx, arr::VirtualPermissiveArray, var)
    call(permissive, unwrap(ctx, arr.body, var), arr.dims...)
end

function lower(ctx::AbstractCompiler, tns::VirtualPermissiveArray, ::DefaultStyle)
    :(PermissiveArray($(ctx(tns.body)), $(tns.dims)))
end

function virtual_size(ctx::AbstractCompiler, arr::VirtualPermissiveArray)
    ifelse.(arr.dims, (auto,), virtual_size(ctx, arr.body))
end

function virtual_resize!(ctx::AbstractCompiler, arr::VirtualPermissiveArray, dims...)
    virtual_resize!(ctx, arr.body, ifelse.(arr.dims, virtual_size(ctx, arr.body), dim))
end

function virtual_fill_value(ctx::AbstractCompiler, arr::VirtualPermissiveArray)
    virtual_fill_value(ctx, arr.body)
end

function instantiate(ctx, arr::VirtualPermissiveArray, mode)
    VirtualPermissiveArray(instantiate(ctx, arr.body, mode), arr.dims)
end

get_style(ctx, node::VirtualPermissiveArray, root) = get_style(ctx, node.body, root)

function popdim(node::VirtualPermissiveArray)
    if length(node.dims) == 1
        return node.body
    else
        return VirtualPermissiveArray(node.body, node.dims[1:(end - 1)])
    end
end

function truncate(ctx, node::VirtualPermissiveArray, ext, ext_2)
    VirtualPermissiveArray(truncate(ctx, node.body, ext, ext_2), node.dims)
end

function get_point_body(ctx, node::VirtualPermissiveArray, ext, idx)
    pass_nothing(get_point_body(ctx, node.body, ext, idx)) do body_2
        popdim(VirtualPermissiveArray(body_2, node.dims))
    end
end

function unwrap_thunk(ctx, node::VirtualPermissiveArray)
    VirtualPermissiveArray(unwrap_thunk(ctx, node.body), node.dims)
end

function get_run_body(ctx, node::VirtualPermissiveArray, ext)
    pass_nothing(get_run_body(ctx, node.body, ext)) do body_2
        popdim(VirtualPermissiveArray(body_2, node.dims))
    end
end

function get_acceptrun_body(ctx, node::VirtualPermissiveArray, ext)
    pass_nothing(get_acceptrun_body(ctx, node.body, ext)) do body_2
        popdim(VirtualPermissiveArray(body_2, node.dims))
    end
end

function get_sequence_phases(ctx, node::VirtualPermissiveArray, ext)
    map(get_sequence_phases(ctx, node.body, ext)) do (keys, body)
        return keys => VirtualPermissiveArray(body, node.dims)
    end
end

function phase_body(ctx, node::VirtualPermissiveArray, ext, ext_2)
    VirtualPermissiveArray(phase_body(ctx, node.body, ext, ext_2), node.dims)
end
phase_range(ctx, node::VirtualPermissiveArray, ext) = phase_range(ctx, node.body, ext)

function get_spike_body(ctx, node::VirtualPermissiveArray, ext, ext_2)
    VirtualPermissiveArray(get_spike_body(ctx, node.body, ext, ext_2), node.dims)
end
function get_spike_tail(ctx, node::VirtualPermissiveArray, ext, ext_2)
    VirtualPermissiveArray(get_spike_tail(ctx, node.body, ext, ext_2), node.dims)
end

function visit_fill_leaf_leaf(node, tns::VirtualPermissiveArray)
    visit_fill_leaf_leaf(node, tns.body)
end
function visit_simplify(node::VirtualPermissiveArray)
    VirtualPermissiveArray(visit_simplify(node.body), node.dims)
end

function get_switch_cases(ctx, node::VirtualPermissiveArray)
    map(get_switch_cases(ctx, node.body)) do (guard, body)
        guard => VirtualPermissiveArray(body, node.dims)
    end
end

stepper_range(ctx, node::VirtualPermissiveArray, ext) = stepper_range(ctx, node.body, ext)
function stepper_body(ctx, node::VirtualPermissiveArray, ext, ext_2)
    VirtualPermissiveArray(stepper_body(ctx, node.body, ext, ext_2), node.dims)
end
stepper_seek(ctx, node::VirtualPermissiveArray, ext) = stepper_seek(ctx, node.body, ext)

jumper_range(ctx, node::VirtualPermissiveArray, ext) = jumper_range(ctx, node.body, ext)
function jumper_body(ctx, node::VirtualPermissiveArray, ext, ext_2)
    VirtualPermissiveArray(jumper_body(ctx, node.body, ext, ext_2), node.dims)
end
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
    if tns.dims[end] && dims[end] != auto
        VirtualPermissiveArray(
            Unfurled(
                tns,
                Sequence([
                    Phase(;
                        stop = (ctx, ext_2) -> call(-, getstart(dims[end]), 1),
                        body = (ctx, ext) -> Run(garb),
                    ),
                    Phase(;
                        stop = (ctx, ext_2) -> getstop(dims[end]),
                        body = (ctx, ext_2) -> truncate(ctx, tns_2, dims[end], ext_2),
                    ),
                    Phase(;
                        body=(ctx, ext_2) -> Run(garb)
                    ),
                ]),
            ),
            tns.dims,
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
