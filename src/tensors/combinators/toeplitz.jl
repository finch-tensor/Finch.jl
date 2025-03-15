struct ToeplitzArray{dim,Body} <: AbstractCombinator
    body::Body
end

ToeplitzArray(body, dim) = ToeplitzArray{dim}(body)
ToeplitzArray{dim}(body::Body) where {dim,Body} = ToeplitzArray{dim,Body}(body)

function Base.show(io::IO, ex::ToeplitzArray{dim}) where {dim}
    print(io, "ToeplitzArray{$dim}($(ex.body))")
end

function labelled_show(io::IO, tns::ToeplitzArray{dim}) where {dim}
    dims = [":" for _ in ndims(tns)]
    dims[dim] = ": + :"
    print(io, "ToeplitzArray [$(join(dims, ", "))]")
end

labelled_children(ex::ToeplitzArray) = [LabelledTree(ex.body)]

struct VirtualToeplitzArray <: AbstractVirtualCombinator
    body
    dim
    VirtualToeplitzArray(body, dim) = begin
        if body isa Thunk
            @assert(false)
        else
            new(body, dim)
        end
    end
end

function distribute(
    ctx::AbstractCompiler, tns::VirtualToeplitzArray, arch, diff, style
)
    VirtualToeplitzArray(distribute(ctx, tns.body, arch, diff, style), tns.dim)
end

function reroot_get(ctx::AbstractCompiler, tns::VirtualToeplitzArray, diff)
    VirtualToeplitzArray(
        reroot_get(ctx, tns.body, diff),
        tns.dim,
    )
end

function is_injective(ctx, lvl::VirtualToeplitzArray)
    sub = is_injective(ctx, lvl.body)
    return [sub[1:(lvl.dim)]..., false, sub[(lvl.dim + 1):end]...]
end
function is_atomic(ctx, lvl::VirtualToeplitzArray)
    (below, overall) = is_atomic(ctx, lvl.body)
    below_2 = [below[1:(lvl.dim)]..., below[lvl.dim], below[(lvl.dim + 1):end]...]
    return (below_2, overall)
end
function is_concurrent(ctx, lvl::VirtualToeplitzArray)
    sub = is_concurrent(ctx, lvl.body)
    return [sub[1:(lvl.dim)]..., false, sub[(lvl.dim + 1):end]...]
end

Base.show(io::IO, ex::VirtualToeplitzArray) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::VirtualToeplitzArray)
    print(io, "VirtualToeplitzArray($(ex.body), $(ex.dim))")
end

function Base.summary(io::IO, ex::VirtualToeplitzArray)
    print(io, "VToeplitz($(summary(ex.body)), $(ex.dim))")
end

FinchNotation.finch_leaf(x::VirtualToeplitzArray) = virtual(x)

function virtualize(ctx, ex, ::Type{ToeplitzArray{dim,Body}}) where {dim,Body}
    VirtualToeplitzArray(virtualize(ctx, :($ex.body), Body), dim)
end

"""
    toeplitz(tns, dim)

Create a `ToeplitzArray` such that
```
    Toeplitz(tns, dim)[i...] == tns[i[1:dim-1]..., i[dim] + i[dim + 1], i[dim + 2:end]...]
```
The ToplitzArray can be thought of as adding a dimension that shifts another dimension of the original tensor.
"""
toeplitz(body, dim) = ToeplitzArray(body, dim)
function virtual_call(ctx, ::typeof(toeplitz), body, dim)
    @assert isliteral(dim)
    VirtualToeplitzArray(body, dim.val)
end

function unwrap(ctx, arr::VirtualToeplitzArray, var)
    call(toeplitz, unwrap(ctx, arr.body, var), arr.dim)
end

function lower(ctx::AbstractCompiler, tns::VirtualToeplitzArray, ::DefaultStyle)
    :(ToeplitzArray($(ctx(tns.body)), $(tns.dim)))
end

function virtual_size(ctx::AbstractCompiler, arr::VirtualToeplitzArray)
    dims = virtual_size(ctx, arr.body)
    return (dims[1:(arr.dim - 1)]..., auto, auto, dims[(arr.dim + 1):end]...)
end
function virtual_resize!(ctx::AbstractCompiler, arr::VirtualToeplitzArray, dims...)
    virtual_resize!(
        ctx, arr.body, dims[1:(arr.dim - 1)]..., auto, dims[(arr.dim + 2):end]...
    )
end

function instantiate(ctx, arr::VirtualToeplitzArray, mode)
    VirtualToeplitzArray(instantiate(ctx, arr.body, mode), arr.dim)
end

get_style(ctx, node::VirtualToeplitzArray, root) = get_style(ctx, node.body, root)

#Note, popdim is NOT recursive, it should only be called on the node itself to
#reflect that the child lost a dimension and perhaps update this wrapper
#accordingly.
function popdim(node::VirtualToeplitzArray, ctx::AbstractCompiler)
    @assert length(virtual_size(ctx, node)) >= node.dim + 1
    return node
end

function truncate(ctx, node::VirtualToeplitzArray, ext, ext_2)
    VirtualToeplitzArray(truncate(ctx, node.body, ext, ext_2), node.dim)
end

function get_point_body(ctx, node::VirtualToeplitzArray, ext, idx)
    pass_nothing(get_point_body(ctx, node.body, ext, idx)) do body_2
        popdim(VirtualToeplitzArray(body_2, node.dim), ctx)
    end
end

function unwrap_thunk(ctx, node::VirtualToeplitzArray)
    VirtualToeplitzArray(unwrap_thunk(ctx, node.body), node.dim)
end

function get_run_body(ctx, node::VirtualToeplitzArray, ext)
    pass_nothing(get_run_body(ctx, node.body, ext)) do body_2
        popdim(VirtualToeplitzArray(body_2, node.dim), ctx)
    end
end

function get_acceptrun_body(ctx, node::VirtualToeplitzArray, ext)
    pass_nothing(get_acceptrun_body(ctx, node.body, ext)) do body_2
        popdim(VirtualToeplitzArray(body_2, node.dim), ctx)
    end
end

function get_sequence_phases(ctx, node::VirtualToeplitzArray, ext)
    map(get_sequence_phases(ctx, node.body, ext)) do (keys, body)
        return keys => VirtualToeplitzArray(body, node.dim)
    end
end

function phase_body(ctx, node::VirtualToeplitzArray, ext, ext_2)
    VirtualToeplitzArray(phase_body(ctx, node.body, ext, ext_2), node.dim)
end
phase_range(ctx, node::VirtualToeplitzArray, ext) = phase_range(ctx, node.body, ext)

function get_spike_body(ctx, node::VirtualToeplitzArray, ext, ext_2)
    VirtualToeplitzArray(get_spike_body(ctx, node.body, ext, ext_2), node.dim)
end
function get_spike_tail(ctx, node::VirtualToeplitzArray, ext, ext_2)
    VirtualToeplitzArray(get_spike_tail(ctx, node.body, ext, ext_2), node.dim)
end

visit_fill_leaf_leaf(node, tns::VirtualToeplitzArray) = visit_fill_leaf_leaf(node, tns.body)
function visit_simplify(node::VirtualToeplitzArray)
    VirtualToeplitzArray(visit_simplify(node.body), node.dim)
end

function get_switch_cases(ctx, node::VirtualToeplitzArray)
    map(get_switch_cases(ctx, node.body)) do (guard, body)
        guard => VirtualToeplitzArray(body, node.dim)
    end
end

stepper_range(ctx, node::VirtualToeplitzArray, ext) = stepper_range(ctx, node.body, ext)
function stepper_body(ctx, node::VirtualToeplitzArray, ext, ext_2)
    VirtualToeplitzArray(stepper_body(ctx, node.body, ext, ext_2), node.dim)
end
stepper_seek(ctx, node::VirtualToeplitzArray, ext) = stepper_seek(ctx, node.body, ext)

jumper_range(ctx, node::VirtualToeplitzArray, ext) = jumper_range(ctx, node.body, ext)
function jumper_body(ctx, node::VirtualToeplitzArray, ext, ext_2)
    VirtualToeplitzArray(jumper_body(ctx, node.body, ext, ext_2), node.dim)
end
jumper_seek(ctx, node::VirtualToeplitzArray, ext) = jumper_seek(ctx, node.body, ext)

function short_circuit_cases(ctx, node::VirtualToeplitzArray, op)
    map(short_circuit_cases(ctx, node.body, op)) do (guard, body)
        guard => VirtualToeplitzArray(body, node.dim)
    end
end

getroot(tns::VirtualToeplitzArray) = getroot(tns.body)

function unfurl(ctx, tns::VirtualToeplitzArray, ext, mode, proto)
    if length(virtual_size(ctx, tns)) == tns.dim + 1
        Unfurled(tns,
            Lookup(;
                body=(ctx, idx) -> VirtualPermissiveArray(
                    VirtualOffsetArray(
                        tns.body, ([literal(0) for _ in 1:(tns.dim - 1)]..., idx)
                    ),
                    ([false for _ in 1:(tns.dim - 1)]..., true),
                ),
            ),
        )
    else
        VirtualToeplitzArray(unfurl(ctx, tns.body, ext, mode, proto), tns.dim)
    end
end
