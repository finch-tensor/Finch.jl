struct ProductArray{dim,Body} <: AbstractCombinator
    body::Body
end

ProductArray(body, dim) = ProductArray{dim}(body)
ProductArray{dim}(body::Body) where {dim,Body} = ProductArray{dim,Body}(body)

function Base.show(io::IO, ex::ProductArray{dim}) where {dim}
    print(io, "ProductArray{$dim}($(ex.body))")
end

function labelled_show(io::IO, tns::ProductArray{dim}) where {dim}
    dims = [":" for _ in ndims(tns)]
    dims[dim] = ": * :"
    print(io, "ProductArray [$(join(dims, ", "))]")
end

function labelled_children(ex::ProductArray)
    [LabelledTree(ex.body)]
end

struct VirtualProductArray <: AbstractVirtualCombinator
    body
    dim
end

function is_injective(ctx, lvl::VirtualProductArray)
    sub = is_injective(ctx, lvl.body)
    return [sub[1:(lvl.dim)]..., false, sub[(lvl.dim + 1):end]...]
end
function is_concurrent(ctx, lvl::VirtualProductArray)
    sub = is_concurrent(ctx, lvl.body)
    return [sub[1:(lvl.dim)]..., false, sub[(lvl.dim + 1):end]...]
end
function is_atomic(ctx, lvl::VirtualProductArray)
    (below, overall) = is_atomic(ctx, lvl.body)
    return (
        [
            below[1:(lvl.dim)]...,
            below[lvl.dim] && below[lvl.dim + 1],
            below[(lvl.dim + 1):end]...,
        ],
        overall,
    )
end

Base.show(io::IO, ex::VirtualProductArray) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::VirtualProductArray)
    print(io, "VirtualProductArray($(ex.body), $(ex.dim))")
end

function Base.summary(io::IO, ex::VirtualProductArray)
    print(io, "VProduct($(summary(ex.body)), $(ex.dim))")
end

FinchNotation.finch_leaf(x::VirtualProductArray) = virtual(x)

function virtualize(ctx, ex, ::Type{ProductArray{dim,Body}}) where {dim,Body}
    VirtualProductArray(virtualize(ctx, :($ex.body), Body), dim)
end

"""
    products(tns, dim)

Create a `ProductArray` such that
```
    products(tns, dim)[i...] == tns[i[1:dim-1]..., i[dim] * i[dim + 1], i[dim + 2:end]...]
```
This is like [`toeplitz`](@ref) but with times instead of plus.
"""
products(body, dim) = ProductArray(body, dim)
function virtual_call(ctx, ::typeof(products), body, dim)
    @assert isliteral(dim)
    VirtualProductArray(body, dim.val)
end

function unwrap(ctx, arr::VirtualProductArray, var)
    call(products, unwrap(ctx, arr.body, var), arr.dim)
end

function lower(ctx::AbstractCompiler, tns::VirtualProductArray, ::DefaultStyle)
    :(ProductArray($(ctx(tns.body)), $(tns.dim)))
end

#virtual_size(ctx::AbstractCompiler, arr::FillLeaf) = (auto,) # this is needed for multidimensional convolution..
#virtual_size(ctx::AbstractCompiler, arr::Simplify) = (auto,)

function virtual_size(ctx::AbstractCompiler, arr::VirtualProductArray)
    dims = virtual_size(ctx, arr.body)
    return (dims[1:(arr.dim - 1)]..., auto, auto, dims[(arr.dim + 1):end]...)
end
function virtual_resize!(ctx::AbstractCompiler, arr::VirtualProductArray, dims...)
    virtual_resize!(
        ctx, arr.body, dims[1:(arr.dim - 1)]..., auto, dims[(arr.dim + 2):end]...
    )
end

function instantiate(arr::VirtualProductArray, ctx, mode)
    VirtualProductArray(instantiate(arr.body, ctx, mode), arr.dim)
end

get_style(ctx, node::VirtualProductArray, root) = get_style(ctx, node.body, root)

function popdim(node::VirtualProductArray, ctx)
    if length(virtual_size(ctx, node)) == node.dim
        return node.body
    else
        return node
    end
end

function truncate(ctx, node::VirtualProductArray, ext, ext_2)
    VirtualProductArray(truncate(ctx, node.body, ext, ext_2), node.dim)
end

function get_point_body(ctx, node::VirtualProductArray, ext, idx)
    pass_nothing(get_point_body(ctx, node.body, ext, idx)) do body_2
        popdim(VirtualProductArray(body_2, node.dim), ctx)
    end
end

function unwrap_thunk(ctx, node::VirtualProductArray)
    VirtualProductArray(unwrap_thunk(ctx, node.body), node.dim)
end

function get_run_body(ctx, node::VirtualProductArray, ext)
    pass_nothing(get_run_body(ctx, node.body, ext)) do body_2
        popdim(VirtualProductArray(body_2, node.dim), ctx)
    end
end

function get_acceptrun_body(ctx, node::VirtualProductArray, ext)
    pass_nothing(get_acceptrun_body(ctx, node.body, ext)) do body_2
        popdim(VirtualProductArray(body_2, node.dim), ctx)
    end
end

function get_sequence_phases(ctx, node::VirtualProductArray, ext)
    map(get_sequence_phases(ctx, node.body, ext)) do (keys, body)
        return keys => VirtualProductArray(body, node.dim)
    end
end

function phase_body(ctx, node::VirtualProductArray, ext, ext_2)
    VirtualProductArray(phase_body(ctx, node.body, ext, ext_2), node.dim)
end
phase_range(ctx, node::VirtualProductArray, ext) = phase_range(ctx, node.body, ext)

function get_spike_body(ctx, node::VirtualProductArray, ext, ext_2)
    VirtualProductArray(get_spike_body(ctx, node.body, ext, ext_2), node.dim)
end
function get_spike_tail(ctx, node::VirtualProductArray, ext, ext_2)
    VirtualProductArray(get_spike_tail(ctx, node.body, ext, ext_2), node.dim)
end

visit_fill_leaf_leaf(node, tns::VirtualProductArray) = visit_fill_leaf_leaf(node, tns.body)
function visit_simplify(node::VirtualProductArray)
    VirtualProductArray(visit_simplify(node.body), node.dim)
end

function get_switch_cases(ctx, node::VirtualProductArray)
    map(get_switch_cases(ctx, node.body)) do (guard, body)
        guard => VirtualProductArray(body, node.dim)
    end
end

function jumper_body(ctx, node::VirtualProductArray, ext)
    VirtualProductArray(jumper_body(ctx, node.body, ext), node.dim)
end
function stepper_body(ctx, node::VirtualProductArray, ext)
    VirtualProductArray(stepper_body(ctx, node.body, ext), node.dim)
end
stepper_seek(ctx, node::VirtualProductArray, ext) = stepper_seek(ctx, node.body, ext)
jumper_seek(ctx, node::VirtualProductArray, ext) = jumper_seek(ctx, node.body, ext)

getroot(tns::VirtualProductArray) = getroot(tns.body)

function unfurl(ctx, tns::VirtualProductArray, ext, mode, proto)
    if length(virtual_size(ctx, tns)) == tns.dim + 1
        Lookup(;
            body=(ctx, idx) -> VirtualPermissiveArray(
                VirtualScaleArray(;
                    body=tns.body,
                    scale=([literal(1) for _ in 1:(tns.dim - 1)]..., idx)
                ),
                ([false for _ in 1:(tns.dim - 1)]..., true),
            ),
        )
    else
        VirtualProductArray(unfurl(ctx, tns.body, ext, mode, proto), tns.dim)
    end
end
