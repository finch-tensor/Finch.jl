struct SwizzleArray{dims,Body} <: AbstractCombinator
    body::Body
end

SwizzleArray(body, dims) = SwizzleArray{dims}(body)
SwizzleArray{dims}(body::Body) where {dims,Body} = SwizzleArray{dims,Body}(body)

Base.ndims(arr::SwizzleArray) = ndims(typeof(arr))
Base.ndims(::Type{SwizzleArray{dims,Body}}) where {dims,Body} = ndims(Body)
Base.eltype(arr::SwizzleArray) = eltype(typeof(arr.body))
Base.eltype(::Type{SwizzleArray{dims,Body}}) where {dims,Body} = eltype(Body)
fill_value(arr::SwizzleArray) = fill_value(typeof(arr))
fill_value(::Type{SwizzleArray{dims,Body}}) where {dims,Body} = fill_value(Body)
Base.similar(arr::SwizzleArray{dims}) where {dims} = SwizzleArray{dims}(similar(arr.body))

isstructequal(a::T, b::T) where {T<:Finch.SwizzleArray} =
    isstructequal(a.body, b.body)

countstored(arr::SwizzleArray) = countstored(arr.body)

Base.size(arr::SwizzleArray{dims}) where {dims} = map(n -> size(arr.body)[n], dims)

function Base.show(io::IO, ex::SwizzleArray{dims}) where {dims}
    print(io, "SwizzleArray($(ex.body), $(dims))")
end

function labelled_show(io::IO, ::SwizzleArray{dims}) where {dims}
    print(io, "SwizzleArray ($(join(dims, ", ")))")
end

labelled_children(ex::SwizzleArray) = [LabelledTree(ex.body)]

struct VirtualSwizzleArray <: AbstractVirtualCombinator
    body
    dims
end

Base.show(io::IO, ex::VirtualSwizzleArray) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::VirtualSwizzleArray)
    print(io, "VirtualSwizzleArray($(ex.body), $(ex.dims))")
end

function Base.summary(io::IO, ex::VirtualSwizzleArray)
    print(io, "VSwizzle($(summary(ex.body)), $(ex.dims))")
end

FinchNotation.finch_leaf(x::VirtualSwizzleArray) = virtual(x)

function virtualize(ctx, ex, ::Type{SwizzleArray{dims,Body}}) where {dims,Body}
    VirtualSwizzleArray(virtualize(ctx, :($ex.body), Body), dims)
end

"""
    swizzle(tns, dims)

Create a `SwizzleArray` to transpose any tensor `tns` such that
```
    swizzle(tns, dims)[i...] == tns[i[dims]]
```
"""
swizzle(body, dims...) = SwizzleArray(body, dims)
function swizzle(body::SwizzleArray{dims}, dims_2...) where {dims}
    SwizzleArray(body.body, ntuple(n -> dims[dims_2[n]], ndims(body)))
end

function virtual_call(ctx, ::typeof(swizzle), body, dims...)
    @assert All(isliteral)(dims)
    VirtualSwizzleArray(body, map(dim -> dim.val, collect(dims)))
end
function unwrap(ctx, arr::VirtualSwizzleArray, var)
    call(swizzle, unwrap(ctx, arr.body, var), arr.dims...)
end

function lower(ctx::AbstractCompiler, tns::VirtualSwizzleArray, ::DefaultStyle)
    :(SwizzleArray($(ctx(tns.body)), $((tns.dims...,))))
end

function virtual_fill_value(ctx::AbstractCompiler, arr::VirtualSwizzleArray)
    virtual_fill_value(ctx, arr.body)
end

function virtual_size(ctx::AbstractCompiler, arr::VirtualSwizzleArray)
    virtual_size(ctx, arr.body)[arr.dims]
end

function virtual_resize!(ctx::AbstractCompiler, arr::VirtualSwizzleArray, dims...)
    virtual_resize!(ctx, arr.body, dims[invperm(arr.dims)]...)
end

function instantiate(ctx, arr::VirtualSwizzleArray, mode)
    VirtualSwizzleArray(instantiate(ctx, arr.body, mode), arr.dims)
end

get_style(ctx, node::VirtualSwizzleArray, root) = get_style(ctx, node.body, root)

getroot(tns::VirtualSwizzleArray) = getroot(tns.body)

function lower_access(ctx::AbstractCompiler, tns::VirtualSwizzleArray, mode)
    lower_access(ctx, tns.body, mode)
end

function lower_assign(ctx::AbstractCompiler, tns::VirtualSwizzleArray, mode, op, rhs)
    lower_assign(ctx, tns.body, mode, op, rhs)
end
