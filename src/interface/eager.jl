using Base: Broadcast
using Base.Broadcast: Broadcasted, BroadcastStyle, AbstractArrayStyle
using Base: broadcasted
using LinearAlgebra

struct FinchStyle{N} <: BroadcastStyle
end
Base.Broadcast.BroadcastStyle(F::Type{<:AbstractTensor}) = FinchStyle{ndims(F)}()
Base.Broadcast.broadcastable(fbr::AbstractTensor) = fbr
function Base.Broadcast.BroadcastStyle(a::FinchStyle{N}, b::FinchStyle{M}) where {M,N}
    FinchStyle{max(M, N)}()
end
function Base.Broadcast.BroadcastStyle(a::LazyStyle{M}, b::FinchStyle{N}) where {M,N}
    LazyStyle{max(M, N)}()
end
function Base.Broadcast.BroadcastStyle(
    a::FinchStyle{N}, b::Broadcast.AbstractArrayStyle{M}
) where {M,N}
    FinchStyle{max(M, N)}()
end

Base.Broadcast.instantiate(bc::Broadcasted{FinchStyle{N}}) where {N} = bc

function Base.copyto!(out, bc::Broadcasted{FinchStyle{N}}) where {N}
    compute(copyto!(out, copy(Broadcasted{LazyStyle{N}}(bc.f, bc.args))))
end

function Base.copy(bc::Broadcasted{FinchStyle{N}}) where {N}
    return compute(copy(Broadcasted{LazyStyle{N}}(bc.f, bc.args)))
end

function Base.reduce(op, src::AbstractTensor; dims=:, init=initial_value(op, eltype(src)))
    res = compute(reduce(op, lazy(src); dims=dims, init=init))
    if dims === Colon()
        return res[]
    else
        return res
    end
end

function Base.mapreduce(
    f,
    op,
    src::AbstractTensor,
    args::Union{AbstractTensor,Base.AbstractArrayOrBroadcasted,Number}...;
    kw...,
)
    reduce(op, broadcasted(f, src, args...); kw...)
end
function Base.map(
    f,
    src::AbstractTensor,
    args::Union{AbstractTensor,Base.AbstractArrayOrBroadcasted,Number}...,
)
    f.(src, args...)
end
function Base.map!(
    dst,
    f,
    src::AbstractTensor,
    args::Union{AbstractTensor,Base.AbstractArrayOrBroadcasted}...,
)
    copyto!(dst, Base.broadcasted(f, src, args...))
end

function Base.reduce(
    op::Function,
    bc::Broadcasted{FinchStyle{N}};
    dims=:,
    init=initial_value(op, return_type(DefaultAlgebra(), bc.f, map(eltype, bc.args)...)),
) where {N}
    res = compute(
        reduce(op, copy(Broadcasted{LazyStyle{N}}(bc.f, bc.args)); dims=dims, init=init)
    )
    if dims === Colon()
        return res[]
    else
        return res
    end
end

function tensordot(
    A::Union{AbstractTensor,AbstractArray},
    B::Union{AbstractTensor,AbstractArray},
    idxs;
    kw...,
)
    compute(tensordot(lazy(A), lazy(B), idxs; kw...))
end

function Base.:+(
    x::AbstractTensor,
    y::Union{Base.AbstractArrayOrBroadcasted,Number},
    z::Union{AbstractTensor,Base.AbstractArrayOrBroadcasted,Number}...,
)
    map(+, x, y, z...)
end
function Base.:+(
    x::Union{Base.AbstractArrayOrBroadcasted,Number},
    y::AbstractTensor,
    z::Union{AbstractTensor,Base.AbstractArrayOrBroadcasted,Number}...,
)
    map(+, y, x, z...)
end
function Base.:+(
    x::AbstractTensor,
    y::AbstractTensor,
    z::Union{AbstractTensor,Base.AbstractArrayOrBroadcasted,Number}...,
)
    map(+, x, y, z...)
end
Base.:*(
    x::AbstractTensor,
    y::Number,
    z::Number...,
) = map(*, x, y, z...)
Base.:*(
    x::Number,
    y::AbstractTensor,
    z::Number...,
) = map(*, y, x, z...)

Base.:*(
    A::AbstractTensor,
    B::Union{AbstractTensor,AbstractArray},
) = tensordot(A, B, (2, 1))
Base.:*(
    A::Union{AbstractTensor,AbstractArray},
    B::AbstractTensor,
) = tensordot(A, B, (2, 1))
Base.:*(
    A::AbstractTensor,
    B::AbstractTensor,
) = tensordot(A, B, (2, 1))

Base.:-(x::AbstractTensor) = map(-, x)

Base.:-(x::AbstractTensor, y::Union{Base.AbstractArrayOrBroadcasted,Number}) = map(-, x, y)
Base.:-(x::Union{Base.AbstractArrayOrBroadcasted,Number}, y::Tensor) = map(-, x, y)
Base.:-(x::AbstractTensor, y::AbstractTensor) = map(-, x, y)

Base.:/(x::AbstractTensor, y::Number) = map(/, x, y)
Base.:/(x::Number, y::AbstractTensor) = map(\, y, x)

const AbstractTensorOrBroadcast = Union{
    <:AbstractTensor,<:Broadcasted{FinchStyle{N}} where {N}
}

Base.sum(arr::AbstractTensorOrBroadcast; kwargs...) = reduce(+, arr; kwargs...)
Base.prod(arr::AbstractTensorOrBroadcast; kwargs...) = reduce(*, arr; kwargs...)
Base.any(arr::AbstractTensorOrBroadcast; kwargs...) = reduce(or, arr; init=false, kwargs...)
Base.all(arr::AbstractTensorOrBroadcast; kwargs...) = reduce(and, arr; init=true, kwargs...)
function Base.minimum(arr::AbstractTensorOrBroadcast; kwargs...)
    reduce(min, arr; init=typemax(broadcast_to_eltype(arr)), kwargs...)
end
function Base.maximum(arr::AbstractTensorOrBroadcast; kwargs...)
    reduce(max, arr; init=typemin(broadcast_to_eltype(arr)), kwargs...)
end

function Base.extrema(arr::AbstractTensorOrBroadcast; kwargs...)
    mapreduce(
        plex,
        min1max2,
        arr;
        init=(typemax(broadcast_to_eltype(arr)), typemin(broadcast_to_eltype(arr))),
        kwargs...,
    )
end

function LinearAlgebra.norm(arr::AbstractTensorOrBroadcast, p::Real=2)
    compute(norm(lazy(arr), p))[]
end

"""
    expanddims(arr::AbstractTensor; dims=:)

Expand the dimensions of an array by inserting a new singleton axis or axes that
will appear at the `dims` position in the expanded array shape.
"""
expanddims(arr::AbstractTensor; dims=:) = compute(expanddims(lazy(arr), dims=dims))

"""
    dropdims(arr::AbstractTensor; dims=:)

Reduces the dimensions of an array by removing the singleton axis or axes that
appear at the `dims` position in the array shape.
"""
Base.dropdims(arr::AbstractTensor; dims=:) = compute(dropdims(lazy(arr), dims=dims))

"""
    argmax(arr::AbstractTensor, dims=:)

Find the index of the maximum value in an array across dims
"""
function Base.argmax(A::AbstractTensor; dims=:)
    res = compute(argmax(lazy(A), dims=dims))
    if dims === Colon()
        return res[]
    else
        return res
    end
end

"""
    argmin(arr::AbstractTensor, dims)

Find the index of the minimum value in an array across dims
"""
function Base.argmin(A::AbstractTensor; dims=:)
    res = compute(argmin(lazy(A), dims=dims))
    if dims === Colon()
        return res[]
    else
        return res
    end
end

"""
    argmax_python(A; dims=:) 

Find the index of the maximum value in an array across dims, following
https://data-apis.org/array-api/latest/API_specification/generated/array_api.argmax.html#argmax,
which is different from Base Julia semantics. This version only accepts either a
single dimension or all dimensions. When dims is a single dimension, the returned
array contains integer indices along that dimension. When dims is all dimensions,
the returned array contains the index of the maximum value in the flattened
array. 
"""
function argmax_python(A::AbstractTensor; dims=:)
    res = compute(argmax_python(lazy(A), dims=dims))
    if dims === Colon()
        return res[]
    else
        return res
    end
end

"""
    argmin_python(A; dims=:) 

Find the index of the minimum value in an array across dims, following
https://data-apis.org/array-api/latest/API_specification/generated/array_api.argmin.html#argmin,
which is different from Base Julia semantics. This version only accepts either a
single dimension or all dimensions. When dims is a single dimension, the returned
array contains integer indices along that dimension. When dims is all dimensions,
the returned array contains the index of the minimum value in the flattened
array. 
"""
function argmin_python(A::AbstractTensor; dims=:)
    res = compute(argmin_python(lazy(A), dims=dims))
    if dims === Colon()
        return res[]
    else
        return res
    end
end

function Statistics.mean(tns::AbstractTensorOrBroadcast; dims=:)
    res = compute(mean(lazy(tns); dims=dims))
    if dims === Colon()
        return res[]
    else
        return res
    end
end

function Statistics.mean(f, tns::AbstractTensorOrBroadcast; dims=:)
    res = compute(mean(f, lazy(tns); dims=dims))
    if dims === Colon()
        return res[]
    else
        return res
    end
end

function Statistics.varm(tns::AbstractTensorOrBroadcast, m; corrected=true, dims=:)
    res = compute(varm(lazy(tns), m; corrected=corrected, dims=dims))
    if dims === Colon()
        return res[]
    else
        return res
    end
end

function Statistics.var(
    tns::AbstractTensorOrBroadcast; corrected=true, mean=nothing, dims=:
)
    res = compute(var(lazy(tns); corrected=corrected, mean=mean, dims=dims))
    if dims === Colon()
        return res[]
    else
        return res
    end
end

function Statistics.stdm(
    tns::AbstractTensorOrBroadcast, m; corrected=true, dims=:
)
    res = compute(stdm(lazy(tns), lazy(m); corrected=corrected, dims=dims))
    if dims === Colon()
        return res[]
    else
        return res
    end
end

function Statistics.std(
    tns::AbstractTensorOrBroadcast; corrected=true, mean=nothing, dims=:
)
    res = compute(std(lazy(tns); corrected=corrected, mean=mean, dims=dims))
    if dims === Colon()
        return res[]
    else
        return res
    end
end
