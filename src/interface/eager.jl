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
expanddims(arr::AbstractTensor; dims=:) = compute(expanddims(lazy(arr); dims=dims))

"""
    dropdims(arr::AbstractTensor; dims=:)

Reduces the dimensions of an array by removing the singleton axis or axes that
appear at the `dims` position in the array shape.
"""
Base.dropdims(arr::AbstractTensor; dims=:) = compute(dropdims(lazy(arr); dims=dims))

"""
    argmax(arr::AbstractTensor, dims=:)

Find the index of the maximum value in an array across dims
"""
function Base.argmax(A::AbstractTensor; dims=:)
    res = compute(argmax(lazy(A); dims=dims))
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
    res = compute(argmin(lazy(A); dims=dims))
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
    res = compute(argmax_python(lazy(A); dims=dims))
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
    res = compute(argmin_python(lazy(A); dims=dims))
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

function reshape_plan(tns, dims)
    if ndims(tns) == 0
        combine_mask = ()
        split_mask = ntuple(i -> (i:i...,), length(dims))
        return combine_mask, split_mask
    end

    if length(dims) == 0
        combine_mask = ndims(tns) == 0 ? () : ((1:ndims(tns)...,),)
        split_mask = ()
        return combine_mask, split_mask
    end

    num_colon = count(x -> x === Colon(), dims)
    if num_colon > 1
        throw(ArgumentError("Only one colon is allowed in the reshape dimensions."))
    end
    if num_colon == 1
        (q, r) = divrem(prod(size(tns)), prod(filter(x -> x !== Colon(), dims)))
        if r != 0
            throw(
                ArgumentError(
                    "The product of the dimensions must be equal to the size of the tensor."
                ),
            )
        end
        dims = (d === Colon() ? q : d for d in dims)
    else
        if prod(dims) != prod(size(tns))
            throw(
                ArgumentError(
                    "The product of the dimensions must be equal to the size of the tensor."
                ),
            )
        end
    end

    prod_mask = cumprod(dims)
    prod_shape = cumprod(size(tns))
    combine_stops = findall(
        i -> prod_shape[i] in prod_mask && (i == ndims(tns) || size(tns)[i + 1] != 1),
        1:ndims(tns),
    )
    combine_mask = (
        map(
            (x, y) -> ((x + 1):y...,), [0, combine_stops[1:(end - 1)]...], combine_stops
        )...,
    )
    split_stops = findall(
        i -> prod_mask[i] in prod_shape && (i == length(dims) || dims[i + 1] != 1),
        1:length(dims),
    )
    split_mask = (
        map((x, y) -> ((x + 1):y...,), [0, split_stops[1:(end - 1)]...], split_stops)...,
    )

    return combine_mask, split_mask
end

combinedims_rep(tns) = tns
function combinedims_rep(tns, dims, mask...)
    res = tns
    for i in 2:length(dims)
        res = combinedims_rep_def(res, get_level_rep(res))
    end
    set_level_rep(res, combinedims_rep(get_level_rep(res), mask...))
end
combinedims_rep_def(tns::SparseData, lvl::SparseData) = SparseData(lvl.lvl)
combinedims_rep_def(tns::SparseData, lvl::DenseData) = SparseData(lvl.lvl)
combinedims_rep_def(tns::SparseData, lvl::ExtrudeData) = SparseData(lvl.lvl)
combinedims_rep_def(tns::SparseData, lvl::RepeatData) = SparseData(lvl.lvl)
combinedims_rep_def(tns::DenseData, lvl::SparseData) = SparseData(lvl.lvl)
combinedims_rep_def(tns::DenseData, lvl::DenseData) = DenseData(lvl.lvl)
combinedims_rep_def(tns::DenseData, lvl::ExtrudeData) = DenseData(lvl.lvl)
combinedims_rep_def(tns::DenseData, lvl::RepeatData) = RepeatData(lvl.lvl)
combinedims_rep_def(tns::RepeatData, lvl::SparseData) = SparseData(lvl.lvl)
combinedims_rep_def(tns::RepeatData, lvl::DenseData) = DenseData(lvl.lvl)
combinedims_rep_def(tns::RepeatData, lvl::ExtrudeData) = RepeatData(lvl.lvl)
combinedims_rep_def(tns::RepeatData, lvl::RepeatData) = RepeatData(lvl.lvl)
combinedims_rep_def(tns::ExtrudeData, lvl) = lvl

splitdims_rep(tns) = tns
splitdims_rep(tns, dims, mask...) = splitdims_rep_def(tns, dims, mask)
function splitdims_rep_def(tns::SparseData, dims, mask)
    res = splitdims_rep(tns.lvl, mask...)
    for dim in dims
        res = SparseData(res)
    end
    res
end
function splitdims_rep_def(tns::DenseData, dims, mask)
    res = splitdims_rep(tns.lvl, mask...)
    for dim in dims
        res = DenseData(res)
    end
    res
end
function splitdims_rep_def(tns::ExtrudeData, dims, mask)
    res = splitdims_rep(tns.lvl, mask...)
    for dim in dims
        res = ExtrudeData(res)
    end
    res
end
function splitdims_rep_def(tns::RepeatData, dims, mask)
    res = RepeatData(splitdims_rep(tns.lvl, mask...))
    for dim in dims[1:(end - 1)]
        res = DenseData(res)
    end
    res
end

function splitdims_rep_def(tns::ElementData, dims, mask)
    res = splitdims_rep(tns, mask...)
    for dim in dims
        res = ExtrudeData(res)
    end
    res
end

@staged function reshape_constructor(tns, dims, combine_mask, split_mask)
    combine_mask = combine_mask.parameters[1]
    split_mask = split_mask.parameters[1]
    dims = dims.parameters
    rep = data_rep(tns)
    rep = splitdims_rep(combinedims_rep(rep, combine_mask...), split_mask...)
    fiber_ctr(rep, [:(dims[$i]) for i in 1:length(dims)]...)
end

@staged function reshape_kernel(dst, src, dims, combine_mask, split_mask)
    combine_mask = combine_mask.parameters[1]
    split_mask = split_mask.parameters[1]
    dims = dims.parameters
    N = ndims(src)
    M = length(dims)

    src_idxs = [Symbol(:i_, n) for n in 1:N]
    src_tmps = [Symbol(:t_, n) for n in 1:N]
    src_dims = [Symbol(:src_dim_, n) for n in 1:N]
    dst_tmps = [Symbol(:s_, m) for m in 1:M]
    dst_idxs = [Symbol(:j_, m) for m in 1:M]
    dst_dims = [Symbol(:dst_dim_, m) for m in 1:M]

    if N == 0
        return quote
            @finch begin
                dst .= 0
                dst[$(fill(1, M)...)] = src[]
            end
            return dst
        end
    end

    for (combine_group, split_group) in zip(combine_mask, split_mask)
        src_tmps[combine_group[end]] = src_idxs[combine_group[end]]
        flat_idx = src_tmps[combine_group[1]]
        if length(combine_group) == 1
            flat_idx = src_idxs[combine_group[1]]
        end
        if length(split_group) == 1
            dst_idxs[split_group[1]] = flat_idx
        else
            dst_idxs[split_group[end]] = dst_tmps[split_group[end - 1]]
        end
    end

    res = quote
        dst[$(dst_idxs...)] = src[$(src_idxs...)]
    end

    for (combine_group, split_group) in zip(combine_mask, split_mask)
        if length(split_group) > 1
            for i in split_group[(end - 1):-1:2]
                res = quote
                    let $(dst_tmps[i]) = fld1($(dst_tmps[i - 1]), $(dst_dims[i]))
                        let $(dst_idxs[i]) =
                                $(dst_tmps[i - 1]) - ($(dst_tmps[i]) - 1) * $(dst_dims[i])
                            $res
                        end
                    end
                end
            end
            i = split_group[1]
            res = quote
                let $(dst_tmps[i]) = fld1($(src_tmps[combine_group[1]]), $(dst_dims[i]))
                    let $(dst_idxs[i]) =
                            $(src_tmps[combine_group[1]]) -
                            ($(dst_tmps[i]) - 1) * $(dst_dims[i])
                        $res
                    end
                end
            end
        end

        for i in combine_group
            if i != combine_group[end]
                res = quote
                    let $(src_tmps[i]) =
                            (($(src_tmps[i + 1]) - 1) * $(src_dims[i])) + $(src_idxs[i])
                        $res
                    end
                end
            end
            res = quote
                for $(src_idxs[i]) in _
                    #for $(src_idxs[i]) = 1:$(src_dims[i])
                    $res
                end
            end
        end
    end

    res = quote
        ($(src_dims...),) = size(src)
        ($(dst_dims...),) = dims
        @finch begin
            dst .= 0
            $res
        end
        return dst
    end

    return unblock(striplines(res))
end

Base.reshape(tns::AbstractTensor, dims::Union{Integer,Colon}...) = reshape(tns, (dims...,))
function Base.reshape(
    tns::SwizzleArray{perm}, dims::Tuple{Vararg{Union{Integer,Colon}}}
) where {perm}
    if perm == 1:ndims(tns)
        return reshape(tns.body, dims...)
    else
        return reshape(permutedims(tns.body, perm), dims)
    end
end
function Base.reshape(tns::AbstractTensor, dims::Tuple{Vararg{Union{Integer,Colon}}})
    num_colon = count(x -> x === Colon(), dims)
    if num_colon > 1
        throw(ArgumentError("Only one colon is allowed in the reshape dimensions."))
    end
    if num_colon == 1
        (q, r) = divrem(prod(size(tns)), prod(filter(x -> x !== Colon(), dims)))
        if r != 0
            throw(
                ArgumentError(
                    "The product of the dimensions must be equal to the size of the tensor."
                ),
            )
        end
        dims = ((d === Colon() ? q : d for d in dims)...,)
    else
        if prod(dims) != prod(size(tns))
            throw(
                ArgumentError(
                    "The product of the dimensions must be equal to the size of the tensor."
                ),
            )
        end
    end

    (combine_mask, split_mask) = reshape_plan(tns, dims)

    if length(dims) == 0
        Tensor(Element(default(tns), tns[fill(1, ndims(tns))...]))
    else
        dst = reshape_constructor(tns, dims, Val(combine_mask), Val(split_mask))
        reshape_kernel(dst, tns, dims, Val(combine_mask), Val(split_mask))
    end
end
function reshape!(dst, src::AbstractTensor, dims::Union{Integer,Colon}...)
    reshape!(dst, src, dims)
end
function reshape!(dst, src::SwizzleArray{perm}, dims::Union{Integer,Colon}) where {perm}
    if perm == 1:ndims(src)
        return reshape!(dst, src.body, dims)
    else
        return reshape!(dst, permutedims(src.body, perm), dims)
    end
end
function reshape!(dst, src::AbstractTensor, dims::Tuple{Vararg{Union{Integer,Colon}}})
    (combine_mask, split_mask) = reshape_plan(tns, dims)
    reshape_kernel(dst, src, dims, Val(combine_mask), Val(split_mask))
end
