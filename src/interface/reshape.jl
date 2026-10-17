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
    (combine_mask, split_mask) = reshape_plan(src, dims)
    reshape_kernel(dst, src, dims, Val(combine_mask), Val(split_mask))
end
