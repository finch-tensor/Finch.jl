struct Drop{Idx}
    idx::Idx
end

function Base.to_indices(A::AbstractTensor, I::Tuple{AbstractVector})
    Base.to_indices(A, axes(A), I)
end

Base.IndexStyle(::Type{<:AbstractTensor}) = Base.IndexCartesian()

"""
    getindex_rep(tns, idxs...)

Return a trait object representing the result of calling getindex(tns, idxs...)
on the tensor represented by `tns`. Assumes traits are in collapsed form.
"""
getindex_rep(tns, idxs...) = collapse_rep(
    getindex_rep_def(tns, map(idx -> ndims(idx) == 0 ? Drop(idx) : idx, idxs)...)
)

getindex_rep_def(fbr::HollowData, idxs...) = HollowData(getindex_rep_def(fbr.lvl, idxs...))

function getindex_rep_def(lvl::SparseData, idx::Drop, idxs...)
    HollowData(getindex_rep_def(lvl.lvl, idxs...))
end
function getindex_rep_def(lvl::SparseData, idx, idxs...)
    HollowData(SparseData(getindex_rep_def(lvl.lvl, idxs...)))
end
function getindex_rep_def(lvl::SparseData, idx::Type{<:Base.Slice}, idxs...)
    SparseData(getindex_rep_def(lvl.lvl, idxs...))
end

getindex_rep_def(lvl::DenseData, idx::Drop, idxs...) = getindex_rep_def(lvl.lvl, idxs...)
function getindex_rep_def(lvl::DenseData, idx, idxs...)
    DenseData(getindex_rep_def(lvl.lvl, idxs...))
end

getindex_rep_def(lvl::RepeatData, idx::Drop, idxs...) = getindex_rep_def(lvl.lvl, idxs...)
function getindex_rep_def(lvl::RepeatData, idx, idxs...)
    RepeatData(getindex_rep_def(lvl.lvl, idxs...))
end

getindex_rep_def(lvl::ElementData) = lvl

function Base.getindex(arr::AbstractTensor, inds::AbstractVector)
    getindex_helper(arr, to_indices(arr, axes(arr), (inds,)))
end
function Base.getindex(arr::AbstractTensor, inds...)
    if nothing in inds && inds isa Tuple{Vararg{Union{Nothing,Colon}}}
        return compute(lazy(arr)[inds...])
    else
        getindex_helper(arr, to_indices(arr, inds))
    end
end

function Base.getindex(arr::SwizzleArray{perm}, inds::AbstractVector) where {perm}
    getindex_helper(arr, to_indices(arr, axes(arr), (inds,)))
end
function Base.getindex(arr::SwizzleArray{perm}, inds...) where {perm}
    if nothing in inds && inds isa Tuple{Vararg{Union{Nothing,Colon}}}
        return compute(lazy(arr)[inds...])
    else
        inds_2 = Base.to_indices(arr, axes(arr), inds)
        perm_2 = collect(invperm(perm))
        res = getindex(arr.body, inds_2[perm_2]...)
        perm_3 = sortperm(filter(n -> ndims(inds_2[n]) > 0, perm_2))
        if issorted(perm_3)
            return res
        else
            return swizzle(res, perm_3...)
        end
    end
end

@staged function getindex_helper(arr, inds)
    inds <: Type{<:Tuple}
    inds = inds.parameters
    @assert ndims(arr) == length(inds)
    N = ndims(arr)

    inds_ndims = ndims.(inds)
    if sum(inds_ndims; init=0) == 0
        return quote
            scl = Scalar($(fill_value(arr)))
            @finch scl[] = arr[inds...]
            return scl[]
        end
    end

    T = eltype(arr)
    syms = [Symbol(:inds_, n) for n in 1:N]
    modes = [Symbol(:mode_, n) for n in 1:N]
    coords = map(1:N) do n
        if ndims(inds[n]) == 0
            syms[n]
        elseif inds[n] <: Base.Slice
            modes[n]
        else
            :($(syms[n])[$(modes[n])])
        end
    end
    dst_modes = modes[filter(n -> ndims(inds[n]) != 0, 1:N)]
    exts = Expr(:block, (:($idx = _) for idx in reverse(dst_modes))...)

    dst = fiber_ctr(getindex_rep(data_rep(arr), inds...))

    quote
        win = $dst
        ($(syms...),) = (inds...,)
        @finch begin
            win .= $(fill_value(arr))
            $(Expr(
                :for,
                exts,
                quote
                    win[$(dst_modes...)] = arr[$(coords...)]
                end,
            ))
        end
        return win
    end
end

function Base.setindex!(arr::AbstractTensor, src, inds...)
    setindex_helper(arr, src, to_indices(arr, inds))
end

function Base.setindex!(arr::SwizzleArray{perm}, v, inds...) where {perm}
    inds_2 = Base.to_indices(arr, inds)
    perm_2 = collect(invperm(perm))
    res = setindex!(arr.body, v, inds_2[perm_2]...)
    arr
end

@staged function setindex_helper(arr, src, inds)
    inds <: Type{<:Tuple}
    inds = inds.parameters
    @assert ndims(arr) == length(inds)
    @assert sum(ndims.(inds)) == 0 || (ndims(src) == sum(ndims.(inds)))
    N = ndims(arr)

    T = eltype(arr)
    syms = [Symbol(:inds_, n) for n in 1:N]
    modes = [Symbol(:mode_, n) for n in 1:N]
    coords = map(1:N) do n
        if ndims(inds[n]) == 0
            syms[n]
        elseif inds[n] <: Base.Slice
            modes[n]
        else
            :($(syms[n])[$(modes[n])])
        end
    end
    src_modes = modes[filter(n -> ndims(inds[n]) != 0, 1:N)]
    exts = Expr(:block, (:($idx = _) for idx in reverse(src_modes))...)
    rhs = sum(ndims.(inds)) == 0 ? :src : :(src[$(src_modes...)])

    quote
        ($(syms...),) = (inds...,)
        @finch begin
            $(Expr(
                :for,
                exts,
                quote
                    arr[$(coords...)] = $rhs
                end,
            ))
        end
        return src
    end
end
