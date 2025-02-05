@staged function copyto_helper!(dst, src)
    ndims(dst) > ndims(src) &&
        throw(DimensionMismatch("more dimensions in destination than source"))
    ndims(dst) < ndims(src) &&
        throw(DimensionMismatch("less dimensions in destination than source"))
    idxs = [Symbol(:i_, n) for n in 1:ndims(dst)]
    exts = Expr(:block, (:($idx = _) for idx in reverse(idxs))...)
    return quote
        @finch mode = :fast begin
            dst .= $(fill_value(dst))
            $(Expr(
                :for,
                exts,
                quote
                    dst[$(idxs...)] = src[$(idxs...)]
                end,
            ))
        end
        return dst
    end
end

Base.copyto!(dst::AbstractTensor, src::AbstractTensor) =
    copyto_helper!(dst, src)

Base.copyto!(dst::AbstractTensor, src::AbstractArray) =
    copyto_helper!(dst, src)

Base.copyto!(dst::AbstractArray, src::AbstractTensor) =
    copyto_helper!(dst, src)

function copyto_swizzled!(dst, src, perm)
    if issorted(perm)
        return copyto_helper!(dst, src)
    else
        tmp = rep_construct(permutedims_rep(data_rep(src), perm))
        tmp = copyto_helper!(swizzle(tmp, invperm(perm)...), src)
        return copyto_helper!(dst, tmp.body)
    end
end

function Base.copyto!(dst::AbstractArray, src::SwizzleArray{dims}) where {dims}
    copyto_swizzled!(dst, src.body, dims)
end

function Base.copyto!(dst::AbstractTensor, src::SwizzleArray{dims}) where {dims}
    copyto_swizzled!(dst, src.body, dims)
end

function Base.copyto!(dst::SwizzleArray{dims}, src::SwizzleArray{dims2}) where {dims,dims2}
    swizzle(copyto!(dst.body, swizzle(src, invperm(dims)...)), dims...)
end

function Base.copyto!(dst::SwizzleArray{dims}, src::AbstractTensor) where {dims}
    swizzle(copyto!(dst.body, swizzle(src, invperm(dims)...)), dims...)
end

function Base.copyto!(dst::SwizzleArray{dims}, src::AbstractArray) where {dims}
    swizzle(copyto!(dst.body, swizzle(src, invperm(dims)...)), dims...)
end

function Base.permutedims(src::AbstractTensor)
    @assert ndims(src) == 2
    permutedims(src, (2, 1))
end

function Base.permutedims(src::AbstractTensor, perm)
    dst = rep_construct(permutedims_rep(data_rep(src), perm))
    dst = copyto_helper!(swizzle(dst, invperm(perm)...), src).body
end

"""
    dropfills(src)

Drop the fill values from `src` and return a new tensor with the same shape and
format.
"""
dropfills(src) = dropfills!(similar(src), src)

@staged function dropfills_helper!(dst, src)
    ndims(dst) > ndims(src) &&
        throw(DimensionMismatch("more dimensions in destination than source"))
    ndims(dst) < ndims(src) &&
        throw(DimensionMismatch("less dimensions in destination than source"))
    idxs = [Symbol(:i_, n) for n in 1:ndims(dst)]
    exts = Expr(:block, (:($idx = _) for idx in reverse(idxs))...)
    T = eltype(dst)
    d = fill_value(dst)
    return quote
        @finch mode = :fast begin
            dst .= $(fill_value(dst))
            $(Expr(
                :for,
                exts,
                quote
                    let tmp = src[$(idxs...)]
                        if !isequal(tmp, $d)
                            dst[$(idxs...)] = tmp
                        end
                    end
                end,
            ))
        end
        return dst
    end
end

"""
    dropfills!(dst, src)

Copy only the non-fill values from `src` into `dst`.
"""
dropfills!(dst::AbstractTensor, src::AbstractTensor) =
    dropfills_helper!(dst, src)

dropfills!(dst::AbstractTensor, src::AbstractArray) =
    dropfills_helper!(dst, src)

dropfills!(dst::AbstractArray, src::AbstractTensor) =
    dropfills_helper!(dst, src)

function dropfills_swizzled!(dst, src, perm)
    if issorted(perm)
        return dropfills_helper!(dst, src)
    else
        tmp = rep_construct(permutedims_rep(data_rep(src), perm))
        tmp = dropfills_helper!(swizzle(tmp, invperm(perm)...), src)
        return copyto_helper!(dst, tmp.body)
    end
end

function dropfills!(dst::AbstractArray, src::SwizzleArray{dims}) where {dims}
    dropfills_swizzled!(dst, src.body, dims)
end

function dropfills!(dst::AbstractTensor, src::SwizzleArray{dims}) where {dims}
    dropfills_swizzled!(dst, src.body, dims)
end

function dropfills!(dst::SwizzleArray{dims}, src::SwizzleArray{dims2}) where {dims,dims2}
    swizzle(dropfills!(dst.body, swizzle(src, invperm(dims)...)), dims...)
end

function dropfills!(dst::SwizzleArray{dims}, src::AbstractTensor) where {dims}
    swizzle(dropfills!(dst.body, swizzle(src, invperm(dims)...)), dims...)
end

function dropfills!(dst::SwizzleArray{dims}, src::AbstractArray) where {dims}
    swizzle(dropfills!(dst.body, swizzle(src, invperm(dims)...)), dims...)
end
