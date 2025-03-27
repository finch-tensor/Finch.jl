@staged function helper_equal(A, B)
    idxs = [Symbol(:i_, n) for n in 1:ndims(A)]
    exts = Expr(:block, (:($idx = _) for idx in reverse(idxs))...)
    return quote
        size(A) == size(B) || return false
        check = Scalar(true)
        @finch $(Expr(
            :for,
            exts,
            quote
                check[] &= (A[$(idxs...)] == B[$(idxs...)])
            end,
        ))
        return check[]
    end
end

function Base.:(==)(A::AbstractTensor, B::AbstractTensor)
    return helper_equal(A, B)
end

function Base.:(==)(A::AbstractTensor, B::AbstractArray)
    return helper_equal(A, B)
end

function Base.:(==)(A::AbstractArray, B::AbstractTensor)
    return helper_equal(A, B)
end

@staged function helper_isequal(A, B)
    idxs = [Symbol(:i_, n) for n in 1:ndims(A)]
    exts = Expr(:block, (:($idx = _) for idx in reverse(idxs))...)
    return quote
        size(A) == size(B) || return false
        check = Scalar(true)
        @finch $(Expr(
            :for,
            exts,
            quote
                check[] &= isequal(A[$(idxs...)], B[$(idxs...)])
            end,
        ))
        return check[]
    end
end

function Base.isequal(A::AbstractTensor, B::AbstractTensor)
    return helper_isequal(A, B)
end

function Base.isequal(A::AbstractTensor, B::AbstractArray)
    return helper_isequal(A, B)
end

function Base.isequal(A::AbstractArray, B::AbstractTensor)
    return helper_isequal(A, B)
end

function helper_argmin(A, dims)
    # @assert 1 <= max(dims) <= ndims(A) && 1 <= min(dims) <= ndims(A)
    if (ndims(A) >= 2)
        return map(x -> x[2], reduce(minby, map(Pair, A, CartesianIndices(size(A))), dims=dims, init=Inf=>CartesianIndex(fill(0, length(size(A)))...)))
    else
        return map(x -> x[2], reduce(minby, map(Pair, A, 1:size(A)[1]), dims=dims, init=Inf=>0))
    end
end

function argmin_python(A; dims, keepdims)
    a = helper_argmin(A, dims)
    if (keepdims)
        return expanddims(a, dims)
    else
        return a
    end
end

function helper_argmin(A)
    return helper_argmin(A, (1:ndims(A)...,))
end

function Base.argmin(A::AbstractTensor; dims)
    return helper_argmin(A, dims)
end

function Base.argmin(A::AbstractArray; dims)
    return helper_argmin(A, dims)
end

function Base.argmin(A::AbstractTensor)
    return helper_argmin(A)
end

function Base.argmin(A::AbstractArray)
    return helper_argmin(A)
end

function helper_argmax(A, dims)
    if (ndims(A) >= 2)
        return map(x -> x[2], reduce(maxby, map(Pair, A, CartesianIndices(size(A))), dims=dims, init=-Inf=>CartesianIndex(fill(0, length(size(A)))...)))
    else
        return map(x -> x[2], reduce(maxby, map(Pair, A, 1:size(A)[1]), dims=dims, init=-Inf=>0))
    end
end

function argmax_python(A; dims, keepdims)
    a = helper_argmax(A, dims)
    if (keepdims)
        return expanddims(a, dims)
    else
        return a
    end
end

function helper_argmax(A)
    return helper_argmax(A, (1:ndims(A)...,))
end

function Base.argmax(A::AbstractTensor; dims)
    return helper_argmax(A, dims)
end

function Base.argmax(A::AbstractArray; dims)
    return helper_argmax(A, dims)
end

function Base.argmax(A::AbstractTensor)
    return helper_argmax(A)
end

function Base.argmax(A::AbstractArray)
    return helper_argmax(A)
end
