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
