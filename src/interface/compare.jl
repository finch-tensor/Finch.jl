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

@staged function helper_argmin(A, dim)
   idxs = [Symbol(:i_, n) for n in 1:ndims(A)]
   exts = Expr(:block, (:($idx = _) for idx in reverse(idxs))...)
   return quote
       x = [Scalar(Inf => CartesianIndex(zeros(Int, ndims(A))...)) for _ in 1:ndims(A)] # for _ in 1:ndims(A)
       @finch $(Expr(
           :for,
           exts,
           quote
               x[1][] <<minby>>= A[$(idxs...)] => CartesianIndex($(idxs...))
               # if x[] < A[$(idxs...)] 
               #      x[] = A[$(idxs...)] => CartesianIndex($(idxs...))
               # end
           end,
       ))
       return x
   end
end

function Base.argmin(A::AbstractTensor, dim::int)
    return helper_argmin(A, dim)
end

function Base.argmin(A::AbstractArray, dim::int)
    return helper_argmin(A, dim)
end

@staged function helper_argmax(A)
   idxs = [Symbol(:i_, n) for n in 1:ndims(A)]
   exts = Expr(:block, (:($idx = _) for idx in reverse(idxs))...)
   return quote
       x = Scalar(-Inf => CartesianIndex(zeros(Int, ndims(A))...))
       @finch $(Expr(
           :for,
           exts,
           quote
               x[] <<maxby>>= A[$(idxs...)] => CartesianIndex($(idxs...))
           end,
       ))
       return x
   end
end

function Base.argmax(A::AbstractTensor, dim::int)
    return helper_argmax(A, dim)
end

function Base.argmax(A::AbstractArray, dim::int)
    return helper_argmax(A, dim)
end
