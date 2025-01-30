module StatisticsExt

using Finch
using Finch: LazyTensor
using Finch: AbstractTensor
using Finch: AbstractTensorOrBroadcast

isdefined(Base, :get_extension) ? (using Statistics) : (using ..Statistics)

"""
"""
    Statistics.mean(tns:: LazyTensor; dims=:) = _mean(identity, tns, dims)

"""
"""
    Statistics.mean(f, tns:: LazyTensor; dims=:) = _mean(f, tns, dims)

"""
"""
    Statistics.mean(tns:: AbstractTensorOrBroadcast; dims=:) = compute(_mean(identity, lazy(tns), dims))

"""
"""
    Statistics.mean(f, tns:: AbstractTensorOrBroadcast; dims=:) = compute(_mean(f, lazy(tns), dims))

function _accumulate(f, tns:: LazyTensor{T, N}, dims=:) where {T, N}
    dims = dims == Colon() ? (1:N) : collect(dims)
    n = mapreduce(i -> tns.shape[i], *, unique(dims); init=1)
    result = reduce(+, f.(tns); dims=dims, init=0)
    return (result, n)
end

function _mean(f, tns:: LazyTensor{T, N}, dims=:) where {T, N}
    result, count = _accumulate(f, tns, dims)
    return result ./ count
end


"""
"""
    Statistics.varm(tns:: LazyTensor, m; corrected::Bool=true, dims=:) = _varm(tns, m, corrected, dims)

"""
"""
    Statistics.varm(tns:: AbstractTensorOrBroadcast, m; corrected::Bool=true, dims=:) = compute(_varm(lazy(tns), m, corrected, dims))


function _varm(tns::LazyTensor{T, N}, m, corrected::Bool=true, dims=:) where {T, N}
    result, count = _accumulate(x -> abs2(x - m), tns, dims)
    return result ./ (count - Int(corrected))
end

"""
"""
    Statistics.var(tns:: LazyTensor; corrected::Bool=true, mean=nothing, dims=:) = _var(tns, corrected, mean, dims)
"""
"""
    Statistics.var(tns:: AbstractTensorOrBroadcast; corrected::Bool=true, mean=nothing, dims=:) = compute(_var(lazy(tns), corrected, mean, dims))

function _var(tns::LazyTensor, corrected::Bool, mean, dims)
  if mean === nothing
      mean = compute(Statistics.mean(tns; dims=dims)).lvl.val[1] #TODO: Find a less hacky solution
  end
  return varm(tns, mean; corrected=corrected, dims=dims)
end

"""
"""
    Statistics.std(tns::Union{LazyTensor, AbstractTensorOrBroadcast}; corrected::Bool=true, mean=nothing, dims=:) =
        sqrt.(var(tns; corrected=corrected, mean=mean, dims=dims))

end
