Base.@propagate_inbounds function binary_search_lb(target, arr, lo, hi)
    result = -1
    while lo <= hi
        mid = div(lo + hi, 2)
        if arr[mid] >= target
            result = mid
            hi = mid - 1
        else
            lo = mid + 1
        end
    end
    return result
end

Base.@propagate_inbounds function binary_search_ub(target, arr, lo, hi)
    result = -1
    while lo <= hi
        mid = div(lo + hi, 2)
        if arr[mid] <= target
            result = mid
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return result
end

Base.@propagate_inbounds function unwrap_dense(gfm, factor, P)
    Threads.@threads for tid in 1:P
        v = gfm[tid]
        olddim = length(v)
        resize!(v, olddim * factor)
        for i in olddim:-1:1
            val = v[i]
            base = (val - 1) * factor
            for j in factor:-1:1
                v[(i - 1) * factor + j] = base + j
            end
        end
    end
end

@inbounds function binary_search(target::Int, arr)
    lo = 1
    hi = length(arr)
    @assert target > 0

    if target >= arr[hi]
        return -1
    end

    while lo <= hi
        mid = div(lo + hi, 2)
        if arr[mid] <= target && arr[mid + 1] > target
            return mid
        elseif arr[mid] > target
            hi = mid
        else
            lo = mid
        end
    end

    return -1
end