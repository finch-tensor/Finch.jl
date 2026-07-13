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