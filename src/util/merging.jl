using AcceleratedKernels

Base.@propagate_inbounds function s_prefix_sum(arr::Vector{Int})
    out = Vector{Int}(undef, length(arr))
    out[1] = arr[1]
    for i in 2:length(arr)
        out[i] = out[i - 1] + arr[i]
    end
    out
end

Base.@propagate_inbounds function binary_search(target::Int, arr)
    lo = 1
    hi = length(arr)
    @assert target > 0
    @assert target < arr[hi]

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

Base.@propagate_inbounds function binary_search_scalar(target, arr)
    lo = 1
    hi = length(arr)

    while lo <= hi
        mid = div(lo + hi, 2)
        if arr[mid] == target
            return mid
        elseif arr[mid] > target
            hi = mid - 1
        else
            lo = mid + 1
        end
    end

    return -1
end

Base.@propagate_inbounds function compute_proc_cutoffs(index, P)
    cutoffs = Vector{Int}(undef, length(index) + 1)
    cutoffs[1] = 1
    for i in 2:length(cutoffs)
        cutoffs[i] = length(index[i - 1])
    end
    s_prefix_sum(cutoffs)
end

Base.@propagate_inbounds function get_permute_idx(proc_id, ptr)
    start = 0

    for i in 1:(proc_id - 1)
        start += length(ptr[i]) - 1
    end

    return start
end

Base.@propagate_inbounds function p_permute(permutation, arr::Vector{T}) where {T}
    shuffled = Vector{T}(undef, length(arr))

    @assert length(permutation) == length(arr)

    Threads.@threads for i in eachindex(permutation)
        shuffled[i] = arr[permutation[i]]
    end

    return shuffled
end