struct MergeFast end
struct MergeNormalization end
struct MergeRandom end
struct MergeDense end

# A restartable, allocation-free iterator over consecutive runs in sorted positions.
struct CoalesceRanges{V}
    positions::V
    start::Int
    stop::Int
end

Base.IteratorSize(::Type{<:CoalesceRanges}) = Base.SizeUnknown()
Base.eltype(::Type{<:CoalesceRanges}) = UnitRange{Int}

@inline function Base.iterate(ranges::CoalesceRanges, r=ranges.start)
    r > ranges.stop && return nothing
    pos_start = pos_stop = Int(ranges.positions[r])
    r += 1
    while r <= ranges.stop && ranges.positions[r] == pos_stop + 1
        pos_stop += 1
        r += 1
    end
    return pos_start:pos_stop, r
end

# Keep the single-range interface as a one-element batch.
function coalesce_fast!(
    tid, pos_offsets, shared_flags, P, lvl, coalescent, was_dense, pos_start, pos_stop
)
    coalesce_fast!(
        tid, pos_offsets, shared_flags, P, lvl, coalescent, was_dense, (pos_start:pos_stop,)
    )
end

function coalesce_dense!(tid, pos_offsets, shared_flags, P, lvl, coalescent, pos_start, pos_stop)
    coalesce_dense!(tid, pos_offsets, shared_flags, P, lvl, coalescent, (pos_start:pos_stop,))
end

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

    if target > arr[hi]
        return -1
    end

    while lo <= hi
        mid = div(lo + hi, 2)
        if arr[mid] <= target && arr[mid + 1] > target
            return mid
        elseif arr[mid] > target
            hi = mid - 1
        else
            lo = mid + 1
        end
    end

    return -1
end


@inbounds function binary_search_offsets(target::Int, arr, lo::Int, hi::Int)
    @assert target > 0

    result = hi
    while lo <= hi
        mid = div(lo + hi, 2)
        if arr[mid + 1] >= target
            result = mid
            hi = mid - 1
        else
            lo = mid + 1
        end
    end

    return result
end

@inbounds function binary_search_first_increase(arr)
    lo = 1
    hi = length(arr)
    target = arr[lo]

    if arr[hi] == target
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

# Translate a parent range to this shard's compacted child slots.
@inline function coalesce_child_range(ptr, shift, cutoff, next_cutoff, range)
    isempty(range) && return 1:0
    local_start = clamp(first(range) - shift, 1, length(ptr))
    local_stop = clamp(last(range) - shift + 1, 1, length(ptr))
    start = cutoff + ptr[local_start] - 1
    stop = cutoff + min(ptr[local_stop] - 1, next_cutoff - cutoff) - 1
    return start:stop
end
