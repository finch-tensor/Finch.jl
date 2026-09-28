# Load-balancing strategies for the normalizing Coalesce merge.
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

# Bands are inclusive bounds on index tuples, compared outermost (last) first.
isempty_band(lb, ub) = isless(reverse(ub), reverse(lb))

# A band with no indices, whatever the shape: its lower bound is past the last
# index. Tasks with empty bands skip accumulation and merging, since tuplemask
# assumes `lb <= ub`.
empty_band(shapes) =
    ((map(one, Base.front(Tuple(shapes)))..., shapes[end] + 1), Tuple(shapes))

"""
    CoalesceBand(lo, hi, lb, ub)

The part of a shard's band still to be enforced below some level. `lb` and `ub`
are the band's inclusive index bounds, innermost first; `last(lb)` bounds the
index of this level's children. Only the subtrees at the local positions `lo`
and `hi` lie on the band's lower and upper edges (0 if none); everything
between them is inside the band.
"""
struct CoalesceBand{B}
    lo::Int
    hi::Int
    lb::B
    ub::B
end

# The band below a level whose children at `lo` and `hi` stay on its edges.
coalesce_band(band::CoalesceBand, lo, hi) =
    CoalesceBand(lo, hi, Base.front(band.lb), Base.front(band.ub))
coalesce_band(::Nothing, lo, hi) = nothing

# The child position on an edge, given this level's first or last entry.
@inline coalesce_edge(edge, pos, i, bound, child) =
    (edge > 0 && pos == edge && i == bound) ? child : 0

"""
    coalesce_shards!(src, dst, P, max_pos, bands)

Merge the `P` shards of `src` into `dst`. Shards must be ordered and disjoint:
everything shard `p` stores precedes, in outermost-first index order, everything
shard `p + 1` stores. The result is their concatenation. The only overlap is at a
band boundary, where neighboring shards can both store the same parent entry;
the earlier shard owns it, and both shards' children land under it.

`bands[tid] = (lb, ub)` bounds shard `tid`'s indices (innermost first). Dense
levels store fill values outside their shard's band, so the bounds clip dense
blocks on the band's edges. With `bands = nothing`, dense blocks instead merge
by copying only non-fill values.

Merging makes two passes over the levels. `setup_coalesce!(lvl, max_pos, dst, P,
shift)` sizes `dst` and returns a plan saying where each shard's positions land
(`dst_pos = pos + shift[p]`) and whether its first entry is shared with the
previous shard. Then, in parallel, `coalesce_shard!(tid, plan, lvl, dst, runs,
band)` copies shard `tid`: `runs` are its parent positions and `band` is a
[`CoalesceBand`](@ref) or `nothing`.
"""
function coalesce_shards!(src, dst, P, max_pos, bands)
    plan = setup_coalesce!(src, max_pos, dst, P, zeros(Int, P))
    Threads.@threads for tid in 1:P
        if isnothing(bands)
            coalesce_shard!(tid, plan, src, dst, (1:max_pos,), nothing)
        elseif !isempty_band(bands[tid]...)
            band = CoalesceBand(1, max_pos, bands[tid]...)
            coalesce_shard!(tid, plan, src, dst, (1:max_pos,), band)
        end
    end
    return dst
end

# Resize `v` to `n`, clearing any new storage as assemble_level! would.
function coalesce_resize!(v, n, fill_value)
    old = length(v)
    resize!(v, n)
    n > old && fill!(view(v, (old + 1):n), fill_value)
    return v
end

@inline function coalesce_copy!(dst, dst_start, src, src_start, count)
    if count <= 16
        @inbounds for k in 0:(count - 1)
            dst[dst_start + k] = src[src_start + k]
        end
    else
        copyto!(dst, dst_start, src, src_start, count)
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
