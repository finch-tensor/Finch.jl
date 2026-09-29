# Load-balancing strategies for the normalizing Coalesce merge.
struct MergeRandom end
struct MergeDense end

# Bands are inclusive bounds on index tuples, compared outermost (last) first.
isempty_band(lb, ub) = isless(reverse(ub), reverse(lb))

# A band with no indices, whatever the shape: its lower bound is past the last
# index. Tasks with empty bands skip accumulation, since tuplemask assumes
# `lb <= ub`.
empty_band(shapes) =
    ((map(one, Base.front(Tuple(shapes)))..., shapes[end] + 1), Tuple(shapes))

# The column-major flat indices between two index tuples.
function band_range(lb, ub, shapes)
    isempty_band(lb, ub) && return 1:0
    flat = LinearIndices(Tuple(shapes))
    return flat[lb...]:flat[ub...]
end

# The number of leaf positions under each position of `lvl`'s parent.
coalesce_leaves(lvl) = prod(level_size(lvl))

"""
    setup_coalesce!(lvl, max_pos, dst, P, shift, overlap)

Allocate destination storage and plan the merge of `P` ordered shards. `max_pos`
is the destination parent-position extent, `shift[p]` translates shard `p`'s
parent positions, and `overlap` indicates that dense leaves may overlap.

Sparse plans report `shared[p]`, the local child position whose index metadata
is already owned by an earlier shard, or `0` when all indices must be written.
`shared_dst[p]` is that child's destination position, also `0` when unshared.
These are positions, not Boolean flags or permutation ranks: a list uses an
`idx` position, a byte map uses a position stored in `srt`, and a hash uses the
`q` in its `(parent, index, q)` entry. Empty shards do not change ownership.

Skip only the shared entry's index metadata. Its children still contribute to
`shared_dst[p]`. Ordinary child positions use the child plan's offset; a shared
position overrides that offset when necessary. Count shared entries with
`shared[p] != 0`, never by subtracting the position itself.

`off[p]` counts earlier shards' owned index entries, and `nnz` counts all owned
entries. Dense and element plans have no sparse index ownership fields; dense
plans delegate through `child`, and element plans use `overlap` when copying.
"""
function setup_coalesce! end

"""
    coalesce_shards!(src, dst, P, max_pos, bands)

Merge the `P` shards of `src` into `dst`. Shards must be ordered and disjoint:
everything shard `p` stores precedes, in outermost-first index order, everything
shard `p + 1` stores. The result is their concatenation. The only overlap is at a
band boundary, where neighboring shards can both store the same parent entry;
the earlier shard owns it, and both shards' children land under it. Dense blocks
can overlap too, since every shard stores fill values outside its band, so
values merge by copying only non-fill values into a destination of fill.

`bands[tid]` is the range of flat (column-major) indices shard `tid` holds, or
`bands` is `nothing` if unknown. Bands keep each shard from scanning, and
conditionally copying, the dense storage outside its band.

Merging makes two passes over the levels. `setup_coalesce!(lvl, max_pos, dst, P,
shift, overlap)` sizes `dst` and returns a plan saying where each shard's
positions land (`dst_pos = pos + shift[p]`), which local child position is shared
(`shared[p]`, or `0`), its destination (`shared_dst[p]`), and whether shards'
leaves can overlap below. Then, in
parallel, `coalesce_shard!(tid, plan, lvl, dst, runs)` copies shard `tid`.
`runs` iterates ranges of leaf positions (positions at the Element level) under
which the shard stores values.
"""
function coalesce_shards!(src, dst, P, max_pos, bands)
    plan = setup_coalesce!(src, max_pos, dst, P, zeros(Int, P), isnothing(bands))
    Threads.@threads for tid in 1:P
        runs = isnothing(bands) ? (1:(max_pos * coalesce_leaves(src))) : bands[tid]
        coalesce_shard!(tid, plan, src, dst, (runs,))
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
