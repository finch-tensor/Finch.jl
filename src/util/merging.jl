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
    ShardShift(offset, src, dst, len)

A shard's parent-position map below a hash: add `offset`, except the `len`
positions starting at `src` move to `dst`. A hash's shared entry keeps an
arbitrary child position, so its children map to the earlier owner's position
instead of following the shard's offset. Levels apply it like an integer shift
(`pos + shift[p]`, and `shift[p] * shape` for the positions below a parent).
"""
struct ShardShift
    offset::Int
    src::Int
    dst::Int
    len::Int
end

Base.:+(pos::Integer, s::ShardShift) =
    s.src <= pos < s.src + s.len ? pos - s.src + s.dst : pos + s.offset
Base.:*(s::ShardShift, n::Integer) =
    ShardShift(s.offset * n, (s.src - 1) * n + 1, (s.dst - 1) * n + 1, s.len * n)

shard_offset(s::Integer) = s
shard_offset(s::ShardShift) = s.offset

"""
    setup_coalesce!(lvl, max_pos, dst, P, shift, overlap)

Allocate destination storage and plan the merge of `P` ordered shards. `max_pos`
is the destination parent-position extent, `shift[p]` translates shard `p`'s
parent positions (an integer, or a `ShardShift` below a hash), and `overlap`
indicates that dense leaves may overlap.

Sparse plans report `shared[p]`, the local child position whose index metadata
is already owned by an earlier shard, or `0` when all indices must be written.
`shared_dst[p]` is that child's destination position, also `0` when unshared.
These are positions, not Boolean flags or permutation ranks: a list uses an
`idx` position, a byte map uses a position stored in `srt`, and a hash uses the
`q` in its `(parent, index, q)` entry. Empty shards do not change ownership.

Skip only the shared entry's index metadata. Its children still contribute to
`shared_dst[p]`. Ordinary child positions use the child plan's offset; a hash
passes its shared position to its child as a `ShardShift`. Count shared entries
with `shared[p] != 0`, never by subtracting the position itself.

`nnz` counts all owned entries, and list plans also report `off[p]`, the owned
index entries of earlier shards. Dense and element plans have no sparse index ownership fields; dense
plans delegate through `child`, and element plans use `overlap` when copying.

`init` lists `(buffer, start, value)` ranges, including child storage. Setup
allocates serially; initialize these ranges before calling `coalesce_shard!`.
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

Merging first plans storage. `setup_coalesce!(lvl, max_pos, dst, P,
shift, overlap)` sizes `dst` and returns a plan saying where each shard's
positions land (`dst_pos = pos + shift[p]`), which local child position is shared
(`shared[p]`, or `0`), its destination (`shared_dst[p]`), and whether shards'
leaves can overlap below. Then, in
parallel, the ranges in `plan.init` are initialized. After initialization finishes,
`coalesce_shard!(tid, plan, lvl, dst, runs)` copies each shard in parallel.
`runs` iterates ranges of leaf positions (positions at the Element level) under
which the shard stores values. Every worker must reach every level, even with an
empty shard: worker `tid` also inserts a hash's output buckets `tid:P:B`.
"""
function coalesce_shards!(src, dst, P, max_pos, bands)
    plan = setup_coalesce!(src, max_pos, dst, P, zeros(Int, P), isnothing(bands))
    if any(init -> init[2] <= length(init[1]), plan.init)
        Threads.@threads for tid in 1:P
            for (buffer, start, value) in plan.init
                n = length(buffer) - start + 1
                lo = start + fld((tid - 1) * n, P)
                hi = start + fld(tid * n, P) - 1
                fill!(view(buffer, lo:hi), value)
            end
        end
    end
    # Initialization must finish before another shard writes shared children.
    Threads.@threads for tid in 1:P
        runs = isnothing(bands) ? (1:(max_pos * coalesce_leaves(src))) : bands[tid]
        coalesce_shard!(tid, plan, src, dst, (runs,))
    end
    return dst
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
