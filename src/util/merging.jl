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
    shard_runs(splits, offsets, rank, parent)

Order the entries of a sparse level's shards for merging. `rank(t, p)` is shard
`t`'s first entry whose parent is at least `p`, and `parent(t, r)` is the parent
of its entry `r`. Return runs `(t, lo, hi, delta)` of shard `t`'s entries `lo:hi`,
in destination order, whose parents all move by `delta`.

Each range of a shard moves as a block. A hash's shared entry keeps an arbitrary
child position, so below a hash, a range can land inside an earlier shard's
range. Cutting every range wherever any range lands leaves runs that meet only
under a parent two shards share, where band order puts the earlier shard's
entries first. A run that continues the one before it merges into it.
"""
function shard_runs(splits, offsets, rank, parent)
    P = length(splits)
    cuts = Int[]
    for t in 1:P, k in eachindex(offsets[t])
        push!(cuts, offsets[t][k], offsets[t][k] + splits[t][k + 1] - splits[t][k])
    end
    unique!(sort!(cuts))
    runs = NTuple{4,Int}[]
    for t in 1:P, k in eachindex(offsets[t])
        start, stop = splits[t][k], splits[t][k + 1]
        delta = offsets[t][k] - start
        lo = rank(t, start)
        for c in view(cuts, (searchsortedlast(cuts, start + delta) + 1):(searchsortedfirst(cuts, stop + delta) - 1))
            hi = rank(t, c - delta) - 1
            lo <= hi && push!(runs, (t, lo, hi, delta))
            lo = hi + 1
        end
        hi = rank(t, stop) - 1
        lo <= hi && push!(runs, (t, lo, hi, delta))
    end
    sort!(runs; by=((t, lo, _, delta),) -> (parent(t, lo) + delta, t))
    merged = empty(runs)
    for (t, lo, hi, delta) in runs
        if !isempty(merged) && last(merged)[1] == t && last(merged)[3] + 1 == lo &&
                last(merged)[4] == delta
            merged[end] = (t, last(merged)[2], hi, delta)
        else
            push!(merged, (t, lo, hi, delta))
        end
    end
    return merged
end

# The same ranges, for the `n` positions below each position.
scale_positions(xs, n) = [(x .- 1) .* n .+ 1 for x in xs]

"""
    setup_coalesce!(lvl, max_pos, dst, P, splits, offsets, overlap)

Allocate destination storage and plan the merge of `P` ordered shards. `max_pos`
is the destination parent-position extent. Shard `p`'s local parent positions
`splits[p][r]:(splits[p][r + 1] - 1)` land in the merged output starting at
`offsets[p][r]`; a hash permutes child positions, so a shard below one can need
several ranges. `overlap` indicates that dense leaves may overlap.

Sparse plans report `shared[p]`, the local child position whose index metadata
is already owned by an earlier shard, or `0` when all indices must be written.
`shared_dst[p]` is that child's destination position, also `0` when unshared.
These are positions, not Boolean flags or permutation ranks: a list uses an
`idx` position, a byte map uses a position stored in `srt`, and a hash uses the
`q` in its `(parent, index, q)` entry. Empty shards do not change ownership.

Skip only the shared entry's index metadata. Its children still contribute to
`shared_dst[p]`. Ordinary child positions use the child plan's offset; a hash
passes its shared position to its child as a range of its own. Count shared entries
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
splits, offsets, overlap)` sizes `dst` and returns a plan saying where each shard's
positions land, which local child position is shared
(`shared[p]`, or `0`), its destination (`shared_dst[p]`), and whether shards'
leaves can overlap below. Then, in
parallel, the ranges in `plan.init` are initialized. After initialization finishes,
`coalesce_shard!(tid, plan, lvl, dst, runs)` copies each shard in parallel.
`runs` iterates ranges of leaf positions (positions at the Element level) under
which the shard stores values. Every worker must reach every level, even with an
empty shard: worker `tid` also inserts a hash's output buckets `tid:P:B`.
"""
function coalesce_shards!(src, dst, P, max_pos, bands)
    plan = setup_coalesce!(
        src, max_pos, dst, P, [[1, max_pos + 1] for _ in 1:P], [[1] for _ in 1:P],
        isnothing(bands),
    )
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
