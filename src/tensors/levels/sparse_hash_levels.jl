"""
    SparseHashLevel{[Ti=Int]}(lvl, [dim], [subtables=1])

A subfiber of a sparse level does not need to represent slices `A[:, ..., :, i]`
which are entirely [`fill_value`](@ref). Instead, only potentially non-fill
slices are stored as subfibers in `lvl`. A hash table records which slices are
stored. Optionally, `dim` is the size of the last dimension, and `subtables`, a
power of two, splits the table into buckets that parallel merges fill
independently.

`Ti` is the type of the last fiber index. Up to 127 overlapping logical writers
can share a tentative entry within one task.

Implementation invariants:

* An entry is keyed by `(p, i)`, a parent position and an index, and owns a
  stable child position `q`. Its record is `key[q] == (p, i, state)`: `0x00`
  means free, `0x01:0x7f` count pending writers, and `0x80` means retained.
  Exceeding 127 pending writers per entry throws an error.
* `tbl_ctrl` and `tbl` form a linear-probing table, split into `subtables`
  contiguous buckets. Every occupied slot, tentative or retained, has the high
  bit of `tbl_ctrl[h]` set, seven hash fingerprint bits below it, and `tbl[h] == q`.
  `0x00` marks an empty slot, which ends a probe. Probes wrap within their bucket.
* Keys hash as `x = a * p + hash(i)`, where `a` is a random odd multiplier shared
  by all hash levels. The low `log2(subtables)` bits of `x` pick the bucket, so
  shifting parents by `delta` rotates buckets by `a * delta`. `x` is linear in
  `p`, and linear probing clusters on linear hashes of structured parents, so the
  rest comes from `y = hash(x)`: its low seven bits are the fingerprint, and the
  bits above them pick the starting slot.
* `tbl_count[b]` counts the entries in bucket `b`, plus keys still pending during
  assembly, which grows the table before any bucket is more than half full.
* Frozen, `length(key)` is the extent of child positions, `ptr[p]:(ptr[p + 1] - 1)`
  indexes `perm`, and `perm[r]` is a child position. Each parent's range is
  sorted by index.
* While assembling, `length(key)` is the child capacity. Only positions through
  the allocated child extent are initialized. A new key gets its child position
  and is inserted tentatively before its child is written. The first dirty
  writer marks the child record retained; later writers cannot discard it.
  Otherwise, the last writer backward-shift deletes the table entry, marks the
  record free, and returns its position to `pool`. Counts stay at stable child
  positions across growth and deletion; only final deletion needs a slot lookup.
* Rehash and freeze scan the child records directly, skipping free positions.
  Coalesce workers initialize holes in their own child ranges. Freeze trims free
  tails and preserves interior positions in `pool` for reuse after thaw.

```jldoctest
julia> tensor_tree(Tensor(Dense(SparseHash(Element(0.0))), [10 0 20; 30 0 0; 0 0 40]))
3×3-Tensor
└─ Dense [:,1:3]
   ├─ [:, 1]: SparseHash (0.0) [1:3]
   │  ├─ [1]: 10.0
   │  └─ [2]: 30.0
   ├─ [:, 2]: SparseHash (0.0) [1:3]
   └─ [:, 3]: SparseHash (0.0) [1:3]
      ├─ [1]: 20.0
      └─ [3]: 40.0

julia> tensor_tree(Tensor(SparseHash(SparseHash(Element(0.0))), [10 0 20; 30 0 0; 0 0 40]))
3×3-Tensor
└─ SparseHash (0.0) [:,1:3]
   ├─ [:, 1]: SparseHash (0.0) [1:3]
   │  ├─ [1]: 10.0
   │  └─ [2]: 30.0
   └─ [:, 3]: SparseHash (0.0) [1:3]
      ├─ [1]: 20.0
      └─ [3]: 40.0

```
"""
struct SparseHashLevel{Ti,Ptr,TblCtrl,Tbl,Key,Pool,Perm,TblCount,Lvl} <: AbstractLevel
    lvl::Lvl
    shape::Ti
    subtables::Int
    ptr::Ptr
    tbl_ctrl::TblCtrl
    tbl::Tbl
    key::Key
    pool::Pool
    perm::Perm
    tbl_count::TblCount
end
const SparseHash = SparseHashLevel

# The buffers a hash level carries, in constructor order after `subtables`.
const SPARSE_HASH_BUFFERS = (:ptr, :tbl_ctrl, :tbl, :key, :pool, :perm, :tbl_count)

SparseHashLevel(lvl, args...) = SparseHashLevel{Int}(lvl, args...)
SparseHashLevel(lvl, shape::Ti, args...) where {Ti} = SparseHashLevel{Ti}(lvl, shape, args...)
function SparseHashLevel{Ti}(lvl, shape=zero(Ti), subtables=1) where {Ti}
    sparse_hash_check_subtables(subtables)
    Tp = postype(lvl)
    SparseHashLevel{Ti}(
        lvl, shape, subtables, Tp[1], UInt8[], Tp[], Tuple{Tp,Ti,UInt8}[], Tp[], Tp[],
        zeros(Int, subtables),
    )
end
function SparseHashLevel{Ti}(
    lvl::Lvl, shape, subtables, ptr::Ptr, tbl_ctrl::TblCtrl, tbl::Tbl, key::Key,
    pool::Pool, perm::Perm, tbl_count::TblCount,
) where {Ti,Ptr,TblCtrl,Tbl,Key,Pool,Perm,TblCount,Lvl}
    sparse_hash_check_subtables(subtables)
    SparseHashLevel{Ti,Ptr,TblCtrl,Tbl,Key,Pool,Perm,TblCount,Lvl}(
        lvl, Ti(shape), Int(subtables), ptr, tbl_ctrl, tbl, key, pool, perm, tbl_count
    )
end

# `lvl` with a new child and shape, keeping its table.
function sparse_hash_with(lvl::SparseHashLevel{Ti}, child, shape=lvl.shape) where {Ti}
    SparseHashLevel{Ti}(
        child, shape, lvl.subtables, (getfield(lvl, f) for f in SPARSE_HASH_BUFFERS)...
    )
end

const SPARSE_HASH_CTRL_EMPTY = 0x00
const SPARSE_HASH_CTRL_FULL = 0x80
const SPARSE_HASH_KEY_FREE = 0x00
const SPARSE_HASH_KEY_RETAINED = 0x80
# Sample once and keep it fixed, including across shards and table resizes.
const SPARSE_HASH_POS_MULTIPLIER = rand(UInt) | one(UInt)

function sparse_hash_check_subtables(subtables)
    if !(subtables isa Integer) || subtables < 1 || !ispow2(subtables)
        throw(ArgumentError("SparseHash subtables must be a positive power of two"))
    end
end

# Slots for a bucket of `n` entries, keeping it at most half full.
sparse_hash_table_capacity(n) = nextpow(2, max(4, 2n))

@inline sparse_hash_hash(p, i) = SPARSE_HASH_POS_MULTIPLIER * (p % UInt) + hash(i)
@inline sparse_hash_hash_subtable(x::UInt, subtables) = Int(x & UInt(subtables - 1)) + 1
@inline sparse_hash_hash_ctrl(x::UInt) = SPARSE_HASH_CTRL_FULL | (hash(x) % UInt8 & 0x7f)
# The bucket's first slot, the probe's starting offset within the bucket, and
# the offset mask for a table of `n` slots. The offset skips the fingerprint bits.
@inline function sparse_hash_hash_slot_parts(x::UInt, n, subtables)
    mask = (n >>> trailing_zeros(subtables)) - 1
    base = Int(x & UInt(subtables - 1)) * (mask + 1) + 1
    return base, Int((hash(x) >>> 7) & UInt(mask)), mask
end

# The slot holding key `(p, i)`, which hashes to `x`, or else the empty slot that
# ends its probe.
@inline function sparse_hash_find(tbl_ctrl, tbl, key, p, i, x, subtables)
    ctrl = sparse_hash_hash_ctrl(x)
    base, off, mask = sparse_hash_hash_slot_parts(x, length(tbl_ctrl), subtables)
    @inbounds while true
        c = tbl_ctrl[base + off]
        c == SPARSE_HASH_CTRL_EMPTY && return base + off
        if c == ctrl
            k = key[tbl[base + off]]
            k[1] == p && k[2] == i && return base + off
        end
        off = (off + 1) & mask
    end
end

# The first empty slot on `x`'s probe, where a key known to be absent goes.
@inline function sparse_hash_vacancy(tbl_ctrl, x, subtables)
    base, off, mask = sparse_hash_hash_slot_parts(x, length(tbl_ctrl), subtables)
    @inbounds while tbl_ctrl[base + off] != SPARSE_HASH_CTRL_EMPTY
        off = (off + 1) & mask
    end
    return base + off
end

# The child position of key `(p, i)`, or zero if it is absent.
@inline function sparse_hash_lookup(tbl_ctrl, tbl, key, p, i, subtables)
    isempty(tbl_ctrl) && return zero(eltype(tbl))
    h = sparse_hash_find(tbl_ctrl, tbl, key, p, i, sparse_hash_hash(p, i), subtables)
    @inbounds return tbl_ctrl[h] == SPARSE_HASH_CTRL_EMPTY ? zero(eltype(tbl)) : tbl[h]
end

# Child records hold both liveness and writer state, so every rebuild scans
# them sequentially. The unused capacity after `qos_stop` is never read.
function sparse_hash_resize!(tbl_ctrl, tbl, key, cap, subtables, qos_stop=length(key))
    empty!(tbl_ctrl)
    resize!(tbl_ctrl, cap)
    fill!(tbl_ctrl, SPARSE_HASH_CTRL_EMPTY)
    empty!(tbl)
    resize!(tbl, cap)
    @inbounds for q in 1:qos_stop
        p, i, state = key[q]
        state == SPARSE_HASH_KEY_FREE && continue
        x = sparse_hash_hash(p, i)
        h = sparse_hash_vacancy(tbl_ctrl, x, subtables)
        tbl_ctrl[h] = sparse_hash_hash_ctrl(x)
        tbl[h] = q
    end
end

# Order the entries for reading: `perm[ptr[p]:(ptr[p + 1] - 1)]` lists parent `p`'s
# child positions by index. Returns the live child extent and trims `key` and
# pooled positions beyond it, so an empty hash also has an empty child.
function sparse_hash_freeze!(ptr, perm, key, pool, pos_stop, qos_stop=length(key))
    n = count(q -> key[q][3] != SPARSE_HASH_KEY_FREE, 1:qos_stop)
    qs = Vector{eltype(perm)}(undef, n)
    ps = Vector{fieldtype(eltype(key), 1)}(undef, n)
    is = Vector{fieldtype(eltype(key), 2)}(undef, n)
    # Collect live child positions, and count parent p's entries in
    # ptr[p + 2], so the prefix sum leaves ptr[p + 1] at parent p's start.
    resize!(ptr, pos_stop + 1)
    fill!(ptr, 0)
    ptr[1] = 1
    k = 0
    @inbounds for q in 1:qos_stop
        p, i, state = key[q]
        state == SPARSE_HASH_KEY_FREE && continue
        k += 1
        qs[k] = q
        ps[k], is[k] = p, i
        ps[k] < pos_stop && (ptr[ps[k] + 2] += 1)
    end
    @inbounds for p in 2:length(ptr)
        ptr[p] += ptr[p - 1]
    end
    # Placing entries in index order leaves each parent's run sorted, and
    # advances ptr[p + 1] from parent p's start to its stop.
    resize!(perm, length(qs))
    @inbounds for k in sortperm(is)
        perm[ptr[ps[k] + 1]] = qs[k]
        ptr[ps[k] + 1] += 1
    end
    extent = isempty(qs) ? 0 : last(qs)
    filter!(q -> q <= extent, pool)
    resize!(key, extent)
    return extent
end

# Join a tentative entry by its stable child position.
@inline function sparse_hash_share!(key, q)
    @inbounds p, i, state = key[q]
    state < SPARSE_HASH_KEY_RETAINED - 0x01 ||
        error("SparseHash supports at most 127 pending writers per entry")
    @inbounds key[q] = (p, i, state + 0x01)
    return nothing
end

# Repair a linear-probing bucket after deleting h. Only slots move: child
# positions, including those held by other unfinished writers, remain stable.
@inline function sparse_hash_delete!(tbl_ctrl, tbl, key, h, x, subtables)
    base, _, mask = sparse_hash_hash_slot_parts(x, length(tbl_ctrl), subtables)
    hole = h - base
    scan = (hole + 1) & mask
    @inbounds while tbl_ctrl[base + scan] != SPARSE_HASH_CTRL_EMPTY
        q = tbl[base + scan]
        p, i, _ = key[q]
        _, home, _ = sparse_hash_hash_slot_parts(
            sparse_hash_hash(p, i), length(tbl_ctrl), subtables
        )
        if ((hole - home) & mask) < ((scan - home) & mask)
            tbl_ctrl[base + hole] = tbl_ctrl[base + scan]
            tbl[base + hole] = q
            hole = scan
        end
        # An entry at home does not end a cluster in ordinary linear probing.
        scan = (scan + 1) & mask
    end
    @inbounds tbl_ctrl[base + hole] = SPARSE_HASH_CTRL_EMPTY
    return nothing
end

# Finish a tentative access through its stable child position. Only deleting
# the last unretained writer needs to locate a table slot.
@inline function sparse_hash_release!(tbl_ctrl, tbl, key, q, x, subtables, dirty)
    @inbounds begin
        p, i, state = key[q]
        state == SPARSE_HASH_KEY_RETAINED && return false
        if dirty
            key[q] = (p, i, SPARSE_HASH_KEY_RETAINED)
        elseif state > 0x01
            key[q] = (p, i, state - 0x01)
        else
            h = sparse_hash_find(tbl_ctrl, tbl, key, p, i, x, subtables)
            sparse_hash_delete!(tbl_ctrl, tbl, key, h, x, subtables)
            key[q] = (p, i, SPARSE_HASH_KEY_FREE)
            return true
        end
    end
    return false
end

# The first rank in `lo:hi` whose index is at least `x`, galloping from `lo`.
Base.@propagate_inbounds function sparse_hash_scansearch(key, perm, x, lo, hi)
    d = one(lo)
    p = lo
    while p < hi && key[perm[p]][2] < x
        d <<= 0x01
        p += d
    end
    lo = p - d
    hi = min(p, hi) + one(lo)
    while lo < hi - one(lo)
        m = lo + ((hi - lo) >>> 0x01)
        key[perm[m]][2] < x ? (lo = m) : (hi = m)
    end
    return hi
end

Base.summary(lvl::SparseHashLevel) = "SparseHash($(summary(lvl.lvl)))"
function similar_level(
    lvl::SparseHashLevel{Ti}, fill_value, eltype::Type, dim, tail...
) where {Ti}
    SparseHashLevel{Ti}(
        similar_level(lvl.lvl, fill_value, eltype, tail...), dim, lvl.subtables
    )
end

coalesce_similar_level(lvl, P) =
    similar_level(lvl, level_fill_value(typeof(lvl)), level_eltype(typeof(lvl)), level_size(lvl)...)
function coalesce_similar_level(lvl::SparseHashLevel{Ti}, P) where {Ti}
    P > 0 || throw(ArgumentError("Coalesce worker count must be positive"))
    SparseHashLevel{Ti}(
        coalesce_similar_level(lvl.lvl, P), lvl.shape, nextpow(2, P)
    )
end

postype(T::Type{<:SparseHashLevel}) = postype(fieldtype(T, :lvl))
Base.resize!(lvl::SparseHashLevel, dims...) =
    sparse_hash_with(lvl, resize!(lvl.lvl, dims[1:(end - 1)]...), dims[end])
pattern!(lvl::SparseHashLevel) = sparse_hash_with(lvl, pattern!(lvl.lvl))
set_fill_value!(lvl::SparseHashLevel, init) =
    sparse_hash_with(lvl, set_fill_value!(lvl.lvl, init))
function transfer(Tm, lvl::SparseHashLevel{Ti}) where {Ti}
    SparseHashLevel{Ti}(
        transfer(Tm, lvl.lvl), lvl.shape, lvl.subtables,
        (transfer(Tm, getfield(lvl, f)) for f in SPARSE_HASH_BUFFERS)...,
    )
end

function countstored_level_at(lvl, pos)
    countstored_level(lvl, pos) - countstored_level(lvl, pos - 1)
end
# The entries under parents `1:pos`, then everything their children store.
function countstored_level(lvl::SparseHashLevel, pos)
    pos == 0 && return countstored_level(lvl.lvl, pos)
    return sum(
        r -> countstored_level_at(lvl.lvl, lvl.perm[r]), 1:(lvl.ptr[pos + 1] - 1); init=0
    )
end

function Base.show(io::IO, lvl::SparseHashLevel{Ti}) where {Ti}
    if get(io, :compact, false)
        print(io, "SparseHash(")
    else
        print(io, "SparseHash{$Ti}(")
    end
    show(io, lvl.lvl)
    print(io, ", ")
    show(IOContext(io, :typeinfo => Ti), lvl.shape)
    print(io, ", ", lvl.subtables)
    if get(io, :compact, false)
        print(io, ", …")
    else
        for f in SPARSE_HASH_BUFFERS
            print(io, ", ")
            show(io, getfield(lvl, f))
        end
    end
    print(io, ")")
end

function labelled_show(io::IO, fbr::SubFiber{<:SparseHashLevel})
    print(io, "SparseHash (", fill_value(fbr), ") [", ":,"^(ndims(fbr) - 1), "1:", size(fbr)[end], "]")
end

function labelled_children(fbr::SubFiber{<:SparseHashLevel})
    lvl = fbr.lvl
    pos = fbr.pos
    pos + 1 > length(lvl.ptr) && return []
    map(lvl.ptr[pos]:(lvl.ptr[pos + 1] - 1)) do r
        q = lvl.perm[r]
        LabelledTree(
            cartesian_label([range_label() for _ in 1:(ndims(fbr) - 1)]..., lvl.key[q][2]),
            SubFiber(lvl.lvl, q),
        )
    end
end

@inline level_ndims(T::Type{<:SparseHashLevel}) = 1 + level_ndims(fieldtype(T, :lvl))
@inline level_size(lvl::SparseHashLevel) = (level_size(lvl.lvl)..., lvl.shape)
@inline level_axes(lvl::SparseHashLevel) = (level_axes(lvl.lvl)..., Base.OneTo(lvl.shape))
@inline level_eltype(T::Type{<:SparseHashLevel}) = level_eltype(fieldtype(T, :lvl))
@inline level_fill_value(T::Type{<:SparseHashLevel}) = level_fill_value(fieldtype(T, :lvl))
data_rep_level(T::Type{<:SparseHashLevel}) = SparseData(data_rep_level(fieldtype(T, :lvl)))

function isstructequal(a::T, b::T) where {T<:SparseHash}
    a.shape == b.shape && a.subtables == b.subtables &&
        all(f -> getfield(a, f) == getfield(b, f), SPARSE_HASH_BUFFERS) &&
        isstructequal(a.lvl, b.lvl)
end

(fbr::AbstractFiber{<:SparseHashLevel})() = fbr
function (fbr::SubFiber{<:SparseHashLevel})(idxs...)
    isempty(idxs) && return fbr
    lvl = fbr.lvl
    q = sparse_hash_lookup(lvl.tbl_ctrl, lvl.tbl, lvl.key, fbr.pos, idxs[end], lvl.subtables)
    q == 0 ? fill_value(fbr) : SubFiber(lvl.lvl, q)(idxs[1:(end - 1)]...)
end

mutable struct VirtualSparseHashLevel <: AbstractVirtualLevel
    tag
    lvl
    Ti
    shape
    subtables
    # Buffers, in the order of SPARSE_HASH_BUFFERS.
    ptr
    tbl_ctrl
    tbl
    key
    pool
    perm
    tbl_count
    # Assembly state: child extent and outstanding tentative-access handles.
    qos_stop
    pending
end

function is_level_injective(ctx, lvl::VirtualSparseHashLevel)
    [is_level_injective(ctx, lvl.lvl)..., false]
end
function is_level_atomic(ctx, lvl::VirtualSparseHashLevel)
    (below, atomic) = is_level_atomic(ctx, lvl.lvl)
    return ([below; [atomic]], atomic)
end
function is_level_concurrent(ctx, lvl::VirtualSparseHashLevel)
    (data, _) = is_level_concurrent(ctx, lvl.lvl)
    return ([data; [false]], false)
end

function virtualize(ctx, ex, T::Type{<:SparseHashLevel{Ti}}, tag=:lvl) where {Ti}
    tag = freshen(ctx, tag)
    buffers = map(f -> freshen(ctx, tag, Symbol(:_, f)), SPARSE_HASH_BUFFERS)
    stop = freshen(ctx, tag, :_stop)
    push_preamble!(
        ctx,
        quote
            $tag = $ex
            $((:($b = $tag.$f) for (b, f) in zip(buffers, SPARSE_HASH_BUFFERS))...)
            $stop = $tag.shape
        end,
    )
    lvl_2 = virtualize(ctx, :($tag.lvl), fieldtype(T, :lvl), tag)
    VirtualSparseHashLevel(
        tag, lvl_2, Ti, value(stop, Int), value(:($tag.subtables), Int),
        buffers..., freshen(ctx, tag, :_qos_stop), freshen(ctx, tag, :_pending),
    )
end

function lower(ctx::AbstractCompiler, lvl::VirtualSparseHashLevel, ::DefaultStyle)
    quote
        $SparseHashLevel{$(lvl.Ti)}(
            $(ctx(lvl.lvl)), $(ctx(lvl.shape)), $(ctx(lvl.subtables)),
            $((getfield(lvl, f) for f in SPARSE_HASH_BUFFERS)...),
        )
    end
end

function distribute_level(
    ctx::AbstractCompiler, lvl::VirtualSparseHashLevel, arch, diff, style
)
    return diff[lvl.tag] = VirtualSparseHashLevel(
        lvl.tag, distribute_level(ctx, lvl.lvl, arch, diff, style), lvl.Ti,
        lvl.shape, lvl.subtables,
        (
            distribute_buffer(ctx, getfield(lvl, f), arch, style) for
            f in SPARSE_HASH_BUFFERS
        )...,
        (freshen(ctx, lvl.tag, f) for f in (:qos_stop, :pending))...,
    )
end

function redistribute(ctx::AbstractCompiler, lvl::VirtualSparseHashLevel, diff)
    get(diff, lvl.tag) do
        fields = map(f -> getfield(lvl, f), fieldnames(VirtualSparseHashLevel))
        VirtualSparseHashLevel(lvl.tag, redistribute(ctx, lvl.lvl, diff), fields[3:end]...)
    end
end

Base.summary(lvl::VirtualSparseHashLevel) = "SparseHash($(summary(lvl.lvl)))"

function virtual_level_size(ctx, lvl::VirtualSparseHashLevel)
    ext = virtual_call(ctx, extent, literal(lvl.Ti(1)), lvl.shape)
    (virtual_level_size(ctx, lvl.lvl)..., ext)
end

function virtual_level_resize!(ctx, lvl::VirtualSparseHashLevel, dims...)
    lvl.shape = getstop(dims[end])
    lvl.lvl = virtual_level_resize!(ctx, lvl.lvl, dims[1:(end - 1)]...)
    lvl
end

virtual_level_eltype(lvl::VirtualSparseHashLevel) = virtual_level_eltype(lvl.lvl)
virtual_level_fill_value(lvl::VirtualSparseHashLevel) = virtual_level_fill_value(lvl.lvl)
@inline sample_dims(lvl::VirtualSparseHashLevel) = 1 + sample_dims(lvl.lvl)
@inline all_dense(::VirtualSparseHashLevel) = false

postype(lvl::VirtualSparseHashLevel) = postype(lvl.lvl)

function declare_level!(ctx::AbstractCompiler, lvl::VirtualSparseHashLevel, pos, init)
    #TODO check that init == fill_value
    push_preamble!(
        ctx,
        quote
            empty!($(lvl.tbl_ctrl))
            empty!($(lvl.tbl))
            empty!($(lvl.key))
            empty!($(lvl.pool))
            fill!($(lvl.tbl_count), 0)
            $(lvl.qos_stop) = 0
            $(lvl.pending) = 0
        end,
    )
    lvl.lvl = declare_level!(ctx, lvl.lvl, literal(postype(lvl)(0)), init)
    return lvl
end

assemble_level!(ctx, lvl::VirtualSparseHashLevel, pos_start, pos_stop) = quote end

function freeze_level!(ctx::AbstractCompiler, lvl::VirtualSparseHashLevel, pos_stop)
    pos_stop = cache!(ctx, :pos_stop, simplify(ctx, pos_stop))
    qos_stop = freshen(ctx, :qos_stop)
    push_preamble!(
        ctx,
        quote
            $(lvl.pending) == 0 || error("SparseHash has unfinished writers during freeze")
            $qos_stop = Finch.sparse_hash_freeze!(
                $(lvl.ptr), $(lvl.perm), $(lvl.key), $(lvl.pool),
                $(ctx(pos_stop)), $(lvl.qos_stop),
            )
        end,
    )
    lvl.lvl = freeze_level!(ctx, lvl.lvl, value(qos_stop))
    return lvl
end

# Frozen, `key` spans exactly the child positions, and `tbl_count` is exact.
function thaw_level!(ctx::AbstractCompiler, lvl::VirtualSparseHashLevel, pos_stop)
    push_preamble!(
        ctx,
        quote
            $(lvl.qos_stop) = length($(lvl.key))
            $(lvl.pending) = 0
        end,
    )
    lvl.lvl = thaw_level!(ctx, lvl.lvl, value(lvl.qos_stop))
    return lvl
end

function unfurl(
    ctx,
    fbr::VirtualSubFiber{VirtualSparseHashLevel},
    ext,
    mode,
    ::Union{typeof(defaultread),typeof(walk)},
)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.tag
    Tp = postype(lvl)
    Ti = lvl.Ti
    my_r = freshen(ctx, tag, :_r)
    my_r_stop = freshen(ctx, tag, :_r_stop)
    my_i = freshen(ctx, tag, :_i)
    my_i1 = freshen(ctx, tag, :_i1)
    my_q = freshen(ctx, tag, :_q)
    idx(r) = :($(lvl.key)[$(lvl.perm)[$r]][2])

    Thunk(;
        preamble=quote
            $my_r = $(lvl.ptr)[$(ctx(pos))]
            $my_r_stop = $(lvl.ptr)[$(ctx(pos)) + $(Tp(1))]
            if $my_r < $my_r_stop
                $my_i = $(idx(my_r))
                $my_i1 = $(idx(:($my_r_stop - $(Tp(1)))))
            else
                $my_i = $(Ti(1))
                $my_i1 = $(Ti(0))
            end
        end,
        body=(ctx) -> Sequence([
            Phase(;
                stop=(ctx, ext) -> value(my_i1),
                body=(ctx, ext) -> Stepper(;
                    seek=(ctx, ext) -> quote
                        if $(idx(my_r)) < $(ctx(getstart(ext)))
                            $my_r = Finch.sparse_hash_scansearch(
                                $(lvl.key), $(lvl.perm), $(ctx(getstart(ext))), $my_r,
                                $my_r_stop - 1,
                            )
                        end
                    end,
                    preamble=quote
                        $my_q = $(lvl.perm)[$my_r]
                        $my_i = $(lvl.key)[$my_q][2]
                    end,
                    stop=(ctx, ext) -> value(my_i),
                    chunk=Spike(;
                        body=FillLeaf(virtual_level_fill_value(lvl)),
                        tail=Simplify(
                            instantiate(ctx, VirtualSubFiber(lvl.lvl, value(my_q, Tp)), mode)
                        ),
                    ),
                    next=(ctx, ext) -> :($my_r += $(Tp(1))),
                ),
            ),
            Phase(; body=(ctx, ext) -> Run(FillLeaf(virtual_level_fill_value(lvl)))),
        ]),
    )
end

function unfurl(
    ctx, fbr::VirtualSubFiber{VirtualSparseHashLevel}, ext, mode, ::typeof(follow)
)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    my_q = freshen(ctx, lvl.tag, :_q)
    Lookup(;
        body=(ctx, i) -> Thunk(;
            preamble=quote
                $my_q = Finch.sparse_hash_lookup(
                    $(lvl.tbl_ctrl), $(lvl.tbl), $(lvl.key), $(ctx(pos)), $(ctx(i)),
                    $(ctx(lvl.subtables)),
                )
            end,
            body=(ctx) -> Switch([
                value(:($my_q != 0)) =>
                    instantiate(ctx, VirtualSubFiber(lvl.lvl, value(my_q, postype(lvl))), mode),
                literal(true) => FillLeaf(virtual_level_fill_value(lvl)),
            ]),
        ),
    )
end

function unfurl(
    ctx,
    fbr::VirtualSubFiber{VirtualSparseHashLevel},
    ext,
    mode,
    proto::Union{typeof(defaultupdate),typeof(extrude)},
)
    unfurl(
        ctx, VirtualHollowSubFiber(fbr.lvl, fbr.pos, freshen(ctx, :null)), ext, mode, proto
    )
end

function unfurl(
    ctx,
    fbr::VirtualHollowSubFiber{VirtualSparseHashLevel},
    ext,
    mode,
    ::Union{typeof(defaultupdate),typeof(extrude)},
)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.tag
    Tp = postype(lvl)
    B = ctx(lvl.subtables)
    (tbl_ctrl, tbl, key, tbl_count, qos_stop) = (
        lvl.tbl_ctrl, lvl.tbl, lvl.key, lvl.tbl_count, lvl.qos_stop
    )
    pending = lvl.pending
    p, i, x, b, h, s, qos, old, q_stop, dirty = map(
        v -> freshen(ctx, tag, v),
        (:_p, :_i, :_x, :_b, :_h, :_s, :_qos, :_old, :_q_stop, :_dirty),
    )

    Thunk(;
        body=(ctx) -> Lookup(;
            body=(ctx, idx) -> Thunk(;
                preamble=quote
                    $p = $(ctx(pos))
                    $i = $(ctx(idx))
                    $x = Finch.sparse_hash_hash($p, $i)
                    $b = Finch.sparse_hash_hash_subtable($x, $B)
                    # Grow before probing, so the probed slot stays valid if the
                    # key turns out to be new.
                    if 2 * $B * ($tbl_count[$b] + 1) > length($tbl_ctrl)
                        Finch.sparse_hash_resize!(
                            $tbl_ctrl, $tbl, $key,
                            max(2 * length($tbl_ctrl), 4 * $B), $B, $qos_stop
                        )
                    end
                    $h = Finch.sparse_hash_find($tbl_ctrl, $tbl, $key, $p, $i, $x, $B)
                    $s = true
                    if $tbl_ctrl[$h] == Finch.SPARSE_HASH_CTRL_EMPTY
                        # A new key: count it in its bucket and give it a child.
                        $tbl_count[$b] += 1
                        $qos = if isempty($(lvl.pool))
                            ($qos_stop += 1)
                        else
                            pop!($(lvl.pool))
                        end
                        if $qos > length($key)
                            $old = length($key) + 1
                            $q_stop = max(2 * length($key), $qos)
                            resize!($key, $q_stop)
                            $(contain(
                                ctx_2 -> assemble_level!(
                                    ctx_2, lvl.lvl, value(old, Tp),
                                    value(q_stop, Tp)
                                ),
                                ctx,
                            ))
                        end
                        $key[$qos] = ($p, $i, 0x01)
                        $tbl_ctrl[$h] = Finch.sparse_hash_hash_ctrl($x)
                        $tbl[$h] = $qos
                    else
                        $qos = $tbl[$h]
                        $s = $key[$qos][3] != Finch.SPARSE_HASH_KEY_RETAINED
                        $s && Finch.sparse_hash_share!($key, $qos)
                    end
                    $s && ($pending += 1)
                    $dirty = false
                end,
                body=(ctx) -> instantiate(
                    ctx, VirtualHollowSubFiber(lvl.lvl, value(qos, Tp), dirty),
                    mode
                ),
                epilogue=quote
                    $dirty && ($(fbr.dirty) = true)
                    if $s
                        if Finch.sparse_hash_release!(
                            $tbl_ctrl, $tbl, $key, $qos, $x, $B, $dirty
                        )
                            $tbl_count[$b] -= 1
                            push!($(lvl.pool), $qos)
                        end
                        $pending -= 1
                    end
                end,
            ),
        ),
    )
end

# A sampled child position names its key directly. Positions freed or pooled
# by unretained writes own no entry, so their samples are redrawn.
function sample(tid, lvl::SparseHashLevel)
    key = lvl.key.data[tid]
    isempty(lvl.perm.data[tid]) &&
        throw(ArgumentError("Cannot sample an empty SparseHash shard"))
    while true
        tup, q = sample(tid, lvl.lvl)
        p, i, state = key[q]
        state == SPARSE_HASH_KEY_RETAINED && return (tup..., i), p
    end
end

# Prepare hash storage and child offsets. Children keep their positions: shard
# `t`'s child `q` lands at `q + child_shift[t]`, except that its shared entry's
# child lands at the owner's `shared_dst[t]`.
function setup_coalesce!(lvl::SparseHashLevel, max_pos, dst, P, shift, overlap)
    key = lvl.key.data
    perm = lvl.perm.data
    ptr = lvl.ptr.data
    B = dst.subtables
    lvl.subtables == B ||
        throw(ArgumentError("Coalescing hashes must use the same bucket count"))
    # Shard t's local rank r, as (parent, index, child position).
    function entry(t, r)
        q = perm[t][r]
        p, i, _ = key[t][q]
        return p, i, q
    end
    child_shift = cumsum([0; [length(key[t]) for t in 1:(P - 1)]])
    max_child_pos = sum(length, key; init=0)

    # Uniform parent shifts rotate whole buckets, so add the frozen counts
    # directly. Moved and shared entries are corrected below.
    bucket(p, i) = sparse_hash_hash_subtable(sparse_hash_hash(p, i), B)
    bucket_shift = [Int((SPARSE_HASH_POS_MULTIPLIER * (shard_offset(s) % UInt)) & UInt(B - 1))
                    for s in shift]
    bucket_counts = zeros(Int, B)
    for t in 1:P, b in 1:B
        bucket_counts[((b - 1 + bucket_shift[t]) & (B - 1)) + 1] += lvl.tbl_count.data[t][b]
    end

    # Shards concatenate, except that a hash above may move a block of a shard's
    # parents into an earlier owner's range (see `ShardShift`). Bands cut in
    # traversal order, so the shard's entries under it form one run of `perm`,
    # which belongs right after the owner's entries under the same block. Only
    # `perm` and `ptr` make room for it: children keep their positions.
    first_rank(t, p) = p <= length(ptr[t]) ? ptr[t][p] : length(perm[t]) + 1
    under(t, lo, len) = first_rank(t, lo):(first_rank(t, lo + len) - 1)
    moved = [1:0 for _ in 1:P]
    owner = zeros(Int, P)
    split = [length(perm[t]) for t in 1:P]
    for t in 1:P
        s = shift[t]
        s isa ShardShift && s.len > 0 || continue
        moved[t] = under(t, s.src, s.len)
        o = owner[t] = findlast(u -> shard_offset(shift[u]) < s.dst, 1:(t - 1))
        split[o] = last(under(o, s.dst - shard_offset(shift[o]), s.len))
        for r in moved[t]
            p, i, _ = entry(t, r)
            bucket_counts[bucket(p + s.offset, i)] -= 1
            bucket_counts[bucket(p + s, i)] += 1
        end
    end

    # Emit pieces, runs of one shard's local ranks, in destination order. Piece
    # `(lo, hi, start, dup, prev)` puts ranks `lo:hi` at consecutive ranks from
    # `start`, skipping `lo` when it repeats the entry before it (`dup`); `prev`
    # is the parent before the piece. Setup reads only the pieces' boundaries.
    pieces = [NTuple{5,Int}[] for _ in 1:P]
    shared = zeros(Int, P)
    shared_dst = zeros(Int, P)
    nnz = 0
    last_key = nothing
    last_child = 0
    function emit!(t, lo, hi)
        lo > hi && return nothing
        dup = false
        if last_key !== nothing
            p, i, q = entry(t, lo)
            if (p + shift[t], i) == last_key
                dup = true
                shared[t] = q
                shared_dst[t] = last_child
                bucket_counts[bucket(last_key...)] -= 1
            end
        end
        push!(pieces[t], (lo, hi, nnz + 1, dup, last_key === nothing ? 0 : first(last_key)))
        nnz += hi - lo + 1 - dup
        p, i, q = entry(t, hi)
        last_key = (p + shift[t], i)
        dup && hi == lo || (last_child = q + child_shift[t])
        return nothing
    end
    for o in 1:P
        m, b = moved[o], split[o]
        emit!(o, 1, min(first(m) - 1, b))
        emit!(o, last(m) + 1, b)
        for t in (o + 1):P
            owner[t] == o && emit!(t, first(moved[t]), last(moved[t]))
        end
        emit!(o, b + 1, first(m) - 1)
        emit!(o, max(last(m), b) + 1, length(perm[o]))
    end

    # Probing cannot leave its bucket, so size every bucket for the busiest.
    capacity = B * sparse_hash_table_capacity(maximum(bucket_counts))
    for (buf, n) in ((dst.ptr, max_pos + 1), (dst.tbl_ctrl, capacity), (dst.tbl, capacity),
                     (dst.key, max_child_pos), (dst.perm, nnz), (dst.pool, 0))
        empty!(buf)
        resize!(buf, n)
    end
    copyto!(dst.tbl_count, bucket_counts)
    child = setup_coalesce!(
        lvl.lvl, max_child_pos, dst.lvl, P,
        [ShardShift(child_shift[t], shared[t], shared_dst[t], shared[t] != 0) for t in 1:P],
        any(!iszero, shared),
    )
    init = ((dst.tbl_ctrl, 1, SPARSE_HASH_CTRL_EMPTY), child.init...)
    nnz == 0 && (init = (init..., (dst.ptr, 1, 1)))
    return (; P, shift, pieces, moved, shared, shared_dst, nnz, child_shift,
        max_child_pos, bucket_counts, bucket_shift, child, init)
end

# Worker `tid` copies shard `tid`'s keys, traversal order, and children, and
# fills output buckets `tid:P:B` from every shard, so each bucket has one writer.
# A uniform parent shift rotates a source bucket onto one output bucket; only a
# shard's moved entries are routed individually.
function coalesce_shard!(tid, plan, lvl::SparseHashLevel, dst, runs)
    key = lvl.key.data[tid]
    perm = lvl.perm.data[tid]
    shift = plan.shift[tid]
    child_shift = plan.child_shift[tid]
    # Each worker copies its child records once, in position order. Holes and
    # shared duplicates stay free; the traversal permutation is filled below.
    @inbounds for q in eachindex(key)
        p, i, state = key[q]
        dst.key[q + child_shift] = if state == SPARSE_HASH_KEY_FREE || q == plan.shared[tid]
            (0, 0, SPARSE_HASH_KEY_FREE)
        else
            (p + shift, i, SPARSE_HASH_KEY_RETAINED)
        end
    end
    for (lo, hi, start, dup, prev) in plan.pieces[tid]
        for r in (lo + dup):hi
            rank = start + r - lo - dup
            q = perm[r]
            p, i = key[q]
            x = p + shift
            dst.perm[rank] = q + child_shift
            for y in (prev + 1):x
                dst.ptr[y] = rank
            end
            if rank == plan.nnz
                for y in (x + 1):length(dst.ptr)
                    dst.ptr[y] = rank + 1
                end
            end
            prev = x
        end
    end

    B = dst.subtables
    # Keys are distinct after dropping shared entries, so place without comparing.
    function place!(t, q)
        q == plan.shared[t] && return nothing
        p, i = lvl.key.data[t][q]
        x = sparse_hash_hash(p + plan.shift[t], i)
        h = sparse_hash_vacancy(dst.tbl_ctrl, x, B)
        dst.tbl_ctrl[h] = sparse_hash_hash_ctrl(x)
        dst.tbl[h] = q + plan.child_shift[t]
        return nothing
    end
    for b in tid:(plan.P):B, t in 1:(plan.P)
        src_ctrl = lvl.tbl_ctrl.data[t]
        src_tbl = lvl.tbl.data[t]
        s = plan.shift[t]
        width = length(src_ctrl) ÷ B
        src_b = ((b - 1 - plan.bucket_shift[t]) & (B - 1)) + 1
        for h in ((src_b - 1) * width + 1):(src_b * width)
            src_ctrl[h] == SPARSE_HASH_CTRL_EMPTY && continue
            q = src_tbl[h]
            p = first(lvl.key.data[t][q])
            s isa ShardShift && s.src <= p < s.src + s.len && continue
            place!(t, q)
        end
        for r in plan.moved[t]
            q = lvl.perm.data[t][r]
            p, i = lvl.key.data[t][q]
            sparse_hash_hash_subtable(sparse_hash_hash(p + s, i), B) == b && place!(t, q)
        end
    end

    # Recurse on every entry, owned or not: a shared entry's children are split
    # between both shards.
    leaves = coalesce_leaves(lvl.lvl)
    child_runs = (((q - 1) * leaves + 1):(q * leaves) for q in perm)
    coalesce_shard!(tid, plan.child, lvl.lvl, dst.lvl, child_runs)
end
