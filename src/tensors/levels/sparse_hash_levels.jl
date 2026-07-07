"""
    SparseHashLevel{[Ti=Int], [SingleWriter=true], [Tp=Int], [Ptr, TblPos, TblIdx, TblCtrl, TblVal, Perm]}(lvl, [dim])

A subfiber of a sparse level does not need to represent slices `A[:, ..., :, i]`
which are entirely [`fill_value`](@ref). Instead, only potentially non-fill
slices are stored as subfibers in `lvl`. A data structure specified by Tbl is used to record which
slices are stored. Optionally, `dim` is the size of the last dimension.

`Ti` is the type of the last fiber index, and `Tp` is the type used for
positions in the level. The types `Ptr` and `Idx` are the types of the
arrays used to store positions and indices.

Implementation invariants:

* `tbl_ctrl` and `tbl_val` form a linear-probing hash table. A full slot has
  the high bit set in `tbl_ctrl[h]`, stores seven high hash bits in the low
  bits, and has a positive `q = tbl_val[h]`. `0x00` is an empty slot and
  terminates a probe.
* Full slots point into the packed entry arrays: `tbl_pos[q]` is the parent
  position and `tbl_idx[q]` is the coordinate. `q == 0` is reserved as the
  missing sentinel.
* The hash bucket uses low hash bits because the table capacity is a power of
  two. The control byte fingerprint uses high hash bits so bucket selection and
  fingerprint screening are independent.
* In frozen/read mode, `ptr[p]:(ptr[p + 1] - 1)` indexes `perm`, and `perm[r]`
  is a `q`. Each parent range is sorted by `tbl_idx[q]`.
* In thawed/update mode, `perm[q] > 0` means `q` is live/dirty and
  `perm[q] == 0` means `q` was touched but has not retained data. Multi-writer
  update mode additionally uses `perm[q] < 0` to link `q` into the free list
  headed by `qos_free`; that encoding assumes `perm` uses a signed position
  type.
* `tbl_count` counts full hash slots.
* `SingleWriter == true` promises that a newly created `(p, i)` has at most one
  writer before it is published to the table. In that case update mode caches
  the insertion slot but delays publishing the missing key until the child
  reports that it retained data.
* `SingleWriter == false` may have several simultaneous writers for the same
  missing key. Generated update code keeps a small linear pending stack of
  `(p, i, q, live-count)` records. Pending entries are not published to the
  hash table; the last live writer either publishes a retained `q` or returns
  it to the multi-writer free list.

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
struct SparseHashLevel{Ti,SingleWriter,Ptr,TblPos,TblIdx,TblCtrl,TblVal,Perm,Lvl} <:
       AbstractLevel
    lvl::Lvl
    shape::Ti
    # In frozen/read mode, ptr is a CSR-style parent pointer into perm.
    ptr::Ptr
    # tbl_val is the open-addressed table: slot => q, with 0 marking empty.
    # tbl_ctrl stores a SwissHash-style control byte per slot:
    #   0x80 set => full, low seven bits => hash fingerprint.
    #   0x00 => empty.
    # tbl_pos and tbl_idx are packed entry arrays indexed by q.
    tbl_pos::TblPos
    tbl_idx::TblIdx
    tbl_ctrl::TblCtrl
    tbl_val::TblVal
    perm::Perm
end

const SparseHash = SparseHashLevel
const SPARSE_HASH_CTRL_EMPTY = UInt8(0x00)
const SPARSE_HASH_CTRL_FULL = UInt8(0x80)
const SPARSE_HASH_CTRL_HASH_MASK = UInt8(0x7f)
const SPARSE_HASH_CTRL_SHIFT = 8 * sizeof(UInt) - 7

@inline sparse_hash_table_capacity(n) = max(4, n <= 1 ? 4 : nextpow(2, 2n))

@inline sparse_hash_hash(p, i) = hash((p, i))
@inline sparse_hash_hash_slot(h::UInt, n) = Int(h & UInt(n - 1)) + 1
@inline sparse_hash_hash_ctrl(h::UInt) =
    SPARSE_HASH_CTRL_FULL |
    UInt8((h >> SPARSE_HASH_CTRL_SHIFT) & UInt(SPARSE_HASH_CTRL_HASH_MASK))

@inline function sparse_hash_table_resize!(tbl_pos, tbl_idx, tbl_ctrl, tbl_val, cap)
    old_val = copy(tbl_val)
    resize!(tbl_ctrl, cap)
    resize!(tbl_val, cap)
    fill!(tbl_ctrl, SPARSE_HASH_CTRL_EMPTY)
    fill!(tbl_val, zero(eltype(tbl_val)))
    @inbounds for h in eachindex(old_val)
        v = old_val[h]
        if v > 0
            sparse_hash_table_insert_noresize!(
                tbl_pos, tbl_idx, tbl_ctrl, tbl_val, tbl_pos[v], tbl_idx[v], v
            )
        end
    end
    return tbl_pos, tbl_idx, tbl_ctrl, tbl_val
end

# An empty slot ends a probe chain.
@inline function sparse_hash_table_lookup_slot(tbl_pos, tbl_idx, tbl_ctrl, tbl_val, p, i)
    isempty(tbl_val) && return 0
    n = length(tbl_val)
    hsh = sparse_hash_hash(p, i)
    ctrl = sparse_hash_hash_ctrl(hsh)
    h = sparse_hash_hash_slot(hsh, n)
    @inbounds for _ in 1:n
        c = tbl_ctrl[h]
        if c == ctrl
            val = tbl_val[h]
            if tbl_pos[val] == p && tbl_idx[val] == i
                return h
            end
        elseif c == SPARSE_HASH_CTRL_EMPTY
            return 0
        end
        h = h == n ? 1 : h + 1
    end
    return 0
end

@inline function sparse_hash_table_lookup_insert_slot(
    tbl_pos, tbl_idx, tbl_ctrl, tbl_val, p, i
)
    isempty(tbl_val) && return 0
    n = length(tbl_val)
    hsh = sparse_hash_hash(p, i)
    ctrl = sparse_hash_hash_ctrl(hsh)
    h = sparse_hash_hash_slot(hsh, n)
    @inbounds for _ in 1:n
        c = tbl_ctrl[h]
        if c == ctrl
            val = tbl_val[h]
            if tbl_pos[val] == p && tbl_idx[val] == i
                return h
            end
        elseif c == SPARSE_HASH_CTRL_EMPTY
            return h
        end
        h = h == n ? 1 : h + 1
    end
    return 0
end

@inline function sparse_hash_table_insert_noresize!(
    tbl_pos, tbl_idx, tbl_ctrl, tbl_val, p, i, v
)
    h = sparse_hash_table_lookup_insert_slot(tbl_pos, tbl_idx, tbl_ctrl, tbl_val, p, i)
    sparse_hash_table_insert_at_slot!(tbl_pos, tbl_idx, tbl_ctrl, tbl_val, h, p, i, v)
    return v
end

@inline function sparse_hash_table_lookup(tbl_pos, tbl_idx, tbl_ctrl, tbl_val, p, i)
    h = sparse_hash_table_lookup_slot(tbl_pos, tbl_idx, tbl_ctrl, tbl_val, p, i)
    h == 0 && return zero(eltype(tbl_val))
    return tbl_val[h]
end

@inline function sparse_hash_table_insert_at_slot!(
    tbl_pos, tbl_idx, tbl_ctrl, tbl_val, h, p, i, v
)
    hsh = sparse_hash_hash(p, i)
    tbl_pos[v] = p
    tbl_idx[v] = i
    tbl_ctrl[h] = sparse_hash_hash_ctrl(hsh)
    tbl_val[h] = v
    return v
end

@inline function sparse_hash_stack_lookup(stk_pos, stk_idx, stk_cnt, stk_stop, p, i)
    @inbounds for s in 1:stk_stop
        if stk_cnt[s] > 0 && stk_pos[s] == p && stk_idx[s] == i
            return s
        end
    end
    return 0
end

@inline function sparse_hash_stack_first_free(stk_cnt, stk_stop)
    @inbounds for s in 1:stk_stop
        if stk_cnt[s] == 0
            return s
        end
    end
    return 0
end

@inline function sparse_hash_stack_push!(
    stk_pos, stk_idx, stk_val, stk_cnt, stk_stop, p, i, v
)
    s = sparse_hash_stack_first_free(stk_cnt, stk_stop)
    if s == 0
        stk_stop += 1
        resize!(stk_pos, stk_stop)
        resize!(stk_idx, stk_stop)
        resize!(stk_val, stk_stop)
        resize!(stk_cnt, stk_stop)
        s = stk_stop
    end
    @inbounds begin
        stk_pos[s] = p
        stk_idx[s] = i
        stk_val[s] = v
        stk_cnt[s] = 1
    end
    return s, stk_stop
end

@inline function sparse_hash_stack_trim(stk_cnt, stk_stop)
    @inbounds while stk_stop > 0 && stk_cnt[stk_stop] == 0
        stk_stop -= 1
    end
    return stk_stop
end

SparseHashLevel(lvl) = SparseHashLevel{Int}(lvl)
SparseHashLevel(lvl, shape::Ti) where {Ti} = SparseHashLevel{Ti}(lvl, shape)
SparseHashLevel{Ti}(lvl) where {Ti} = SparseHashLevel{Ti,true}(lvl)
SparseHashLevel{Ti}(lvl, shape) where {Ti} = SparseHashLevel{Ti,true}(lvl, shape)
function SparseHashLevel{Ti,SingleWriter}(lvl) where {Ti,SingleWriter}
    SparseHashLevel{Ti,SingleWriter}(lvl, zero(Ti))
end
function SparseHashLevel{Ti,SingleWriter}(lvl, shape) where {Ti,SingleWriter}
    SparseHashLevel{Ti,SingleWriter}(
        lvl,
        shape,
        postype(lvl)[1],
        postype(lvl)[],
        Ti[],
        UInt8[],
        postype(lvl)[],
        postype(lvl)[],
    )
end

function SparseHashLevel{Ti,SingleWriter}(
    lvl::Lvl,
    shape,
    ptr::Ptr,
    tbl_pos::TblPos,
    tbl_idx::TblIdx,
    tbl_ctrl::TblCtrl,
    tbl_val::TblVal,
    perm::Perm,
) where {Ti,SingleWriter,Ptr,TblPos,TblIdx,TblCtrl,TblVal,Perm,Lvl}
    SparseHashLevel{Ti,SingleWriter,Ptr,TblPos,TblIdx,TblCtrl,TblVal,Perm,Lvl}(
        lvl, shape, ptr, tbl_pos, tbl_idx, tbl_ctrl, tbl_val, perm
    )
end

Base.summary(lvl::SparseHashLevel) = "SparseHash($(summary(lvl.lvl)))"
function similar_level(
    lvl::SparseHashLevel{Ti,SingleWriter}, fill_value, eltype::Type, dim, tail...
) where {Ti,SingleWriter}
    SparseHashLevel{Ti,SingleWriter}(
        similar_level(lvl.lvl, fill_value, eltype, tail...), dim
    )
end

function postype(
    ::Type{SparseHashLevel{Ti,SingleWriter,Ptr,TblPos,TblIdx,TblCtrl,TblVal,Perm,Lvl}}
) where {Ti,SingleWriter,Ptr,TblPos,TblIdx,TblCtrl,TblVal,Perm,Lvl}
    return postype(Lvl)
end

function Base.resize!(
    lvl::SparseHashLevel{Ti,SingleWriter}, dims...
) where {Ti,SingleWriter}
    SparseHashLevel{Ti,SingleWriter}(
        resize!(lvl.lvl, dims[1:(end - 1)]...),
        dims[end],
        lvl.ptr,
        lvl.tbl_pos,
        lvl.tbl_idx,
        lvl.tbl_ctrl,
        lvl.tbl_val,
        lvl.perm,
    )
end

function transfer(
    Tm,
    lvl::SparseHashLevel{Ti,SingleWriter,Ptr,TblPos,TblIdx,TblCtrl,TblVal,Perm,Lvl},
) where {Ti,SingleWriter,Ptr,TblPos,TblIdx,TblCtrl,TblVal,Perm,Lvl}
    lvl_2 = transfer(Tm, lvl.lvl)
    ptr_2 = transfer(Tm, lvl.ptr)
    tbl_pos_2 = transfer(Tm, lvl.tbl_pos)
    tbl_idx_2 = transfer(Tm, lvl.tbl_idx)
    tbl_ctrl_2 = transfer(Tm, lvl.tbl_ctrl)
    tbl_val_2 = transfer(Tm, lvl.tbl_val)
    perm_2 = transfer(Tm, lvl.perm)
    return SparseHashLevel{Ti,SingleWriter}(
        lvl_2, lvl.shape, ptr_2, tbl_pos_2, tbl_idx_2, tbl_ctrl_2, tbl_val_2, perm_2
    )
end

function countstored_level(lvl::SparseHashLevel, pos)
    pos == 0 && return countstored_level(lvl.lvl, pos)
    countstored_level(lvl.lvl, lvl.ptr[pos + 1] - 1)
end

function pattern!(lvl::SparseHashLevel{Ti,SingleWriter}) where {Ti,SingleWriter}
    SparseHashLevel{Ti,SingleWriter}(
        pattern!(lvl.lvl),
        lvl.shape,
        lvl.ptr,
        lvl.tbl_pos,
        lvl.tbl_idx,
        lvl.tbl_ctrl,
        lvl.tbl_val,
        lvl.perm,
    )
end

function set_fill_value!(
    lvl::SparseHashLevel{Ti,SingleWriter}, init
) where {Ti,SingleWriter}
    SparseHashLevel{Ti,SingleWriter}(
        set_fill_value!(lvl.lvl, init),
        lvl.shape,
        lvl.ptr,
        lvl.tbl_pos,
        lvl.tbl_idx,
        lvl.tbl_ctrl,
        lvl.tbl_val,
        lvl.perm,
    )
end

function Base.show(io::IO, lvl::SparseHashLevel{Ti,SingleWriter}) where {Ti,SingleWriter}
    if get(io, :compact, false)
        print(io, "SparseHash(")
    elseif SingleWriter
        print(io, "SparseHash{$Ti}(")
    else
        print(io, "SparseHash{$Ti, false}(")
    end
    show(io, lvl.lvl)
    print(io, ", ")
    show(IOContext(io, :typeinfo => Ti), lvl.shape)
    print(io, ", ")
    if get(io, :compact, false)
        print(io, "…")
    else
        show(io, lvl.ptr)
        print(io, ", ")
        show(io, lvl.tbl_pos)
        print(io, ", ")
        show(io, lvl.tbl_idx)
        print(io, ", ")
        show(io, lvl.tbl_ctrl)
        print(io, ", ")
        show(io, lvl.tbl_val)
        print(io, ", ")
        show(io, lvl.perm)
    end
    print(io, ")")
end

function labelled_show(io::IO, fbr::SubFiber{<:SparseHashLevel})
    print(
        io,
        "SparseHash (",
        fill_value(fbr),
        ") [",
        ":,"^(ndims(fbr) - 1),
        "1:",
        size(fbr)[end],
        "]",
    )
end

function labelled_children(fbr::SubFiber{<:SparseHashLevel})
    lvl = fbr.lvl
    pos = fbr.pos
    pos + 1 > length(lvl.ptr) && return []
    map(lvl.ptr[pos]:(lvl.ptr[pos + 1] - 1)) do qos
        q = lvl.perm[qos]
        LabelledTree(
            cartesian_label(
                [range_label() for _ in 1:(ndims(fbr) - 1)]...,
                lvl.tbl_idx[q],
            ),
            SubFiber(lvl.lvl, q),
        )
    end
end

@inline level_ndims(
    ::Type{<:SparseHashLevel{Ti,SingleWriter,Ptr,TblPos,TblIdx,TblCtrl,TblVal,Perm,Lvl}}
) where {Ti,SingleWriter,Ptr,TblPos,TblIdx,TblCtrl,TblVal,Perm,Lvl} =
    1 + level_ndims(Lvl)
@inline level_size(lvl::SparseHashLevel) = (level_size(lvl.lvl)..., lvl.shape)
@inline level_axes(lvl::SparseHashLevel) = (level_axes(lvl.lvl)..., Base.OneTo(lvl.shape))
@inline level_eltype(
    ::Type{<:SparseHashLevel{Ti,SingleWriter,Ptr,TblPos,TblIdx,TblCtrl,TblVal,Perm,Lvl}}
) where {Ti,SingleWriter,Ptr,TblPos,TblIdx,TblCtrl,TblVal,Perm,Lvl} = level_eltype(Lvl)
@inline level_fill_value(
    ::Type{<:SparseHashLevel{Ti,SingleWriter,Ptr,TblPos,TblIdx,TblCtrl,TblVal,Perm,Lvl}}
) where {Ti,SingleWriter,Ptr,TblPos,TblIdx,TblCtrl,TblVal,Perm,Lvl} =
    level_fill_value(Lvl)
function data_rep_level(
    ::Type{<:SparseHashLevel{Ti,SingleWriter,Ptr,TblPos,TblIdx,TblCtrl,TblVal,Perm,Lvl}}
) where {Ti,SingleWriter,Ptr,TblPos,TblIdx,TblCtrl,TblVal,Perm,Lvl}
    SparseData(data_rep_level(Lvl))
end

function isstructequal(a::T, b::T) where {T<:SparseHash}
    a.shape == b.shape &&
        a.tbl_pos == b.tbl_pos &&
        a.tbl_idx == b.tbl_idx &&
        a.tbl_ctrl == b.tbl_ctrl &&
        a.tbl_val == b.tbl_val &&
        a.perm == b.perm &&
        isstructequal(a.lvl, b.lvl)
end

(fbr::AbstractFiber{<:SparseHashLevel})() = fbr
function (fbr::SubFiber{<:SparseHashLevel{Ti}})(idxs...) where {Ti}
    isempty(idxs) && return fbr
    lvl = fbr.lvl
    p = fbr.pos
    crds = [lvl.tbl_idx[lvl.perm[q]] for q in lvl.ptr[p]:(lvl.ptr[p + 1] - 1)]
    r = searchsorted(crds, idxs[end])
    q = lvl.ptr[p] + first(r) - 1
    length(r) == 0 ? fill_value(fbr) :
    SubFiber(lvl.lvl, lvl.perm[q])(idxs[1:(end - 1)]...)
end

mutable struct VirtualSparseHashLevel <: AbstractVirtualLevel
    tag
    lvl
    Ti
    single_writer
    ptr
    tbl_pos
    tbl_idx
    tbl_ctrl
    tbl_val
    perm
    shape
    qos_stop
    qos_free
    tbl_count
    stk_pos
    stk_idx
    stk_val
    stk_cnt
    stk_stop
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

function virtualize(
    ctx,
    ex,
    ::Type{SparseHashLevel{Ti,SingleWriter,Ptr,TblPos,TblIdx,TblCtrl,TblVal,Perm,Lvl}},
    tag=:lvl,
) where {Ti,SingleWriter,Ptr,TblPos,TblIdx,TblCtrl,TblVal,Perm,Lvl}
    tag = freshen(ctx, tag)
    ptr = freshen(ctx, tag, :_ptr)
    tbl_pos = freshen(ctx, tag, :_tbl_pos)
    tbl_idx = freshen(ctx, tag, :_tbl_idx)
    tbl_ctrl = freshen(ctx, tag, :_tbl_ctrl)
    tbl_val = freshen(ctx, tag, :_tbl_val)
    perm = freshen(ctx, tag, :_perm)
    stk_pos = freshen(ctx, tag, :_stk_pos)
    stk_idx = freshen(ctx, tag, :_stk_idx)
    stk_val = freshen(ctx, tag, :_stk_val)
    stk_cnt = freshen(ctx, tag, :_stk_cnt)
    stop = freshen(ctx, tag, :_stop)
    push_preamble!(
        ctx,
        quote
            $tag = $ex
            $ptr = $tag.ptr
            $tbl_pos = $tag.tbl_pos
            $tbl_idx = $tag.tbl_idx
            $tbl_ctrl = $tag.tbl_ctrl
            $tbl_val = $tag.tbl_val
            $perm = $tag.perm
            $(
                if SingleWriter
                    nothing
                else
                    quote
                        $stk_pos = similar($tbl_pos, 0)
                        $stk_idx = similar($tbl_idx, 0)
                        $stk_val = similar($tbl_val, 0)
                        $stk_cnt = Int[]
                    end
                end
            )
            $stop = $tag.shape
        end,
    )
    qos_stop = freshen(ctx, tag, :_qos_stop)
    qos_free = freshen(ctx, tag, :_qos_free)
    tbl_count = freshen(ctx, tag, :_tbl_count)
    stk_stop = freshen(ctx, tag, :_stk_stop)
    shape = value(stop, Int)
    lvl_2 = virtualize(ctx, :($tag.lvl), Lvl, tag)
    VirtualSparseHashLevel(
        tag, lvl_2, Ti, SingleWriter, ptr, tbl_pos, tbl_idx, tbl_ctrl, tbl_val, perm,
        shape, qos_stop, qos_free, tbl_count, stk_pos, stk_idx, stk_val, stk_cnt,
        stk_stop,
    )
end
function lower(ctx::AbstractCompiler, lvl::VirtualSparseHashLevel, ::DefaultStyle)
    quote
        $SparseHashLevel{$(lvl.Ti),$(lvl.single_writer)}(
            $(ctx(lvl.lvl)),
            $(ctx(lvl.shape)),
            $(lvl.ptr),
            $(lvl.tbl_pos),
            $(lvl.tbl_idx),
            $(lvl.tbl_ctrl),
            $(lvl.tbl_val),
            $(lvl.perm),
        )
    end
end

function distribute_level(
    ctx::AbstractCompiler, lvl::VirtualSparseHashLevel, arch, diff, style
)
    return diff[lvl.tag] = VirtualSparseHashLevel(
        lvl.tag,
        distribute_level(ctx, lvl.lvl, arch, diff, style),
        lvl.Ti,
        lvl.single_writer,
        distribute_buffer(ctx, lvl.ptr, arch, style),
        distribute_buffer(ctx, lvl.tbl_pos, arch, style),
        distribute_buffer(ctx, lvl.tbl_idx, arch, style),
        distribute_buffer(ctx, lvl.tbl_ctrl, arch, style),
        distribute_buffer(ctx, lvl.tbl_val, arch, style),
        distribute_buffer(ctx, lvl.perm, arch, style),
        lvl.shape,
        lvl.qos_stop,
        lvl.qos_free,
        lvl.tbl_count,
        lvl.stk_pos,
        lvl.stk_idx,
        lvl.stk_val,
        lvl.stk_cnt,
        lvl.stk_stop,
    )
end

function redistribute(ctx::AbstractCompiler, lvl::VirtualSparseHashLevel, diff)
    get(
        diff,
        lvl.tag,
        VirtualSparseHashLevel(
            lvl.tag,
            redistribute(ctx, lvl.lvl, diff),
            lvl.Ti,
            lvl.single_writer,
            lvl.ptr,
            lvl.tbl_pos,
            lvl.tbl_idx,
            lvl.tbl_ctrl,
            lvl.tbl_val,
            lvl.perm,
            lvl.shape,
            lvl.qos_stop,
            lvl.qos_free,
            lvl.tbl_count,
            lvl.stk_pos,
            lvl.stk_idx,
            lvl.stk_val,
            lvl.stk_cnt,
            lvl.stk_stop,
        ),
    )
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

postype(lvl::VirtualSparseHashLevel) = postype(lvl.lvl)

function declare_level!(ctx::AbstractCompiler, lvl::VirtualSparseHashLevel, pos, init)
    #TODO check that init == fill_value
    Tp = postype(lvl)
    qos = freshen(ctx, lvl.tag, :qos)
    push_preamble!(
        ctx,
        quote
            empty!($(lvl.tbl_pos))
            empty!($(lvl.tbl_idx))
            empty!($(lvl.tbl_ctrl))
            empty!($(lvl.tbl_val))
            $qos = $(Tp(0))
            $(lvl.qos_stop) = 0
            $(
                if lvl.single_writer
                    nothing
                else
                    quote
                        $(lvl.qos_free) = 0
                    end
                end
            )
            $(lvl.tbl_count) = 0
            $(lvl.stk_stop) = 0
            resize!($(lvl.perm), 0)
        end,
    )
    lvl.lvl = declare_level!(ctx, lvl.lvl, value(qos, Tp), init)
    return lvl
end

function assemble_level!(ctx, lvl::VirtualSparseHashLevel, pos_start, pos_stop)
    pos_start = ctx(cache!(ctx, :p_start, pos_start))
    pos_stop = ctx(cache!(ctx, :p_start, pos_stop))
end

function freeze_level!(ctx::AbstractCompiler, lvl::VirtualSparseHashLevel, pos_stop)
    Tp = postype(lvl)
    Ti = lvl.Ti
    pos_stop = cache!(ctx, :pos_stop, simplify(ctx, pos_stop))
    qos_stop = freshen(ctx, :qos_stop)
    p = freshen(ctx, :p)
    q = freshen(ctx, :q)
    v = freshen(ctx, :v)
    qos_max = freshen(ctx, :qos_max)
    tbl_count = lvl.tbl_count
    h = freshen(ctx, :h)
    r = freshen(ctx, :r)
    idx_tmp = freshen(ctx, :idx_tmp)
    val_tmp = freshen(ctx, :val_tmp)
    shuffler = freshen(ctx, :shuffler)
    push_preamble!(
        ctx,
        quote
            # Count bucket sizes in ptr[p + 1]; ptr[1] is the fixed first start.
            resize!($(lvl.ptr), $(ctx(pos_stop)) + 1)
            $(lvl.ptr)[1] = 1
            Finch.fill_range!($(lvl.ptr), 0, 2, $(ctx(pos_stop)) + 1)
            $q = 0
            $qos_max = $(Tp(0))
            for $h in eachindex($(lvl.tbl_val))
                $v = $(lvl.tbl_val)[$h]
                if $v > 0
                    $p = $(lvl.tbl_pos)[$v]
                    $q += 1
                    $qos_max = max($qos_max, $v)
                    $(lvl.ptr)[$p + 1] += 1
                end
            end
            $tbl_count = $q
            $val_tmp = Vector{$Tp}(undef, $tbl_count)
            $q = 0
            for $h in eachindex($(lvl.tbl_val))
                $v = $(lvl.tbl_val)[$h]
                if $v > 0
                    $q += 1
                    $val_tmp[$q] = $v
                end
            end
            resize!($(lvl.tbl_pos), $qos_max)
            resize!($(lvl.tbl_idx), $qos_max)
            # After the prefix sum, ptr[p] is the final start for bucket p,
            # and ptr[p + 1] is the final stop for bucket p.
            for $p in 2:($(ctx(pos_stop)) + 1)
                $(lvl.ptr)[$p] += $(lvl.ptr)[$p - 1]
            end
            resize!($(lvl.perm), $tbl_count)
            $idx_tmp = Vector{$Ti}(undef, $tbl_count)
            @inbounds for $q in eachindex($val_tmp)
                $v = $val_tmp[$q]
                $idx_tmp[$q] = $(lvl.tbl_idx)[$v]
            end
            # Sort all live q values by coordinate, then scatter by parent.
            # Filtering this globally sorted stream into parent ranges leaves
            # each parent range sorted by tbl_idx[q].
            $shuffler = sortperm($idx_tmp)
            @inbounds for $q in $shuffler
                $v = $val_tmp[$q]
                $p = $(lvl.tbl_pos)[$v]
                $r = $(lvl.ptr)[$p]
                $(lvl.perm)[$r] = $v
                # Advancing ptr[p] turns bucket p's start into bucket p's stop.
                $(lvl.ptr)[$p] += 1
            end
            # Shift the restored stops right: ptr[p + 1] becomes bucket p's stop,
            # and ptr[1] is reset to the first start.
            for $p in ($(ctx(pos_stop)) + 1):-1:2
                $(lvl.ptr)[$p] = $(lvl.ptr)[$p - 1]
            end
            $(lvl.ptr)[1] = 1
            $(lvl.stk_stop) == 0 ||
                error("SparseHash pending writer stack is not empty during freeze")
            $qos_stop = $qos_max
        end,
    )
    lvl.lvl = freeze_level!(ctx, lvl.lvl, value(qos_stop))
    return lvl
end

function thaw_level!(ctx::AbstractCompiler, lvl::VirtualSparseHashLevel, pos_stop)
    p = freshen(ctx, :p)
    q = freshen(ctx, :q)
    v = freshen(ctx, :v)
    h = freshen(ctx, :h)
    tbl_count = lvl.tbl_count
    push_preamble!(
        ctx,
        quote
            $(lvl.qos_stop) = 0
            $(
                if lvl.single_writer
                    nothing
                else
                    quote
                        $(lvl.qos_free) = 0
                    end
                end
            )
            $q = 0
            for $h in eachindex($(lvl.tbl_val))
                $v = $(lvl.tbl_val)[$h]
                if $v > 0
                    $q += 1
                    $(lvl.qos_stop) = max($(lvl.qos_stop), $v)
                end
            end
            $tbl_count = $q
            $(lvl.stk_stop) = 0
            Finch.resize_if_smaller!($(lvl.perm), $(lvl.qos_stop))
            Finch.fill_range!($(lvl.perm), 0, 1, $(lvl.qos_stop))
            # Restore update-mode perm: live q maps to itself.
            for $h in eachindex($(lvl.tbl_val))
                $v = $(lvl.tbl_val)[$h]
                if $v > 0
                    $(lvl.perm)[$v] = $v
                end
            end
            $(
                if lvl.single_writer
                    nothing
                else
                    quote
                        # Link unused q values into the multi-writer free list.
                        for $p in ($(lvl.qos_stop)):-1:1
                            if $(lvl.perm)[$p] == 0
                                $(lvl.perm)[$p] = -$(lvl.qos_free)
                                $(lvl.qos_free) = $p
                            end
                        end
                    end
                end
            )
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
    my_i = freshen(ctx, tag, :_i)
    my_q = freshen(ctx, tag, :_q)
    my_q_stop = freshen(ctx, tag, :_q_stop)
    my_i1 = freshen(ctx, tag, :_i1)
    my_v = freshen(ctx, tag, :_v)

    Thunk(;
        preamble=quote
            $my_q = $(lvl.ptr)[$(ctx(pos))]
            $my_q_stop = $(lvl.ptr)[$(ctx(pos)) + $(Tp(1))]
            if $my_q < $my_q_stop
                $my_i = $(lvl.tbl_idx)[$(lvl.perm)[$my_q]]
                $my_i1 = $(lvl.tbl_idx)[$(lvl.perm)[$my_q_stop - $(Tp(1))]]
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
                        if $(lvl.tbl_idx)[$(lvl.perm)[$my_q]] < $(ctx(getstart(ext)))
                            $my_q = Finch.scansearch_perm(
                                $(lvl.tbl_idx),
                                $(lvl.perm),
                                $(ctx(getstart(ext))),
                                $my_q,
                                $my_q_stop - 1,
                            )
                            $my_i = $(lvl.tbl_idx)[$(lvl.perm)[$my_q]]
                        end
                    end,
                    preamble=quote
                        $my_i = $(lvl.tbl_idx)[$(lvl.perm)[$my_q]]
                        $my_v = $(lvl.perm)[$my_q]
                    end,
                    stop=(ctx, ext) -> value(my_i),
                    chunk=Spike(;
                        body=FillLeaf(virtual_level_fill_value(lvl)),
                        tail=Simplify(
                            instantiate(
                                ctx, VirtualSubFiber(lvl.lvl, value(my_v, Ti)), mode
                            ),
                        ),
                    ),
                    next=(ctx, ext) -> :($my_q += $(Tp(1))),
                ),
            ),
            Phase(;
                body=(ctx, ext) -> Run(FillLeaf(virtual_level_fill_value(lvl)))
            ),
        ]),
    )
end

function unfurl(
    ctx, fbr::VirtualSubFiber{VirtualSparseHashLevel}, ext, mode, ::typeof(follow)
)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.tag
    Tp = postype(lvl)
    my_q = freshen(ctx, tag, :_q)

    Lookup(;
        body=(ctx, i) -> Thunk(;
            preamble=quote
                $my_q = Finch.sparse_hash_table_lookup(
                    $(lvl.tbl_pos),
                    $(lvl.tbl_idx),
                    $(lvl.tbl_ctrl),
                    $(lvl.tbl_val),
                    $(ctx(pos)),
                    $(ctx(i)),
                )
            end,
            body=(ctx) -> Switch(
                [
                    value(:($my_q != 0)) => instantiate(
                        ctx, VirtualSubFiber(lvl.lvl, value(my_q, Tp)), mode
                    )
                    literal(true) => FillLeaf(virtual_level_fill_value(lvl))
                ],
            ),
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
    qos = freshen(ctx, tag, :_qos)
    qos_stop = lvl.qos_stop
    qos_free = lvl.qos_free
    dirty = freshen(ctx, tag, :_dirty)
    p = freshen(ctx, tag, :_p)
    q_stop = freshen(ctx, tag, :_q_stop)
    old = freshen(ctx, tag, :_old)
    tbl_cap = freshen(ctx, tag, :_tbl_cap)
    tbl_pos = freshen(ctx, tag, :_tbl_pos)
    tbl_idx = freshen(ctx, tag, :_tbl_idx)
    tbl_ctrl = freshen(ctx, tag, :_tbl_ctrl)
    tbl_val = freshen(ctx, tag, :_tbl_val)
    tbl_slot = freshen(ctx, tag, :_tbl_slot)
    stk_slot = freshen(ctx, tag, :_stk_slot)

    Thunk(;
        body=(ctx) -> Lookup(;
            body=(ctx, idx) -> Thunk(;
                preamble=quote
                    $tbl_pos = $(lvl.tbl_pos)
                    $tbl_idx = $(lvl.tbl_idx)
                    $tbl_ctrl = $(lvl.tbl_ctrl)
                    $tbl_val = $(lvl.tbl_val)
                    if $(
                        if lvl.single_writer
                            :($qos_stop == length($(lvl.perm)))
                        else
                            :($qos_free == 0 && $qos_stop == length($(lvl.perm)))
                        end
                    )
                        $old = length($(lvl.perm)) + 1
                        $p = $old
                        $q_stop = max(length($(lvl.perm)) << 1, $qos_stop + 1)
                        Finch.resize_if_smaller!($(lvl.perm), $q_stop)
                        Finch.resize_if_smaller!($(lvl.tbl_pos), $q_stop)
                        Finch.resize_if_smaller!($(lvl.tbl_idx), $q_stop)
                        Finch.fill_range!($(lvl.perm), 0, $old, $q_stop)
                        $tbl_cap = Finch.sparse_hash_table_capacity($q_stop)
                        Finch.sparse_hash_table_resize!(
                            $tbl_pos, $tbl_idx, $tbl_ctrl, $tbl_val, $tbl_cap
                        )
                        $(contain(
                            ctx_2 -> assemble_level!(
                                ctx_2,
                                lvl.lvl,
                                value(p, Tp),
                                value(q_stop, Tp),
                            ),
                            ctx,
                        ))
                    end
                    $stk_slot = 0
                    $tbl_slot = Finch.sparse_hash_table_lookup_insert_slot(
                        $tbl_pos,
                        $tbl_idx,
                        $tbl_ctrl,
                        $tbl_val,
                        $(ctx(pos)),
                        $(ctx(idx)),
                    )
                    $qos = $(Tp(0))
                    if $tbl_slot != 0
                        $qos = $tbl_val[$tbl_slot]
                    end
                    $(
                        if lvl.single_writer
                            nothing
                        else
                            quote
                                if $qos == 0
                                    $stk_slot = Finch.sparse_hash_stack_lookup(
                                        $(lvl.stk_pos),
                                        $(lvl.stk_idx),
                                        $(lvl.stk_cnt),
                                        $(lvl.stk_stop),
                                        $(ctx(pos)),
                                        $(ctx(idx)),
                                    )
                                    if $stk_slot != 0
                                        $qos = $(lvl.stk_val)[$stk_slot]
                                        $(lvl.stk_cnt)[$stk_slot] += 1
                                    end
                                end
                            end
                        end
                    )
                    if $qos == 0
                        # If the qos is not in the table or pending stack, allocate it.
                        $(
                            if lvl.single_writer
                                quote
                                    $qos = $qos_stop + 1
                                    $qos_stop = $qos
                                end
                            else
                                quote
                                    if $qos_free != 0
                                        $qos = $qos_free
                                        $qos_free = -$(lvl.perm)[$qos]
                                        $(lvl.perm)[$qos] = 0
                                    else
                                        $qos = $qos_stop + 1
                                        $qos_stop = $qos
                                    end
                                    ($stk_slot, $(lvl.stk_stop)) =
                                        Finch.sparse_hash_stack_push!(
                                            $(lvl.stk_pos),
                                            $(lvl.stk_idx),
                                            $(lvl.stk_val),
                                            $(lvl.stk_cnt),
                                            $(lvl.stk_stop),
                                            $(ctx(pos)),
                                            $(ctx(idx)),
                                            $qos,
                                        )
                                end
                            end
                        )
                    end
                    $dirty = false
                end,
                body=(ctx) -> instantiate(
                    ctx,
                    VirtualHollowSubFiber(lvl.lvl, value(qos, Tp), dirty),
                    mode,
                ),
                epilogue=if lvl.single_writer
                    quote
                        if $dirty
                            if $(lvl.perm)[$qos] == 0
                                Finch.sparse_hash_table_insert_at_slot!(
                                    $tbl_pos,
                                    $tbl_idx,
                                    $tbl_ctrl,
                                    $tbl_val,
                                    $tbl_slot,
                                    $(ctx(pos)),
                                    $(ctx(idx)),
                                    $qos,
                                )
                                $(lvl.tbl_count) += 1
                                $(lvl.perm)[$qos] = $qos
                            end
                            $(fbr.dirty) = true
                        end
                    end
                else
                    quote
                        if $dirty
                            if $(lvl.perm)[$qos] <= 0
                                $(lvl.perm)[$qos] = $qos
                            end
                            $(fbr.dirty) = true
                        end
                        if $stk_slot != 0
                            $(lvl.stk_cnt)[$stk_slot] -= 1
                            if $(lvl.stk_cnt)[$stk_slot] == 0
                                if $(lvl.perm)[$qos] > 0
                                    $tbl_slot = Finch.sparse_hash_table_lookup_insert_slot(
                                        $tbl_pos,
                                        $tbl_idx,
                                        $tbl_ctrl,
                                        $tbl_val,
                                        $(lvl.stk_pos)[$stk_slot],
                                        $(lvl.stk_idx)[$stk_slot],
                                    )
                                    Finch.sparse_hash_table_insert_at_slot!(
                                        $tbl_pos,
                                        $tbl_idx,
                                        $tbl_ctrl,
                                        $tbl_val,
                                        $tbl_slot,
                                        $(lvl.stk_pos)[$stk_slot],
                                        $(lvl.stk_idx)[$stk_slot],
                                        $qos,
                                    )
                                    $(lvl.tbl_count) += 1
                                else
                                    $(lvl.perm)[$qos] = -$qos_free
                                    $qos_free = $qos
                                end
                                $(lvl.stk_stop) = Finch.sparse_hash_stack_trim(
                                    $(lvl.stk_cnt), $(lvl.stk_stop)
                                )
                            end
                        end
                    end
                end,
            ),
        ),
    )
end
