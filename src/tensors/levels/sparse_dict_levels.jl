"""
    SparseDictLevel{[Ti=Int], [Tp=Int], [Ptr, Idx, Val, TblPos, TblIdx, TblVal]}(lvl, [dim])

A subfiber of a sparse level does not need to represent slices `A[:, ..., :, i]`
which are entirely [`fill_value`](@ref). Instead, only potentially non-fill
slices are stored as subfibers in `lvl`.  A datastructure specified by Tbl is used to record which
slices are stored. Optionally, `dim` is the size of the last dimension.

`Ti` is the type of the last fiber index, and `Tp` is the type used for
positions in the level. The types `Ptr` and `Idx` are the types of the
arrays used to store positions and indicies.

```jldoctest
julia> tensor_tree(Tensor(Dense(SparseDict(Element(0.0))), [10 0 20; 30 0 0; 0 0 40]))
3×3-Tensor
└─ Dense [:,1:3]
   ├─ [:, 1]: SparseDict (0.0) [1:3]
   │  ├─ [1]: 10.0
   │  └─ [2]: 30.0
   ├─ [:, 2]: SparseDict (0.0) [1:3]
   └─ [:, 3]: SparseDict (0.0) [1:3]
      ├─ [1]: 20.0
      └─ [3]: 40.0

julia> tensor_tree(Tensor(SparseDict(SparseDict(Element(0.0))), [10 0 20; 30 0 0; 0 0 40]))
3×3-Tensor
└─ SparseDict (0.0) [:,1:3]
   ├─ [:, 1]: SparseDict (0.0) [1:3]
   │  ├─ [1]: 10.0
   │  └─ [2]: 30.0
   └─ [:, 3]: SparseDict (0.0) [1:3]
      ├─ [1]: 20.0
      └─ [3]: 40.0

```
"""
struct SparseDictLevel{Ti,Ptr,Idx,Val,TblPos,TblIdx,TblVal,Lvl} <: AbstractLevel
    lvl::Lvl
    shape::Ti
    ptr::Ptr
    idx::Idx
    val::Val
    tbl_pos::TblPos
    tbl_idx::TblIdx
    tbl_val::TblVal
end

const SparseDict = SparseDictLevel

@inline sparse_dict_table_capacity(n) = max(4, n <= 1 ? 4 : nextpow(2, 2n))

@inline function sparse_dict_hash_slot(p, i, n)
    return Int(mod(hash((p, i)), UInt(n))) + 1
end

function sparse_dict_table_count(tbl_val)
    n = 0
    @inbounds for v in tbl_val
        n += v > 0
    end
    return n
end

function sparse_dict_table_resize!(tbl_pos, tbl_idx, tbl_val, cap)
    old_pos = copy(tbl_pos)
    old_idx = copy(tbl_idx)
    old_val = copy(tbl_val)
    resize!(tbl_pos, cap)
    resize!(tbl_idx, cap)
    resize!(tbl_val, cap)
    fill!(tbl_val, zero(eltype(tbl_val)))
    @inbounds for h in eachindex(old_val)
        v = old_val[h]
        if v != 0
            sparse_dict_table_insert_noresize!(tbl_pos, tbl_idx, tbl_val, old_pos[h], old_idx[h], v)
        end
    end
    return tbl_pos, tbl_idx, tbl_val
end

function sparse_dict_table_insert_noresize!(tbl_pos, tbl_idx, tbl_val, p, i, v)
    n = length(tbl_val)
    h = sparse_dict_hash_slot(p, i, n)
    @inbounds for _ in 1:n
        val = tbl_val[h]
        if val == 0
            tbl_pos[h] = p
            tbl_idx[h] = i
            tbl_val[h] = v
            return v
        elseif tbl_pos[h] == p && tbl_idx[h] == i
            tbl_val[h] = v
            return v
        end
        h = h == n ? 1 : h + 1
    end
    error("SparseDict linear-probing table is full")
end

function sparse_dict_table_lookup(tbl_pos, tbl_idx, tbl_val, p, i)
    isempty(tbl_val) && return zero(eltype(tbl_val))
    n = length(tbl_val)
    h = sparse_dict_hash_slot(p, i, n)
    @inbounds for _ in 1:n
        val = tbl_val[h]
        val == 0 && return zero(eltype(tbl_val))
        if tbl_pos[h] == p && tbl_idx[h] == i
            return val
        end
        h = h == n ? 1 : h + 1
    end
    return zero(eltype(tbl_val))
end

function sparse_dict_table_rebuild!(tbl_pos, tbl_idx, tbl_val, ptr, idx, val, pos_stop)
    nnz = isempty(ptr) ? 0 : ptr[pos_stop + 1] - 1
    sparse_dict_table_resize!(tbl_pos, tbl_idx, tbl_val, sparse_dict_table_capacity(nnz))
    @inbounds for p in 1:pos_stop
        for q in ptr[p]:(ptr[p + 1] - 1)
            sparse_dict_table_insert_noresize!(tbl_pos, tbl_idx, tbl_val, p, idx[q], val[q])
        end
    end
    return tbl_pos, tbl_idx, tbl_val
end

SparseDictLevel(lvl) = SparseDictLevel{Int}(lvl)
SparseDictLevel(lvl, shape::Ti) where {Ti} = SparseDictLevel{Ti}(lvl, shape)
SparseDictLevel{Ti}(lvl) where {Ti} = SparseDictLevel{Ti}(lvl, zero(Ti))
function SparseDictLevel{Ti}(lvl, shape) where {Ti}
    SparseDictLevel{Ti}(
        lvl,
        shape,
        postype(lvl)[1],
        Ti[],
        postype(lvl)[],
        postype(lvl)[],
        Ti[],
        postype(lvl)[],
    )
end

function SparseDictLevel{Ti}(
    lvl::Lvl,
    shape,
    ptr::Ptr,
    idx::Idx,
    val::Val,
    tbl_pos::TblPos,
    tbl_idx::TblIdx,
    tbl_val::TblVal,
) where {Ti,Ptr,Idx,Val,TblPos,TblIdx,TblVal,Lvl}
    SparseDictLevel{Ti,Ptr,Idx,Val,TblPos,TblIdx,TblVal,Lvl}(
        lvl, shape, ptr, idx, val, tbl_pos, tbl_idx, tbl_val
    )
end

Base.summary(lvl::SparseDictLevel) = "SparseDict($(summary(lvl.lvl)))"
function similar_level(lvl::SparseDictLevel, fill_value, eltype::Type, dim, tail...)
    SparseDict(similar_level(lvl.lvl, fill_value, eltype, tail...), dim)
end

function postype(
    ::Type{SparseDictLevel{Ti,Ptr,Idx,Val,TblPos,TblIdx,TblVal,Lvl}}
) where {Ti,Ptr,Idx,Val,TblPos,TblIdx,TblVal,Lvl}
    return postype(Lvl)
end

function Base.resize!(lvl::SparseDictLevel{Ti}, dims...) where {Ti}
    SparseDictLevel{Ti}(
        resize!(lvl.lvl, dims[1:(end - 1)]...),
        dims[end],
        lvl.ptr,
        lvl.idx,
        lvl.val,
        lvl.tbl_pos,
        lvl.tbl_idx,
        lvl.tbl_val,
    )
end

function transfer(
    Tm, lvl::SparseDictLevel{Ti,Ptr,Idx,Val,TblPos,TblIdx,TblVal,Lvl}
) where {Ti,Ptr,Idx,Val,TblPos,TblIdx,TblVal,Lvl}
    lvl_2 = transfer(Tm, lvl.lvl)
    ptr_2 = transfer(Tm, lvl.ptr)
    idx_2 = transfer(Tm, lvl.idx)
    val_2 = transfer(Tm, lvl.val)
    tbl_pos_2 = transfer(Tm, lvl.tbl_pos)
    tbl_idx_2 = transfer(Tm, lvl.tbl_idx)
    tbl_val_2 = transfer(Tm, lvl.tbl_val)
    return SparseDictLevel{Ti}(
        lvl_2, lvl.shape, ptr_2, idx_2, val_2, tbl_pos_2, tbl_idx_2, tbl_val_2
    )
end

function countstored_level(lvl::SparseDictLevel, pos)
    pos == 0 && return countstored_level(lvl.lvl, pos)
    countstored_level(lvl.lvl, lvl.ptr[pos + 1] - 1)
end

function pattern!(lvl::SparseDictLevel{Ti}) where {Ti}
    SparseDictLevel{Ti}(
        pattern!(lvl.lvl),
        lvl.shape,
        lvl.ptr,
        lvl.idx,
        lvl.val,
        lvl.tbl_pos,
        lvl.tbl_idx,
        lvl.tbl_val,
    )
end

function set_fill_value!(lvl::SparseDictLevel{Ti}, init) where {Ti}
    SparseDictLevel{Ti}(
        set_fill_value!(lvl.lvl, init),
        lvl.shape,
        lvl.ptr,
        lvl.idx,
        lvl.val,
        lvl.tbl_pos,
        lvl.tbl_idx,
        lvl.tbl_val,
    )
end

function Base.show(io::IO, lvl::SparseDictLevel{Ti}) where {Ti}
    if get(io, :compact, false)
        print(io, "SparseDict(")
    else
        print(io, "SparseDict{$Ti}(")
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
        show(io, lvl.idx)
        print(io, ", ")
        show(io, lvl.val)
        print(io, ", ")
        show(io, lvl.tbl_pos)
        print(io, ", ")
        show(io, lvl.tbl_idx)
        print(io, ", ")
        show(io, lvl.tbl_val)
    end
    print(io, ")")
end

function labelled_show(io::IO, fbr::SubFiber{<:SparseDictLevel})
    print(
        io,
        "SparseDict (",
        fill_value(fbr),
        ") [",
        ":,"^(ndims(fbr) - 1),
        "1:",
        size(fbr)[end],
        "]",
    )
end

function labelled_children(fbr::SubFiber{<:SparseDictLevel})
    lvl = fbr.lvl
    pos = fbr.pos
    pos + 1 > length(lvl.ptr) && return []
    map(lvl.ptr[pos]:(lvl.ptr[pos + 1] - 1)) do qos
        LabelledTree(
            cartesian_label([range_label() for _ in 1:(ndims(fbr) - 1)]..., lvl.idx[qos]),
            SubFiber(lvl.lvl, lvl.val[qos]),
        )
    end
end

@inline level_ndims(
    ::Type{<:SparseDictLevel{Ti,Ptr,Idx,Val,TblPos,TblIdx,TblVal,Lvl}}
) where {Ti,Ptr,Idx,Val,TblPos,TblIdx,TblVal,Lvl} = 1 + level_ndims(Lvl)
@inline level_size(lvl::SparseDictLevel) = (level_size(lvl.lvl)..., lvl.shape)
@inline level_axes(lvl::SparseDictLevel) = (level_axes(lvl.lvl)..., Base.OneTo(lvl.shape))
@inline level_eltype(
    ::Type{<:SparseDictLevel{Ti,Ptr,Idx,Val,TblPos,TblIdx,TblVal,Lvl}}
) where {Ti,Ptr,Idx,Val,TblPos,TblIdx,TblVal,Lvl} = level_eltype(Lvl)
@inline level_fill_value(
    ::Type{<:SparseDictLevel{Ti,Ptr,Idx,Val,TblPos,TblIdx,TblVal,Lvl}}
) where {Ti,Ptr,Idx,Val,TblPos,TblIdx,TblVal,Lvl} = level_fill_value(Lvl)
function data_rep_level(
    ::Type{<:SparseDictLevel{Ti,Ptr,Idx,Val,TblPos,TblIdx,TblVal,Lvl}}
) where {Ti,Ptr,Idx,Val,TblPos,TblIdx,TblVal,Lvl}
    SparseData(data_rep_level(Lvl))
end

function isstructequal(a::T, b::T) where {T<:SparseDict}
    a.shape == b.shape &&
        a.tbl_pos == b.tbl_pos &&
        a.tbl_idx == b.tbl_idx &&
        a.tbl_val == b.tbl_val &&
        isstructequal(a.lvl, b.lvl)
end

(fbr::AbstractFiber{<:SparseDictLevel})() = fbr
function (fbr::SubFiber{<:SparseDictLevel{Ti}})(idxs...) where {Ti}
    isempty(idxs) && return fbr
    lvl = fbr.lvl
    p = fbr.pos
    crds = @view lvl.idx[lvl.ptr[p]:(lvl.ptr[p + 1] - 1)]
    r = searchsorted(crds, idxs[end])
    q = lvl.ptr[p] + first(r) - 1
    length(r) == 0 ? fill_value(fbr) : SubFiber(lvl.lvl, lvl.val[q])(idxs[1:(end - 1)]...)
end

mutable struct VirtualSparseDictLevel <: AbstractVirtualLevel
    tag
    lvl
    Ti
    ptr
    idx
    val
    tbl_pos
    tbl_idx
    tbl_val
    shape
    qos_stop
    qos_free
end

function is_level_injective(ctx, lvl::VirtualSparseDictLevel)
    [is_level_injective(ctx, lvl.lvl)..., false]
end
function is_level_atomic(ctx, lvl::VirtualSparseDictLevel)
    (below, atomic) = is_level_atomic(ctx, lvl.lvl)
    return ([below; [atomic]], atomic)
end
function is_level_concurrent(ctx, lvl::VirtualSparseDictLevel)
    (data, _) = is_level_concurrent(ctx, lvl.lvl)
    return ([data; [false]], false)
end

function virtualize(
    ctx,
    ex,
    ::Type{SparseDictLevel{Ti,Ptr,Idx,Val,TblPos,TblIdx,TblVal,Lvl}},
    tag=:lvl,
) where {Ti,Ptr,Idx,Val,TblPos,TblIdx,TblVal,Lvl}
    tag = freshen(ctx, tag)
    ptr = freshen(ctx, tag, :_ptr)
    idx = freshen(ctx, tag, :_idx)
    val = freshen(ctx, tag, :_val)
    tbl_pos = freshen(ctx, tag, :_tbl_pos)
    tbl_idx = freshen(ctx, tag, :_tbl_idx)
    tbl_val = freshen(ctx, tag, :_tbl_val)
    stop = freshen(ctx, tag, :_stop)
    push_preamble!(
        ctx,
        quote
            $tag = $ex
            $ptr = $tag.ptr
            $idx = $tag.idx
            $val = $tag.val
            $tbl_pos = $tag.tbl_pos
            $tbl_idx = $tag.tbl_idx
            $tbl_val = $tag.tbl_val
            $stop = $tag.shape
        end,
    )
    qos_stop = freshen(ctx, tag, :_qos_stop)
    qos_free = freshen(ctx, tag, :_qos_free)
    shape = value(stop, Int)
    lvl_2 = virtualize(ctx, :($tag.lvl), Lvl, tag)
    VirtualSparseDictLevel(
        tag, lvl_2, Ti, ptr, idx, val, tbl_pos, tbl_idx, tbl_val, shape, qos_stop,
        qos_free,
    )
end
function lower(ctx::AbstractCompiler, lvl::VirtualSparseDictLevel, ::DefaultStyle)
    quote
        $SparseDictLevel{$(lvl.Ti)}(
            $(ctx(lvl.lvl)),
            $(ctx(lvl.shape)),
            $(lvl.ptr),
            $(lvl.idx),
            $(lvl.val),
            $(lvl.tbl_pos),
            $(lvl.tbl_idx),
            $(lvl.tbl_val),
        )
    end
end

function distribute_level(
    ctx::AbstractCompiler, lvl::VirtualSparseDictLevel, arch, diff, style
)
    return diff[lvl.tag] = VirtualSparseDictLevel(
        lvl.tag,
        distribute_level(ctx, lvl.lvl, arch, diff, style),
        lvl.Ti,
        distribute_buffer(ctx, lvl.ptr, arch, style),
        distribute_buffer(ctx, lvl.idx, arch, style),
        distribute_buffer(ctx, lvl.val, arch, style),
        distribute_buffer(ctx, lvl.tbl_pos, arch, style),
        distribute_buffer(ctx, lvl.tbl_idx, arch, style),
        distribute_buffer(ctx, lvl.tbl_val, arch, style),
        lvl.shape,
        lvl.qos_stop,
        lvl.qos_free,
    )
end

function redistribute(ctx::AbstractCompiler, lvl::VirtualSparseDictLevel, diff)
    get(
        diff,
        lvl.tag,
        VirtualSparseDictLevel(
            lvl.tag,
            redistribute(ctx, lvl.lvl, diff),
            lvl.Ti,
            lvl.ptr,
            lvl.idx,
            lvl.val,
            lvl.tbl_pos,
            lvl.tbl_idx,
            lvl.tbl_val,
            lvl.shape,
            lvl.qos_stop,
            lvl.qos_free,
        ),
    )
end

Base.summary(lvl::VirtualSparseDictLevel) = "SparseDict($(summary(lvl.lvl)))"

function virtual_level_size(ctx, lvl::VirtualSparseDictLevel)
    ext = virtual_call(ctx, extent, literal(lvl.Ti(1)), lvl.shape)
    (virtual_level_size(ctx, lvl.lvl)..., ext)
end

function virtual_level_resize!(ctx, lvl::VirtualSparseDictLevel, dims...)
    lvl.shape = getstop(dims[end])
    lvl.lvl = virtual_level_resize!(ctx, lvl.lvl, dims[1:(end - 1)]...)
    lvl
end

virtual_level_eltype(lvl::VirtualSparseDictLevel) = virtual_level_eltype(lvl.lvl)
virtual_level_fill_value(lvl::VirtualSparseDictLevel) = virtual_level_fill_value(lvl.lvl)

postype(lvl::VirtualSparseDictLevel) = postype(lvl.lvl)

function declare_level!(ctx::AbstractCompiler, lvl::VirtualSparseDictLevel, pos, init)
    #TODO check that init == fill_value
    Ti = lvl.Ti
    Tp = postype(lvl)
    qos = freshen(ctx, tag, :qos)
    push_preamble!(
        ctx,
        quote
            empty!($(lvl.tbl_pos))
            empty!($(lvl.tbl_idx))
            empty!($(lvl.tbl_val))
            $qos = $(Tp(0))
            $(lvl.qos_stop) = 0
            $(lvl.qos_free) = 0
            resize!($(lvl.val), 0)
        end,
    )
    lvl.lvl = declare_level!(ctx, lvl.lvl, value(qos, Tp), init)
    return lvl
end

function assemble_level!(ctx, lvl::VirtualSparseDictLevel, pos_start, pos_stop)
    pos_start = ctx(cache!(ctx, :p_start, pos_start))
    pos_stop = ctx(cache!(ctx, :p_start, pos_stop))
    Tp = postype(lvl)
    old = freshen(ctx, lvl.tag, :old)
    tbl_cap = freshen(ctx, lvl.tag, :tbl_cap)

    quote
        $old = length($(lvl.val)) + 1
        Finch.resize_if_smaller!($(lvl.val), $pos_stop)
        if $old <= $pos_stop
            Finch.fill_range!($(lvl.val), 0, $old, $pos_stop)
        end
        $tbl_cap = Finch.sparse_dict_table_capacity($pos_stop)
        if length($(lvl.tbl_val)) < $tbl_cap
            Finch.sparse_dict_table_resize!(
                $(lvl.tbl_pos), $(lvl.tbl_idx), $(lvl.tbl_val), $tbl_cap
            )
        end
        $(contain(
            ctx_2 -> assemble_level!(ctx_2, lvl.lvl, value(old, Tp), value(pos_stop, Tp)),
            ctx,
        ))
    end
end

function freeze_level!(ctx::AbstractCompiler, lvl::VirtualSparseDictLevel, pos_stop)
    p = freshen(ctx, :p)
    Tp = postype(lvl)
    Ti = lvl.Ti
    pos_stop = cache!(ctx, :pos_stop, simplify(ctx, pos_stop))
    qos_stop = freshen(ctx, :qos_stop)
    p = freshen(ctx, :p)
    q = freshen(ctx, :q)
    r = freshen(ctx, :r)
    i = freshen(ctx, :i)
    v = freshen(ctx, :v)
    idx_tmp = freshen(ctx, :idx_tmp)
    val_tmp = freshen(ctx, :val_tmp)
    perm = freshen(ctx, :perm)
    pdx_tmp = freshen(ctx, :pdx_tmp)
    entry = freshen(ctx, :entry)
    ptr_2 = freshen(ctx, :ptr_2)
    qos_max = freshen(ctx, :qos_max)
    tbl_len = freshen(ctx, :tbl_len)
    h = freshen(ctx, :h)
    push_preamble!(
        ctx,
        quote
            $tbl_len = Finch.sparse_dict_table_count($(lvl.tbl_val))
            resize!($(lvl.ptr), $(ctx(pos_stop)) + 1)
            $(lvl.ptr)[1] = 1
            Finch.fill_range!($(lvl.ptr), 0, 2, $(ctx(pos_stop)) + 1)
            $pdx_tmp = Vector{$Tp}(undef, $tbl_len)
            resize!($(lvl.idx), $tbl_len)
            $idx_tmp = Vector{$Ti}(undef, $tbl_len)
            $val_tmp = Vector{$Tp}(undef, $tbl_len)
            $q = 0
            $qos_max = $(Tp(0))
            for $h in eachindex($(lvl.tbl_val))
                $v = $(lvl.tbl_val)[$h]
                if $v > 0
                    $p = $(lvl.tbl_pos)[$h]
                    $i = $(lvl.tbl_idx)[$h]
                    $q += 1
                    $idx_tmp[$q] = $i
                    $val_tmp[$q] = $v
                    $qos_max = max($qos_max, $v)
                    $pdx_tmp[$q] = $p
                    $(lvl.ptr)[$p + 1] += 1
                end
            end
            # In read mode, val[1:length(tbl)] stores child positions; the tail
            # encodes free qoses from older tables.
            $p = $qos_max
            $q = $(lvl.qos_free)
            while $q != 0
                $v = -$(lvl.val)[$q]
                if $q <= $tbl_len
                    while $(lvl.val)[$p] <= 0
                        $p -= 1
                    end
                    $(lvl.val)[$p] = $q
                    $p -= 1
                end
                $q = $v
            end
            resize!($(lvl.val), $qos_max)
            for $p in 2:($(ctx(pos_stop)) + 1)
                $(lvl.ptr)[$p] += $(lvl.ptr)[$p - 1]
            end
            $perm = sortperm($idx_tmp)
            $ptr_2 = copy($(lvl.ptr))
            for $q in $perm
                $p = $pdx_tmp[$q]
                $r = $ptr_2[$p]
                $(lvl.idx)[$r] = $idx_tmp[$q]
                $(lvl.val)[$r] = $val_tmp[$q]
                $ptr_2[$p] += 1
            end
            $qos_stop = $qos_max
        end,
    )
    lvl.lvl = freeze_level!(ctx, lvl.lvl, value(qos_stop))
    return lvl
end

function thaw_level!(ctx::AbstractCompiler, lvl::VirtualSparseDictLevel, pos_stop)
    p = freshen(ctx, :p)
    v = freshen(ctx, :v)
    tbl_len = freshen(ctx, :tbl_len)
    pos_stop = ctx(cache!(ctx, :pos_stop, simplify(ctx, pos_stop)))
    push_preamble!(
        ctx,
        quote
            $(lvl.qos_stop) = length($(lvl.val))
            $(lvl.qos_free) = 0
            $tbl_len = Finch.sparse_dict_table_count($(lvl.tbl_val))
            for $p in ($tbl_len + 1):$(lvl.qos_stop)
                $v = $(lvl.val)[$p]
                if $v <= 0
                    $v = $p
                end
                $(lvl.val)[$v] = -$(lvl.qos_free)
                $(lvl.qos_free) = $v
            end
            for $v in $(lvl.tbl_val)
                if $v > 0
                    $(lvl.val)[$v] = $v
                end
            end
        end,
    )
    lvl.lvl = thaw_level!(ctx, lvl.lvl, value(lvl.qos_stop))
    return lvl
end

function unfurl(
    ctx,
    fbr::VirtualSubFiber{VirtualSparseDictLevel},
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
                $my_i = $(lvl.idx)[$my_q]
                $my_i1 = $(lvl.idx)[$my_q_stop - $(Tp(1))]
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
                        if $(lvl.idx)[$my_q] < $(ctx(getstart(ext)))
                            $my_q = Finch.scansearch(
                                $(lvl.idx),
                                $(ctx(getstart(ext))),
                                $my_q,
                                $my_q_stop - 1,
                            )
                            $my_i = $(lvl.idx)[$my_q]
                        end
                    end,
                    preamble=quote
                        $my_i = $(lvl.idx)[$my_q]
                        $my_v = $(lvl.val)[$my_q]
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
    ctx, fbr::VirtualSubFiber{VirtualSparseDictLevel}, ext, mode, ::typeof(follow)
)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.tag
    Tp = postype(lvl)
    my_q = freshen(ctx, tag, :_q)

    Lookup(;
        body=(ctx, i) -> Thunk(;
            preamble=quote
                $my_q = Finch.sparse_dict_table_lookup(
                    $(lvl.tbl_pos),
                    $(lvl.tbl_idx),
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
    fbr::VirtualSubFiber{VirtualSparseDictLevel},
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
    fbr::VirtualHollowSubFiber{VirtualSparseDictLevel},
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
    tbl_pos = freshen(ctx, tag, :_tbl_pos)
    tbl_idx = freshen(ctx, tag, :_tbl_idx)
    tbl_val = freshen(ctx, tag, :_tbl_val)

    Thunk(;
        body=(ctx) -> Lookup(;
            body=(ctx, idx) -> Thunk(;
                preamble=quote
                    $tbl_pos = $(lvl.tbl_pos)
                    $tbl_idx = $(lvl.tbl_idx)
                    $tbl_val = $(lvl.tbl_val)
                    $qos = Finch.sparse_dict_table_lookup(
                        $tbl_pos, $tbl_idx, $tbl_val, $(ctx(pos)), $(ctx(idx))
                    )
                    if $qos == 0
                        #If the qos is not in the table, we need to add it.
                        #We need to commit it to the table in the event that
                        #another accessor tries to write it in the same loop.
                        if $qos_free != 0
                            $qos = $qos_free
                            $qos_free = -$(lvl.val)[$qos]
                            $(lvl.val)[$qos] = 0
                        else
                            $qos = $qos_stop + 1
                            $qos_stop = $qos
                        end
                        if $qos > length($(lvl.val))
                            $p = length($(lvl.val)) + 1
                            $q_stop = max(length($(lvl.val)) << 1, $qos)
                            $(contain(
                                ctx_2 -> assemble_level!(
                                    ctx_2,
                                    lvl,
                                    value(p, Tp),
                                    value(q_stop, Tp),
                                ),
                                ctx,
                            ))
                        end
                        Finch.sparse_dict_table_insert_noresize!(
                            $tbl_pos, $tbl_idx, $tbl_val, $(ctx(pos)), $(ctx(idx)), $qos
                        )
                    end
                    $dirty = false
                end,
                body=(ctx) -> instantiate(
                    ctx,
                    VirtualHollowSubFiber(lvl.lvl, value(qos, Tp), dirty),
                    mode,
                ),
                epilogue=quote
                    if $dirty
                        if $(lvl.val)[$qos] <= 0
                            $(lvl.val)[$qos] = $qos
                        end
                        $(fbr.dirty) = true
                    elseif $(lvl.val)[$qos] == 0 #here, val is being used as a dirty bit
                        $(lvl.val)[$qos] = $qos
                    end
                end,
            ),
        ),
    )
end

function coalesce_level!(
    lvl::SparseDictLevel, global_fbr_map, local_fbr_map, task_map, factor, P, coalescent
)
    if factor > 1
        global_fbr_map, local_fbr_map, task_map = unroll_dense_coalesce(
            global_fbr_map, local_fbr_map, task_map, factor, P
        )
        factor = 1
    end

    #lvl.idx and lvl.ptr should be MultiChannelBuffers
    idx = lvl.idx.data
    ptr = lvl.ptr.data
    # val = lvl.val.data
    max_level_dim = global_fbr_map[length(global_fbr_map)]
    cutoffs = compute_proc_cutoffs(idx, P)

    #Don't merge zero-ed arrays.
    if cutoffs[P + 1] == 1
        return coalescent
    end

    pos_map, idx_map, lfm, tm = gen_pos_idx_map_hash(
        global_fbr_map,
        local_fbr_map,
        task_map,
        ptr,
        idx,
        cutoffs,
        P,
        lvl.tbl_pos.data,
        lvl.tbl_idx.data,
        lvl.tbl_val.data,
    )
    global_fbr_map, local_fbr_map, task_map = process_next_lvl_hash(
        pos_map, idx_map, tm, lfm, P, max_level_dim, coalescent.ptr, coalescent.idx,
        coalescent.val, coalescent.tbl_pos, coalescent.tbl_idx, coalescent.tbl_val,
    )

    coalesce_level!(
        lvl.lvl, global_fbr_map, local_fbr_map, task_map, factor, P, coalescent.lvl
    )
end

Base.@propagate_inbounds function gen_pos_idx_map_hash(
    global_fbr_map,
    local_fbr_map,
    task_map,
    ptr,
    index,
    cutoffs,
    P,
    tbl_pos,
    tbl_idx,
    tbl_val,
)
    ordering = Base.Order.By(j -> (task_map[j], local_fbr_map[j]))
    sorter = AcceleratedKernels.sortperm(collect(1:length(task_map)); order=ordering)

    nnz = cutoffs[length(cutoffs)] - 1
    merged_positions = Vector{Int}(undef, nnz)
    merged_indices = Vector{Int}(undef, nnz)

    task_map2 = Vector{Int}(undef, nnz)
    local_fbr_map2 = Vector{Int}(undef, nnz)

    chk_size = fld(nnz + P - 1, P)
    Threads.@threads for tid in 1:P
        init = (tid - 1) * chk_size + 1
        proc_id = binary_search(init, cutoffs)

        if proc_id < 1
            continue
        end

        idx_id = init - cutoffs[proc_id] + 1

        local_fbr = binary_search(idx_id, ptr[proc_id])

        tag = get_permute_idx(proc_id, ptr) + local_fbr

        @assert local_fbr > 0
        @assert tag > 0

        global_fbr = global_fbr_map[sorter[tag]]

        j = 0
        for i in 0:(chk_size - 1)
            offset = init + i
            if offset > nnz
                break
            end

            nz_id = j + idx_id
            idx = index[proc_id][nz_id]
            merged_positions[offset] = global_fbr
            merged_indices[offset] = idx
            task_map2[offset] = proc_id
            local_fbr_map2[offset] = sparse_dict_table_lookup(
                tbl_pos[proc_id], tbl_idx[proc_id], tbl_val[proc_id], local_fbr, idx
            )

            if nz_id >= length(index[proc_id]) && proc_id < P
                proc_id += 1
                while proc_id < P && length(index[proc_id]) < 1
                    proc_id += 1
                end

                if length(index[proc_id]) < 1
                    break
                end

                idx_id = 1
                j = 0

                local_fbr = binary_search(idx_id, ptr[proc_id])
                tag = get_permute_idx(proc_id, ptr) + local_fbr

                global_fbr = global_fbr_map[sorter[tag]]
            elseif nz_id + 1 >= ptr[proc_id][local_fbr + 1] &&
                local_fbr + 1 < length(ptr[proc_id]) &&
                ptr[proc_id][local_fbr + 1] < ptr[proc_id][length(ptr[proc_id])]
                local_fbr = binary_search(nz_id + 1, ptr[proc_id])

                tag = get_permute_idx(proc_id, ptr) + local_fbr
                global_fbr = global_fbr_map[sorter[tag]]
                j += 1
            else
                j += 1
            end
        end
    end
    return merged_positions, merged_indices, local_fbr_map2, task_map2
end

Base.@propagate_inbounds function process_next_lvl_hash(
    merged_positions, merged_indices, task_map, local_fbr_map, P, max_level_dim, lvl_ptr,
    lvl_idx, lvl_val, lvl_tbl_pos, lvl_tbl_idx, lvl_tbl_val,
)
    ordering = Base.Order.By(j -> (merged_positions[j], merged_indices[j]))
    shuffler = AcceleratedKernels.sortperm(
        collect(1:length(merged_positions)); order=ordering
    )

    nnz = length(local_fbr_map)
    global_fbr_map2 = Vector{Int}(undef, nnz)

    merged_positions_s = p_permute(shuffler, merged_positions)
    merged_indices_s = p_permute(shuffler, merged_indices)
    task_map = p_permute(shuffler, task_map)
    local_fbr_map = p_permute(shuffler, local_fbr_map)

    uq_ptr = zeros(Int, P + 1)
    uq_idx = zeros(Int, P + 1)

    chk_size = fld(nnz + P - 1, P)

    Threads.@threads for tid in 1:P
        init = (tid - 1) * chk_size + 1

        if init > length(merged_positions_s)
            continue
        end

        seen = 0
        prev =
            init > 1 ? (merged_positions_s[init - 1], merged_indices_s[init - 1]) : (-1, -1)
        prev_ptr = init > 1 ? merged_positions_s[init - 1] : 1
        seen_ptr = 0

        for i in 0:(chk_size - 1)
            offset = init + i
            if offset > nnz
                break
            end

            tup = (merged_positions_s[offset], merged_indices_s[offset])
            if tup != prev
                prev = tup
                seen += 1
            end

            p = merged_positions_s[offset]
            if prev_ptr != p
                seen_ptr += (p - prev_ptr)
                prev_ptr = p
            end
        end
        uq_idx[tid + 1] = seen
        uq_ptr[tid + 1] = seen_ptr
    end
    uq_ptr_s = s_prefix_sum(uq_ptr)
    uq_idx_s = s_prefix_sum(uq_idx)

    Finch.resize_if_smaller!(lvl_ptr, max_level_dim + 1)
    fill!(lvl_ptr, 0)
    Finch.resize_if_smaller!(lvl_idx, uq_idx_s[length(uq_idx_s)])
    Finch.resize_if_smaller!(lvl_val, uq_idx_s[length(uq_idx_s)])

    Threads.@threads for tid in 1:P
        init = (tid - 1) * chk_size + 1
        if init > length(merged_positions_s)
            continue
        end
        seen_ptr = uq_ptr_s[tid] + 2
        seen_idx = uq_idx_s[tid] + 1
        prev =
            init > 1 ? (merged_positions_s[init - 1], merged_indices_s[init - 1]) : (1, -1)

        for i in 0:(chk_size - 1)
            offset = init + i
            if offset > nnz
                break
            end

            while seen_ptr < merged_positions_s[offset]
                lvl_ptr[seen_ptr] = seen_idx
                seen_ptr += 1
            end

            tup = (merged_positions_s[offset], merged_indices_s[offset])
            if tup != prev
                lvl_idx[seen_idx] = tup[2]
                lvl_val[seen_idx] = seen_idx

                p = merged_positions_s[offset]
                if prev[1] != p
                    lvl_ptr[seen_ptr] = seen_idx
                    seen_ptr += 1
                end
                prev = tup
                seen_idx += 1
            end
            global_fbr_map2[offset] = seen_idx - 1
        end
    end

    lvl_ptr[1] = 1
    i = length(lvl_ptr)
    while lvl_ptr[i] == 0
        lvl_ptr[i] = length(lvl_idx) + 1
        i -= 1
    end
    return global_fbr_map2, local_fbr_map, task_map
end
