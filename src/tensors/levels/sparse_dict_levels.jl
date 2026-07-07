"""
    SparseDictLevel{[Ti=Int], [Tp=Int], [Ptr, TblPos, TblIdx, TblVal, Perm]}(lvl, [dim])

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
struct SparseDictLevel{Ti,Ptr,TblPos,TblIdx,TblVal,Perm,Lvl} <: AbstractLevel
    lvl::Lvl
    shape::Ti
    ptr::Ptr
    tbl_pos::TblPos
    tbl_idx::TblIdx
    tbl_val::TblVal
    perm::Perm
end

const SparseDict = SparseDictLevel

@inline sparse_dict_table_capacity(n) = max(4, n <= 1 ? 4 : nextpow(2, 2n))

@inline function sparse_dict_hash_slot(p, i, n)
    return Int(mod(hash((p, i)), UInt(n))) + 1
end

@inline function sparse_dict_table_resize!(tbl_pos, tbl_idx, tbl_val, cap)
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

@inline function sparse_dict_table_insert_slot_noresize!(tbl_pos, tbl_idx, tbl_val, p, i, v)
    n = length(tbl_val)
    h = sparse_dict_hash_slot(p, i, n)
    @inbounds for _ in 1:n
        val = tbl_val[h]
        if val == 0
            tbl_pos[h] = p
            tbl_idx[h] = i
            tbl_val[h] = v
            return h
        elseif tbl_pos[h] == p && tbl_idx[h] == i
            tbl_val[h] = v
            return h
        end
        h = h == n ? 1 : h + 1
    end
    error("SparseDict linear-probing table is full")
end

@inline function sparse_dict_table_insert_noresize!(tbl_pos, tbl_idx, tbl_val, p, i, v)
    sparse_dict_table_insert_slot_noresize!(tbl_pos, tbl_idx, tbl_val, p, i, v)
    return v
end

@inline function sparse_dict_table_lookup_slot(tbl_pos, tbl_idx, tbl_val, p, i)
    isempty(tbl_val) && return 0
    n = length(tbl_val)
    h = sparse_dict_hash_slot(p, i, n)
    @inbounds for _ in 1:n
        val = tbl_val[h]
        if val == 0 || (tbl_pos[h] == p && tbl_idx[h] == i)
            return h
        end
        h = h == n ? 1 : h + 1
    end
    error("SparseDict linear-probing table is full")
end

@inline function sparse_dict_table_lookup(tbl_pos, tbl_idx, tbl_val, p, i)
    h = sparse_dict_table_lookup_slot(tbl_pos, tbl_idx, tbl_val, p, i)
    h == 0 && return zero(eltype(tbl_val))
    return tbl_val[h]
end

@inline function sparse_dict_table_insert_at_slot!(tbl_pos, tbl_idx, tbl_val, h, p, i, v)
    tbl_pos[h] = p
    tbl_idx[h] = i
    tbl_val[h] = v
    return v
end

function sparse_dict_table_sort_perm!(perm, tbl_pos, tbl_idx)
    sort!(perm; by=h -> (tbl_pos[h], tbl_idx[h]))
    return perm
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

function sparse_dict_table_rebuild_perm!(
    perm, tbl_pos, tbl_idx, tbl_val, ptr, idx, val, pos_stop
)
    nnz = isempty(ptr) ? 0 : ptr[pos_stop + 1] - 1
    sparse_dict_table_resize!(tbl_pos, tbl_idx, tbl_val, sparse_dict_table_capacity(nnz))
    resize!(perm, nnz)
    @inbounds for p in 1:pos_stop
        for q in ptr[p]:(ptr[p + 1] - 1)
            perm[q] = sparse_dict_table_insert_slot_noresize!(
                tbl_pos, tbl_idx, tbl_val, p, idx[q], val[q]
            )
        end
    end
    return perm, tbl_pos, tbl_idx, tbl_val
end

SparseDictLevel(lvl) = SparseDictLevel{Int}(lvl)
SparseDictLevel(lvl, shape::Ti) where {Ti} = SparseDictLevel{Ti}(lvl, shape)
SparseDictLevel{Ti}(lvl) where {Ti} = SparseDictLevel{Ti}(lvl, zero(Ti))
function SparseDictLevel{Ti}(lvl, shape) where {Ti}
    SparseDictLevel{Ti}(
        lvl,
        shape,
        postype(lvl)[1],
        postype(lvl)[],
        Ti[],
        postype(lvl)[],
        postype(lvl)[],
    )
end

function SparseDictLevel{Ti}(
    lvl::Lvl,
    shape,
    ptr::Ptr,
    tbl_pos::TblPos,
    tbl_idx::TblIdx,
    tbl_val::TblVal,
    perm::Perm,
) where {Ti,Ptr,TblPos,TblIdx,TblVal,Perm,Lvl}
    SparseDictLevel{Ti,Ptr,TblPos,TblIdx,TblVal,Perm,Lvl}(
        lvl, shape, ptr, tbl_pos, tbl_idx, tbl_val, perm
    )
end

Base.summary(lvl::SparseDictLevel) = "SparseDict($(summary(lvl.lvl)))"
function similar_level(lvl::SparseDictLevel, fill_value, eltype::Type, dim, tail...)
    SparseDict(similar_level(lvl.lvl, fill_value, eltype, tail...), dim)
end

function postype(
    ::Type{SparseDictLevel{Ti,Ptr,TblPos,TblIdx,TblVal,Perm,Lvl}}
) where {Ti,Ptr,TblPos,TblIdx,TblVal,Perm,Lvl}
    return postype(Lvl)
end

function Base.resize!(lvl::SparseDictLevel{Ti}, dims...) where {Ti}
    SparseDictLevel{Ti}(
        resize!(lvl.lvl, dims[1:(end - 1)]...),
        dims[end],
        lvl.ptr,
        lvl.tbl_pos,
        lvl.tbl_idx,
        lvl.tbl_val,
        lvl.perm,
    )
end

function transfer(
    Tm, lvl::SparseDictLevel{Ti,Ptr,TblPos,TblIdx,TblVal,Perm,Lvl}
) where {Ti,Ptr,TblPos,TblIdx,TblVal,Perm,Lvl}
    lvl_2 = transfer(Tm, lvl.lvl)
    ptr_2 = transfer(Tm, lvl.ptr)
    tbl_pos_2 = transfer(Tm, lvl.tbl_pos)
    tbl_idx_2 = transfer(Tm, lvl.tbl_idx)
    tbl_val_2 = transfer(Tm, lvl.tbl_val)
    perm_2 = transfer(Tm, lvl.perm)
    return SparseDictLevel{Ti}(
        lvl_2, lvl.shape, ptr_2, tbl_pos_2, tbl_idx_2, tbl_val_2, perm_2
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
        lvl.tbl_pos,
        lvl.tbl_idx,
        lvl.tbl_val,
        lvl.perm,
    )
end

function set_fill_value!(lvl::SparseDictLevel{Ti}, init) where {Ti}
    SparseDictLevel{Ti}(
        set_fill_value!(lvl.lvl, init),
        lvl.shape,
        lvl.ptr,
        lvl.tbl_pos,
        lvl.tbl_idx,
        lvl.tbl_val,
        lvl.perm,
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
        show(io, lvl.tbl_pos)
        print(io, ", ")
        show(io, lvl.tbl_idx)
        print(io, ", ")
        show(io, lvl.tbl_val)
        print(io, ", ")
        show(io, lvl.perm)
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
            cartesian_label(
                [range_label() for _ in 1:(ndims(fbr) - 1)]...,
                lvl.tbl_idx[lvl.perm[qos]],
            ),
            SubFiber(lvl.lvl, lvl.tbl_val[lvl.perm[qos]]),
        )
    end
end

@inline level_ndims(
    ::Type{<:SparseDictLevel{Ti,Ptr,TblPos,TblIdx,TblVal,Perm,Lvl}}
) where {Ti,Ptr,TblPos,TblIdx,TblVal,Perm,Lvl} = 1 + level_ndims(Lvl)
@inline level_size(lvl::SparseDictLevel) = (level_size(lvl.lvl)..., lvl.shape)
@inline level_axes(lvl::SparseDictLevel) = (level_axes(lvl.lvl)..., Base.OneTo(lvl.shape))
@inline level_eltype(
    ::Type{<:SparseDictLevel{Ti,Ptr,TblPos,TblIdx,TblVal,Perm,Lvl}}
) where {Ti,Ptr,TblPos,TblIdx,TblVal,Perm,Lvl} = level_eltype(Lvl)
@inline level_fill_value(
    ::Type{<:SparseDictLevel{Ti,Ptr,TblPos,TblIdx,TblVal,Perm,Lvl}}
) where {Ti,Ptr,TblPos,TblIdx,TblVal,Perm,Lvl} = level_fill_value(Lvl)
function data_rep_level(
    ::Type{<:SparseDictLevel{Ti,Ptr,TblPos,TblIdx,TblVal,Perm,Lvl}}
) where {Ti,Ptr,TblPos,TblIdx,TblVal,Perm,Lvl}
    SparseData(data_rep_level(Lvl))
end

function isstructequal(a::T, b::T) where {T<:SparseDict}
    a.shape == b.shape &&
        a.tbl_pos == b.tbl_pos &&
        a.tbl_idx == b.tbl_idx &&
        a.tbl_val == b.tbl_val &&
        a.perm == b.perm &&
        isstructequal(a.lvl, b.lvl)
end

(fbr::AbstractFiber{<:SparseDictLevel})() = fbr
function (fbr::SubFiber{<:SparseDictLevel{Ti}})(idxs...) where {Ti}
    isempty(idxs) && return fbr
    lvl = fbr.lvl
    p = fbr.pos
    crds = [lvl.tbl_idx[lvl.perm[q]] for q in lvl.ptr[p]:(lvl.ptr[p + 1] - 1)]
    r = searchsorted(crds, idxs[end])
    q = lvl.ptr[p] + first(r) - 1
    length(r) == 0 ? fill_value(fbr) :
    SubFiber(lvl.lvl, lvl.tbl_val[lvl.perm[q]])(idxs[1:(end - 1)]...)
end

mutable struct VirtualSparseDictLevel <: AbstractVirtualLevel
    tag
    lvl
    Ti
    ptr
    tbl_pos
    tbl_idx
    tbl_val
    perm
    shape
    qos_stop
    qos_free
    tbl_count
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
    ::Type{SparseDictLevel{Ti,Ptr,TblPos,TblIdx,TblVal,Perm,Lvl}},
    tag=:lvl,
) where {Ti,Ptr,TblPos,TblIdx,TblVal,Perm,Lvl}
    tag = freshen(ctx, tag)
    ptr = freshen(ctx, tag, :_ptr)
    tbl_pos = freshen(ctx, tag, :_tbl_pos)
    tbl_idx = freshen(ctx, tag, :_tbl_idx)
    tbl_val = freshen(ctx, tag, :_tbl_val)
    perm = freshen(ctx, tag, :_perm)
    stop = freshen(ctx, tag, :_stop)
    push_preamble!(
        ctx,
        quote
            $tag = $ex
            $ptr = $tag.ptr
            $tbl_pos = $tag.tbl_pos
            $tbl_idx = $tag.tbl_idx
            $tbl_val = $tag.tbl_val
            $perm = $tag.perm
            $stop = $tag.shape
        end,
    )
    qos_stop = freshen(ctx, tag, :_qos_stop)
    qos_free = freshen(ctx, tag, :_qos_free)
    tbl_count = freshen(ctx, tag, :_tbl_count)
    shape = value(stop, Int)
    lvl_2 = virtualize(ctx, :($tag.lvl), Lvl, tag)
    VirtualSparseDictLevel(
        tag, lvl_2, Ti, ptr, tbl_pos, tbl_idx, tbl_val, perm, shape, qos_stop,
        qos_free, tbl_count,
    )
end
function lower(ctx::AbstractCompiler, lvl::VirtualSparseDictLevel, ::DefaultStyle)
    quote
        $SparseDictLevel{$(lvl.Ti)}(
            $(ctx(lvl.lvl)),
            $(ctx(lvl.shape)),
            $(lvl.ptr),
            $(lvl.tbl_pos),
            $(lvl.tbl_idx),
            $(lvl.tbl_val),
            $(lvl.perm),
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
        distribute_buffer(ctx, lvl.tbl_pos, arch, style),
        distribute_buffer(ctx, lvl.tbl_idx, arch, style),
        distribute_buffer(ctx, lvl.tbl_val, arch, style),
        distribute_buffer(ctx, lvl.perm, arch, style),
        lvl.shape,
        lvl.qos_stop,
        lvl.qos_free,
        lvl.tbl_count,
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
            lvl.tbl_pos,
            lvl.tbl_idx,
            lvl.tbl_val,
            lvl.perm,
            lvl.shape,
            lvl.qos_stop,
            lvl.qos_free,
            lvl.tbl_count,
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
            $(lvl.tbl_count) = 0
            resize!($(lvl.perm), 0)
        end,
    )
    lvl.lvl = declare_level!(ctx, lvl.lvl, value(qos, Tp), init)
    return lvl
end

function assemble_level!(ctx, lvl::VirtualSparseDictLevel, pos_start, pos_stop)
    pos_start = ctx(cache!(ctx, :p_start, pos_start))
    pos_stop = ctx(cache!(ctx, :p_start, pos_stop))
end

function freeze_level!(ctx::AbstractCompiler, lvl::VirtualSparseDictLevel, pos_stop)
    p = freshen(ctx, :p)
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
    push_preamble!(
        ctx,
        quote
            resize!($(lvl.ptr), $(ctx(pos_stop)) + 1)
            $(lvl.ptr)[1] = 1
            Finch.fill_range!($(lvl.ptr), 0, 2, $(ctx(pos_stop)) + 1)
            $q = 0
            $qos_max = $(Tp(0))
            for $h in eachindex($(lvl.tbl_val))
                $v = $(lvl.tbl_val)[$h]
                if $v > 0
                    $p = $(lvl.tbl_pos)[$h]
                    $q += 1
                    $qos_max = max($qos_max, $v)
                    $(lvl.ptr)[$p + 1] += 1
                end
            end
            # In read mode, val[1:length(tbl)] stores child positions; the tail
            # encodes free qoses from older tables.
            $p = $qos_max
            $q = $(lvl.qos_free)
            while $q != 0
                $v = -$(lvl.perm)[$q]
                if $q <= $tbl_count
                    while $(lvl.perm)[$p] <= 0
                        $p -= 1
                    end
                    $(lvl.perm)[$p] = $q
                    $p -= 1
                end
                $q = $v
            end
            resize!($(lvl.perm), $tbl_count)
            $q = 0
            for $h in eachindex($(lvl.tbl_val))
                if $(lvl.tbl_val)[$h] > 0
                    $q += 1
                    $(lvl.perm)[$q] = $h
                end
            end
            for $p in 2:($(ctx(pos_stop)) + 1)
                $(lvl.ptr)[$p] += $(lvl.ptr)[$p - 1]
            end
            Finch.sparse_dict_table_sort_perm!($(lvl.perm), $(lvl.tbl_pos), $(lvl.tbl_idx))
            $qos_stop = $qos_max
        end,
    )
    lvl.lvl = freeze_level!(ctx, lvl.lvl, value(qos_stop))
    return lvl
end

function thaw_level!(ctx::AbstractCompiler, lvl::VirtualSparseDictLevel, pos_stop)
    p = freshen(ctx, :p)
    v = freshen(ctx, :v)
    tbl_count = lvl.tbl_count
    pos_stop = ctx(cache!(ctx, :pos_stop, simplify(ctx, pos_stop)))
    push_preamble!(
        ctx,
        quote
            $(lvl.qos_stop) = length($(lvl.perm))
            $(lvl.qos_free) = 0
            $tbl_count = length($(lvl.perm))
            for $p in ($tbl_count + 1):$(lvl.qos_stop)
                $v = $(lvl.perm)[$p]
                if $v <= 0
                    $v = $p
                end
                $(lvl.perm)[$v] = -$(lvl.qos_free)
                $(lvl.qos_free) = $v
            end
            for $v in $(lvl.tbl_val)
                if $v > 0
                    $(lvl.perm)[$v] = $v
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
                        $my_v = $(lvl.tbl_val)[$(lvl.perm)[$my_q]]
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
    old = freshen(ctx, tag, :_old)
    tbl_cap = freshen(ctx, tag, :_tbl_cap)
    tbl_pos = freshen(ctx, tag, :_tbl_pos)
    tbl_idx = freshen(ctx, tag, :_tbl_idx)
    tbl_val = freshen(ctx, tag, :_tbl_val)
    tbl_slot = freshen(ctx, tag, :_tbl_slot)

    Thunk(;
        body=(ctx) -> Lookup(;
            body=(ctx, idx) -> Thunk(;
                preamble=quote
                    $tbl_pos = $(lvl.tbl_pos)
                    $tbl_idx = $(lvl.tbl_idx)
                    $tbl_val = $(lvl.tbl_val)
                    if $qos_stop == length($(lvl.perm))
                        $old = length($(lvl.perm)) + 1
                        $p = $old
                        $q_stop = max(length($(lvl.perm)) << 1, $qos_stop + 1)
                        Finch.resize_if_smaller!($(lvl.perm), $q_stop)
                        Finch.fill_range!($(lvl.perm), 0, $old, $q_stop)
                        $tbl_cap = Finch.sparse_dict_table_capacity($q_stop)
                        Finch.sparse_dict_table_resize!(
                            $tbl_pos, $tbl_idx, $tbl_val, $tbl_cap
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
                    $tbl_slot = Finch.sparse_dict_table_lookup_slot(
                        $tbl_pos, $tbl_idx, $tbl_val, $(ctx(pos)), $(ctx(idx))
                    )
                    $qos = $tbl_val[$tbl_slot]
                    if $qos == 0
                        #If the qos is not in the table, we need to add it.
                        #We need to commit it to the table in the event that
                        #another accessor tries to write it in the same loop.
                        if $qos_free != 0
                            $qos = $qos_free
                            $qos_free = -$(lvl.perm)[$qos]
                            $(lvl.perm)[$qos] = 0
                        else
                            $qos = $qos_stop + 1
                            $qos_stop = $qos
                        end
                        Finch.sparse_dict_table_insert_at_slot!(
                            $tbl_pos,
                            $tbl_idx,
                            $tbl_val,
                            $tbl_slot,
                            $(ctx(pos)),
                            $(ctx(idx)),
                            $qos,
                        )
                        $(lvl.tbl_count) += 1
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
                        if $(lvl.perm)[$qos] <= 0
                            $(lvl.perm)[$qos] = $qos
                        end
                        $(fbr.dirty) = true
                    elseif $(lvl.perm)[$qos] == 0 #here, perm is being used as a dirty bit
                        $(lvl.perm)[$qos] = $qos
                    end
                end,
            ),
        ),
    )
end
