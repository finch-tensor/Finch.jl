"""
    SparseDictLevel{[Ti=Int], [Tp=Int], [Ptr, Idx, Val, Tbl, Pool=Dict]}(lvl, [dim])

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
struct SparseDictLevel{Ti, Ptr, Idx, Val, Tbl, Pool, Lvl} <: AbstractLevel
    lvl::Lvl
    shape::Ti
    ptr::Ptr
    idx::Idx
    val::Val
    tbl::Tbl
    pool::Pool
end

const SparseDict = SparseDictLevel

SparseDictLevel(lvl) = SparseDictLevel{Int}(lvl)
SparseDictLevel(lvl, shape::Ti) where {Ti} = SparseDictLevel{Ti}(lvl, shape)
SparseDictLevel{Ti}(lvl) where {Ti} = SparseDictLevel{Ti}(lvl, zero(Ti))
SparseDictLevel{Ti}(lvl, shape) where {Ti} = SparseDictLevel{Ti}(lvl, shape, postype(lvl)[1], Ti[], postype(lvl)[], Dict{Tuple{postype(lvl), Ti}, postype(lvl)}(), postype(lvl)[])

SparseDictLevel{Ti}(lvl::Lvl, shape, ptr::Ptr, idx::Idx, val::Val, tbl::Tbl, pool::Pool) where {Ti, Ptr, Idx, Val, Tbl, Pool, Lvl} =
    SparseDictLevel{Ti, Ptr, Idx, Val, Tbl, Pool, Lvl}(lvl, shape, ptr, idx, val, tbl, pool)

Base.summary(lvl::SparseDictLevel) = "SparseDict($(summary(lvl.lvl)))"
similar_level(lvl::SparseDictLevel, fill_value, eltype::Type, dim, tail...) =
    SparseDict(similar_level(lvl.lvl, fill_value, eltype, tail...), dim)

function postype(::Type{SparseDictLevel{Ti, Ptr, Idx, Val, Tbl, Pool, Lvl}}) where {Ti, Ptr, Idx, Val, Tbl, Pool, Lvl}
    return postype(Lvl)
end

Base.resize!(lvl::SparseDictLevel{Ti}, dims...) where {Ti} =
    SparseDictLevel{Ti}(resize!(lvl.lvl, dims[1:end-1]...), dims[end], lvl.ptr, lvl.idx, lvl.val, lvl.tbl, lvl.pool)

function moveto(lvl::SparseDictLevel{Ti, Ptr, Idx, Val, Tbl, Pool, Lvl}, Tm) where {Ti, Ptr, Idx, Val, Tbl, Pool, Lvl}
    lvl_2 = moveto(lvl.lvl, Tm)
    ptr_2 = moveto(lvl.ptr, Tm)
    idx_2 = moveto(lvl.idx, Tm)
    val_2 = moveto(lvl.val, Tm)
    tbl_2 = moveto(lvl.tbl, Tm)
    pool_2 = moveto(lvl.pool, Tm)
    return SparseDictLevel{Ti}(lvl_2, lvl.shape, ptr_2, idx_2, val_2, tbl_2, pool_2)
end

function countstored_level(lvl::SparseDictLevel, pos)
    pos == 0 && return countstored_level(lvl.lvl, pos)
    countstored_level(lvl.lvl, lvl.ptr[pos + 1]  - 1)
end

pattern!(lvl::SparseDictLevel{Ti}) where {Ti} =
    SparseDictLevel{Ti}(pattern!(lvl.lvl), lvl.shape, lvl.ptr, lvl.idx, lvl.val, lvl.tbl, lvl.pool)

set_fill_value!(lvl::SparseDictLevel{Ti}, init) where {Ti} =
    SparseDictLevel{Ti}(set_fill_value!(lvl.lvl, init), lvl.shape, lvl.ptr, lvl.idx, lvl.val, lvl.tbl, lvl.pool)

function Base.show(io::IO, lvl::SparseDictLevel{Ti}) where {Ti}
    if get(io, :compact, false)
        print(io, "SparseDict(")
    else
        print(io, "SparseDict{$Ti}(")
    end
    show(io, lvl.lvl)
    print(io, ", ")
    show(IOContext(io, :typeinfo=>Ti), lvl.shape)
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
        show(io, lvl.tbl)
        print(io, ", ")
        show(io, lvl.pool)
    end
    print(io, ")")
end

labelled_show(io::IO, fbr::SubFiber{<:SparseDictLevel}) =
    print(io, "SparseDict (", fill_value(fbr), ") [", ":,"^(ndims(fbr) - 1), "1:", size(fbr)[end], "]")

function labelled_children(fbr::SubFiber{<:SparseDictLevel})
    lvl = fbr.lvl
    pos = fbr.pos
    pos + 1 > length(lvl.ptr) && return []
    map(lvl.ptr[pos]:lvl.ptr[pos + 1] - 1) do qos
        LabelledTree(cartesian_label([range_label() for _ = 1:ndims(fbr) - 1]..., lvl.idx[qos]), SubFiber(lvl.lvl, lvl.val[qos]))
    end
end

@inline level_ndims(::Type{<:SparseDictLevel{Ti, Ptr, Idx, Val, Tbl, Pool, Lvl}}) where {Ti, Ptr, Idx, Val, Tbl, Pool, Lvl} = 1 + level_ndims(Lvl)
@inline level_size(lvl::SparseDictLevel) = (level_size(lvl.lvl)..., lvl.shape)
@inline level_axes(lvl::SparseDictLevel) = (level_axes(lvl.lvl)..., Base.OneTo(lvl.shape))
@inline level_eltype(::Type{<:SparseDictLevel{Ti, Ptr, Idx, Val, Tbl, Pool, Lvl}}) where {Ti, Ptr, Idx, Val, Tbl, Pool, Lvl} = level_eltype(Lvl)
@inline level_fill_value(::Type{<:SparseDictLevel{Ti, Ptr, Idx, Val, Tbl, Pool, Lvl}}) where {Ti, Ptr, Idx, Val, Tbl, Pool, Lvl} = level_fill_value(Lvl)
data_rep_level(::Type{<:SparseDictLevel{Ti, Ptr, Idx, Val, Tbl, Pool, Lvl}}) where {Ti, Ptr, Idx, Val, Tbl, Pool, Lvl} = SparseData(data_rep_level(Lvl))

(fbr::AbstractFiber{<:SparseDictLevel})() = fbr
function (fbr::SubFiber{<:SparseDictLevel{Ti}})(idxs...) where {Ti}
    isempty(idxs) && return fbr
    lvl = fbr.lvl
    p = fbr.pos
    crds = @view lvl.idx[lvl.ptr[p]:lvl.ptr[p + 1] - 1]
    r = searchsorted(crds, idxs[end])
    q = lvl.ptr[p] + first(r) - 1
    length(r) == 0 ? fill_value(fbr) : SubFiber(lvl.lvl, lvl.val[q])(idxs[1:end-1]...)
end

mutable struct VirtualSparseDictLevel <: AbstractVirtualLevel
    lvl
    ex
    Ti
    ptr
    idx
    val
    tbl
    pool
    shape
    qos_stop
end

is_level_injective(ctx, lvl::VirtualSparseDictLevel) = [is_level_injective(ctx, lvl.lvl)..., false]
function is_level_atomic(ctx, lvl::VirtualSparseDictLevel)
    (below, atomic) = is_level_atomic(ctx, lvl.lvl)
    return ([below; [atomic]], atomic)
end
function is_level_concurrent(ctx, lvl::VirtualSparseDictLevel)
    (data, _) = is_level_concurrent(ctx, lvl.lvl)
    return ([data; [false]], false)
end

function virtualize(ctx, ex, ::Type{SparseDictLevel{Ti, Ptr, Idx, Val, Tbl, Pool, Lvl}}, tag=:lvl) where {Ti, Ptr, Idx, Val, Tbl, Pool, Lvl}
    sym = freshen(ctx, tag)
    ptr = freshen(ctx, tag, :_ptr)
    idx = freshen(ctx, tag, :_idx)
    val = freshen(ctx, tag, :_val)
    tbl = freshen(ctx, tag, :_tbl)
    pool = freshen(ctx, tag, :_pool)
    qos_stop = freshen(ctx, tag, :_qos_stop)
    push_preamble!(ctx, quote
        $sym = $ex
        $ptr = $sym.ptr
        $idx = $sym.idx
        $val = $sym.val
        $tbl = $sym.tbl
        $pool = $sym.pool
        $qos_stop = length($tbl)
    end)
    lvl_2 = virtualize(ctx, :($sym.lvl), Lvl, sym)
    shape = value(:($sym.shape), Int)
    VirtualSparseDictLevel(lvl_2, sym, Ti, ptr, idx, val, tbl, pool, shape, qos_stop)
end
function lower(ctx::AbstractCompiler, lvl::VirtualSparseDictLevel, ::DefaultStyle)
    quote
        $SparseDictLevel{$(lvl.Ti)}(
            $(ctx(lvl.lvl)),
            $(ctx(lvl.shape)),
            $(lvl.ptr),
            $(lvl.idx),
            $(lvl.val),
            $(lvl.tbl),
            $(lvl.pool),
        )
    end
end

Base.summary(lvl::VirtualSparseDictLevel) = "SparseDict($(summary(lvl.lvl)))"

function virtual_level_size(ctx, lvl::VirtualSparseDictLevel)
    ext = make_extent(lvl.Ti, literal(lvl.Ti(1)), lvl.shape)
    (virtual_level_size(ctx, lvl.lvl)..., ext)
end

function virtual_level_resize!(ctx, lvl::VirtualSparseDictLevel, dims...)
    lvl.shape = getstop(dims[end])
    lvl.lvl = virtual_level_resize!(ctx, lvl.lvl, dims[1:end-1]...)
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
    push_preamble!(ctx, quote
        empty!($(lvl.tbl))
        empty!($(lvl.pool))
        $qos = $(Tp(0))
        $(lvl.qos_stop) = 0
    end)
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
    r = freshen(ctx, :r)
    i = freshen(ctx, :i)
    v = freshen(ctx, :v)
    idx_tmp = freshen(ctx, :idx_tmp)
    val_tmp = freshen(ctx, :val_tmp)
    perm = freshen(ctx, :perm)
    pdx_tmp = freshen(ctx, :pdx_tmp)
    entry = freshen(ctx, :entry)
    ptr_2 = freshen(ctx, :ptr_2)
    push_preamble!(ctx, quote
        resize!($(lvl.ptr), $(ctx(pos_stop)) + 1)
        $(lvl.ptr)[1] = 1
        Finch.fill_range!($(lvl.ptr), 0, 2, $(ctx(pos_stop)) + 1)
        $pdx_tmp = Vector{$Tp}(undef, length($(lvl.tbl)))
        resize!($(lvl.idx), length($(lvl.tbl)))
        resize!($(lvl.val), length($(lvl.tbl)))
        $idx_tmp = Vector{$Ti}(undef, length($(lvl.tbl)))
        $val_tmp = Vector{$Tp}(undef, length($(lvl.tbl)))
        $q = 0
        for $entry in pairs($(lvl.tbl))
            (($p, $i), $v) = $entry
            $q += 1
            $idx_tmp[$q] = $i
            $val_tmp[$q] = $v
            $pdx_tmp[$q] = $p
            $(lvl.ptr)[$p + 1] += 1
        end
        for $p = 2:$(ctx(pos_stop)) + 1
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
        $qos_stop = $(lvl.ptr)[$(ctx(pos_stop)) + 1] - 1
    end)
    lvl.lvl = freeze_level!(ctx, lvl.lvl, value(qos_stop))
    return lvl
end

function thaw_level!(ctx::AbstractCompiler, lvl::VirtualSparseDictLevel, pos_stop)
    p = freshen(ctx, :p)
    pos_stop = ctx(cache!(ctx, :pos_stop, simplify(ctx, pos_stop)))
    push_preamble!(ctx, quote
        $(lvl.qos_stop) = $(lvl.ptr)[$(ctx(pos_stop)) + 1] - 1
    end)
    lvl.lvl = thaw_level!(ctx, lvl.lvl, value(lvl.qos_stop))
    return lvl
end

function virtual_moveto_level(ctx::AbstractCompiler, lvl::VirtualSparseDictLevel, arch)
    ptr_2 = freshen(ctx, lvl.ptr)
    idx_2 = freshen(ctx, lvl.idx)
    tbl_2 = freshen(ctx, lvl.tbl_2)
    push_preamble!(ctx, quote
        $tbl_2 = $(lvl.tbl)
        $(lvl.tbl) = $moveto($(lvl.tbl), $(ctx(arch)))
    end)
    push_epilogue!(ctx, quote
        $(lvl.tbl) = $tbl_2
    end)
    virtual_moveto_level(ctx, lvl.lvl, arch)
end

function unfurl(ctx, fbr::VirtualSubFiber{VirtualSparseDictLevel}, ext, mode::Reader, ::Union{typeof(defaultread), typeof(walk)})
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    Tp = postype(lvl)
    Ti = lvl.Ti
    my_i = freshen(ctx, tag, :_i)
    my_q = freshen(ctx, tag, :_q)
    my_q_stop = freshen(ctx, tag, :_q_stop)
    my_i1 = freshen(ctx, tag, :_i1)
    my_v = freshen(ctx, tag, :_v)

    Thunk(
        preamble = quote
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
        body = (ctx) -> Sequence([
            Phase(
                stop = (ctx, ext) -> value(my_i1),
                body = (ctx, ext) -> Stepper(
                    seek = (ctx, ext) -> quote
                        if $(lvl.idx)[$my_q] < $(ctx(getstart(ext)))
                            $my_q = Finch.scansearch($(lvl.idx), $(ctx(getstart(ext))), $my_q, $my_q_stop - 1)
                            $my_i = $(lvl.idx)[$my_q]
                        end
                    end,
                    preamble = quote
                        $my_i = $(lvl.idx)[$my_q]
                        $my_v = $(lvl.val)[$my_q]
                    end,
                    stop = (ctx, ext) -> value(my_i),
                    chunk = Spike(
                        body = FillLeaf(virtual_level_fill_value(lvl)),
                        tail = Simplify(instantiate(ctx, VirtualSubFiber(lvl.lvl, value(my_v, Ti)), mode)),
                    ),
                    next = (ctx, ext) -> :($my_q += $(Tp(1)))
                )
            ),
            Phase(
                body = (ctx, ext) -> Run(FillLeaf(virtual_level_fill_value(lvl)))
            )
        ])
    )
end

function unfurl(ctx, fbr::VirtualSubFiber{VirtualSparseDictLevel}, ext, mode::Reader, ::typeof(follow))
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    Tp = postype(lvl)
    my_q = freshen(ctx, tag, :_q)

    Lookup(
        body = (ctx, i) -> Thunk(
            preamble = quote
                $my_q = get($(lvl.tbl), ($(ctx(pos)), $(ctx(i))), 0)
            end,
            body = (ctx) -> Switch([
                value(:($my_q != 0)) => instantiate(ctx, VirtualSubFiber(lvl.lvl, value(my_q, Tp)), mode)
                literal(true) => FillLeaf(virtual_level_fill_value(lvl))
            ])
        )
    )
end

unfurl(ctx, fbr::VirtualSubFiber{VirtualSparseDictLevel}, ext, mode::Updater, proto) = begin
    unfurl(ctx, VirtualHollowSubFiber(fbr.lvl, fbr.pos, freshen(ctx, :null)), ext, mode, proto)
end
function unfurl(ctx, fbr::VirtualHollowSubFiber{VirtualSparseDictLevel}, ext, mode::Updater, ::Union{typeof(defaultupdate), typeof(extrude)})
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    Tp = postype(lvl)
    qos = freshen(ctx, tag, :_qos)
    qos_stop = lvl.qos_stop
    dirty = freshen(ctx, tag, :_dirty)

    Thunk(
        body = (ctx) -> Lookup(
            body = (ctx, idx) -> Thunk(
                preamble = quote
                    $qos = get($(lvl.tbl), ($(ctx(pos)), $(ctx(idx))), 0)
                    if $qos == 0
                        #If the qos is not in the table, we need to add it.
                        #We need to commit it to the table in the event that
                        #another accessor tries to write it in the same loop.
                        if !isempty($(lvl.pool))
                            $qos = pop!($(lvl.pool))
                        else
                            $qos = length($(lvl.tbl)) + 1
                            if $qos > $qos_stop
                                $qos_stop = max($qos_stop << 1, 1)
                                $(contain(ctx_2->assemble_level!(ctx_2, lvl.lvl, value(qos, Tp), value(qos_stop, Tp)), ctx))
                                Finch.resize_if_smaller!($(lvl.val), $qos_stop)
                                Finch.fill_range!($(lvl.val), 0, $qos, $qos_stop)
                            end
                        end
                        $(lvl.tbl)[($(ctx(pos)), $(ctx(idx)))] = $qos
                    end
                    $dirty = false
                end,
                body = (ctx) -> instantiate(ctx, VirtualHollowSubFiber(lvl.lvl, value(qos, Tp), dirty), mode),
                epilogue = quote
                    if $dirty
                        $(lvl.val)[$qos] = $qos
                        $(fbr.dirty) = true
                    elseif $(lvl.val)[$qos] == 0 #here, val is being used as a dirty bit
                        push!($(lvl.pool), $qos)
                        delete!($(lvl.tbl), ($(ctx(pos)), $(ctx(idx))))
                    end
                end
            )
        ),
    )
end
