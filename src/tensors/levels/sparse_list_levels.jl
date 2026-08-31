"""
    SparseListLevel{[Ti=Int], [Ptr, Idx]}(lvl, [dim])

A subfiber of a sparse level does not need to represent slices `A[:, ..., :, i]`
which are entirely [`fill_value`](@ref). Instead, only potentially non-fill
slices are stored as subfibers in `lvl`.  A sorted list is used to record which
slices are stored. Optionally, `dim` is the size of the last dimension.

`Ti` is the type of the last tensor index, and `Tp` is the type used for
positions in the level. The types `Ptr` and `Idx` are the types of the
arrays used to store positions and indicies.

```jldoctest
julia> tensor_tree(Tensor(Dense(SparseList(Element(0.0))), [10 0 20; 30 0 0; 0 0 40]))
3×3-Tensor
└─ Dense [:,1:3]
   ├─ [:, 1]: SparseList (0.0) [1:3]
   │  ├─ [1]: 10.0
   │  └─ [2]: 30.0
   ├─ [:, 2]: SparseList (0.0) [1:3]
   └─ [:, 3]: SparseList (0.0) [1:3]
      ├─ [1]: 20.0
      └─ [3]: 40.0

julia> tensor_tree(Tensor(SparseList(SparseList(Element(0.0))), [10 0 20; 30 0 0; 0 0 40]))
3×3-Tensor
└─ SparseList (0.0) [:,1:3]
   ├─ [:, 1]: SparseList (0.0) [1:3]
   │  ├─ [1]: 10.0
   │  └─ [2]: 30.0
   └─ [:, 3]: SparseList (0.0) [1:3]
      ├─ [1]: 20.0
      └─ [3]: 40.0

```
"""
struct SparseListLevel{Ti,Ptr,Idx,Lvl} <: AbstractLevel
    lvl::Lvl
    shape::Ti
    ptr::Ptr
    idx::Idx
end
const SparseList = SparseListLevel
SparseListLevel(lvl) = SparseListLevel{Int}(lvl)
SparseListLevel(lvl, shape::Ti) where {Ti} = SparseListLevel{Ti}(lvl, shape)
SparseListLevel{Ti}(lvl) where {Ti} = SparseListLevel{Ti}(lvl, zero(Ti))
function SparseListLevel{Ti}(lvl, shape) where {Ti}
    SparseListLevel{Ti}(lvl, shape, postype(lvl)[1], Ti[])
end

function SparseListLevel{Ti}(lvl::Lvl, shape, ptr::Ptr, idx::Idx) where {Ti,Lvl,Ptr,Idx}
    SparseListLevel{Ti,Ptr,Idx,Lvl}(lvl, shape, ptr, idx)
end

Base.summary(lvl::SparseListLevel) = "SparseList($(summary(lvl.lvl)))"
function similar_level(lvl::SparseListLevel, fill_value, eltype::Type, dim, tail...)
    SparseList(similar_level(lvl.lvl, fill_value, eltype, tail...), dim)
end

function postype(::Type{SparseListLevel{Ti,Ptr,Idx,Lvl}}) where {Ti,Ptr,Idx,Lvl}
    return postype(Lvl)
end

function transfer(Tm, lvl::SparseListLevel{Ti,Ptr,Idx,Lvl}) where {Ti,Ptr,Idx,Lvl}
    lvl_2 = transfer(Tm, lvl.lvl)
    ptr_2 = transfer(Tm, lvl.ptr)
    idx_2 = transfer(Tm, lvl.idx)
    return SparseListLevel{Ti}(lvl_2, lvl.shape, ptr_2, idx_2)
end

function countstored_level(lvl::SparseListLevel, pos)
    countstored_level(lvl.lvl, lvl.ptr[pos + 1] - 1)
end

function countstored_level(lvl::SparseListLevel, pos, idxs, dim, proc, exact)
    my_ptr = lvl.ptr.data[proc]
    my_idx = lvl.idx.data[proc]
    q_start = my_ptr[pos]
    q_stop = my_ptr[pos + 1] - 1
    idx = exact ? idxs[length(idxs) - dim] : lvl.shape
    
    ##First, find if there are indices underneath the target index.
    r = binary_search_ub(idx, my_idx, q_start, q_stop)

    ##If not, do a cumulative sum of the previous position (or return 0 if at position 1, as there is no position 0)
    ##If yes, that becomes the new position, we recurse on that.
    if r == -1
        q_start == 1 && return 0
        count = countstored_level(lvl.lvl, q_start - 1, idxs, dim + 1, proc, exact & false)
    elseif my_idx[r] == idx
        count = countstored_level(lvl.lvl, r, idxs, dim + 1, proc, exact & true)
    else
        count = countstored_level(lvl.lvl, r, idxs, dim + 1, proc, exact & false)
    end

    return count
end

function pattern!(lvl::SparseListLevel{Ti}) where {Ti}
    SparseListLevel{Ti}(pattern!(lvl.lvl), lvl.shape, lvl.ptr, lvl.idx)
end

function set_fill_value!(lvl::SparseListLevel{Ti}, init) where {Ti}
    SparseListLevel{Ti}(set_fill_value!(lvl.lvl, init), lvl.shape, lvl.ptr, lvl.idx)
end

function Base.resize!(lvl::SparseListLevel{Ti}, dims...) where {Ti}
    SparseListLevel{Ti}(resize!(lvl.lvl, dims[1:(end - 1)]...), dims[end], lvl.ptr, lvl.idx)
end

function Base.show(io::IO, lvl::SparseListLevel{Ti,Ptr,Idx,Lvl}) where {Ti,Lvl,Idx,Ptr}
    if get(io, :compact, false)
        print(io, "SparseList(")
    else
        print(io, "SparseList{$Ti}(")
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
    end
    print(io, ")")
end

function labelled_show(io::IO, fbr::SubFiber{<:SparseListLevel})
    print(
        io,
        "SparseList (",
        fill_value(fbr),
        ") [",
        ":,"^(ndims(fbr) - 1),
        "1:",
        size(fbr)[end],
        "]",
    )
end

function labelled_children(fbr::SubFiber{<:SparseListLevel})
    lvl = fbr.lvl
    pos = fbr.pos
    pos + 1 > length(lvl.ptr) && return []
    map(lvl.ptr[pos]:(lvl.ptr[pos + 1] - 1)) do qos
        LabelledTree(
            cartesian_label([range_label() for _ in 1:(ndims(fbr) - 1)]..., lvl.idx[qos]),
            SubFiber(lvl.lvl, qos),
        )
    end
end

@inline level_ndims(::Type{<:SparseListLevel{Ti,Ptr,Idx,Lvl}}) where {Ti,Ptr,Idx,Lvl} =
    1 + level_ndims(Lvl)
@inline level_size(lvl::SparseListLevel) = (level_size(lvl.lvl)..., lvl.shape)
@inline level_axes(lvl::SparseListLevel) = (level_axes(lvl.lvl)..., Base.OneTo(lvl.shape))
@inline level_eltype(::Type{<:SparseListLevel{Ti,Ptr,Idx,Lvl}}) where {Ti,Ptr,Idx,Lvl} =
    level_eltype(
        Lvl
    )
@inline level_fill_value(::Type{<:SparseListLevel{Ti,Ptr,Idx,Lvl}}) where {Ti,Ptr,Idx,Lvl} =
    level_fill_value(
        Lvl
    )
function data_rep_level(::Type{<:SparseListLevel{Ti,Ptr,Idx,Lvl}}) where {Ti,Ptr,Idx,Lvl}
    SparseData(data_rep_level(Lvl))
end

function isstructequal(a::T, b::T) where {T<:SparseList}
    a.shape == b.shape &&
        a.ptr == b.ptr &&
        a.idx == b.idx &&
        isstructequal(a.lvl, b.lvl)
end

(fbr::AbstractFiber{<:SparseListLevel})() = fbr
function (fbr::SubFiber{<:SparseListLevel{Ti}})(idxs...) where {Ti}
    isempty(idxs) && return fbr
    lvl = fbr.lvl
    p = fbr.pos
    r = searchsorted(@view(lvl.idx[lvl.ptr[p]:(lvl.ptr[p + 1] - 1)]), idxs[end])
    q = lvl.ptr[p] + first(r) - 1
    fbr_2 = SubFiber(lvl.lvl, q)
    length(r) == 0 ? fill_value(fbr_2) : fbr_2(idxs[1:(end - 1)]...)
end

mutable struct VirtualSparseListLevel <: AbstractVirtualLevel
    tag
    lvl
    Ti
    ptr
    idx
    shape
    qos_fill
    qos_stop
    prev_pos
end

function is_level_injective(ctx, lvl::VirtualSparseListLevel)
    [is_level_injective(ctx, lvl.lvl)..., false]
end
function is_level_atomic(ctx, lvl::VirtualSparseListLevel)
    (below, atomic) = is_level_atomic(ctx, lvl.lvl)
    return ([below; [atomic]], atomic)
end
function is_level_concurrent(ctx, lvl::VirtualSparseListLevel)
    (data, _) = is_level_concurrent(ctx, lvl.lvl)
    return ([data; [false]], false)
end

function virtualize(
    ctx, ex, ::Type{SparseListLevel{Ti,Ptr,Idx,Lvl}}, tag=:lvl
) where {Ti,Ptr,Idx,Lvl}
    tag = freshen(ctx, tag)
    ptr = freshen(ctx, tag, :_ptr)
    idx = freshen(ctx, tag, :_idx)
    stop = freshen(ctx, tag, :_stop)
    push_preamble!(
        ctx,
        quote
            $tag = $ex
            $ptr = $tag.ptr
            $idx = $tag.idx
            $stop = $tag.shape
        end,
    )
    shape = value(stop, Int)
    lvl_2 = virtualize(ctx, :($tag.lvl), Lvl, tag)
    qos_fill = freshen(ctx, tag, :_qos_fill)
    qos_stop = freshen(ctx, tag, :_qos_stop)
    prev_pos = freshen(ctx, tag, :_prev_pos)
    VirtualSparseListLevel(
        tag, lvl_2, Ti, ptr, idx, shape, qos_fill, qos_stop, prev_pos
    )
end
function lower(ctx::AbstractCompiler, lvl::VirtualSparseListLevel, ::DefaultStyle)
    quote
        $SparseListLevel{$(lvl.Ti)}(
            $(ctx(lvl.lvl)),
            $(ctx(lvl.shape)),
            $(lvl.ptr),
            $(lvl.idx),
        )
    end
end

function distribute_level(
    ctx::AbstractCompiler, lvl::VirtualSparseListLevel, arch, diff, style
)
    return diff[lvl.tag] = VirtualSparseListLevel(
        lvl.tag,
        distribute_level(ctx, lvl.lvl, arch, diff, style),
        lvl.Ti,
        distribute_buffer(ctx, lvl.ptr, arch, style),
        distribute_buffer(ctx, lvl.idx, arch, style),
        lvl.shape,
        lvl.qos_fill,
        lvl.qos_stop,
        lvl.prev_pos,
    )
end

function redistribute(ctx::AbstractCompiler, lvl::VirtualSparseListLevel, diff)
    get(
        diff,
        lvl.tag,
        VirtualSparseListLevel(
            lvl.tag,
            redistribute(ctx, lvl.lvl, diff),
            lvl.Ti,
            lvl.ptr,
            lvl.idx,
            lvl.shape,
            lvl.qos_fill,
            lvl.qos_stop,
            lvl.prev_pos,
        ),
    )
end

Base.summary(lvl::VirtualSparseListLevel) = "SparseList($(summary(lvl.lvl)))"

function virtual_level_size(ctx, lvl::VirtualSparseListLevel)
    ext = virtual_call(ctx, extent, literal(lvl.Ti(1)), lvl.shape)
    (virtual_level_size(ctx, lvl.lvl)..., ext)
end

function virtual_level_resize!(ctx, lvl::VirtualSparseListLevel, dims...)
    lvl.shape = getstop(dims[end])
    lvl.lvl = virtual_level_resize!(ctx, lvl.lvl, dims[1:(end - 1)]...)
    lvl
end

virtual_level_eltype(lvl::VirtualSparseListLevel) = virtual_level_eltype(lvl.lvl)
virtual_level_fill_value(lvl::VirtualSparseListLevel) = virtual_level_fill_value(lvl.lvl)

postype(lvl::VirtualSparseListLevel) = postype(lvl.lvl)

function declare_level!(ctx::AbstractCompiler, lvl::VirtualSparseListLevel, pos, init)
    #TODO check that init == fill_value
    Ti = lvl.Ti
    Tp = postype(lvl)
    push_preamble!(
        ctx,
        quote
            $(lvl.qos_fill) = $(Tp(0))
            $(lvl.qos_stop) = $(Tp(0))
        end,
    )
    if issafe(get_mode_flag(ctx))
        push_preamble!(
            ctx,
            quote
                $(lvl.prev_pos) = $(Tp(0))
            end,
        )
    end
    lvl.lvl = declare_level!(ctx, lvl.lvl, literal(Tp(0)), init)
    return lvl
end

function assemble_level!(ctx, lvl::VirtualSparseListLevel, pos_start, pos_stop)
    pos_start = ctx(cache!(ctx, :p_start, pos_start))
    pos_stop = ctx(cache!(ctx, :p_start, pos_stop))
    return quote
        Finch.resize_if_smaller!($(lvl.ptr), $pos_stop + 1)
        Finch.fill_range!($(lvl.ptr), 0, $pos_start + 1, $pos_stop + 1)
    end
end

function freeze_level!(ctx::AbstractCompiler, lvl::VirtualSparseListLevel, pos_stop)
    p = freshen(ctx, :p)
    pos_stop = ctx(cache!(ctx, :pos_stop, simplify(ctx, pos_stop)))
    qos_stop = freshen(ctx, :qos_stop)
    push_preamble!(
        ctx,
        quote
            resize!($(lvl.ptr), $pos_stop + 1)
            for $p in 1:($pos_stop)
                $(lvl.ptr)[$p + 1] += $(lvl.ptr)[$p]
            end
            $qos_stop = $(lvl.ptr)[$pos_stop + 1] - 1
            resize!($(lvl.idx), $qos_stop)
        end,
    )
    lvl.lvl = freeze_level!(ctx, lvl.lvl, value(qos_stop))
    return lvl
end

function thaw_level!(ctx::AbstractCompiler, lvl::VirtualSparseListLevel, pos_stop)
    p = freshen(ctx, :p)
    pos_stop = ctx(cache!(ctx, :pos_stop, simplify(ctx, pos_stop)))
    qos_stop = freshen(ctx, :qos_stop)
    push_preamble!(
        ctx,
        quote
            $(lvl.qos_fill) = $(lvl.ptr)[$pos_stop + 1] - 1
            $(lvl.qos_stop) = $(lvl.qos_fill)
            $qos_stop = $(lvl.qos_fill)
            $(
                if issafe(get_mode_flag(ctx))
                    quote
                        $(lvl.prev_pos) =
                            Finch.scansearch(
                                $(lvl.ptr), $(lvl.qos_stop) + 1, 1, $pos_stop
                            ) - 1
                    end
                end
            )
            for $p in ($pos_stop):-1:1
                $(lvl.ptr)[$p + 1] -= $(lvl.ptr)[$p]
            end
        end,
    )
    lvl.lvl = thaw_level!(ctx, lvl.lvl, value(qos_stop))
    return lvl
end

function unfurl(
    ctx,
    fbr::VirtualSubFiber{VirtualSparseListLevel},
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
                        end
                    end,
                    preamble=:($my_i = $(lvl.idx)[$my_q]),
                    stop=(ctx, ext) -> value(my_i),
                    chunk=Spike(;
                        body=FillLeaf(virtual_level_fill_value(lvl)),
                        tail=Simplify(
                            instantiate(
                                ctx, VirtualSubFiber(lvl.lvl, value(my_q, Ti)), mode
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
    ctx, fbr::VirtualSubFiber{VirtualSparseListLevel}, ext, mode, ::typeof(follow)
)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.tag
    Tp = postype(lvl)
    my_q = freshen(ctx, tag, :_q)
    my_q_stop = freshen(ctx, tag, :_q_stop)
    my_qos = freshen(ctx, tag, :_qos)
    Thunk(;
        preamble=quote
            $my_q = $(lvl.ptr)[$(ctx(pos))]
        end,
        body=(ctx) -> Lookup(;
            body=(ctx, i) -> Thunk(;
                preamble=quote
                    $my_q = max($my_q, $(lvl.ptr)[$(ctx(pos))])
                    $my_q_stop = $(lvl.ptr)[$(ctx(pos)) + $(Tp(1))]
                    $my_qos = scansearch($(lvl.idx), $(ctx(i)), $my_q, $my_q_stop - 1)
                    $my_q = min($my_q_stop - 1, $my_qos)
                end,
                body=(ctx) -> Switch([
                    value(:($my_qos < $my_q_stop && $(lvl.idx)[$my_qos] == $(ctx(i)))) => VirtualSubFiber(lvl.lvl, value(my_qos, Tp)),
                    literal(true) => FillLeaf(virtual_level_fill_value(lvl)),
                ]),
            ),
        ),
    )
end

function unfurl(
    ctx, fbr::VirtualSubFiber{VirtualSparseListLevel}, ext, mode, ::typeof(gallop)
)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.tag
    Tp = postype(lvl)
    Ti = lvl.Ti
    my_i = freshen(ctx, tag, :_i)
    my_q = freshen(ctx, tag, :_q)
    my_q_stop = freshen(ctx, tag, :_q_stop)
    my_i1 = freshen(ctx, tag, :_i1)
    my_i2 = freshen(ctx, tag, :_i2)
    my_i3 = freshen(ctx, tag, :_i3)
    my_i4 = freshen(ctx, tag, :_i4)

    Thunk(;
        preamble=quote
            $my_q = $(lvl.ptr)[$(ctx(pos))]
            $my_q_stop = $(lvl.ptr)[$(ctx(pos)) + 1]
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
                body=(ctx, ext) -> Jumper(;
                    seek=(ctx, ext) -> quote
                        if $(lvl.idx)[$my_q] < $(ctx(getstart(ext)))
                            $my_q = Finch.scansearch(
                                $(lvl.idx),
                                $(ctx(getstart(ext))),
                                $my_q,
                                $my_q_stop - 1,
                            )
                        end
                    end,
                    preamble=:($my_i2 = $(lvl.idx)[$my_q]),
                    stop=(ctx, ext) -> value(my_i2),
                    chunk=Spike(;
                        body=FillLeaf(virtual_level_fill_value(lvl)),
                        tail=instantiate(
                            ctx, VirtualSubFiber(lvl.lvl, value(my_q, Ti)), mode
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
    ctx,
    fbr::VirtualSubFiber{VirtualSparseListLevel},
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
    fbr::VirtualHollowSubFiber{VirtualSparseListLevel},
    ext,
    mode,
    ::Union{typeof(defaultupdate),typeof(extrude)},
)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.tag
    Tp = postype(lvl)
    qos = freshen(ctx, tag, :_qos)
    qos_fill = lvl.qos_fill
    qos_stop = lvl.qos_stop
    dirty = freshen(ctx, tag, :dirty)

    Thunk(;
        preamble=quote
            $qos = $qos_fill + 1
            $(
                if issafe(get_mode_flag(ctx))
                    quote
                        $(lvl.prev_pos) < $(ctx(pos)) || throw(
                            $FinchProtocolError(
                                "SparseListLevels cannot be updated multiple times"
                            ),
                        )
                    end
                end
            )
        end,
        body=(ctx) -> Lookup(;
            body=(ctx, idx) -> Thunk(;
                preamble=quote
                    if $qos > $qos_stop
                        $qos_stop = max($qos_stop << 1, 1)
                        Finch.resize_if_smaller!($(lvl.idx), $qos_stop)
                        $(contain(
                            ctx_2 -> assemble_level!(
                                ctx_2,
                                lvl.lvl,
                                value(qos, Tp),
                                value(qos_stop, Tp),
                            ),
                            ctx,
                        ))
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
                        $(fbr.dirty) = true
                        $(lvl.idx)[$qos] = $(ctx(idx))
                        $qos += $(Tp(1))
                        $(
                            if issafe(get_mode_flag(ctx))
                                quote
                                    $(lvl.prev_pos) = $(ctx(pos))
                                end
                            end
                        )
                    end
                end,
            ),
        ),
        epilogue=quote
            $(lvl.ptr)[$(ctx(pos)) + 1] += $qos - $qos_fill - 1
            $qos_fill = $qos - 1
        end,
    )
end

function coalesce_level!(
    lvl::SparseListLevel, global_fbr_map, factor, max_dim, P, coalescent, mode
)
    idx = lvl.idx.data
    ptr = lvl.ptr.data
    
    lvl_ptr = coalescent.ptr
    lvl_idx = coalescent.idx
    max_idx = lvl.shape

    if sum(length, idx) < 1
        return nothing
    end

    if factor > 1
        unwrap_dense(global_fbr_map, factor, P)
        was_dense = true
        factor = 1
    else
        was_dense = false
    end

    if mode == :weak
        gfm2, max_dim2 = merge_splist_weak(global_fbr_map, ptr, idx, P, max_dim, max_idx, was_dense, lvl_ptr, lvl_idx)
    else
        gfm2, max_dim2 = merge_splist(global_fbr_map, ptr, idx, P, max_dim, max_idx, was_dense, lvl_ptr, lvl_idx)
    end

    coalesce_level!(
        lvl.lvl, gfm2, factor, max_dim2, P, coalescent.lvl, mode
    )
end

function setup_coalesce!(lvl::SparseListLevel, max_pos, coalescent)
    lvl_ptr = coalescent.ptr
    lvl_idx = coalescent.idx
    nnz = sum(length, lvl.idx.data)
    if nnz < 1
        return false
    end
    resize!(lvl_idx, nnz)
    resize!(lvl_ptr, max_pos + 1) ##maybe need fill 0

    lvl_ptr[1] = 1
    lvl_ptr[end] = nnz + 1
    
    setup_coalesce!(lvl.lvl, nnz, coalescent.lvl)
end

function coalesce_fast!(tid, meta, P, lvl::SparseListLevel, coalescent, was_dense)
    ptr = lvl.ptr.data
    idx = lvl.idx.data
    lvl_ptr = coalescent.ptr
    lvl_idx = coalescent.idx

    fastmerge_splist(tid, ptr, idx, P, lvl_ptr, lvl_idx, meta, was_dense)
    coalesce_fast!(tid, meta, P, lvl.lvl, coalescent.lvl, false)
end

@inbounds function fastmerge_splist(tid, ptr, idx, P, lvl_ptr, lvl_idx, pos_offsets, was_dense)
    nnz_cutoffs = Vector{Int}(undef, P + 1)
    nnz_cutoffs[1] = 1
    for p in 2:P+1
        nnz_cutoffs[p] = nnz_cutoffs[p - 1] + length(idx[p - 1])
    end
    nnz = nnz_cutoffs[end] - 1
    max_pos = length(lvl_ptr) - 1

    base, rem = divrem(nnz, P)
    offset = (tid - 1) * base + min(tid - 1, rem)
    chunksize = base + (tid <= rem ? 1 : 0)
    work_lb = 1 + offset

    proc_id_lower = binary_search(work_lb, nnz_cutoffs)
    nz_id_lower = work_lb - nnz_cutoffs[proc_id_lower] + 1

    if was_dense
        ##Optimize this pass, currently O(P * npos / P), way too slow.
        proc = proc_id_lower
        idx_read = nz_id_lower
        idx_write = nnz_cutoffs[proc] + nz_id_lower - 1
        ceil = idx_write + chunksize
        while idx_write < ceil
            lvl_idx[idx_write] = idx[proc][idx_read]
            idx_read += 1
            idx_write += 1

            if idx_read > length(idx[proc])
                idx_read = 1
                proc += 1
            end
        end

        pos_base, pos_rem = divrem(max_pos - 1, P)
        pos_offset = (tid - 1) * pos_base + min(tid - 1, pos_rem)
        pos_chunksize = pos_base + (tid <= pos_rem ? 1 : 0)
        pos_lb = 2 + pos_offset
        pos_ub = pos_lb + pos_chunksize - 1

        for pos in pos_lb:pos_ub
            total = 1
            for p in 1:P
                total += ptr[p][pos] - 1
            end
            lvl_ptr[pos] = total
        end
    else
        work_ub = work_lb + chunksize - 1
        proc_id_upper = binary_search(work_ub, nnz_cutoffs)
        nz_id_upper = work_ub - nnz_cutoffs[proc_id_upper] + 1
        lfbr_lower = binary_search(nz_id_lower, ptr[proc_id_lower])
        lfbr_upper = binary_search(nz_id_upper, ptr[proc_id_upper])

        pos_lb = pos_offsets[tid][proc_id_lower] + lfbr_lower - 1
        pos_ub = min(pos_offsets[tid][proc_id_upper] + lfbr_upper - 1, max_pos - 1)

        if nz_id_upper < ptr[proc_id_upper][lfbr_upper + 1] - 1
            shares_border = true
        elseif lfbr_upper < length(ptr[proc_id_upper]) - 1
            shares_border = false
        elseif proc_id_upper < P
            shares_border = pos_offsets[tid][proc_id_upper + 1] == pos_ub
        else
            shares_border = false
        end

        ##copy idx
        proc = proc_id_lower
        idx_read = nz_id_lower
        idx_write = nnz_cutoffs[proc] + nz_id_lower - 1
        ceil = idx_write + chunksize
        while idx_write < ceil
            lvl_idx[idx_write] = idx[proc][idx_read]
            idx_read += 1
            idx_write += 1

            if idx_read > length(idx[proc])
                idx_read = 1
                proc += 1
            end
        end

        ##copy pos
        proc = proc_id_lower
        pos_read = lfbr_lower

        pos_write = 2
        for p in 1:proc - 1
            pos_write += length(ptr[p]) - 1
            pos_offsets[tid][p + 1] == pos_offsets[tid][p] + length(ptr[p]) - 2 && (pos_write -= 1)
        end
        pos_write += lfbr_lower - 1

        ceil = 3
        for p in 1:proc_id_upper - 1
            ceil += length(ptr[p]) - 1
            pos_offsets[tid][p + 1] == pos_offsets[tid][p] + length(ptr[p]) - 2 && (ceil -= 1)
        end
        ceil += lfbr_upper - 1
        shares_border && (ceil -= 1)

        prefix = ptr[proc][pos_read] + nnz_cutoffs[proc] - 1
        while pos_write < ceil
            delta = ptr[proc][pos_read + 1] - ptr[proc][pos_read ]
            prefix += delta
            lvl_ptr[pos_write] = prefix
            pos_write += 1
            pos_read += 1

            if pos_read > length(ptr[proc]) - 1
                pos_read = 1
                old_proc = proc
                proc += 1
                if proc > P
                    break
                end
                if pos_offsets[tid][old_proc + 1] == pos_offsets[tid][old_proc] + length(ptr[old_proc]) - 2
                    pos_write -= 1
                end
            end
        end

        for p in 1:P
            pos_offsets[tid][p] = nnz_cutoffs[p]
        end
    end
end

Base.@propagate_inbounds function merge_splist(gfm, ptr, idx, P, max_pos, max_idx, was_dense, lvl_ptr, lvl_idx)
    resize!(lvl_ptr, max_pos + 1)
    fill!(lvl_ptr, 0)
    uq_pairs = Vector{Int}(undef, P + 1)
    uq_pairs[1] = 0
    gfm2 = Vector{Vector{Int}}(undef, P)
    nnz = 0
    for p in 1:P
        gfm2[p] = Vector{Int}(undef, length(idx[p]))
        nnz += length(idx[p])
    end
    prefixes = Vector{Int}(undef, P + 1)
    prefixes[1] = 1

    chk_size = fld(max_pos + P - 1, P)
    chk_nnz = fld(nnz + P - 1, P)
    Threads.@threads for tid in 1:P
        uq_pairs[tid + 1] = 0
        pos = (tid - 1) * chk_size + 1 #global position to be merged
        cap = min(tid * chk_size + 1, max_pos + 1)
        work_lb = (tid - 1) * chk_nnz
        work_ub = min(tid * chk_nnz, nnz) ##Processor is responsible for [pos_lb, pos_ub) nonzeroes

        if tid == 1
            lb = (1, 1)
        else
            lb = find_split(work_lb, max_pos, max_idx, ptr, idx, gfm, P)
        end

        if tid == P
            ub = (max_pos, max_idx)
        else
            split = find_split(work_ub, max_pos, max_idx, ptr, idx, gfm, P)
            ub = split[2] > 1 ? (split[1], split[2] - 1) : (split[1] - 1, max_idx)
        end
    
        pos = lb[1]
        cap = ub[1] + 1
        idxlb = lb[2]
        idxub = ub[2]

        posdata = Vector{Tuple{Int, Int, Int, Int}}(undef, P + 1)
        ord = Base.Order.Lt((i, j) -> lowerfbr(posdata[i], posdata[j]))
        heap = BinaryHeap{Int}(ord)
        sizehint!(heap, P)

        for proc in 1:P
            lo, hi = 1, length(gfm[proc])
            lfbr = binary_search_lb(pos, gfm[proc], lo, hi)

            if lfbr < 1
                continue
            end

            ##skip zeroes.
            while lfbr < length(ptr[proc]) && ptr[proc][lfbr + 1] - ptr[proc][lfbr] < 1 && gfm[proc][lfbr] < cap
                lfbr += 1
            end
            if lfbr >= length(ptr[proc])
                continue
            end
            adj_pos = gfm[proc][lfbr]
            if adj_pos >= cap
                continue
            end
            if adj_pos == pos
                i = binary_search_lb(idxlb, idx[proc], ptr[proc][lfbr], ptr[proc][lfbr+1] - 1)
                if i < 1
                    lfbr += 1
                    while lfbr < length(ptr[proc]) && ptr[proc][lfbr+1] - ptr[proc][lfbr] < 1 && gfm[proc][lfbr] < cap
                        lfbr += 1
                    end
                    (lfbr >= length(ptr[proc])) && continue
                    adj_pos = gfm[proc][lfbr]
                    adj_pos >= cap && continue
                    i = ptr[proc][lfbr]
                end
            else
                i = ptr[proc][lfbr]
            end
            posdata[proc] = (adj_pos, idx[proc][i], lfbr, i - ptr[proc][lfbr])
            push!(heap, proc)
        end
        
        posdata[end] = (typemax(Int), typemax(Int), -1, -1)
        push!(heap, P + 1)

        c_proc = pop!(heap)
        c_pos, c_idx, c_lfbr, c_nz = posdata[c_proc]
        prev = (c_pos, -1)
        seen = 0
        start_pos = was_dense ? pos : c_pos
        deferred = false
        while !isempty(heap)
            if c_pos == cap - 1 && c_idx > idxub
                deferred = true #another thread owns the data
                c_proc = pop!(heap)
                c_pos, c_idx, c_lfbr, c_nz = posdata[c_proc]
                continue
            end
            if (c_pos, c_idx) != prev
                ##New position, otherwise just a new index
                if prev[1] != c_pos
                    lvl_ptr[prev[1]+1] = seen
                    seen = 0
                end
                seen += 1
                uq_pairs[tid+1] += 1
                prev = (c_pos, c_idx)
            end
            delta = ptr[c_proc][c_lfbr + 1] - ptr[c_proc][c_lfbr]
            c_nz += 1

            if c_nz >= delta
                c_nz = 0
                c_lfbr += 1

                while c_lfbr < length(ptr[c_proc]) && ptr[c_proc][c_lfbr + 1] - ptr[c_proc][c_lfbr] < 1 && gfm[c_proc][c_lfbr] < cap
                    lvl_ptr[gfm[c_proc][c_lfbr] + 1] = 0
                    c_lfbr += 1
                end
            end

            if c_lfbr < length(ptr[c_proc])
                c_gpos = gfm[c_proc][c_lfbr]
                if c_gpos < cap
                    c_idx = idx[c_proc][ptr[c_proc][c_lfbr] + c_nz]
                    posdata[c_proc] = (c_gpos, c_idx, c_lfbr, c_nz)
                    push!(heap,  c_proc)
                end
            end
 
            c_proc = pop!(heap)
            c_pos, c_idx, c_lfbr, c_nz = posdata[c_proc]
        end

        boundary = cap - 1
        is_writer = (prev[1] < boundary) || !deferred
        if is_writer
            lvl_ptr[prev[1] + 1] = seen
            cap = prev[1] + 1
        else
            cap = boundary
        end

        if tid == P
            cap = length(lvl_ptr)
        end

        for p in start_pos+2:cap
            lvl_ptr[p] = lvl_ptr[p] + lvl_ptr[p-1]
        end

        prefixes[tid + 1] = uq_pairs[tid + 1]
    end
    for p in 2:P + 1
        uq_pairs[p] += uq_pairs[p - 1]
        prefixes[p] += prefixes[p - 1]
    end

    resize!(lvl_idx, uq_pairs[end])
    lvl_ptr[1] = 1

    ##Phase 2: Compute idx and gfm2.
    Threads.@threads for tid in 1:P
        pos = (tid - 1) * chk_size + 1 #global position to be merged
        cap = min(tid * chk_size + 1, max_pos + 1)
        work_lb = (tid - 1) * chk_nnz
        work_ub = min(tid * chk_nnz, nnz) ##Processor is responsible for [pos_lb, pos_ub) nonzeroes

        if tid == 1
            lb = (1, 1)
        else
            lb = find_split(work_lb, max_pos, max_idx, ptr, idx, gfm, P)
        end

        if tid == P
            ub = (max_pos, max_idx)
        else
            split = find_split(work_ub, max_pos, max_idx, ptr, idx, gfm, P)
            ub = split[2] > 1 ? (split[1], split[2] - 1) : (split[1] - 1, max_idx)
        end

        pos = lb[1]
        cap = ub[1] + 1
        idxlb = lb[2]
        idxub = ub[2]

        posdata = Vector{Tuple{Int, Int, Int, Int}}(undef, P + 1)
        ord = Base.Order.Lt((i, j) -> lowerfbr(posdata[i], posdata[j]))
        heap = BinaryHeap{Int}(ord)
        sizehint!(heap, P)
        ##Can probably reduce this preprocessing.
        for proc in 1:P
            lo, hi = 1, length(gfm[proc])
            lfbr = binary_search_lb(pos, gfm[proc], lo, hi)
            if lfbr < 1
                continue
            end

            ##skip zeroes.
            while lfbr < length(ptr[proc]) && ptr[proc][lfbr + 1] - ptr[proc][lfbr] < 1 && gfm[proc][lfbr] < cap
                lfbr += 1
            end
            if lfbr >= length(ptr[proc])
                continue
            end
            adj_pos = gfm[proc][lfbr]
            if adj_pos >= cap
                continue
            end
            if adj_pos == pos
                i = binary_search_lb(idxlb, idx[proc], ptr[proc][lfbr], ptr[proc][lfbr+1] - 1)
                if i < 1
                    lfbr += 1
                    while lfbr < length(ptr[proc]) && ptr[proc][lfbr+1] - ptr[proc][lfbr] < 1 && gfm[proc][lfbr] < cap
                        lfbr += 1
                    end
                    (lfbr >= length(ptr[proc])) && continue
                    adj_pos = gfm[proc][lfbr]
                    adj_pos >= cap && continue
                    i = ptr[proc][lfbr]
                end
            else
                i = ptr[proc][lfbr]
            end
            posdata[proc] = (adj_pos, idx[proc][i], lfbr, i - ptr[proc][lfbr])
            push!(heap, proc)
        end
        posdata[end] = (typemax(Int), typemax(Int), -1, -1)
        push!(heap, P + 1)

        c_proc = pop!(heap)
        c_pos, c_idx, c_lfbr, c_nz = posdata[c_proc]
        prev = (c_pos, -1)
        seen = 0
        start_pos = was_dense ? pos : c_pos
        deferred = false

        while !isempty(heap)
            if c_pos == cap - 1 && c_idx > idxub
                deferred = true
                c_proc = pop!(heap)
                c_pos, c_idx, c_lfbr, c_nz = posdata[c_proc]
                continue
            end
            if (c_pos, c_idx) != prev
                ##Every unique pair is a unique index.
                lvl_idx[uq_pairs[tid] + seen + 1] = c_idx
                seen += 1
                prev = (c_pos, c_idx)
            end
            gfm2[c_proc][ptr[c_proc][c_lfbr] + c_nz] = uq_pairs[tid] + seen

            delta = ptr[c_proc][c_lfbr + 1] - ptr[c_proc][c_lfbr]
            c_nz += 1

            if c_nz >= delta
                c_nz = 0
                c_lfbr += 1

                while c_lfbr < length(ptr[c_proc]) && ptr[c_proc][c_lfbr + 1] - ptr[c_proc][c_lfbr] < 1
                    c_lfbr += 1
                end
            end
            
            if c_lfbr < length(ptr[c_proc])
                c_gpos = gfm[c_proc][c_lfbr]
                if c_gpos < cap
                    c_idx = idx[c_proc][ptr[c_proc][c_lfbr] + c_nz]
                    posdata[c_proc] = (c_gpos, c_idx, c_lfbr, c_nz)
                    push!(heap,  c_proc)
                end
            end
 
            c_proc = pop!(heap)
            c_pos, c_idx, c_lfbr, c_nz = posdata[c_proc]
        end

        boundary = cap - 1
        is_writer = (prev[1] < boundary) || !deferred
        cap = is_writer ? prev[1] + 1 : boundary

        if tid == P
            cap = length(lvl_ptr)
        end

        for p in start_pos+1:cap
            lvl_ptr[p] += prefixes[tid]
        end
    end
    return gfm2, uq_pairs[end]
end

Base.@propagate_inbounds function merge_splist_weak(gfm, ptr, idx, P, max_pos, max_idx, was_dense, lvl_ptr, lvl_idx)
    resize!(lvl_ptr, max_pos + 1)
    fill!(lvl_ptr, 0)
    gfm2 = Vector{Vector{Int}}(undef, P)
    nnz = 0
    for p in 1:P
        gfm2[p] = Vector{Int}(undef, length(idx[p]))
        nnz += length(idx[p])
    end
    resize!(lvl_idx, nnz) ##Assumption: no duplicates.

    chk_nnz = fld(nnz + P - 1, P)
    Threads.@threads for tid in 1:P
        work_lb = (tid - 1) * chk_nnz
        work_ub = min(tid * chk_nnz, nnz)

        if tid == 1
            lb = (1, 1)
            uq_pairs_prior = 0
        else
            lb = find_split(work_lb, max_pos, max_idx, ptr, idx, gfm, P)
            uq_pairs_prior = total_idx(lb[2], lb[1], ptr, idx, gfm, P) - 1 ##Assuming no duplicates, EVERY prior (p, i) pair is unique.
        end

        if tid == P
            ub = (max_pos, max_idx)
        else
            split = find_split(work_ub, max_pos, max_idx, ptr, idx, gfm, P)
            ub = split[2] > 1 ? (split[1], split[2] - 1) : (split[1] - 1, max_idx)
        end

        pos = lb[1]
        cap = ub[1] + 1
        idxlb = lb[2]
        idxub = ub[2]

        posdata = Vector{Tuple{Int, Int, Int, Int}}(undef, P + 1)
        ord = Base.Order.Lt((i, j) -> lowerfbr(posdata[i], posdata[j]))
        heap = BinaryHeap{Int}(ord)
        sizehint!(heap, P)

        for proc in 1:P
            lo, hi = 1, length(gfm[proc])
            lfbr = binary_search_lb(pos, gfm[proc], lo, hi)

            if lfbr < 1
                continue
            end

            ##skip zeroes.
            while lfbr < length(ptr[proc]) && ptr[proc][lfbr+1] - ptr[proc][lfbr] < 1 && gfm[proc][lfbr] < cap
                lfbr += 1
            end
            if lfbr >= length(ptr[proc])
                continue
            end
            adj_pos = gfm[proc][lfbr]
            if adj_pos >= cap
                continue
            end
            if adj_pos == pos
                i = binary_search_lb(idxlb, idx[proc], ptr[proc][lfbr], ptr[proc][lfbr+1] - 1)
                if i < 1
                    lfbr += 1
                    while lfbr < length(ptr[proc]) && ptr[proc][lfbr+1] - ptr[proc][lfbr] < 1 && gfm[proc][lfbr] < cap
                        lfbr += 1
                    end
                    (lfbr >= length(ptr[proc])) && continue
                    adj_pos = gfm[proc][lfbr]
                    adj_pos >= cap && continue
                    i = ptr[proc][lfbr]
                end
            else
                i = ptr[proc][lfbr]
            end
            posdata[proc] = (adj_pos, idx[proc][i], lfbr, i - ptr[proc][lfbr])
            push!(heap, proc)
        end
        
        posdata[end] = (typemax(Int), typemax(Int), -1, -1)
        push!(heap, P + 1)

        c_proc = pop!(heap)
        c_pos, c_idx, c_lfbr, c_nz = posdata[c_proc]
        prev = (c_pos, -1)
        seen_pos = 0
        seen_idx = 0
        start_pos = was_dense ? pos : c_pos
        deferred = false
        while !isempty(heap)
            if c_pos == cap - 1 && c_idx > idxub
                deferred = true #another thread owns the data
                c_proc = pop!(heap)
                c_pos, c_idx, c_lfbr, c_nz = posdata[c_proc]
                continue
            end
            
            ##We assume every pair is unique, at least an idx update happens every iteration.
            if prev[1] != c_pos
                lvl_ptr[prev[1]+1] = seen_pos
                seen_pos = 0
            end
            seen_pos += 1
            lvl_idx[uq_pairs_prior + seen_idx + 1] = c_idx
            seen_idx += 1
            gfm2[c_proc][ptr[c_proc][c_lfbr] + c_nz] = uq_pairs_prior + seen_idx
            prev = (c_pos, c_idx)

              delta = ptr[c_proc][c_lfbr + 1] - ptr[c_proc][c_lfbr]
            c_nz += 1

            if c_nz >= delta
                c_nz = 0
                c_lfbr += 1

                while c_lfbr < length(ptr[c_proc]) && ptr[c_proc][c_lfbr + 1] - ptr[c_proc][c_lfbr] < 1
                    c_lfbr += 1
                end
            end
            
            if c_lfbr < length(ptr[c_proc])
                c_gpos = gfm[c_proc][c_lfbr]
                if c_gpos < cap
                    c_idx = idx[c_proc][ptr[c_proc][c_lfbr] + c_nz]
                    posdata[c_proc] = (c_gpos, c_idx, c_lfbr, c_nz)
                    push!(heap,  c_proc)
                end
            end
 
            c_proc = pop!(heap)
            c_pos, c_idx, c_lfbr, c_nz = posdata[c_proc]
        end

        boundary = cap - 1
        is_writer = (prev[1] < boundary) || !deferred
        if is_writer
            lvl_ptr[prev[1] + 1] = seen_pos
            cap = prev[1] + 1
        else
            cap = boundary
        end

        if tid == P
            cap = length(lvl_ptr)
        end
        
        for p in start_pos+2:cap
            lvl_ptr[p] = lvl_ptr[p] + lvl_ptr[p-1]
        end

        for p in start_pos+1:cap
            lvl_ptr[p] += uq_pairs_prior + 1
        end
    end
    lvl_ptr[1] = 1
    return gfm2, nnz
end

Base.@propagate_inbounds function lowerfbr(a::Tuple{Int,Int,Int,Int}, b::Tuple{Int,Int,Int,Int})
    if a[1] < b[1]
        return true
    elseif a[1] == b[1] && a[2] < b[2]
        return true
    else
        return false
    end
end

Base.@propagate_inbounds function total_pos(candidate, ptr, gfm, P)
    total = 0
    for proc in 1:P
        lfbr = binary_search_lb(candidate, gfm[proc], 1, length(gfm[proc]))
        lfbr < 1 && continue
        total += ptr[proc][lfbr+1] - 1
    end
    return total
end

Base.@propagate_inbounds function total_idx(candidate, pos, ptr, idx, gfm, P)
    total = 0
    for proc in 1:P
        lfbr = binary_search_lb(pos, gfm[proc], 1, length(gfm[proc]))
        lfbr < 1 && continue
        total += ptr[proc][lfbr] - 1

        lo_b = ptr[proc][lfbr]
        hi_b = ptr[proc][lfbr+1] - 1
        hi_b < lo_b && continue  # empty fiber, nothing more to add

        gidx = binary_search_lb(candidate, idx[proc], lo_b, hi_b)
        if gidx < 1
            total += hi_b - lo_b + 1   #if we "own" more idx than present, claim all the indices.
        else
            total += gidx - lo_b
            if idx[proc][gidx] == candidate
                total += 1
            end
        end
    end
    return total
end

Base.@propagate_inbounds function find_split(target, max_pos, max_idx, ptr, idx, gfm, P)
    posx = -1
    lo, hi = 1, max_pos
    while lo <= hi
        candidate = div(lo + hi, 2)
        total = total_pos(candidate, ptr, gfm, P)
        if total >= target
            posx = candidate
            hi = candidate - 1
        else
            lo = candidate + 1
        end
    end
    posx == -1 && return (max_pos, max_idx)

    idxx = -1
    lo, hi = 1, max_idx
    while lo <= hi
        candidate = div(lo + hi, 2)
        total = total_idx(candidate, posx, ptr, idx, gfm, P)
        if total >= target
            idxx = candidate
            hi = candidate - 1
        else
            lo = candidate + 1
        end
    end
    return (posx, idxx)
end