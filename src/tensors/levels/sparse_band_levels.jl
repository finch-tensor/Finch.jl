"""
SparseBandLevel{[Ti=Int], [Ptr, Idx, Ofs]}(lvl, [dim])

Like the [`SparseBlockListLevel`](@ref), but stores only a single block, and fills in zeros.

```jldoctest
julia> Tensor(Dense(SparseBand(Element(0.0))), [10 0 20; 30 40 0; 0 0 50])
Dense [:,1:3]
├─[:,1]: SparseList (0.0) [1:3]
│ ├─[1]: 10.0
│ ├─[2]: 30.0
├─[:,2]: SparseList (0.0) [1:3]
├─[:,3]: SparseList (0.0) [1:3]
│ ├─[1]: 20.0
│ ├─[3]: 40.0
"""
struct SparseBandLevel{Ti, Ptr<:AbstractVector, Idx<:AbstractVector, Ofs<:AbstractVector, Lvl} <: AbstractLevel
    lvl::Lvl
    shape::Ti
    ptr::Ptr
    idx::Idx
    ofs::Ofs
end

const SparseBand = SparseBandLevel
SparseBandLevel(lvl::Lvl) where {Lvl} = SparseBandLevel{Int}(lvl)
SparseBandLevel(lvl, shape, args...) = SparseBandLevel{typeof(shape)}(lvl, shape, args...)
SparseBandLevel{Ti}(lvl) where {Ti} = SparseBandLevel{Ti}(lvl, zero(Ti))
SparseBandLevel{Ti}(lvl, shape) where {Ti} = SparseBandLevel{Ti}(lvl, shape, postype(lvl)[1], Ti[], postype(lvl)[])
SparseBandLevel{Ti}(lvl::Lvl, shape, ptr::Ptr, idx::Idx, ofs::Ofs) where {Ti, Lvl, Ptr, Idx, Ofs} =
    SparseBandLevel{Ti, Ptr, Idx, Ofs, Lvl}(lvl, Ti(shape), ptr, idx, ofs)

function postype(::Type{SparseBandLevel{Ti, Ptr, Idx, Ofs, Lvl}}) where {Ti, Ptr, Idx, Ofs, Lvl}
    return postype(Lvl)
end

function moveto(lvl::SparseBandLevel{Ti}, device) where {Ti}
    lvl_2 = moveto(lvl.lvl, device)
    ptr_2 = moveto(lvl.ptr, device)
    idx_2 = moveto(lvl.idx, device)
    ofs_2 = moveto(lvl.ofs, device)
    return SparseBandLevel{Ti}(lvl_2, lvl.shape, ptr_2, idx_2, ofs_2)
end

Base.summary(lvl::SparseBandLevel) = "SparseBand($(summary(lvl.lvl)))"
similar_level(lvl::SparseBandLevel, fill_value, eltype::Type, dim, tail...) =
    SparseBand(similar_level(lvl.lvl, fill_value, eltype, tail...), dim)

pattern!(lvl::SparseBandLevel{Ti}) where {Ti} =
    SparseBandLevel{Ti}(pattern!(lvl.lvl), lvl.shape, lvl.ptr, lvl.idx, lvl.ofs)

function countstored_level(lvl::SparseBandLevel, pos)
    countstored_level(lvl.lvl, lvl.ofs[lvl.ptr[pos + 1]]-1)
end

set_fill_value!(lvl::SparseBandLevel{Ti}, init) where {Ti} =
    SparseBandLevel{Ti}(set_fill_value!(lvl.lvl, init), lvl.shape, lvl.ptr, lvl.idx, lvl.ofs)

Base.resize!(lvl::SparseBandLevel{Ti}, dims...) where {Ti} =
    SparseBandLevel{Ti}(resize!(lvl.lvl, dims[1:end-1]...), dims[end], lvl.ptr, lvl.idx, lvl.ofs)

function Base.show(io::IO, lvl::SparseBandLevel{Ti, Ptr, Idx, Ofs, Lvl}) where {Ti, Ptr, Idx, Ofs, Lvl}
    if get(io, :compact, false)
        print(io, "SparseBand(")
    else
        print(io, "SparseBand{$Ti}(")
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
        show(io, lvl.ofs)
    end
    print(io, ")")
end

labelled_show(io::IO, fbr::SubFiber{<:SparseBandLevel}) =
    print(io, "SparseBand (", fill_value(fbr), ") [", ":,"^(ndims(fbr) - 1), "1:", size(fbr)[end], "]")

function labelled_children(fbr::SubFiber{<:SparseBandLevel})
    lvl = fbr.lvl
    pos = fbr.pos
    pos + 1 > length(lvl.ptr) && return []
    res = []
    for r = lvl.ptr[pos]:lvl.ptr[pos + 1] - 1
        i = lvl.idx[r]
        qos = lvl.ofs[r]
        l = lvl.ofs[r + 1] - lvl.ofs[r]
        for qos = lvl.ofs[r]:lvl.ofs[r + 1] - 1
            push!(res, LabelledTree(cartesian_label([range_label() for _ = 1:ndims(fbr) - 1]..., i - (lvl.ofs[r + 1] - 1) + qos), SubFiber(lvl.lvl, qos)))
        end
    end
    res
end

@inline level_ndims(::Type{<:SparseBandLevel{Ti, Ptr, Idx, Ofs, Lvl}}) where {Ti, Ptr, Idx, Ofs, Lvl} = 1 + level_ndims(Lvl)
@inline level_size(lvl::SparseBandLevel) = (level_size(lvl.lvl)..., lvl.shape)
@inline level_axes(lvl::SparseBandLevel) = (level_axes(lvl.lvl)..., Base.OneTo(lvl.shape))
@inline level_eltype(::Type{<:SparseBandLevel{Ti, Ptr, Idx, Ofs, Lvl}}) where {Ti, Ptr, Idx, Ofs, Lvl} = level_eltype(Lvl)
@inline level_fill_value(::Type{<:SparseBandLevel{Ti, Ptr, Idx, Ofs, Lvl}}) where {Ti, Ptr, Idx, Ofs, Lvl} = level_fill_value(Lvl)
data_rep_level(::Type{<:SparseBandLevel{Ti, Ptr, Idx, Ofs, Lvl}}) where {Ti, Ptr, Idx, Ofs, Lvl} = SparseData(data_rep_level(Lvl))

(fbr::AbstractFiber{<:SparseBandLevel})() = fbr
function (fbr::SubFiber{<:SparseBandLevel})(idxs...)
    isempty(idxs) && return fbr
    lvl = fbr.lvl
    p = fbr.pos
    r = lvl.ptr[p] + searchsortedfirst(@view(lvl.idx[lvl.ptr[p]:lvl.ptr[p + 1] - 1]), idxs[end]) - 1
    r < lvl.ptr[p + 1] || return fill_value(fbr)
    q = lvl.ofs[r + 1] - 1 - lvl.idx[r] + idxs[end]
    q >= lvl.ofs[r] || return fill_value(fbr)
    fbr_2 = SubFiber(lvl.lvl, q)
    return fbr_2(idxs[1:end-1]...)
end

mutable struct VirtualSparseBandLevel <: AbstractVirtualLevel
    lvl
    ex
    Ti
    shape
    qos_fill
    qos_stop
    ros_fill
    ros_stop
    dirty
    ptr
    idx
    ofs
    prev_pos
end

is_level_injective(ctx, lvl::VirtualSparseBandLevel) = [is_level_injective(ctx, lvl.lvl)..., false]
function is_level_atomic(ctx, lvl::VirtualSparseBandLevel)
    (below, atomic) = is_level_atomic(ctx, lvl.lvl)
    return ([below; [atomic]], atomic)
end
function is_level_concurrent(ctx, lvl::VirtualSparseBandLevel)
    (data, _) = is_level_concurrent(ctx, lvl.lvl)
    return ([data; [false]], false)
end

postype(lvl::VirtualSparseBandLevel) = postype(lvl.lvl)


function virtualize(ctx, ex, ::Type{SparseBandLevel{Ti, Ptr, Idx, Ofs, Lvl}}, tag=:lvl) where {Ti, Ptr, Idx, Ofs, Lvl}
    sym = freshen(ctx, tag)
    shape = value(:($sym.shape), Int)
    qos_fill = freshen(ctx, sym, :_qos_fill)
    qos_stop = freshen(ctx, sym, :_qos_stop)
    ros_fill = freshen(ctx, sym, :_ros_fill)
    ros_stop = freshen(ctx, sym, :_ros_stop)
    dirty = freshen(ctx, sym, :_dirty)
    ptr = freshen(ctx, tag, :_ptr)
    idx = freshen(ctx, tag, :_idx)
    ofs = freshen(ctx, tag, :_ofs)
    push_preamble!(ctx, quote
        $sym = $ex
        $ptr = $sym.ptr
        $idx = $sym.idx
        $ofs = $sym.ofs
    end)
    prev_pos = freshen(ctx, sym, :_prev_pos)
    lvl_2 = virtualize(ctx, :($sym.lvl), Lvl, sym)
    VirtualSparseBandLevel(lvl_2, sym, Ti, shape, qos_fill, qos_stop, ros_fill, ros_stop, dirty, ptr, idx, ofs, prev_pos)
end
function lower(ctx::AbstractCompiler, lvl::VirtualSparseBandLevel, ::DefaultStyle)
    quote
        $SparseBandLevel{$(lvl.Ti)}(
            $(ctx(lvl.lvl)),
            $(ctx(lvl.shape)),
            $(lvl.ptr),
            $(lvl.idx),
            $(lvl.ofs),
        )
    end
end

Base.summary(lvl::VirtualSparseBandLevel) = "SparseBand($(summary(lvl.lvl)))"

function virtual_level_size(ctx, lvl::VirtualSparseBandLevel)
    ext = Extent(literal(lvl.Ti(1)), lvl.shape)
    (virtual_level_size(ctx, lvl.lvl)..., ext)
end

function virtual_level_resize!(ctx, lvl::VirtualSparseBandLevel, dims...)
    lvl.shape = getstop(dims[end])
    lvl.lvl = virtual_level_resize!(ctx, lvl.lvl, dims[1:end-1]...)
    lvl
end

virtual_level_eltype(lvl::VirtualSparseBandLevel) = virtual_level_eltype(lvl.lvl)
virtual_level_fill_value(lvl::VirtualSparseBandLevel) = virtual_level_fill_value(lvl.lvl)

function virtual_moveto_level(ctx::AbstractCompiler, lvl::VirtualSparseBandLevel, arch)
    ptr_2 = freshen(ctx, lvl.ptr)
    tbl_2 = freshen(ctx, lvl.tbl)
    ofs_2 = freshen(ctx, lvl.ofs)
    push_preamble!(ctx, quote
        $ptr_2 = $(lvl.ptr)
        $tbl_2 = $(lvl.tbl)
        $ofs_2 = $(lvl.ofs)
        $(lvl.ptr) = $moveto($(lvl.ptr), $(ctx(arch)))
        $(lvl.tbl) = $moveto($(lvl.tbl), $(ctx(arch)))
        $(lvl.ofs) = $moveto($(lvl.ofs), $(ctx(arch)))
    end)
    push_epilogue!(ctx, quote
        $(lvl.ptr) = $ptr_2
        $(lvl.tbl) = $tbl_2
        $(lvl.ofs) = $ofs_2
    end)
    virtual_moveto_level(ctx, lvl.lvl, arch)
end

function declare_level!(ctx::AbstractCompiler, lvl::VirtualSparseBandLevel, pos, init)
    Tp = postype(lvl)
    Ti = lvl.Ti
    push_preamble!(ctx, quote
        $(lvl.qos_fill) = $(Tp(0))
        $(lvl.qos_stop) = $(Tp(0))
        $(lvl.ros_fill) = $(Tp(0))
        $(lvl.ros_stop) = $(Tp(0))
        Finch.resize_if_smaller!($(lvl.ofs), 1)
        $(lvl.ofs)[1] = 1
    end)
    if issafe(get_mode_flag(ctx))
        push_preamble!(ctx, quote
            $(lvl.prev_pos) = $(Tp(0))
        end)
    end
    lvl.lvl = declare_level!(ctx, lvl.lvl, literal(Tp(0)), init)
    return lvl
end

function assemble_level!(ctx, lvl::VirtualSparseBandLevel, pos_start, pos_stop)
    pos_start = ctx(cache!(ctx, :p_start, pos_start))
    pos_stop = ctx(cache!(ctx, :p_start, pos_stop))
    return quote
        Finch.resize_if_smaller!($(lvl.ptr), $pos_stop + 1)
        Finch.fill_range!($(lvl.ptr), 0, $pos_start + 1, $pos_stop + 1)
    end
end

function freeze_level!(ctx::AbstractCompiler, lvl::VirtualSparseBandLevel, pos_stop)
    p = freshen(ctx, :p)
    Tp = postype(lvl)
    pos_stop = ctx(cache!(ctx, :pos_stop, simplify(ctx, pos_stop)))
    ros_stop = freshen(ctx, :ros_stop)
    qos_stop = freshen(ctx, :qos_stop)
    push_preamble!(ctx, quote
        resize!($(lvl.ptr), $pos_stop + 1)
        for $p = 2:($pos_stop + 1)
            $(lvl.ptr)[$p] += $(lvl.ptr)[$p - 1]
        end
        $ros_stop = $(lvl.ptr)[$pos_stop + 1] - 1
        resize!($(lvl.idx), $ros_stop)
        resize!($(lvl.ofs), $ros_stop + 1)
        $qos_stop = $(lvl.ofs)[$ros_stop + 1] - $(Tp(1))
    end)
    lvl.lvl = freeze_level!(ctx, lvl.lvl, value(qos_stop))
    return lvl
end

function unfurl(ctx, fbr::VirtualSubFiber{VirtualSparseBandLevel}, ext, mode, ::Union{typeof(defaultread), typeof(walk)})
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    Tp = postype(lvl)
    Ti = lvl.Ti
    my_i = freshen(ctx, tag, :_i)
    my_i_start = freshen(ctx, tag, :_i)
    my_r = freshen(ctx, tag, :_r)
    my_r_stop = freshen(ctx, tag, :_r_stop)
    my_q = freshen(ctx, tag, :_q)
    my_q_stop = freshen(ctx, tag, :_q_stop)
    my_q_ofs = freshen(ctx, tag, :_q_ofs)
    my_i1 = freshen(ctx, tag, :_i1)

    Unfurled(
        arr = fbr,
        body = Thunk(
            preamble = quote
                $my_r = $(lvl.ptr)[$(ctx(pos))]
                $my_r_stop = $(lvl.ptr)[$(ctx(pos)) + $(Tp(1))] - 1
                if $my_r <= $my_r_stop
                    $my_i1 = $(lvl.idx)[$my_r]
                    $my_q_stop = $(lvl.ofs)[$my_r + $(Tp(1))]
                    $my_i_start = $my_i1 - ($my_q_stop - $(lvl.ofs)[$my_r] - 1)
                    $my_q_ofs = $my_q_stop - $my_i1 - $(Tp(1))
                else
                    $my_i_start = $(Ti(1))
                    $my_i1 = $(Ti(0))
                    $my_q_stop = $(Ti(0))
                    $my_q = $(Ti(0))
                end
            end,
            body = (ctx) -> Sequence([
                Phase(
                    stop = (ctx, ext) -> call(-, value(my_i_start), 1),
                    body = (ctx, ext) -> Run(FillLeaf(virtual_level_fill_value(lvl))),
                ),
                Phase(
                    stop = (ctx, ext) -> value(my_i1),
                    body = (ctx, ext) -> Lookup(
                        body = (ctx, i) -> Thunk(
                            preamble = :($my_q = $my_q_ofs + $(ctx(i))),
                            body = (ctx) -> instantiate(ctx, VirtualSubFiber(lvl.lvl, value(my_q, Tp)), mode),
                        )
                    )
                ),
                Phase(
                    body = (ctx, ext) -> Run(FillLeaf(virtual_level_fill_value(lvl))),
                ),
            ])
        )
    )
end

unfurl(ctx, fbr::VirtualSubFiber{VirtualSparseBandLevel}, ext, mode, proto::Union{typeof(defaultupdate), typeof(extrude)}) =
    unfurl(ctx, VirtualHollowSubFiber(fbr.lvl, fbr.pos, freshen(ctx, :null)), ext, mode, proto)
function unfurl(ctx, fbr::VirtualHollowSubFiber{VirtualSparseBandLevel}, ext, mode, ::Union{typeof(defaultupdate), typeof(extrude)})
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    Tp = postype(lvl)
    Ti = lvl.Ti
    my_p = freshen(ctx, tag, :_p)
    my_q = freshen(ctx, tag, :_q)
    my_i_prev = freshen(ctx, tag, :_i_prev)
    my_i_set = freshen(ctx, tag, :_i_set)
    qos = freshen(ctx, tag, :_qos)
    qos_set = freshen(ctx, tag, :_qos_set)
    ros = freshen(ctx, tag, :_ros)
    qos_fill = lvl.qos_fill
    qos_stop = lvl.qos_stop
    ros_fill = lvl.ros_fill
    ros_stop = lvl.ros_stop
    dirty = freshen(ctx, tag, :dirty)
    qos_2 = freshen(ctx, tag, :_qos_2)

    Unfurled(
        arr = fbr,
        body = Thunk(
            preamble = quote
                $ros = $ros_fill
                $qos = $qos_fill + 1
                $qos_set = $qos_fill
                $my_i_prev = $(Ti(-1))
                $my_i_set = $(Ti(-1))
                $(if issafe(get_mode_flag(ctx))
                    quote
                        $(lvl.prev_pos) < $(ctx(pos)) || throw(FinchProtocolError("SparseBandLevels cannot be updated multiple times"))
                    end
                end)
            end,
            body = (ctx) -> Lookup(
                body = (ctx, idx) -> Thunk(
                    preamble = quote
                        if $my_i_prev > 0
                            $(if issafe(get_mode_flag(ctx))
                                quote
                                    if $(ctx(idx)) < $my_i_prev
                                        throw(FinchProtocolError("SparseBandLevels cannot be updated out of order"))
                                    end
                                end
                            end)
                            $qos = $(ctx(idx)) - $my_i_prev + $qos_fill + 1
                        end
                        if $qos > $qos_stop
                            $qos_2 = $qos_stop + 1
                            while $qos > $qos_stop
                                $qos_stop = max($qos_stop << 1, 1)
                            end
                            $(contain(ctx_2->assemble_level!(ctx_2, lvl.lvl, value(qos_2, Tp), value(qos_stop, Tp)), ctx))
                        end
                        $dirty = false
                    end,
                    body = (ctx) -> instantiate(ctx, VirtualHollowSubFiber(lvl.lvl, value(qos, Tp), dirty), mode),
                    epilogue = quote
                        if $dirty
                            $(fbr.dirty) = true
                            if $my_i_prev <= 0
                                $my_i_prev = $(ctx(idx))
                            end
                            $my_i_set = $(ctx(idx))
                            $qos_set = $qos
                        end
                    end
                )
            ),
            epilogue = quote
                if $my_i_prev > 0
                    $ros += 1
                    if $ros > $ros_stop
                        $ros_stop = max($ros_stop << 1, 1)
                        Finch.resize_if_smaller!($(lvl.idx), $ros_stop)
                        Finch.resize_if_smaller!($(lvl.ofs), $ros_stop + 1)
                    end
                    $qos = $qos_set
                    $(lvl.idx)[$(ros)] = $my_i_set
                    $(lvl.ofs)[$(ros) + 1] = $qos + 1
                    $(if issafe(get_mode_flag(ctx))
                        quote
                            $(lvl.prev_pos) = $(ctx(pos))
                        end
                    end)
                    $qos_fill = $qos
                end
                $(lvl.ptr)[$(ctx(pos)) + 1] += $ros - $ros_fill
                $ros_fill = $ros
            end
        )
    )
end
