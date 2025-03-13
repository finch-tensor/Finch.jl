"""
SparseBlockListLevel{[Ti=Int], [Ptr, Idx, Ofs]}(lvl, [dim])

Like the [`SparseListLevel`](@ref), but contiguous subfibers are stored together in blocks.

```jldoctest
julia> Tensor(Dense(SparseBlockList(Element(0.0))), [10 0 20; 30 0 0; 0 0 40])
Dense [:,1:3]
├─[:,1]: SparseList (0.0) [1:3]
│ ├─[1]: 10.0
│ ├─[2]: 30.0
├─[:,2]: SparseList (0.0) [1:3]
├─[:,3]: SparseList (0.0) [1:3]
│ ├─[1]: 20.0
│ ├─[3]: 40.0

julia> Tensor(SparseBlockList(SparseBlockList(Element(0.0))), [10 0 20; 30 0 0; 0 0 40])
SparseList (0.0) [:,1:3]
├─[:,1]: SparseList (0.0) [1:3]
│ ├─[1]: 10.0
│ ├─[2]: 30.0
├─[:,3]: SparseList (0.0) [1:3]
│ ├─[1]: 20.0
│ ├─[3]: 40.0
"""
struct SparseBlockListLevel{
    Ti,Ptr<:AbstractVector,Idx<:AbstractVector,Ofs<:AbstractVector,Lvl
} <: AbstractLevel
    lvl::Lvl
    shape::Ti
    ptr::Ptr
    idx::Idx
    ofs::Ofs
end

const SparseBlockList = SparseBlockListLevel
SparseBlockListLevel(lvl::Lvl) where {Lvl} = SparseBlockListLevel{Int}(lvl)
function SparseBlockListLevel(lvl, shape, args...)
    SparseBlockListLevel{typeof(shape)}(lvl, shape, args...)
end
SparseBlockListLevel{Ti}(lvl) where {Ti} = SparseBlockListLevel{Ti}(lvl, zero(Ti))
function SparseBlockListLevel{Ti}(lvl, shape) where {Ti}
    SparseBlockListLevel{Ti}(lvl, shape, postype(lvl)[1], Ti[], postype(lvl)[])
end
function SparseBlockListLevel{Ti}(
    lvl::Lvl, shape, ptr::Ptr, idx::Idx, ofs::Ofs
) where {Ti,Lvl,Ptr,Idx,Ofs}
    SparseBlockListLevel{Ti,Ptr,Idx,Ofs,Lvl}(lvl, Ti(shape), ptr, idx, ofs)
end

function postype(
    ::Type{SparseBlockListLevel{Ti,Ptr,Idx,Ofs,Lvl}}
) where {Ti,Ptr,Idx,Ofs,Lvl}
    return postype(Lvl)
end

function transfer(lvl::SparseBlockListLevel{Ti}, device, style) where {Ti}
    lvl_2 = transfer(lvl.lvl, device, style)
    ptr_2 = transfer(lvl.ptr, device, style)
    idx_2 = transfer(lvl.idx, device, style)
    ofs_2 = transfer(lvl.ofs, device, style)
    return SparseBlockListLevel{Ti}(lvl_2, lvl.shape, ptr_2, idx_2, ofs_2)
end

Base.summary(lvl::SparseBlockListLevel) = "SparseBlockList($(summary(lvl.lvl)))"
function similar_level(lvl::SparseBlockListLevel, fill_value, eltype::Type, dim, tail...)
    SparseBlockList(similar_level(lvl.lvl, fill_value, eltype, tail...), dim)
end

function pattern!(lvl::SparseBlockListLevel{Ti}) where {Ti}
    SparseBlockListLevel{Ti}(pattern!(lvl.lvl), lvl.shape, lvl.ptr, lvl.idx, lvl.ofs)
end

function countstored_level(lvl::SparseBlockListLevel, pos)
    countstored_level(lvl.lvl, lvl.ofs[lvl.ptr[pos + 1]] - 1)
end

function set_fill_value!(lvl::SparseBlockListLevel{Ti}, init) where {Ti}
    SparseBlockListLevel{Ti}(
        set_fill_value!(lvl.lvl, init), lvl.shape, lvl.ptr, lvl.idx, lvl.ofs
    )
end

function Base.resize!(lvl::SparseBlockListLevel{Ti}, dims...) where {Ti}
    SparseBlockListLevel{Ti}(
        resize!(lvl.lvl, dims[1:(end - 1)]...), dims[end], lvl.ptr, lvl.idx, lvl.ofs
    )
end

function Base.show(
    io::IO, lvl::SparseBlockListLevel{Ti,Ptr,Idx,Ofs,Lvl}
) where {Ti,Ptr,Idx,Ofs,Lvl}
    if get(io, :compact, false)
        print(io, "SparseBlockList(")
    else
        print(io, "SparseBlockList{$Ti}(")
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
        show(io, lvl.ofs)
    end
    print(io, ")")
end

function labelled_show(io::IO, fbr::SubFiber{<:SparseBlockListLevel})
    print(
        io,
        "SparseBlockList (",
        fill_value(fbr),
        ") [",
        ":,"^(ndims(fbr) - 1),
        "1:",
        size(fbr)[end],
        "]",
    )
end

function labelled_children(fbr::SubFiber{<:SparseBlockListLevel})
    lvl = fbr.lvl
    pos = fbr.pos
    pos + 1 > length(lvl.ptr) && return []
    res = []
    for r in lvl.ptr[pos]:(lvl.ptr[pos + 1] - 1)
        i = lvl.idx[r]
        qos = lvl.ofs[r]
        l = lvl.ofs[r + 1] - lvl.ofs[r]
        for qos in lvl.ofs[r]:(lvl.ofs[r + 1] - 1)
            push!(
                res,
                LabelledTree(
                    cartesian_label(
                        [range_label() for _ in 1:(ndims(fbr) - 1)]...,
                        i - (lvl.ofs[r + 1] - 1) + qos,
                    ),
                    SubFiber(lvl.lvl, qos),
                ),
            )
        end
    end
    res
end

@inline level_ndims(
    ::Type{<:SparseBlockListLevel{Ti,Ptr,Idx,Ofs,Lvl}}
) where {Ti,Ptr,Idx,Ofs,Lvl} = 1 + level_ndims(Lvl)
@inline level_size(lvl::SparseBlockListLevel) = (level_size(lvl.lvl)..., lvl.shape)
@inline level_axes(lvl::SparseBlockListLevel) =
    (level_axes(lvl.lvl)..., Base.OneTo(lvl.shape))
@inline level_eltype(
    ::Type{<:SparseBlockListLevel{Ti,Ptr,Idx,Ofs,Lvl}}
) where {Ti,Ptr,Idx,Ofs,Lvl} = level_eltype(Lvl)
@inline level_fill_value(
    ::Type{<:SparseBlockListLevel{Ti,Ptr,Idx,Ofs,Lvl}}
) where {Ti,Ptr,Idx,Ofs,Lvl} = level_fill_value(Lvl)
function data_rep_level(
    ::Type{<:SparseBlockListLevel{Ti,Ptr,Idx,Ofs,Lvl}}
) where {Ti,Ptr,Idx,Ofs,Lvl}
    SparseData(data_rep_level(Lvl))
end

function isstructequal(a::T, b::T) where {T<:SparseBlockList}
    a.shape == b.shape &&
        a.ptr == b.ptr &&
        a.idx == b.idx &&
        a.ofs == b.ofs &&
        isstructequal(a.lvl, b.lvl)
end

(fbr::AbstractFiber{<:SparseBlockListLevel})() = fbr
function (fbr::SubFiber{<:SparseBlockListLevel})(idxs...)
    isempty(idxs) && return fbr
    lvl = fbr.lvl
    p = fbr.pos
    r =
        lvl.ptr[p] +
        searchsortedfirst(@view(lvl.idx[lvl.ptr[p]:(lvl.ptr[p + 1] - 1)]), idxs[end]) - 1
    r < lvl.ptr[p + 1] || return fill_value(fbr)
    q = lvl.ofs[r + 1] - 1 - lvl.idx[r] + idxs[end]
    q >= lvl.ofs[r] || return fill_value(fbr)
    fbr_2 = SubFiber(lvl.lvl, q)
    return fbr_2(idxs[1:(end - 1)]...)
end

mutable struct VirtualSparseBlockListLevel <: AbstractVirtualLevel
    tag
    lvl
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

function reroot_set!(ctx::AbstractCompiler, lvl::VirtualSparseBlockListLevel, diff)
    diff[lvl.tag] = lvl
    reroot_set!(ctx, lvl.lvl, diff)
end

function reroot_get(ctx::AbstractCompiler, lvl::VirtualSparseBlockListLevel, diff)
    get(
        diff,
        lvl.tag,
        VirtualSparseBlockListLevel(
            lvl.tag,
            reroot_get(ctx, lvl.lvl, diff),
            lvl.Ti,
            lvl.qos_fill,
            lvl.qos_stop,
            lvl.ros_fill,
            lvl.ros_stop,
            lvl.dirty,
            lvl.ptr,
            lvl.idx,
            lvl.ofs,
            lvl.prev_pos,
        ),
    )
end

function is_level_injective(ctx, lvl::VirtualSparseBlockListLevel)
    [is_level_injective(ctx, lvl.lvl)..., false]
end
function is_level_atomic(ctx, lvl::VirtualSparseBlockListLevel)
    (below, atomic) = is_level_atomic(ctx, lvl.lvl)
    return ([below; [atomic]], atomic)
end
function is_level_concurrent(ctx, lvl::VirtualSparseBlockListLevel)
    (data, _) = is_level_concurrent(ctx, lvl.lvl)
    return ([data; [false]], false)
end
postype(lvl::VirtualSparseBlockListLevel) = postype(lvl.lvl)

function virtualize(
    ctx, ex, ::Type{SparseBlockListLevel{Ti,Ptr,Idx,Ofs,Lvl}}, tag=:lvl
) where {Ti,Ptr,Idx,Ofs,Lvl}
    tag = freshen(ctx, tag)
    qos_fill = freshen(ctx, tag, :_qos_fill)
    qos_stop = freshen(ctx, tag, :_qos_stop)
    ros_fill = freshen(ctx, tag, :_ros_fill)
    ros_stop = freshen(ctx, tag, :_ros_stop)
    dirty = freshen(ctx, tag, :_dirty)
    ptr = freshen(ctx, tag, :_ptr)
    idx = freshen(ctx, tag, :_idx)
    ofs = freshen(ctx, tag, :_ofs)
    stop = freshen(ctx, tag, :_stop)
    push_preamble!(
        ctx,
        quote
            $tag = $ex
            $ptr = $tag.ptr
            $idx = $tag.idx
            $ofs = $tag.ofs
            $stop = $tag.shape
        end,
    )
    shape = value(stop, Int)
    prev_pos = freshen(ctx, tag, :_prev_pos)
    lvl_2 = virtualize(ctx, :($tag.lvl), Lvl, tag)
    VirtualSparseBlockListLevel(
        tag,
        lvl_2,
        Ti,
        shape,
        qos_fill,
        qos_stop,
        ros_fill,
        ros_stop,
        dirty,
        ptr,
        idx,
        ofs,
        prev_pos,
    )
end
function lower(ctx::AbstractCompiler, lvl::VirtualSparseBlockListLevel, ::DefaultStyle)
    quote
        $SparseBlockListLevel{$(lvl.Ti)}(
            $(ctx(lvl.lvl)),
            $(ctx(lvl.shape)),
            $(lvl.ptr),
            $(lvl.idx),
            $(lvl.ofs),
        )
    end
end

Base.summary(lvl::VirtualSparseBlockListLevel) = "SparseBlockList($(summary(lvl.lvl)))"

function virtual_level_size(ctx, lvl::VirtualSparseBlockListLevel)
    ext = Extent(literal(lvl.Ti(1)), lvl.shape)
    (virtual_level_size(ctx, lvl.lvl)..., ext)
end

function virtual_level_resize!(ctx, lvl::VirtualSparseBlockListLevel, dims...)
    lvl.shape = getstop(dims[end])
    lvl.lvl = virtual_level_resize!(ctx, lvl.lvl, dims[1:(end - 1)]...)
    lvl
end

virtual_level_eltype(lvl::VirtualSparseBlockListLevel) = virtual_level_eltype(lvl.lvl)
function virtual_level_fill_value(lvl::VirtualSparseBlockListLevel)
    virtual_level_fill_value(lvl.lvl)
end

function distribute_level(
    ctx::AbstractCompiler, lvl::VirtualSparseBlockListLevel, arch, style
)
    ptr_2 = freshen(ctx, lvl.ptr)
    tbl_2 = freshen(ctx, lvl.tbl)
    ofs_2 = freshen(ctx, lvl.ofs)
    push_preamble!(
        ctx,
        quote
            $ptr_2 = $transfer($(lvl.ptr), $(ctx(arch)), $style)
            $tbl_2 = $transfer($(lvl.tbl), $(ctx(arch)), $style)
            $ofs_2 = $transfer($(lvl.ofs), $(ctx(arch)), $style)
        end,
    )
    lvl_2 = distribute_level(ctx, lvl.lvl, arch, style)
    VirtualSparseBlockListLevel(
        lvl.tag,
        lvl_2,
        lvl.Ti,
        lvl.shape,
        lvl.qos_fill,
        lvl.qos_stop,
        lvl.ros_fill,
        lvl.ros_stop,
        lvl.dirty,
        ptr_2,
        tbl_2,
        ofs_2,
        lvl.prev_pos,
    )
end

function declare_level!(ctx::AbstractCompiler, lvl::VirtualSparseBlockListLevel, pos, init)
    Tp = postype(lvl)
    Ti = lvl.Ti
    push_preamble!(
        ctx,
        quote
            $(lvl.qos_fill) = $(Tp(0))
            $(lvl.qos_stop) = $(Tp(0))
            $(lvl.ros_fill) = $(Tp(0))
            $(lvl.ros_stop) = $(Tp(0))
            Finch.resize_if_smaller!($(lvl.ofs), 1)
            $(lvl.ofs)[1] = 1
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

function assemble_level!(ctx, lvl::VirtualSparseBlockListLevel, pos_start, pos_stop)
    pos_start = ctx(cache!(ctx, :p_start, pos_start))
    pos_stop = ctx(cache!(ctx, :p_start, pos_stop))
    return quote
        Finch.resize_if_smaller!($(lvl.ptr), $pos_stop + 1)
        Finch.fill_range!($(lvl.ptr), 0, $pos_start + 1, $pos_stop + 1)
    end
end

function freeze_level!(ctx::AbstractCompiler, lvl::VirtualSparseBlockListLevel, pos_stop)
    p = freshen(ctx, :p)
    Tp = postype(lvl)
    pos_stop = ctx(cache!(ctx, :pos_stop, simplify(ctx, pos_stop)))
    ros_stop = freshen(ctx, :ros_stop)
    qos_stop = freshen(ctx, :qos_stop)
    push_preamble!(
        ctx,
        quote
            resize!($(lvl.ptr), $pos_stop + 1)
            for $p in 2:($pos_stop + 1)
                $(lvl.ptr)[$p] += $(lvl.ptr)[$p - 1]
            end
            $ros_stop = $(lvl.ptr)[$pos_stop + 1] - 1
            resize!($(lvl.idx), $ros_stop)
            resize!($(lvl.ofs), $ros_stop + 1)
            $qos_stop = $(lvl.ofs)[$ros_stop + 1] - $(Tp(1))
        end,
    )
    lvl.lvl = freeze_level!(ctx, lvl.lvl, value(qos_stop))
    return lvl
end

function unfurl(
    ctx,
    fbr::VirtualSubFiber{VirtualSparseBlockListLevel},
    ext,
    mode,
    ::Union{typeof(defaultread),typeof(walk)},
)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.tag
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

    Thunk(;
        preamble=quote
            $my_r = $(lvl.ptr)[$(ctx(pos))]
            $my_r_stop = $(lvl.ptr)[$(ctx(pos)) + $(Tp(1))]
            if $my_r < $my_r_stop
                $my_i = $(lvl.idx)[$my_r]
                $my_i1 = $(lvl.idx)[$my_r_stop - $(Tp(1))]
            else
                $my_i = $(Ti(1))
                $my_i1 = $(Ti(0))
            end
        end,
        body=(ctx) -> Sequence([
            Phase(;
                stop = (ctx, ext) -> value(my_i1),
                body = (ctx, ext) -> Stepper(;
                seek=(ctx, ext) -> quote
                    if $(lvl.idx)[$my_r] < $(ctx(getstart(ext)))
                        $my_r = Finch.scansearch($(lvl.idx), $(ctx(getstart(ext))), $my_r, $my_r_stop - 1)
                    end
                end,
                preamble=quote
                    $my_i = $(lvl.idx)[$my_r]
                    $my_q_stop = $(lvl.ofs)[$my_r + $(Tp(1))]
                    $my_i_start = $my_i - ($my_q_stop - $(lvl.ofs)[$my_r])
                    $my_q_ofs = $my_q_stop - $my_i - $(Tp(1))
                end,
                stop=(ctx, ext) -> value(my_i),
                body=(ctx, ext) -> Thunk(;
                body=(ctx) -> Sequence([
                Phase(;
                stop = (ctx, ext) -> value(my_i_start),
                body = (ctx, ext) -> Run(FillLeaf(virtual_level_fill_value(lvl)))
            ),
                Phase(;
                body=(ctx, ext) -> Lookup(;
                body=(ctx, i) -> Thunk(;
                preamble=:($my_q = $my_q_ofs + $(ctx(i))),
                body=(ctx) -> instantiate(ctx, VirtualSubFiber(lvl.lvl, value(my_q, Tp)), mode)
            )
            )
            )
            ]),
                epilogue=quote
                    $my_r += ($(ctx(getstop(ext))) == $my_i)
                end
            )
            ),
            ),
            Phase(;
                body=(ctx, ext) -> Run(FillLeaf(virtual_level_fill_value(lvl)))
            ),
        ]),
    )
end

function unfurl(
    ctx, fbr::VirtualSubFiber{VirtualSparseBlockListLevel}, ext, mode, ::typeof(gallop)
)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.tag
    Tp = postype(lvl)
    Ti = lvl.Ti
    my_i = freshen(ctx, tag, :_i)
    my_j = freshen(ctx, tag, :_j)
    my_i_start = freshen(ctx, tag, :_i)
    my_r = freshen(ctx, tag, :_r)
    my_r_stop = freshen(ctx, tag, :_r_stop)
    my_q = freshen(ctx, tag, :_q)
    my_q_stop = freshen(ctx, tag, :_q_stop)
    my_q_ofs = freshen(ctx, tag, :_q_ofs)
    my_i1 = freshen(ctx, tag, :_i1)

    Thunk(;
        preamble=quote
            $my_r = $(lvl.ptr)[$(ctx(pos))]
            $my_r_stop = $(lvl.ptr)[$(ctx(pos)) + $(Tp(1))]
            if $my_r < $my_r_stop
                $my_i = $(lvl.idx)[$my_r]
                $my_i1 = $(lvl.idx)[$my_r_stop - $(Tp(1))]
            else
                $my_i = $(Ti(1))
                $my_i1 = $(Ti(0))
            end
        end,
        body=(ctx) -> Sequence([
            Phase(;
                stop = (ctx, ext) -> value(my_i1),
                body = (ctx, ext) -> Jumper(;
                seek=(ctx, ext) -> quote
                    if $(lvl.idx)[$my_r] < $(ctx(getstart(ext)))
                        $my_r = Finch.scansearch($(lvl.idx), $(ctx(getstart(ext))), $my_r, $my_r_stop - 1)
                    end
                end,
                preamble=quote
                    $my_i = $(lvl.idx)[$my_r]
                    $my_q_stop = $(lvl.ofs)[$my_r + $(Tp(1))]
                    $my_i_start = $my_i - ($my_q_stop - $(lvl.ofs)[$my_r])
                    $my_q_ofs = $my_q_stop - $my_i - $(Tp(1))
                end,
                stop=(ctx, ext) -> value(my_i),
                chunk=Sequence([
                Phase(;
                stop = (ctx, ext) -> value(my_i_start),
                body = (ctx, ext) -> Run(FillLeaf(virtual_level_fill_value(lvl)))
            ),
                Phase(;
                body=(ctx, ext) -> Lookup(;
                body=(ctx, i) -> Thunk(;
                preamble=:($my_q = $my_q_ofs + $(ctx(i))),
                body=(ctx) -> instantiate(ctx, VirtualSubFiber(lvl.lvl, value(my_q, Tp)), mode)
            )
            )
            )
            ]),
                next=(ctx, ext) -> :($my_r += $(Tp(1)))
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
    fbr::VirtualSubFiber{VirtualSparseBlockListLevel},
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
    fbr::VirtualHollowSubFiber{VirtualSparseBlockListLevel},
    ext,
    mode,
    ::Union{typeof(defaultupdate),typeof(extrude)},
)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.tag
    Tp = postype(lvl)
    Ti = lvl.Ti
    my_p = freshen(ctx, tag, :_p)
    my_q = freshen(ctx, tag, :_q)
    my_i_prev = freshen(ctx, tag, :_i_prev)
    qos = freshen(ctx, tag, :_qos)
    ros = freshen(ctx, tag, :_ros)
    qos_fill = lvl.qos_fill
    qos_stop = lvl.qos_stop
    ros_fill = lvl.ros_fill
    ros_stop = lvl.ros_stop
    dirty = freshen(ctx, tag, :dirty)

    Thunk(;
        preamble = quote
            $ros = $ros_fill
            $qos = $qos_fill + 1
            $my_i_prev = $(Ti(-1))
            $(if issafe(get_mode_flag(ctx))
                quote
                    $(lvl.prev_pos) < $(ctx(pos)) || throw(FinchProtocolError("SparseBlockListLevels cannot be updated multiple times"))
                end
            end)
        end,
        body     = (ctx) -> Lookup(;
        body=(ctx, idx) -> Thunk(;
        preamble = quote
            if $qos > $qos_stop
                $qos_stop = max($qos_stop << 1, 1)
                $(contain(ctx_2 -> assemble_level!(ctx_2, lvl.lvl, value(qos, Tp), value(qos_stop, Tp)), ctx))
            end
            $dirty = false
        end,
        body     = (ctx) -> instantiate(ctx, VirtualHollowSubFiber(lvl.lvl, value(qos, Tp), dirty), mode),
        epilogue = quote
            if $dirty
                $(fbr.dirty) = true
                if $(ctx(idx)) > $my_i_prev + $(Ti(1))
                    $ros += $(Tp(1))
                    if $ros > $ros_stop
                        $ros_stop = max($ros_stop << 1, 1)
                        Finch.resize_if_smaller!($(lvl.idx), $ros_stop)
                        Finch.resize_if_smaller!($(lvl.ofs), $ros_stop + 1)
                    end
                end
                $(lvl.idx)[$ros] = $my_i_prev = $(ctx(idx))
                $(qos) += $(Tp(1))
                $(lvl.ofs)[$ros + 1] = $qos
                $(if issafe(get_mode_flag(ctx))
                    quote
                        $(lvl.prev_pos) = $(ctx(pos))
                    end
                end)
            end
        end
    )
    ),
        epilogue = quote
            $(lvl.ptr)[$(ctx(pos)) + 1] = $ros - $ros_fill
            $ros_fill = $ros
            $qos_fill = $qos - 1
        end,
    )
end
