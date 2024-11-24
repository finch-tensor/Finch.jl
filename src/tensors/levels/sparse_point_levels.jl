"""
    SparsePointLevel{[Ti=Int], [Ptr, Idx]}(lvl, [dim])

A subfiber of a SparsePoint level does not need to represent slices `A[:, ..., :, i]`
which are entirely [`fill_value`](@ref). Instead, only potentially non-fill
slices are stored as subfibers in `lvl`. A main difference compared to SparseList
level is that SparsePoint level only stores a 'single' non-fill slice. It emits
an error if the program tries to write multiple (>=2) coordinates into SparsePoint.

`Ti` is the type of the last tensor index. The types `Ptr` and `Idx` are the
types of the arrays used to store positions and indicies.

```jldoctest
julia> tensor_tree(Tensor(Dense(SparsePoint(Element(0.0))), [10 0 0; 0 20 0; 0 0 30]))
3×3-Tensor
└─ Dense [:,1:3]
   ├─ [:, 1]: SparsePoint (0.0) [1:3]
   │  └─ 10.0
   ├─ [:, 2]: SparsePoint (0.0) [1:3]
   │  └─ 20.0
   └─ [:, 3]: SparsePoint (0.0) [1:3]
      └─ 30.0

julia> tensor_tree(Tensor(SparsePoint(Dense(Element(0.0))), [0 0 0; 0 0 30; 0 0 30]))
3×3-Tensor
└─ SparsePoint (0.0) [:,1:3]
   └─ Dense [1:3]
      ├─ [1]: 0.0
      ├─ [2]: 30.0
      └─ [3]: 30.0

```
"""
struct SparsePointLevel{Ti, Ptr, Idx, Lvl} <: AbstractLevel
    lvl::Lvl
    shape::Ti
    ptr::Ptr
    idx::Idx
end
const SparsePoint = SparsePointLevel
SparsePointLevel(lvl) = SparsePointLevel{Int}(lvl)
SparsePointLevel(lvl, shape::Ti) where {Ti} = SparsePointLevel{Ti}(lvl, shape)
SparsePointLevel{Ti}(lvl) where {Ti} = SparsePointLevel{Ti}(lvl, zero(Ti))
SparsePointLevel{Ti}(lvl, shape) where {Ti} = SparsePointLevel{Ti}(lvl, shape, postype(lvl)[1], Ti[])

SparsePointLevel{Ti}(lvl::Lvl, shape, ptr::Ptr, idx::Idx) where {Ti, Lvl, Ptr, Idx} =
    SparsePointLevel{Ti, Ptr, Idx, Lvl}(lvl, shape, ptr, idx)

Base.summary(lvl::SparsePointLevel) = "SparsePoint($(summary(lvl.lvl)))"
similar_level(lvl::SparsePointLevel, fill_value, eltype::Type, dim, tail...) =
    SparsePoint(similar_level(lvl.lvl, fill_value, eltype, tail...), dim)

function postype(::Type{SparsePointLevel{Ti, Ptr, Idx, Lvl}}) where {Ti, Ptr, Idx, Lvl}
    return postype(Lvl)
end

function moveto(lvl::SparsePointLevel{Ti, Ptr, Idx, Lvl}, Tm) where {Ti, Ptr, Idx, Lvl}
    lvl_2 = moveto(lvl.lvl, Tm)
    ptr_2 = moveto(lvl.ptr, Tm)
    idx_2 = moveto(lvl.idx, Tm)
    return SparsePointLevel{Ti}(lvl_2, lvl.shape, ptr_2, idx_2)
end

function countstored_level(lvl::SparsePointLevel, pos)
    countstored_level(lvl.lvl, lvl.ptr[pos + 1] - 1)
end

pattern!(lvl::SparsePointLevel{Ti}) where {Ti} =
    SparsePointLevel{Ti}(pattern!(lvl.lvl), lvl.shape, lvl.ptr, lvl.idx)

set_fill_value!(lvl::SparsePointLevel{Ti}, init) where {Ti} =
    SparsePointLevel{Ti}(set_fill_value!(lvl.lvl, init), lvl.shape, lvl.ptr, lvl.idx)

Base.resize!(lvl::SparsePointLevel{Ti}, dims...) where {Ti} =
    SparsePointLevel{Ti}(resize!(lvl.lvl, dims[1:end-1]...), dims[end], lvl.ptr, lvl.idx)

function Base.show(io::IO, lvl::SparsePointLevel{Ti, Ptr, Idx, Lvl}) where {Ti, Lvl, Idx, Ptr}
    if get(io, :compact, false)
        print(io, "SparsePoint(")
    else
        print(io, "SparsePoint{$Ti}(")
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
    end
    print(io, ")")
end

labelled_show(io::IO, fbr::SubFiber{<:SparsePointLevel}) =
    print(io, "SparsePoint (", fill_value(fbr), ") [", ":,"^(ndims(fbr) - 1), "1:", size(fbr)[end], "]")

function labelled_children(fbr::SubFiber{<:SparsePointLevel})
    lvl = fbr.lvl
    pos = fbr.pos
    pos + 1 > length(lvl.ptr) && return []
    map(lvl.ptr[pos]:lvl.ptr[pos + 1] - 1) do qos
        cartesian_label([range_label() for _ = 1:ndims(fbr) - 1]..., lvl.idx[qos])
        LabelledTree(SubFiber(lvl.lvl, qos))
    end
end

@inline level_ndims(::Type{<:SparsePointLevel{Ti, Ptr, Idx, Lvl}}) where {Ti, Ptr, Idx, Lvl} = 1 + level_ndims(Lvl)
@inline level_size(lvl::SparsePointLevel) = (level_size(lvl.lvl)..., lvl.shape)
@inline level_axes(lvl::SparsePointLevel) = (level_axes(lvl.lvl)..., Base.OneTo(lvl.shape))
@inline level_eltype(::Type{<:SparsePointLevel{Ti, Ptr, Idx, Lvl}}) where {Ti, Ptr, Idx, Lvl} = level_eltype(Lvl)
@inline level_fill_value(::Type{<:SparsePointLevel{Ti, Ptr, Idx, Lvl}}) where {Ti, Ptr, Idx, Lvl} = level_fill_value(Lvl)
data_rep_level(::Type{<:SparsePointLevel{Ti, Ptr, Idx, Lvl}}) where {Ti, Ptr, Idx, Lvl} = SparseData(data_rep_level(Lvl))

(fbr::AbstractFiber{<:SparsePointLevel})() = fbr
function (fbr::SubFiber{<:SparsePointLevel{Ti}})(idxs...) where {Ti}
    isempty(idxs) && return fbr
    lvl = fbr.lvl
    p = fbr.pos
    r = searchsorted(@view(lvl.idx[lvl.ptr[p]:lvl.ptr[p + 1] - 1]), idxs[end])
    q = lvl.ptr[p] + first(r) - 1
    fbr_2 = SubFiber(lvl.lvl, q)
    length(r) == 0 ? fill_value(fbr_2) : fbr_2(idxs[1:end-1]...)
end

mutable struct VirtualSparsePointLevel <: AbstractVirtualLevel
    lvl
    ex
    Ti
    ptr
    idx
    shape
    qos_fill
    qos_stop
    prev_pos
end

is_level_injective(ctx, lvl::VirtualSparsePointLevel) = [is_level_injective(ctx, lvl.lvl)..., false]

function is_level_atomic(ctx, lvl::VirtualSparsePointLevel)
    (below, atomic) = is_level_atomic(ctx, lvl.lvl)
    return ([below; [atomic]], atomic)
end
function is_level_concurrent(ctx, lvl::VirtualSparsePointLevel)
    (data, _) = is_level_concurrent(ctx, lvl.lvl)
    return ([data; [false]], false)
end

function virtualize(ctx, ex, ::Type{SparsePointLevel{Ti, Ptr, Idx, Lvl}}, tag=:lvl) where {Ti, Ptr, Idx, Lvl}
    sym = freshen(ctx, tag)
    ptr = freshen(ctx, tag, :_ptr)
    idx = freshen(ctx, tag, :_idx)
    push_preamble!(ctx, quote
        $sym = $ex
        $ptr = $sym.ptr
        $idx = $sym.idx
    end)
    lvl_2 = virtualize(ctx, :($sym.lvl), Lvl, sym)
    shape = value(:($sym.shape), Int)
    qos_fill = freshen(ctx, sym, :_qos_fill)
    qos_stop = freshen(ctx, sym, :_qos_stop)
    prev_pos = freshen(ctx, sym, :_prev_pos)
    VirtualSparsePointLevel(lvl_2, sym, Ti, ptr, idx, shape, qos_fill, qos_stop, prev_pos)
end
function lower(ctx::AbstractCompiler, lvl::VirtualSparsePointLevel, ::DefaultStyle)
    quote
        $SparsePointLevel{$(lvl.Ti)}(
            $(ctx(lvl.lvl)),
            $(ctx(lvl.shape)),
            $(lvl.ptr),
            $(lvl.idx),
        )
    end
end

Base.summary(lvl::VirtualSparsePointLevel) = "SparsePoint($(summary(lvl.lvl)))"

function virtual_level_size(ctx, lvl::VirtualSparsePointLevel)
    ext = make_extent(lvl.Ti, literal(lvl.Ti(1)), lvl.shape)
    (virtual_level_size(ctx, lvl.lvl)..., ext)
end

function virtual_level_resize!(ctx, lvl::VirtualSparsePointLevel, dims...)
    lvl.shape = getstop(dims[end])
    lvl.lvl = virtual_level_resize!(ctx, lvl.lvl, dims[1:end-1]...)
    lvl
end

virtual_level_eltype(lvl::VirtualSparsePointLevel) = virtual_level_eltype(lvl.lvl)
virtual_level_fill_value(lvl::VirtualSparsePointLevel) = virtual_level_fill_value(lvl.lvl)

postype(lvl::VirtualSparsePointLevel) = postype(lvl.lvl)

function declare_level!(ctx::AbstractCompiler, lvl::VirtualSparsePointLevel, pos, init)
    #TODO check that init == fill_value
    Ti = lvl.Ti
    Tp = postype(lvl)
    push_preamble!(ctx, quote
        $(lvl.qos_fill) = $(Tp(0))
        $(lvl.qos_stop) = $(Tp(0))
    end)
    if issafe(get_mode_flag(ctx))
        push_preamble!(ctx, quote
            $(lvl.prev_pos) = $(Tp(0))
        end)
    end
    lvl.lvl = declare_level!(ctx, lvl.lvl, literal(Tp(0)), init)
    return lvl
end

function assemble_level!(ctx, lvl::VirtualSparsePointLevel, pos_start, pos_stop)
    pos_start = ctx(cache!(ctx, :p_start, pos_start))
    pos_stop = ctx(cache!(ctx, :p_start, pos_stop))
    return quote
        Finch.resize_if_smaller!($(lvl.ptr), $pos_stop + 1)
        Finch.fill_range!($(lvl.ptr), 0, $pos_start + 1, $pos_stop + 1)
    end
end

function freeze_level!(ctx::AbstractCompiler, lvl::VirtualSparsePointLevel, pos_stop)
    p = freshen(ctx, :p)
    pos_stop = ctx(cache!(ctx, :pos_stop, simplify(ctx, pos_stop)))
    qos_stop = freshen(ctx, :qos_stop)
    push_preamble!(ctx, quote
        resize!($(lvl.ptr), $pos_stop + 1)
        for $p = 1:$pos_stop
            $(lvl.ptr)[$p + 1] += $(lvl.ptr)[$p]
        end
        $qos_stop = $(lvl.ptr)[$pos_stop + 1] - 1
        resize!($(lvl.idx), $qos_stop)
    end)
    lvl.lvl = freeze_level!(ctx, lvl.lvl, value(qos_stop))
    return lvl
end

function thaw_level!(ctx::AbstractCompiler, lvl::VirtualSparsePointLevel, pos_stop)
    p = freshen(ctx, :p)
    pos_stop = ctx(cache!(ctx, :pos_stop, simplify(ctx, pos_stop)))
    qos_stop = freshen(ctx, :qos_stop)
    push_preamble!(ctx, quote
        $(lvl.qos_fill) = $(lvl.ptr)[$pos_stop + 1] - 1
        $(lvl.qos_stop) = $(lvl.qos_fill)
        $qos_stop = $(lvl.qos_fill)
        $(if issafe(get_mode_flag(ctx))
            quote
                $(lvl.prev_pos) = Finch.scansearch($(lvl.ptr), $(lvl.qos_stop) + 1, 1, $pos_stop) - 1
            end
        end)
        for $p = $pos_stop:-1:1
            $(lvl.ptr)[$p + 1] -= $(lvl.ptr)[$p]
        end
    end)
    lvl.lvl = thaw_level!(ctx, lvl.lvl, value(qos_stop))
    return lvl
end

function virtual_moveto_level(ctx::AbstractCompiler, lvl::VirtualSparsePointLevel, arch)
    ptr_2 = freshen(ctx, lvl.ptr)
    idx_2 = freshen(ctx, lvl.idx)
    push_preamble!(ctx, quote
        $ptr_2 = $(lvl.ptr)
        $idx_2 = $(lvl.idx)
        $(lvl.ptr) = $moveto($(lvl.ptr), $(ctx(arch)))
        $(lvl.idx) = $moveto($(lvl.idx), $(ctx(arch)))
    end)
    push_epilogue!(ctx, quote
        $(lvl.ptr) = $ptr_2
        $(lvl.idx) = $idx_2
    end)
    virtual_moveto_level(ctx, lvl.lvl, arch)
end

function unfurl(ctx, fbr::VirtualSubFiber{VirtualSparsePointLevel}, ext, mode, ::Union{typeof(defaultread), typeof(walk)})
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    Tp = postype(lvl)
    Ti = lvl.Ti
    my_i = freshen(ctx, tag, :_i)
    my_q = freshen(ctx, tag, :_q)
    my_q_stop = freshen(ctx, tag, :_q_stop)

    Thunk(
        preamble = quote
            $my_q = $(lvl.ptr)[$(ctx(pos))]
            $my_q_stop = $(lvl.ptr)[$(ctx(pos)) + $(Tp(1))]
            if $my_q < $my_q_stop
                $my_i = $(lvl.idx)[$my_q]
            else
                $my_i = $(Ti(0))
            end
        end,
        body = (ctx) -> Sequence([
            Phase(
                start = (ctx, ext) -> literal(lvl.Ti(1)),
                stop = (ctx, ext) -> value(my_i),
                body = (ctx, ext) -> truncate(
                    ctx,
                    Spike(
                        body = FillLeaf(virtual_level_fill_value(lvl)),
                        tail = instantiate(ctx, VirtualSubFiber(lvl.lvl, value(my_q, Ti)), mode)
                    ),
                    similar_extent(ext, getstart(ext), value(my_i)),
                    ext
                )
            ),
            Phase(
                stop = (ctx, ext) -> lvl.shape,
                body = (ctx, ext) -> Run(FillLeaf(virtual_level_fill_value(lvl)))
            )
        ])
    )
end

unfurl(ctx, fbr::VirtualSubFiber{VirtualSparsePointLevel}, ext, mode, proto::Union{typeof(defaultupdate), typeof(extrude)}) = 
    unfurl(ctx, VirtualHollowSubFiber(fbr.lvl, fbr.pos, freshen(ctx, :null)), ext, mode, proto)
function unfurl(ctx, fbr::VirtualHollowSubFiber{VirtualSparsePointLevel}, ext, mode, ::Union{typeof(defaultupdate), typeof(extrude)})
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    Tp = postype(lvl)
    qos = freshen(ctx, tag, :_qos)
    qos_fill = lvl.qos_fill
    qos_stop = lvl.qos_stop
    dirty = freshen(ctx, tag, :dirty)

    Thunk(
        preamble = quote
            $qos = $qos_fill + 1
            $(lvl.ptr)[$(ctx(pos)) + 1] == 0 || throw(FinchProtocolError("SparsePointLevels can only be updated once"))
            $(if issafe(get_mode_flag(ctx))
                quote
                    $(lvl.prev_pos) < $(ctx(pos)) || throw(FinchProtocolError("SparsePointLevels cannot be updated multiple times"))
                end
            end)
        end,
        body = (ctx) -> Lookup(
            body = (ctx, idx) -> Thunk(
                preamble = quote
                    if $qos > $qos_stop
                        $qos_stop = max($qos_stop << 1, 1)
                        Finch.resize_if_smaller!($(lvl.idx), $qos_stop)
                        $(contain(ctx_2->assemble_level!(ctx_2, lvl.lvl, value(qos, Tp), value(qos_stop, Tp)), ctx))
                    end
                    $dirty = false
                end,
                body = (ctx) -> instantiate(ctx, VirtualHollowSubFiber(lvl.lvl, value(qos, Tp), dirty), mode),
                epilogue = quote
                    if $dirty
                        $(fbr.dirty) = true
                        $qos == $qos_fill + 1 || throw(FinchProtocolError("SparsePointLevels can only be updated once"))
                        $(lvl.idx)[$qos] = $(ctx(idx))
                        $qos += $(Tp(1))
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
            $(lvl.ptr)[$(ctx(pos)) + 1] += $qos - $qos_fill - 1
            $qos_fill = $qos - 1
        end
    )
end
