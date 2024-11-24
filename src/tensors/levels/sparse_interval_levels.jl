"""
    SparseIntervalLevel{[Ti=Int], [Ptr, Left, Right]}(lvl, [dim])

The SparseIntervalLevel represent runs of equivalent slices `A[:, ..., :, i]`
which are not entirely [`fill_value`](@ref). A main difference compared to SparseRunList
level is that SparseInterval level only stores a 'single' non-fill run. It emits
an error if the program tries to write multiple (>=2) runs into SparseInterval.

`Ti` is the type of the last tensor index. The types `Ptr`, `Left`, and 'Right'
are the types of the arrays used to store positions and endpoints.

```jldoctest
julia> tensor_tree(Tensor(SparseInterval(Element(0)), [0, 10, 0]))
3-Tensor
└─ SparseInterval (0) [1:3]
   └─ [2:2]: 10

julia> x = Tensor(SparseInterval(Element(0)), 10);

julia> @finch begin for i = extent(3,6); x[~i] = 1 end end;

julia> tensor_tree(x)
10-Tensor
└─ SparseInterval (0) [1:10]
   └─ [3:6]: 1

```
"""
struct SparseIntervalLevel{Ti, Ptr<:AbstractVector, Left<:AbstractVector, Right<:AbstractVector, Lvl} <: AbstractLevel
    lvl::Lvl
    shape::Ti
    ptr::Ptr
    left::Left
    right::Right
end

const SparseInterval = SparseIntervalLevel
SparseIntervalLevel(lvl::Lvl) where {Lvl} = SparseIntervalLevel{Int}(lvl)
SparseIntervalLevel(lvl, shape::Ti, args...) where {Ti} = SparseIntervalLevel{Ti}(lvl, shape, args...)
SparseIntervalLevel{Ti}(lvl) where {Ti} = SparseIntervalLevel{Ti}(lvl, zero(Ti))
SparseIntervalLevel{Ti}(lvl, shape) where {Ti} = SparseIntervalLevel{Ti}(lvl, shape, postype(lvl)[1], Ti[], Ti[])

SparseIntervalLevel{Ti}(lvl::Lvl, shape, ptr::Ptr, left::Left, right::Right) where {Ti, Lvl, Ptr, Left, Right} =
    SparseIntervalLevel{Ti, Ptr, Left, Right, Lvl}(lvl, shape, ptr, left, right)

Base.summary(lvl::SparseIntervalLevel) = "SparseInterval($(summary(lvl.lvl)))"
similar_level(lvl::SparseIntervalLevel, fill_value, eltype::Type, dim, tail...) =
    SparseInterval(similar_level(lvl.lvl, fill_value, eltype, tail...), dim)

function memtype(::Type{SparseIntervalLevel{Ti, Ptr, Left, Right, Lvl}}) where {Ti, Ptr, Left, Right, Lvl}
    return Ti
end

function postype(::Type{SparseIntervalLevel{Ti, Ptr, Left, Right, Lvl}}) where {Ti, Ptr, Left, Right, Lvl}
    return postype(Lvl)
end

function moveto(lvl::SparseIntervalLevel{Ti, Ptr, Left, Right, Lvl}, Tm) where {Ti, Ptr, Left, Right, Lvl}
    lvl_2 = moveto(lvl.lvl, Tm)
    ptr_2 = moveto(lvl.ptr, Tm)
    left_2 = moveto(lvl.left, Tm)
    right_2 = moveto(lvl.right, Tm)
    return SparseIntervalLevel{Ti}(lvl_2, lvl.shape, ptr_2, left_2, right_2)
end

function countstored_level(lvl::SparseIntervalLevel, pos)
    countstored_level(lvl.lvl, lvl.ptr[pos + 1] - 1)
end

pattern!(lvl::SparseIntervalLevel{Ti}) where {Ti} =
    SparseIntervalLevel{Ti}(pattern!(lvl.lvl), lvl.shape, lvl.ptr, lvl.left, lvl.right)

set_fill_value!(lvl::SparseIntervalLevel{Ti}, init) where {Ti} =
    SparseIntervalLevel{Ti}(set_fill_value!(lvl.lvl, init), lvl.shape, lvl.ptr, lvl.left, lvl.right)

Base.resize!(lvl::SparseIntervalLevel{Ti}, dims...) where {Ti} =
    SparseIntervalLevel{Ti}(resize!(lvl.lvl, dims[1:end-1]...), dims[end], lvl.ptr, lvl.left, lvl.right)

function Base.show(io::IO, lvl::SparseIntervalLevel{Ti, Ptr, Left, Right, Lvl}) where {Ti, Lvl, Left, Right, Ptr}
    if get(io, :compact, false)
        print(io, "SparseInterval(")
    else
        print(io, "SparseInterval{$Ti}(")
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
        show(io, lvl.left)
        print(io, ", ")
        show(io, lvl.right)
    end
    print(io, ")")
end

labelled_show(io::IO, fbr::SubFiber{<:SparseIntervalLevel}) =
    print(io, "SparseInterval (", fill_value(fbr), ") [", ":,"^(ndims(fbr) - 1), "1:", size(fbr)[end], "]")

function labelled_children(fbr::SubFiber{<:SparseIntervalLevel})
    lvl = fbr.lvl
    pos = fbr.pos
    pos + 1 > length(lvl.ptr) && return []
    map(lvl.ptr[pos]:lvl.ptr[pos + 1] - 1) do qos
        LabelledTree(cartesian_label([range_label() for _ = 1:ndims(fbr) - 1]..., range_label(lvl.left[qos], lvl.right[qos])), SubFiber(lvl.lvl, qos))
    end
end

@inline level_ndims(::Type{<:SparseIntervalLevel{Ti, Ptr, Left, Right, Lvl}}) where {Ti, Ptr, Left, Right, Lvl} = 1 + level_ndims(Lvl)
@inline level_size(lvl::SparseIntervalLevel) = (level_size(lvl.lvl)..., lvl.shape)
@inline level_axes(lvl::SparseIntervalLevel) = (level_axes(lvl.lvl)..., Base.OneTo(lvl.shape))
@inline level_eltype(::Type{<:SparseIntervalLevel{Ti, Ptr, Left, Right, Lvl}}) where {Ti, Ptr, Left, Right, Lvl} = level_eltype(Lvl)
@inline level_fill_value(::Type{<:SparseIntervalLevel{Ti, Ptr, Left, Right, Lvl}}) where {Ti, Ptr, Left, Right, Lvl}= level_fill_value(Lvl)
data_rep_level(::Type{<:SparseIntervalLevel{Ti, Ptr, Left, Right, Lvl}}) where {Ti, Ptr, Left, Right, Lvl} = SparseData(data_rep_level(Lvl))

(fbr::AbstractFiber{<:SparseIntervalLevel})() = fbr
function (fbr::SubFiber{<:SparseIntervalLevel})(idxs...)
    isempty(idxs) && return fbr
    lvl = fbr.lvl
    p = fbr.pos
    r1 = searchsortedlast(@view(lvl.left[lvl.ptr[p]:lvl.ptr[p + 1] - 1]), idxs[end])
    r2 = searchsortedfirst(@view(lvl.right[lvl.ptr[p]:lvl.ptr[p + 1] - 1]), idxs[end])
    q = lvl.ptr[p] + first(r1) - 1
    fbr_2 = SubFiber(lvl.lvl, q)
    r1 != r2 ? fill_value(fbr_2) : fbr_2(idxs[1:end-1]...)
end


mutable struct VirtualSparseIntervalLevel <: AbstractVirtualLevel
    lvl
    ex
    Ti
    ptr
    left
    right
    shape
    qos_fill
    qos_stop
    prev_pos
end

is_level_injective(ctx, lvl::VirtualSparseIntervalLevel) = [false, is_level_injective(ctx, lvl.lvl)...]
function is_level_atomic(ctx, lvl::VirtualSparseIntervalLevel)
    (below, atomic) = is_level_atomic(ctx, lvl.lvl)
    return ([below; [atomic]], atomic)
end
function is_level_concurrent(ctx, lvl::VirtualSparseIntervalLevel)
    (data, concurrent) = is_level_concurrent(ctx, lvl.lvl)
    return ([data; [false]], false)
end

function virtualize(ctx, ex, ::Type{SparseIntervalLevel{Ti, Ptr, Left, Right, Lvl}}, tag=:lvl) where {Ti, Ptr, Left, Right, Lvl}
    sym = freshen(ctx, tag)
    ptr = freshen(ctx, tag, :_ptr)
    left = freshen(ctx, tag, :_left)
    right = freshen(ctx, tag, :_right)
    push_preamble!(ctx, quote
        $sym = $ex
        $ptr = $sym.ptr
        $left = $sym.left
        $right = $sym.right
    end)
    lvl_2 = virtualize(ctx, :($sym.lvl), Lvl, sym)
    shape = value(:($sym.shape), Int)
    qos_fill = freshen(ctx, sym, :_qos_fill)
    qos_stop = freshen(ctx, sym, :_qos_stop)
    prev_pos = freshen(ctx, sym, :_prev_pos)
    VirtualSparseIntervalLevel(lvl_2, sym, Ti, ptr, left, right, shape, qos_fill, qos_stop, prev_pos)
end
function lower(ctx::AbstractCompiler, lvl::VirtualSparseIntervalLevel, ::DefaultStyle)
    quote
        $SparseIntervalLevel{$(lvl.Ti)}(
            $(ctx(lvl.lvl)),
            $(ctx(lvl.shape)),
            $(lvl.ex).ptr,
            $(lvl.ex).left,
            $(lvl.ex).right,
        )
    end
end

Base.summary(lvl::VirtualSparseIntervalLevel) = "SparseInterval($(summary(lvl.lvl)))"

function virtual_level_size(ctx, lvl::VirtualSparseIntervalLevel)
    ext = make_extent(lvl.Ti, literal(lvl.Ti(1)), lvl.shape)
    (virtual_level_size(ctx, lvl.lvl)..., ext)
end

function virtual_level_resize!(ctx, lvl::VirtualSparseIntervalLevel, dims...)
    lvl.shape = getstop(dims[end])
    lvl.lvl = virtual_level_resize!(ctx, lvl.lvl, dims[1:end-1]...)
    lvl
end


virtual_level_eltype(lvl::VirtualSparseIntervalLevel) = virtual_level_eltype(lvl.lvl)
virtual_level_fill_value(lvl::VirtualSparseIntervalLevel) = virtual_level_fill_value(lvl.lvl)
postype(lvl::VirtualSparseIntervalLevel) = postype(lvl.lvl)

function declare_level!(ctx::AbstractCompiler, lvl::VirtualSparseIntervalLevel, pos, init)
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

function assemble_level!(ctx, lvl::VirtualSparseIntervalLevel, pos_start, pos_stop)
    pos_start = ctx(cache!(ctx, :p_start, pos_start))
    pos_stop = ctx(cache!(ctx, :p_start, pos_stop))
    return quote
        Finch.resize_if_smaller!($(lvl.ptr), $pos_stop + 1)
        Finch.fill_range!($(lvl.ptr), 0, $pos_start + 1, $pos_stop + 1)
    end
end

function freeze_level!(ctx::AbstractCompiler, lvl::VirtualSparseIntervalLevel, pos_stop)
    p = freshen(ctx, :p)
    pos_stop = ctx(cache!(ctx, :pos_stop, simplify(ctx, pos_stop)))
    qos_stop = freshen(ctx, :qos_stop)
    push_preamble!(ctx, quote
        resize!($(lvl.ptr), $pos_stop + 1)
        for $p = 1:($pos_stop)
           $(lvl.ptr)[$p + 1] += $(lvl.ptr)[$p]
        end
        $qos_stop = $(lvl.ptr)[$pos_stop + 1] - 1
        resize!($(lvl.left), $qos_stop)
        resize!($(lvl.right), $qos_stop)
    end)
    lvl.lvl = freeze_level!(ctx, lvl.lvl, value(qos_stop))
    return lvl
end

function thaw_level!(ctx::AbstractCompiler, lvl::VirtualSparseIntervalLevel, pos_stop)
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

function unfurl(ctx, fbr::VirtualSubFiber{VirtualSparseIntervalLevel}, ext, mode, ::Union{typeof(defaultread), typeof(walk)})
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    Tp = postype(lvl)
    Ti = lvl.Ti
    my_i_end = freshen(ctx, tag, :_i_end)
    my_i_stop = freshen(ctx, tag, :_i_stop)
    my_i_start = freshen(ctx, tag, :_i_start)
    my_q = freshen(ctx, tag, :_q)
    my_q_stop = freshen(ctx, tag, :_q_stop)

    Thunk(
        preamble = quote
            $my_q = $(lvl.ptr)[$(ctx(pos))]
            $my_q_stop = $(lvl.ptr)[$(ctx(pos)) + $(Tp(1))]
            if $my_q < $my_q_stop
                $my_i_start = $(lvl.left)[$my_q]
                $my_i_stop = $(lvl.right)[$my_q]
            else
                $my_i_start= $(Ti(1))
                $my_i_stop= $(Ti(0))
            end
        end,
        body = (ctx) -> Sequence([
            Phase(
                start = (ctx, ext) -> literal(lvl.Ti(1)),
                stop = (ctx, ext) -> call(-, value(my_i_start, lvl.Ti), getunit(ext)),
                body = (ctx, ext) -> Run(FillLeaf(virtual_level_fill_value(lvl))),
            ),
            Phase(
                stop = (ctx, ext) -> value(my_i_stop, lvl.Ti),
                body = (ctx, ext) -> Run(Simplify(instantiate(ctx, VirtualSubFiber(lvl.lvl, value(my_q)), mode))),
            ),
            Phase(
                stop = (ctx, ext) -> lvl.shape,
                body = (ctx, ext) -> Run(FillLeaf(virtual_level_fill_value(lvl)))
            )
        ])
    )
end

unfurl(ctx, fbr::VirtualSubFiber{VirtualSparseIntervalLevel}, ext, mode, proto::Union{typeof(defaultupdate), typeof(extrude)}) =
    unfurl(ctx, VirtualHollowSubFiber(fbr.lvl, fbr.pos, freshen(ctx, :null)), ext, mode, proto::Union{typeof(defaultupdate), typeof(extrude)})

function unfurl(ctx, fbr::VirtualHollowSubFiber{VirtualSparseIntervalLevel}, ext, mode, ::Union{typeof(defaultupdate), typeof(extrude)})
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    Tp = postype(lvl)
    Ti = lvl.Ti
    qos = freshen(ctx, tag, :_qos)
    qos_fill = lvl.qos_fill
    qos_stop = lvl.qos_stop
    dirty = freshen(ctx, tag, :dirty)

    Thunk(
        preamble = quote
            $qos = $qos_fill + 1
            $(lvl.ptr)[$(ctx(pos)) + 1] == 0 ||
                throw(FinchProtocolError("SparseIntervalLevels can only be updated once"))
        end,

        body = (ctx) -> AcceptRun(
            body = (ctx, ext) -> Thunk(
                preamble = quote
                    if $qos > $qos_stop
                        $qos_stop = max($qos_stop << 1, 1)
                        Finch.resize_if_smaller!($(lvl.left), $qos_stop)
                        Finch.resize_if_smaller!($(lvl.right), $qos_stop)
                        $(contain(ctx_2->assemble_level!(ctx_2, lvl.lvl, value(qos, Tp), value(qos_stop, Tp)), ctx))
                    end
                    $dirty = false
                end,
                body = (ctx) -> instantiate(ctx, VirtualHollowSubFiber(lvl.lvl, value(qos, Tp), dirty), mode),
                epilogue = quote
                    if $dirty
                        $(fbr.dirty) = true
                        $qos == $qos_fill + 1 || throw(FinchProtocolError("SparseIntervalLevels can only be updated once"))
                        $(lvl.left)[$qos] = $(ctx(getstart(ext)))
                        $(lvl.right)[$qos] = $(ctx(getstop(ext)))
                        $(qos) += $(Tp(1))
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
            $(lvl.ptr)[$(ctx(pos)) + 1] = $qos - $qos_fill - 1
            $qos_fill = $qos - 1
        end
    )
end
