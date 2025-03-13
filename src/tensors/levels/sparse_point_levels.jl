"""
    SparsePointLevel{[Ti=Int], [Idx]}(lvl, [dim])

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
   │  └─ [1]: 10.0
   ├─ [:, 2]: SparsePoint (0.0) [1:3]
   │  └─ [2]: 20.0
   └─ [:, 3]: SparsePoint (0.0) [1:3]
      └─ [3]: 30.0

julia> tensor_tree(Tensor(SparsePoint(Dense(Element(0.0))), [0 0 0; 0 0 30; 0 0 30]))
3×3-Tensor
└─ SparsePoint (0.0) [:,1:3]
   └─ [:, 3]: Dense [1:3]
      ├─ [1]: 0.0
      ├─ [2]: 30.0
      └─ [3]: 30.0

```
"""
struct SparsePointLevel{Ti,Idx,Lvl} <: AbstractLevel
    lvl::Lvl
    shape::Ti
    idx::Idx
end
const SparsePoint = SparsePointLevel
SparsePointLevel(lvl) = SparsePointLevel{Int}(lvl)
SparsePointLevel(lvl, shape::Ti) where {Ti} = SparsePointLevel{Ti}(lvl, shape)
SparsePointLevel{Ti}(lvl) where {Ti} = SparsePointLevel{Ti}(lvl, zero(Ti))
SparsePointLevel{Ti}(lvl, shape) where {Ti} = SparsePointLevel{Ti}(lvl, shape, Ti[])

function SparsePointLevel{Ti}(lvl::Lvl, shape, idx::Idx) where {Ti,Lvl,Idx}
    SparsePointLevel{Ti,Idx,Lvl}(lvl, shape, idx)
end

Base.summary(lvl::SparsePointLevel) = "SparsePoint($(summary(lvl.lvl)))"
function similar_level(lvl::SparsePointLevel, fill_value, eltype::Type, dim, tail...)
    SparsePoint(similar_level(lvl.lvl, fill_value, eltype, tail...), dim)
end

function postype(::Type{SparsePointLevel{Ti,Idx,Lvl}}) where {Ti,Idx,Lvl}
    return postype(Lvl)
end

function transfer(lvl::SparsePointLevel{Ti,Idx,Lvl}, Tm, style) where {Ti,Idx,Lvl}
    lvl_2 = transfer(Tm, lvl.lvl)
    idx_2 = transfer(Tm, lvl.idx)
    return SparsePointLevel{Ti}(lvl_2, lvl.shape, idx_2)
end

function countstored_level(lvl::SparsePointLevel, pos)
    countstored_level(lvl.lvl, pos)
end

function pattern!(lvl::SparsePointLevel{Ti}) where {Ti}
    SparsePointLevel{Ti}(pattern!(lvl.lvl), lvl.shape, lvl.idx)
end

function set_fill_value!(lvl::SparsePointLevel{Ti}, init) where {Ti}
    SparsePointLevel{Ti}(set_fill_value!(lvl.lvl, init), lvl.shape, lvl.idx)
end

function Base.resize!(lvl::SparsePointLevel{Ti}, dims...) where {Ti}
    SparsePointLevel{Ti}(resize!(lvl.lvl, dims[1:(end - 1)]...), dims[end], lvl.idx)
end

function Base.show(io::IO, lvl::SparsePointLevel{Ti,Idx,Lvl}) where {Ti,Lvl,Idx}
    if get(io, :compact, false)
        print(io, "SparsePoint(")
    else
        print(io, "SparsePoint{$Ti}(")
    end
    show(io, lvl.lvl)
    print(io, ", ")
    show(IOContext(io, :typeinfo => Ti), lvl.shape)
    print(io, ", ")
    if !get(io, :compact, false)
        show(io, lvl.idx)
    else
        print(io, "…")
    end
    print(io, ")")
end

function labelled_show(io::IO, fbr::SubFiber{<:SparsePointLevel})
    print(
        io,
        "SparsePoint (",
        fill_value(fbr),
        ") [",
        ":,"^(ndims(fbr) - 1),
        "1:",
        size(fbr)[end],
        "]",
    )
end

function labelled_children(fbr::SubFiber{<:SparsePointLevel})
    lvl = fbr.lvl
    pos = fbr.pos
    [
        LabelledTree(
            cartesian_label(
                [range_label() for _ in 1:(ndims(fbr) - 1)]..., max(lvl.idx[pos], 1)
            ),
            SubFiber(lvl.lvl, pos),
        ),
    ]
end

@inline level_ndims(::Type{<:SparsePointLevel{Ti,Idx,Lvl}}) where {Ti,Idx,Lvl} =
    1 + level_ndims(Lvl)
@inline level_size(lvl::SparsePointLevel) = (level_size(lvl.lvl)..., lvl.shape)
@inline level_axes(lvl::SparsePointLevel) = (level_axes(lvl.lvl)..., Base.OneTo(lvl.shape))
@inline level_eltype(::Type{<:SparsePointLevel{Ti,Idx,Lvl}}) where {Ti,Idx,Lvl} =
    level_eltype(Lvl)
@inline level_fill_value(::Type{<:SparsePointLevel{Ti,Idx,Lvl}}) where {Ti,Idx,Lvl} =
    level_fill_value(Lvl)
function data_rep_level(::Type{<:SparsePointLevel{Ti,Idx,Lvl}}) where {Ti,Idx,Lvl}
    SparseData(data_rep_level(Lvl))
end

function isstructequal(a::T, b::T) where {T<:SparsePoint}
    a.shape == b.shape &&
        a.idx == b.idx &&
        isstructequal(a.lvl, b.lvl)
end

(fbr::AbstractFiber{<:SparsePointLevel})() = fbr
function (fbr::SubFiber{<:SparsePointLevel{Ti}})(idxs...) where {Ti}
    isempty(idxs) && return fbr
    lvl = fbr.lvl
    if idxs[end] == lvl.idx[fbr.pos]
        return SubFiber(lvl.lvl, fbr.pos)(idxs[1:(end - 1)]...)
    else
        fill_value(fbr)
    end
end

mutable struct VirtualSparsePointLevel <: AbstractVirtualLevel
    tag
    lvl
    Ti
    idx
    shape
end

function reroot_set!(ctx::AbstractCompiler, lvl::VirtualSparsePointLevel, diff)
    diff[lvl.tag] = lvl
    reroot_set!(ctx, lvl.lvl, diff)
end

function reroot_get(ctx::AbstractCompiler, lvl::VirtualSparsePointLevel, diff)
    get(
        diff,
        lvl.tag,
        VirtualSparsePointLevel(
            lvl.tag,
            reroot_get(ctx, lvl.lvl, diff),
            lvl.Ti,
            lvl.idx,
            lvl.shape,
        ),
    )
end

function is_level_injective(ctx, lvl::VirtualSparsePointLevel)
    [is_level_injective(ctx, lvl.lvl)..., false]
end

function is_level_atomic(ctx, lvl::VirtualSparsePointLevel)
    (below, atomic) = is_level_atomic(ctx, lvl.lvl)
    return ([below; [atomic]], atomic)
end
function is_level_concurrent(ctx, lvl::VirtualSparsePointLevel)
    (data, _) = is_level_concurrent(ctx, lvl.lvl)
    return ([data; [false]], false)
end

function virtualize(
    ctx, ex, ::Type{SparsePointLevel{Ti,Idx,Lvl}}, tag=:lvl
) where {Ti,Idx,Lvl}
    tag = freshen(ctx, tag)
    idx = freshen(ctx, tag, :_idx)
    stop = freshen(ctx, tag, :_stop)
    push_preamble!(
        ctx,
        quote
            $tag = $ex
            $idx = $tag.idx
            $stop = $tag.shape
        end,
    )
    shape = value(stop, Int)
    lvl_2 = virtualize(ctx, :($tag.lvl), Lvl, tag)
    VirtualSparsePointLevel(tag, lvl_2, Ti, idx, shape)
end
function lower(ctx::AbstractCompiler, lvl::VirtualSparsePointLevel, ::DefaultStyle)
    quote
        $SparsePointLevel{$(lvl.Ti)}(
            $(ctx(lvl.lvl)),
            $(ctx(lvl.shape)),
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
    lvl.lvl = virtual_level_resize!(ctx, lvl.lvl, dims[1:(end - 1)]...)
    lvl
end

virtual_level_eltype(lvl::VirtualSparsePointLevel) = virtual_level_eltype(lvl.lvl)
virtual_level_fill_value(lvl::VirtualSparsePointLevel) = virtual_level_fill_value(lvl.lvl)

postype(lvl::VirtualSparsePointLevel) = postype(lvl.lvl)

function declare_level!(ctx::AbstractCompiler, lvl::VirtualSparsePointLevel, pos, init)
    #TODO check that init == fill_value
    Ti = lvl.Ti
    Tp = postype(lvl)
    lvl.lvl = declare_level!(ctx, lvl.lvl, literal(Tp(0)), init)
    return lvl
end

function assemble_level!(ctx, lvl::VirtualSparsePointLevel, pos_start, pos_stop)
    pos_start = cache!(ctx, :p_start, pos_start)
    pos_stop = cache!(ctx, :p_start, pos_stop)
    Ti = lvl.Ti
    return quote
        Finch.resize_if_smaller!($(lvl.idx), $(ctx(pos_stop)))
        Finch.fill_range!($(lvl.idx), $(Ti(0)), $(ctx(pos_start)), $(ctx(pos_stop)))
        $(assemble_level!(ctx, lvl.lvl, pos_start, pos_stop))
    end
end

function freeze_level!(ctx::AbstractCompiler, lvl::VirtualSparsePointLevel, pos_stop)
    p = freshen(ctx, :p)
    pos_stop = ctx(cache!(ctx, :pos_stop, simplify(ctx, pos_stop)))
    push_preamble!(
        ctx,
        quote
            resize!($(lvl.idx), $pos_stop)
        end,
    )
    lvl.lvl = freeze_level!(ctx, lvl.lvl, value(pos_stop))
    return lvl
end

function thaw_level!(ctx::AbstractCompiler, lvl::VirtualSparsePointLevel, pos_stop)
    p = freshen(ctx, :p)
    pos_stop = ctx(cache!(ctx, :pos_stop, simplify(ctx, pos_stop)))
    lvl.lvl = thaw_level!(ctx, lvl.lvl, value(pos_stop))
    return lvl
end

function distribute_level(
    ctx::AbstractCompiler, lvl::VirtualSparsePointLevel, arch, style
)
    ptr_2 = freshen(ctx, lvl.ptr)
    idx_2 = freshen(ctx, lvl.idx)
    push_preamble!(
        ctx,
        quote
            $idx_2 = $transfer($(ctx(arch)), $(lvl.idx))
        end,
    )
    lvl_2 = distribute_level(ctx, lvl.lvl, arch, style)
    return VirtualSparsePointLevel(lvl.tag, lvl_2, lvl.Ti, idx_2, lvl.shape)
end

function unfurl(
    ctx,
    fbr::VirtualSubFiber{VirtualSparsePointLevel},
    ext,
    mode,
    ::Union{typeof(defaultread),typeof(walk)},
)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.tag
    Tp = postype(lvl)
    Ti = lvl.Ti
    my_i = freshen(ctx, tag, :_i)
    pos = cache!(ctx, :pos, simplify(ctx, pos))

    Thunk(;
        preamble=quote
            $my_i = max($(lvl.idx)[$(ctx(pos))], $(Ti(1)))
        end,
        body=(ctx) -> Sequence([
            Phase(;
                start=(ctx, ext) -> literal(lvl.Ti(1)),
                stop=(ctx, ext) -> value(my_i),
                body=(ctx, ext) -> truncate(
                    ctx,
                    Spike(;
                        body = FillLeaf(virtual_level_fill_value(lvl)),
                        tail = instantiate(ctx, VirtualSubFiber(lvl.lvl, pos), mode),
                    ),
                    similar_extent(ext, getstart(ext), value(my_i)),
                    ext,
                ),
            ),
            Phase(;
                stop = (ctx, ext) -> lvl.shape,
                body = (ctx, ext) -> Run(FillLeaf(virtual_level_fill_value(lvl))),
            ),
        ]),
    )
end

function unfurl(
    ctx,
    fbr::VirtualSubFiber{VirtualSparsePointLevel},
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
    fbr::VirtualHollowSubFiber{VirtualSparsePointLevel},
    ext,
    mode,
    ::Union{typeof(defaultupdate),typeof(extrude)},
)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.tag
    dirty = freshen(ctx, tag, :dirty)
    Tp = postype(lvl)
    pos = cache!(ctx, :pos, simplify(ctx, pos))

    Lookup(;
        body=(ctx, idx) -> Thunk(;
            preamble = quote
                $dirty = false
            end,
            body     = (ctx) -> instantiate(ctx, VirtualHollowSubFiber(lvl.lvl, pos, dirty), mode),
            epilogue = quote
                if $dirty
                    $(fbr.dirty) = true
                    if $(issafe(get_mode_flag(ctx)))
                        @assert $(lvl.idx)[$(ctx(pos))] == 0 || $(lvl.idx)[$(ctx(pos))] == $(ctx(idx))
                    end
                    $(lvl.idx)[$(ctx(pos))] = $(ctx(idx))
                end
            end,
        ),
    )
end
