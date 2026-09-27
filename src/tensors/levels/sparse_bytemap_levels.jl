"""
    SparseByteMapLevel{[Ti=Int], [Ptr, Tbl]}(lvl, [dims])

Like the [`SparseListLevel`](@ref), but a dense bitmap is used to encode
which slices are stored. This allows the ByteMap level to support random access.

`Ti` is the type of the last tensor index, and `Tp` is the type used for
positions in the level.

```jldoctest
julia> tensor_tree(Tensor(Dense(SparseByteMap(Element(0.0))), [10 0 20; 30 0 0; 0 0 40]))
3×3-Tensor
└─ Dense [:,1:3]
   ├─ [:, 1]: SparseByteMap (0.0) [1:3]
   │  ├─ [1]: 10.0
   │  └─ [2]: 30.0
   ├─ [:, 2]: SparseByteMap (0.0) [1:3]
   └─ [:, 3]: SparseByteMap (0.0) [1:3]
      ├─ [1]: 20.0
      └─ [3]: 40.0

julia> tensor_tree(Tensor(SparseByteMap(SparseByteMap(Element(0.0))), [10 0 20; 30 0 0; 0 0 40]))
3×3-Tensor
└─ SparseByteMap (0.0) [:,1:3]
   ├─ [:, 1]: SparseByteMap (0.0) [1:3]
   │  ├─ [1]: 10.0
   │  └─ [2]: 30.0
   └─ [:, 3]: SparseByteMap (0.0) [1:3]
      ├─ [1]: 20.0
      └─ [3]: 40.0
```
"""
struct SparseByteMapLevel{Ti,Ptr,Tbl,Srt,Lvl} <: AbstractLevel
    lvl::Lvl
    shape::Ti
    ptr::Ptr
    tbl::Tbl
    srt::Srt
end
const SparseByteMap = SparseByteMapLevel
SparseByteMapLevel(lvl::Lvl) where {Lvl} = SparseByteMapLevel{Int}(lvl)
function SparseByteMapLevel(lvl, shape, args...)
    SparseByteMapLevel{typeof(shape)}(lvl, shape, args...)
end
SparseByteMapLevel{Ti}(lvl) where {Ti} = SparseByteMapLevel{Ti}(lvl, zero(Ti))
function SparseByteMapLevel{Ti}(lvl, shape) where {Ti}
    SparseByteMapLevel{Ti}(lvl, shape, postype(lvl)[1], Bool[], postype(lvl)[])
end
function SparseByteMapLevel{Ti}(
    lvl::Lvl, shape, ptr::Ptr, tbl::Tbl, srt::Srt
) where {Ti,Lvl,Ptr,Tbl,Srt}
    SparseByteMapLevel{Ti,Ptr,Tbl,Srt,Lvl}(lvl, shape, ptr, tbl, srt)
end

Base.summary(lvl::SparseByteMapLevel) = "SparseByteMap($(summary(lvl.lvl)))"
function similar_level(lvl::SparseByteMapLevel, fill_value, eltype::Type, dims...)
    SparseByteMap(
        similar_level(lvl.lvl, fill_value, eltype, dims[1:(end - 1)]...), dims[end]
    )
end

function postype(::Type{SparseByteMapLevel{Ti,Ptr,Tbl,Srt,Lvl}}) where {Ti,Ptr,Tbl,Srt,Lvl}
    return postype(Lvl)
end

function transfer(device, lvl::SparseByteMapLevel{Ti}) where {Ti}
    lvl_2 = transfer(device, lvl.lvl)
    ptr_2 = transfer(device, lvl.ptr)
    tbl_2 = transfer(device, lvl.tbl)
    srt_2 = transfer(device, lvl.srt)
    return SparseByteMapLevel{Ti}(lvl_2, lvl.shape, ptr_2, tbl_2, srt_2)
end

function pattern!(lvl::SparseByteMapLevel{Ti}) where {Ti}
    SparseByteMapLevel{Ti}(pattern!(lvl.lvl), lvl.shape, lvl.ptr, lvl.tbl, lvl.srt)
end

function set_fill_value!(lvl::SparseByteMapLevel{Ti}, init) where {Ti}
    SparseByteMapLevel{Ti}(
        set_fill_value!(lvl.lvl, init), lvl.shape, lvl.ptr, lvl.tbl, lvl.srt
    )
end

function Base.resize!(lvl::SparseByteMapLevel{Ti}, dims...) where {Ti}
    SparseByteMapLevel{Ti}(
        resize!(lvl.lvl, dims[1:(end - 1)]...), dims[end], lvl.ptr, lvl.tbl, lvl.srt
    )
end

function countstored_level(lvl::SparseByteMapLevel, pos)
    countstored_level(lvl.lvl, pos * lvl.shape)
end

function countstored_level(lvl::SparseByteMapLevel, pos, idxs, dim, proc, exact)
    idx = exact ? idxs[length(idxs) - dim] : lvl.shape
    q = (pos - 1) * lvl.shape + idx
    countstored_level(lvl.lvl, q, idxs, dim + 1, proc, exact)
end

function Base.show(
    io::IO, lvl::SparseByteMapLevel{Ti,Ptr,Tbl,Srt,Lvl}
) where {Ti,Ptr,Tbl,Srt,Lvl}
    if get(io, :compact, false)
        print(io, "SparseByteMap(")
    else
        print(io, "SparseByteMap{$Ti}(")
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
        show(io, lvl.tbl)
        print(io, ", ")
        show(io, lvl.srt)
    end
    print(io, ")")
end

function labelled_show(io::IO, fbr::SubFiber{<:SparseByteMapLevel})
    print(
        io,
        "SparseByteMap (",
        fill_value(fbr),
        ") [",
        ":,"^(ndims(fbr) - 1),
        "1:",
        size(fbr)[end],
        "]",
    )
end

function labelled_children(fbr::SubFiber{<:SparseByteMapLevel})
    lvl = fbr.lvl
    pos = fbr.pos
    pos + 1 > length(lvl.ptr) && return []
    Tp = postype(lvl)
    q_offset = (Tp(pos) - one(Tp)) * Tp(lvl.shape)
    map(lvl.ptr[pos]:(lvl.ptr[pos + 1] - 1)) do qos
        srt_entry = lvl.srt[qos]
        LabelledTree(
            cartesian_label(
                [range_label() for _ in 1:(ndims(fbr) - 1)]...,
                srt_entry - q_offset,
            ),
            SubFiber(lvl.lvl, srt_entry),
        )
    end
end

@inline level_ndims(
    ::Type{<:SparseByteMapLevel{Ti,Ptr,Tbl,Srt,Lvl}}
) where {Ti,Ptr,Tbl,Srt,Lvl} = 1 + level_ndims(Lvl)
@inline level_size(lvl::SparseByteMapLevel) = (level_size(lvl.lvl)..., lvl.shape)
@inline level_axes(lvl::SparseByteMapLevel) = (
    level_axes(lvl.lvl)..., Base.OneTo(lvl.shape)
)
@inline level_eltype(
    ::Type{<:SparseByteMapLevel{Ti,Ptr,Tbl,Srt,Lvl}}
) where {Ti,Ptr,Tbl,Srt,Lvl} = level_eltype(Lvl)
@inline level_fill_value(
    ::Type{<:SparseByteMapLevel{Ti,Ptr,Tbl,Srt,Lvl}}
) where {Ti,Ptr,Tbl,Srt,Lvl} = level_fill_value(Lvl)
function data_rep_level(
    ::Type{<:SparseByteMapLevel{Ti,Ptr,Tbl,Srt,Lvl}}
) where {Ti,Ptr,Tbl,Srt,Lvl}
    SparseData(data_rep_level(Lvl))
end
function isstructequal(a::T, b::T) where {T<:SparseByteMap}
    a.shape == b.shape &&
        a.ptr == b.ptr &&
        a.tbl == b.tbl &&
        a.srt == b.srt &&
        isstructequal(a.lvl, b.lvl)
end

(fbr::AbstractFiber{<:SparseByteMapLevel})() = fbr
function (fbr::SubFiber{<:SparseByteMapLevel{Ti}})(idxs...) where {Ti}
    isempty(idxs) && return fbr
    lvl = fbr.lvl
    p = fbr.pos
    q = (p - 1) * lvl.shape + idxs[end]
    if lvl.tbl[q]
        fbr_2 = SubFiber(lvl.lvl, q)
        fbr_2(idxs[1:(end - 1)]...)
    else
        fill_value(fbr)
    end
end

mutable struct VirtualSparseByteMapLevel <: AbstractVirtualLevel
    tag
    lvl
    Ti
    ptr
    tbl
    srt
    shape
    qos_fill
    qos_stop
end

function is_level_injective(ctx, lvl::VirtualSparseByteMapLevel)
    [is_level_injective(ctx, lvl.lvl)..., false]
end
function is_level_atomic(ctx, lvl::VirtualSparseByteMapLevel)
    (below, atomic) = is_level_atomic(ctx, lvl.lvl)
    return ([below; [atomic]], atomic)
end
function is_level_concurrent(ctx, lvl::VirtualSparseByteMapLevel)
    (data, _) = is_level_concurrent(ctx, lvl.lvl)
    return ([data; [false]], false)
end

function virtualize(
    ctx, ex, ::Type{SparseByteMapLevel{Ti,Ptr,Tbl,Srt,Lvl}}, tag=:lvl
) where {Ti,Ptr,Tbl,Srt,Lvl}
    tag = freshen(ctx, tag)
    shape = value(:($tag.shape), Int)
    qos_fill = freshen(ctx, tag, :_qos_fill)
    qos_stop = freshen(ctx, tag, :_qos_stop)
    ptr = freshen(ctx, tag, :_ptr)
    tbl = freshen(ctx, tag, :_tbl)
    srt = freshen(ctx, tag, :_srt)
    stop = freshen(ctx, tag, :_stop)
    push_preamble!(
        ctx,
        quote
            $tag = $ex
            $ptr = $tag.ptr
            $tbl = $tag.tbl
            $srt = $tag.srt
            $qos_stop = $qos_fill = length($tag.srt)
            $stop = $tag.shape
        end,
    )
    shape = value(stop, Int)
    lvl_2 = virtualize(ctx, :($tag.lvl), Lvl, tag)
    VirtualSparseByteMapLevel(
        tag, lvl_2, Ti, ptr, tbl, srt, shape, qos_fill, qos_stop
    )
end
function lower(ctx::AbstractCompiler, lvl::VirtualSparseByteMapLevel, ::DefaultStyle)
    quote
        $SparseByteMapLevel{$(lvl.Ti)}(
            $(ctx(lvl.lvl)),
            $(ctx(lvl.shape)),
            $(lvl.ptr),
            $(lvl.tbl),
            $(lvl.srt),
        )
    end
end

function distribute_level(
    ctx::AbstractCompiler, lvl::VirtualSparseByteMapLevel, arch, diff, style
)
    diff[lvl.tag] = VirtualSparseByteMapLevel(
        lvl.tag,
        distribute_level(ctx, lvl.lvl, arch, diff, style),
        lvl.Ti,
        distribute_buffer(ctx, lvl.ptr, arch, style),
        distribute_buffer(ctx, lvl.tbl, arch, style),
        distribute_buffer(ctx, lvl.srt, arch, style),
        lvl.shape,
        distribute_buffer(ctx, lvl.qos_fill, arch, style),
        # lvl.qos_fill,
        # lvl.qos_stop,
        distribute_buffer(ctx, lvl.qos_stop, arch, style),
    )
end

function redistribute(ctx::AbstractCompiler, lvl::VirtualSparseByteMapLevel, diff)
    get(
        diff,
        lvl.tag,
        VirtualSparseByteMapLevel(
            lvl.tag,
            redistribute(ctx, lvl.lvl, diff),
            lvl.Ti,
            lvl.ptr,
            lvl.tbl,
            lvl.srt,
            lvl.shape,
            lvl.qos_fill,
            lvl.qos_stop,
        ),
    )
end

Base.summary(lvl::VirtualSparseByteMapLevel) = "SparseByteMap($(summary(lvl.lvl)))"

function virtual_level_size(ctx, lvl::VirtualSparseByteMapLevel)
    ext = VirtualExtent(literal(lvl.Ti(1)), lvl.shape)
    (virtual_level_size(ctx, lvl.lvl)..., ext)
end

function virtual_level_resize!(ctx, lvl::VirtualSparseByteMapLevel, dims...)
    lvl.shape = getstop(dims[end])
    lvl.lvl = virtual_level_resize!(ctx, lvl.lvl, dims[1:(end - 1)]...)
    lvl
end

virtual_level_eltype(lvl::VirtualSparseByteMapLevel) = virtual_level_eltype(lvl.lvl)
virtual_level_fill_value(lvl::VirtualSparseByteMapLevel) = virtual_level_fill_value(lvl.lvl)
@inline sample_dims(lvl::VirtualSparseByteMapLevel) = 1 + sample_dims(lvl.lvl)
@inline all_dense(lvl::VirtualSparseByteMapLevel) = true & all_dense(lvl.lvl)

postype(lvl::VirtualSparseByteMapLevel) = postype(lvl.lvl)

function sparse_bytemap_parent_position(
    ctx, lvl::VirtualSparseByteMapLevel, q, pos_stop, srt_shape
)
    Tp = postype(lvl)
    if prove(ctx, call(==, pos_stop, 1))
        return nothing, :($(Tp(1)))
    elseif prove(ctx, call(==, lvl.shape, 1))
        return nothing, q
    else
        return :($srt_shape = $(Tp)($(ctx(lvl.shape)))),
        :(
            fld($q - $(Tp(1)), $srt_shape) + $(Tp(1))
        )
    end
end

function declare_level!(ctx::AbstractCompiler, lvl::VirtualSparseByteMapLevel, pos, init)
    Ti = lvl.Ti
    Tp = postype(lvl)
    r = freshen(ctx, lvl.tag, :_r)
    p = freshen(ctx, lvl.tag, :_p)
    q = freshen(ctx, lvl.tag, :_q)
    srt_shape = freshen(ctx, lvl.tag, :_srt_shape)
    (srt_shape_init, parent_position) = sparse_bytemap_parent_position(
        ctx, lvl, q, pos, srt_shape
    )
    push_preamble!(
        ctx,
        quote
            $srt_shape_init
            for $r in 1:($(lvl.qos_fill))
                $q = $(lvl.srt)[$r]
                $p = $parent_position
                $(lvl.ptr)[$p] = $(Tp(0))
                $(lvl.ptr)[$p + 1] = $(Tp(0))
                $(lvl.tbl)[$q] = false
                if $(supports_reassembly(lvl.lvl))
                    $(contain(
                        ctx_2 ->
                            assemble_level!(ctx_2, lvl.lvl, value(q, Tp), value(q, Tp)),
                        ctx,
                    ))
                end
            end
            $(lvl.qos_fill) = 0
            if $(!supports_reassembly(lvl.lvl))
                $(lvl.qos_stop) = $(Tp(0))
            end
            $(lvl.ptr)[1] = 1
        end,
    )
    if !supports_reassembly(lvl.lvl)
        lvl.lvl = declare_level!(ctx, lvl.lvl, call(*, pos, lvl.shape), init)
        push_preamble!(
            ctx,
            contain(
                ctx_2 -> assemble_level!(
                    ctx_2, lvl.lvl, literal(Tp(1)), call(*, pos, lvl.shape)
                ),
                ctx,
            ),
        )
    end
    return lvl
end

function thaw_level!(ctx::AbstractCompiler, lvl::VirtualSparseByteMapLevel, pos)
    Ti = lvl.Ti
    Tp = postype(lvl)
    p = freshen(ctx, lvl.tag, :_p)
    lvl.lvl = thaw_level!(ctx, lvl.lvl, call(*, pos, lvl.shape))
    return lvl
end

function assemble_level!(ctx, lvl::VirtualSparseByteMapLevel, pos_start, pos_stop)
    Ti = lvl.Ti
    Tp = postype(lvl)
    pos_start = ctx(cache!(ctx, :p_start, pos_start))
    pos_stop = ctx(cache!(ctx, :p_start, pos_stop))
    q_start = freshen(ctx, lvl.tag, :q_start)
    q_stop = freshen(ctx, lvl.tag, :q_stop)
    q = freshen(ctx, lvl.tag, :q)
    old = freshen(ctx, lvl.tag, :old)

    quote
        $q_start = ($(ctx(pos_start)) - $(Tp(1))) * $(ctx(lvl.shape)) + $(Tp(1))
        $q_stop = $(ctx(pos_stop)) * $(ctx(lvl.shape))
        Finch.resize_if_smaller!($(lvl.ptr), $pos_stop + 1)
        Finch.fill_range!($(lvl.ptr), 0, $pos_start + 1, $pos_stop + 1)
        $old = $q_start
        Finch.resize_if_smaller!($(lvl.tbl), $q_stop)
        Finch.fill_range!($(lvl.tbl), false, $old, $q_stop)
        $(contain(
            ctx_2 -> assemble_level!(ctx_2, lvl.lvl, value(old, Tp), value(q_stop, Tp)), ctx
        ))
    end
end

function freeze_level!(ctx::AbstractCompiler, lvl::VirtualSparseByteMapLevel, pos_stop)
    r = freshen(ctx, lvl.tag, :_r)
    p = freshen(ctx, lvl.tag, :_p)
    p_prev = freshen(ctx, lvl.tag, :_p_prev)
    srt_shape = freshen(ctx, lvl.tag, :_srt_shape)
    pos_stop = cache!(ctx, :pos_stop, pos_stop)
    Ti = lvl.Ti
    Tp = postype(lvl)
    srt_entry = :($(lvl.srt)[$r])
    (srt_shape_init, parent_position) = sparse_bytemap_parent_position(
        ctx, lvl, srt_entry, pos_stop, srt_shape
    )
    push_preamble!(
        ctx,
        quote
            resize!($(lvl.ptr), $(ctx(pos_stop)) + 1)
            resize!($(lvl.tbl), $(ctx(pos_stop)) * $(ctx(lvl.shape)))
            resize!($(lvl.srt), $(lvl.qos_fill))
            sort!($(lvl.srt))
            $srt_shape_init
            $p_prev = $(Tp(0))
            for $r in 1:($(lvl.qos_fill))
                $p = $parent_position
                if $p != $p_prev
                    $(lvl.ptr)[$p_prev + 1] = $r
                    $(lvl.ptr)[$p] = $r
                end
                $p_prev = $p
            end
            $(lvl.ptr)[$p_prev + 1] = $(lvl.qos_fill) + 1
            $(lvl.qos_stop) = $(lvl.qos_fill)
        end,
    )
    lvl.lvl = freeze_level!(ctx, lvl.lvl, call(*, pos_stop, lvl.shape))
    return lvl
end

function unfurl(
    ctx,
    fbr::VirtualSubFiber{VirtualSparseByteMapLevel},
    ext,
    mode,
    ::Union{typeof(defaultread),typeof(walk)},
)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.tag
    Ti = lvl.Ti
    Tp = postype(lvl)
    my_i = freshen(ctx, tag, :_i)
    my_q = freshen(ctx, tag, :_q)
    my_q_offset = freshen(ctx, tag, :_q_offset)
    my_r = freshen(ctx, tag, :_r)
    my_r_stop = freshen(ctx, tag, :_r_stop)
    my_i_stop = freshen(ctx, tag, :_i_stop)

    Unfurled(;
        arr=fbr,
        body=Thunk(;
            preamble=quote
                $my_q_offset = ($(ctx(pos)) - $(Tp(1))) * $(Tp)($(ctx(lvl.shape)))
                $my_r = $(lvl.ptr)[$(ctx(pos))]
                $my_r_stop = $(lvl.ptr)[$(ctx(pos)) + 1]
                if $my_r != 0 && $my_r < $my_r_stop
                    $my_i = $(lvl.srt)[$my_r] - $my_q_offset
                    $my_i_stop = $(lvl.srt)[$my_r_stop - 1] - $my_q_offset
                else
                    $my_i = $(Tp(1))
                    $my_i_stop = $(Tp(0))
                end
            end,
            body=(ctx) -> Sequence([
                Phase(;
                    stop=(ctx, ext) -> value(my_i_stop),
                    body=(ctx, ext) -> Stepper(;
                        seek=(ctx, ext) -> quote
                            while $my_r + $(Tp(1)) < $my_r_stop &&
                                $(lvl.srt)[$my_r] <
                                $my_q_offset + $(Tp)($(ctx(getstart(ext))))
                                $my_r += $(Tp(1))
                            end
                        end,
                        preamble=:(
                            $my_i = $(lvl.srt)[$my_r] - $my_q_offset
                        ),
                        stop=(ctx, ext) -> value(my_i),
                        chunk=Spike(;
                            body=FillLeaf(virtual_level_fill_value(lvl)),
                            tail=Thunk(;
                                preamble=:($my_q = $my_q_offset + $my_i),
                                body=(ctx) -> instantiate(
                                    ctx,
                                    VirtualSubFiber(lvl.lvl, value(my_q, lvl.Ti)),
                                    mode,
                                ),
                            ),
                        ),
                        next=(ctx, ext) -> :($my_r += $(Tp(1))),
                    ),
                ),
                Phase(;
                    body=(ctx, ext) -> Run(FillLeaf(virtual_level_fill_value(lvl)))
                ),
            ]),
        ),
    )
end

function unfurl(
    ctx, fbr::VirtualSubFiber{VirtualSparseByteMapLevel}, ext, mode, ::typeof(gallop)
)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.tag
    Ti = lvl.Ti
    Tp = postype(lvl)
    my_i = freshen(ctx, tag, :_i)
    my_q = freshen(ctx, tag, :_q)
    my_q_offset = freshen(ctx, tag, :_q_offset)
    my_r = freshen(ctx, tag, :_r)
    my_r_stop = freshen(ctx, tag, :_r_stop)
    my_i_stop = freshen(ctx, tag, :_i_stop)
    my_j = freshen(ctx, tag, :_j)

    Unfurled(;
        arr=fbr,
        body=Thunk(;
            preamble=quote
                $my_q_offset = ($(ctx(pos)) - $(Tp(1))) * $(Tp)($(ctx(lvl.shape)))
                $my_r = $(lvl.ptr)[$(ctx(pos))]
                $my_r_stop = $(lvl.ptr)[$(ctx(pos)) + 1]
                if $my_r != 0 && $my_r < $my_r_stop
                    $my_i = $(lvl.srt)[$my_r] - $my_q_offset
                    $my_i_stop = $(lvl.srt)[$my_r_stop - 1] - $my_q_offset
                else
                    $my_i = $(Tp(1))
                    $my_i_stop = $(Tp(0))
                end
            end,
            body=(ctx) -> Sequence([
                Phase(;
                    stop=(ctx, ext) -> value(my_i_stop),
                    body=(ctx, ext) -> Jumper(;
                        seek=(ctx, ext) -> quote
                            while $my_r + $(Tp(1)) < $my_r_stop &&
                                $(lvl.srt)[$my_r] <
                                $my_q_offset + $(Tp)($(ctx(getstart(ext))))
                                $my_r += $(Tp(1))
                            end
                        end,
                        preamble=:(
                            $my_i = $(lvl.srt)[$my_r] - $my_q_offset
                        ),
                        stop=(ctx, ext) -> value(my_i),
                        chunk=Spike(;
                            body=FillLeaf(virtual_level_fill_value(lvl)),
                            tail=Thunk(;
                                preamble=:($my_q = $my_q_offset + $my_i),
                                body=(ctx) -> instantiate(
                                    ctx,
                                    VirtualSubFiber(lvl.lvl, value(my_q, lvl.Ti)),
                                    mode,
                                ),
                            ),
                        ),
                        next=(ctx, ext) -> :($my_r += $(Tp(1))),
                    ),
                ),
                Phase(;
                    body=(ctx, ext) -> Run(FillLeaf(virtual_level_fill_value(lvl)))
                ),
            ]),
        ),
    )
end

function unfurl(
    ctx, fbr::VirtualSubFiber{VirtualSparseByteMapLevel}, ext, mode, ::typeof(follow)
)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.tag
    my_q = freshen(ctx, tag, :_q)
    q = pos
    Ti = lvl.Ti

    Unfurled(;
        arr=fbr,
        body=Lookup(;
            body=(ctx, i) -> Thunk(;
                preamble=quote
                    $my_q = ($(ctx(q)) - $(Ti(1))) * $(ctx(lvl.shape)) + $(ctx(i))
                end,
                body=(ctx) -> Switch([
                    value(:($(lvl.tbl)[$my_q])) => instantiate(
                        ctx, VirtualSubFiber(lvl.lvl, value(my_q)), mode
                    ),
                    literal(true) => FillLeaf(virtual_level_fill_value(lvl)),
                ]),
            ),
        ),
    )
end

function unfurl(
    ctx,
    fbr::VirtualSubFiber{VirtualSparseByteMapLevel},
    ext,
    mode,
    proto::Union{typeof(defaultupdate),typeof(extrude),typeof(laminate)},
)
    unfurl(
        ctx, VirtualHollowSubFiber(fbr.lvl, fbr.pos, freshen(ctx, :null)), ext, mode, proto
    )
end
function unfurl(
    ctx,
    fbr::VirtualHollowSubFiber{VirtualSparseByteMapLevel},
    ext,
    mode,
    ::Union{typeof(defaultupdate),typeof(extrude),typeof(laminate)},
)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.tag
    Tp = postype(lvl)
    my_q = freshen(ctx, tag, :_q)
    dirty = freshen(ctx, :dirty)

    Unfurled(;
        arr=fbr,
        body=Lookup(;
            body=(ctx, idx) -> Thunk(;
                preamble=quote
                    $my_q = ($(ctx(pos)) - $(Tp(1))) * $(ctx(lvl.shape)) + $(ctx(idx))
                    $dirty = false
                end,
                body=(ctx) -> instantiate(
                    ctx,
                    VirtualHollowSubFiber(lvl.lvl, value(my_q, lvl.Ti), dirty),
                    mode,
                ),
                epilogue=quote
                    if $dirty
                        $(fbr.dirty) = true
                        if !$(lvl.tbl)[$my_q]
                            $(lvl.tbl)[$my_q] = true
                            $(lvl.qos_fill) += 1
                            if $(lvl.qos_fill) > $(lvl.qos_stop)
                                $(lvl.qos_stop) = max($(lvl.qos_stop) << 1, 1)
                                Finch.resize_if_smaller!($(lvl.srt), $(lvl.qos_stop))
                            end
                            $(lvl.srt)[$(lvl.qos_fill)] = $my_q
                        end
                    end
                end,
            ),
        ),
    )
end

function sample(tid, lvl::SparseByteMapLevel)
    tup, idx = sample(tid, lvl.lvl)
    idx_2 = mod1(idx, lvl.shape)
    pos_2 = fld(idx - 1, lvl.shape) + 1
    return (tup..., idx_2), pos_2
end

@inbounds function setup_coalesce!(
    lvl::SparseByteMapLevel, max_pos, coalescent, P, style::MergeFast
)
    lvl_ptr = coalescent.ptr
    lvl_tbl = coalescent.tbl
    lvl_srt = coalescent.srt

    nnz = sum(length, lvl.srt.data)
    if nnz < 1
        return false
    end

    resize!(lvl_ptr, max_pos + 1)
    resize!(lvl_tbl, max_pos * lvl.shape)
    resize!(lvl_srt, nnz)

    lvl_ptr[1] = 1

    setup_coalesce!(lvl.lvl, length(lvl_tbl), coalescent.lvl, P, style)
end

@inbounds function setup_coalesce!(
    lvl::SparseByteMapLevel, max_pos, coalescent, P, style::MergeNormalization;
    pos_map=nothing, was_dense=false,
)
    lvl_ptr = coalescent.ptr
    lvl_tbl = coalescent.tbl
    lvl_srt = coalescent.srt

    nnz = sum(length, lvl.srt.data)
    if nnz < 1
        return false
    end

    for p in 1:P - 1
        srt_p = lvl.srt.data[p]
        srt_next = lvl.srt.data[p + 1]
        if !isempty(srt_p) && !isempty(srt_next) && srt_p[end] == srt_next[1]
            nnz -= 1
            srt_p[end] = -1
        end
    end


    resize!(lvl_ptr, max_pos + 1)
    resize!(lvl_tbl, max_pos * lvl.shape)
    resize!(lvl_srt, nnz)

    lvl_ptr[1] = 1
    if max_pos == 1
        lvl_ptr[end] = nnz + 1
    end

    setup_coalesce!(
        lvl.lvl, length(lvl_tbl), coalescent.lvl, P, style; pos_map, was_dense=true
    )
end

function coalesce_fast!(
    tid, pos_offsets, shared_flags, P, lvl::SparseByteMapLevel, coalescent, was_dense
)
    ptr = lvl.ptr.data
    srt = lvl.srt.data
    tbl = lvl.tbl.data
    lvl_ptr = coalescent.ptr
    lvl_tbl = coalescent.tbl
    lvl_srt = coalescent.srt

    work_start, work_stop = fastmerge_spbytemap!(
        tid, pos_offsets, shared_flags, ptr, srt, tbl, P, lvl.shape, lvl_ptr, lvl_srt, lvl_tbl
    )
    coalesce_fast!(
        tid, pos_offsets, shared_flags, P, lvl.lvl, coalescent.lvl, true,
        CoalesceRanges(lvl_srt, work_start, work_stop),
    )
end

function coalesce_dense!(
    tid, pos_offsets, shared_flags, P, lvl::SparseByteMapLevel, coalescent
)
    ptr = lvl.ptr.data
    srt = lvl.srt.data
    tbl = lvl.tbl.data
    lvl_ptr = coalescent.ptr
    lvl_tbl = coalescent.tbl
    lvl_srt = coalescent.srt

    work_start, work_stop = fastmerge_spbytemap!(
        tid, pos_offsets, shared_flags, ptr, srt, tbl, P, lvl.shape, lvl_ptr, lvl_srt, lvl_tbl
    )
    cutoff = 1
    for proc in 1:P
        shared = !isempty(srt[proc]) && srt[proc][end] < 0
        next_cutoff = cutoff + length(srt[proc]) - shared
        start = max(work_start, cutoff)
        stop = min(work_stop, next_cutoff - 1)
        if start <= stop
            coalesce_dense!(
                proc, pos_offsets, shared_flags, P, lvl.lvl, coalescent.lvl,
                CoalesceRanges(lvl_srt, start, stop),
            )
        end
        cutoff = next_cutoff
    end
end

@inbounds function fastmerge_spbytemap!(
    tid, pos_offsets, shared_flags, ptr, srt, tbl, P, shape, lvl_ptr, lvl_srt, lvl_tbl
)
    nnz_cutoffs = Vector{Int}(undef, P + 1)
    nnz_cutoffs[1] = 1
    for p in 2:P+1
        nnz_cutoffs[p] = nnz_cutoffs[p - 1] + length(srt[p - 1])
        if !isempty(srt[p - 1]) && srt[p - 1][end] < 0
            nnz_cutoffs[p] -= 1
        end
    end
    nnz = nnz_cutoffs[end] - 1
    max_pos = length(lvl_ptr) - 1

    base, rem = divrem(nnz, P)
    offset = (tid - 1) * base + min(tid - 1, rem)
    chunksize = base + (tid <= rem ? 1 : 0)

    if chunksize > 0
        work_lb = 1 + offset
        work_ub = work_lb + chunksize - 1

        proc_id_lower = binary_search(work_lb, nnz_cutoffs)
        nz_id_lower = work_lb - nnz_cutoffs[proc_id_lower] + 1
        proc_id_upper = binary_search(work_ub, nnz_cutoffs)
        nz_id_upper = work_ub - nnz_cutoffs[proc_id_upper] + 1

        lfbr_lower = binary_search(nz_id_lower, ptr[proc_id_lower])
        lfbr_upper = binary_search(nz_id_upper, ptr[proc_id_upper])

        pos_lb = pos_offsets[tid][proc_id_lower + 1] + lfbr_lower - 1
        pos_ub = min(pos_offsets[tid][proc_id_upper + 1] + lfbr_upper - 1, max_pos)


        if nz_id_upper < ptr[proc_id_upper][lfbr_upper + 1] - 1
            shares_border = true
        elseif lfbr_upper < length(ptr[proc_id_upper]) - 1
            shares_border = false
        elseif proc_id_upper < P
            shares_border = pos_offsets[tid][proc_id_upper + 2] == pos_ub
        else
            shares_border = false
        end

        proc = proc_id_lower
        srt_read = nz_id_lower
        srt_write = work_lb
        srt_ceil = srt_write + chunksize
        while srt_write < srt_ceil
            raw_start = pos_offsets[tid][proc + 1] - (shared_flags[tid][proc] ? 1 : 0)
            pos_shift = (raw_start - 1) * shape
            ele = srt[proc][srt_read] + pos_shift
            if ele > 0
                lvl_tbl[ele] = true
                lvl_srt[srt_write] = ele
                srt_write += 1
            end
            srt_read += 1

            if srt_read > length(srt[proc])
                srt_read = 1
                proc += 1
            end
        end

        if max_pos != 1
            proc = proc_id_lower
            pos_read = lfbr_lower

            pos_write = 2
            for p in 1:proc - 1
                pos_write += length(ptr[p]) - 1
                shared_flags[tid][p] && (pos_write -= 1)
            end
            pos_write += lfbr_lower - 1

            ceil = 3
            for p in 1:proc_id_upper - 1
                ceil += length(ptr[p]) - 1
                shared_flags[tid][p] && (ceil -= 1)
            end
            ceil += lfbr_upper - 1
            shares_border && (ceil -= 1)

            prefix = ptr[proc][pos_read] + nnz_cutoffs[proc] - 1
            while pos_write < ceil
                delta = ptr[proc][pos_read + 1] - ptr[proc][pos_read]
                mul = delta > 0
                prefix += delta * mul
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
                    if shared_flags[tid][old_proc]
                        pos_write -= 1
                    end
                end
            end
        end
    end

    last_pos = 0
    for p in 1:P
        ancestor_shared = shared_flags[tid][p]
        last_pos += length(ptr[p]) - 1
        pos_offsets[tid][p + 1] = last_pos
        if ancestor_shared
            last_pos -= 1
        end
    end
    return 1 + offset, offset + chunksize
end

function coalesce_fast!(
    tid, pos_offsets, shared_flags, P, lvl::SparseByteMapLevel, coalescent,
    was_dense, ranges,
)
    ptr = lvl.ptr.data
    srt = lvl.srt.data
    shifts = coalesce_parent_shifts(
        tid, pos_offsets, shared_flags, P, length(coalescent.ptr) - 1, was_dense
    )
    cutoffs = ones(Int, P + 1)
    for p in 1:P
        shared = !isempty(srt[p]) && srt[p][end] < 0
        cutoffs[p + 1] = cutoffs[p] + length(srt[p]) - shared
    end
    for range in ranges
        isempty(range) && continue
        for pos in first(range):(last(range) + 1)
            prefix = 1
            for p in 1:P
                local_pos = clamp(pos - shifts[p], 1, length(ptr[p]))
                prefix += min(ptr[p][local_pos] - 1, cutoffs[p + 1] - cutoffs[p])
            end
            coalescent.ptr[pos] = prefix
        end
    end
    slots = (coalesce_child_range(ptr[p], shifts[p], cutoffs[p], cutoffs[p + 1], r)
        for p in 1:P for r in ranges)
    for p in 1:P, range in ranges
        for r in coalesce_child_range(ptr[p], shifts[p], cutoffs[p], cutoffs[p + 1], range)
            q = srt[p][r - cutoffs[p] + 1] + shifts[p] * lvl.shape
            coalescent.tbl[q] = true
            coalescent.srt[r] = q
        end
    end
    child_ranges = Iterators.flatten(
        CoalesceRanges(coalescent.srt, first(r), last(r)) for r in slots
    )
    coalesce_fast!(
        tid, pos_offsets, shared_flags, P, lvl.lvl, coalescent.lvl, true, child_ranges
    )
end

function coalesce_dense!(
    tid, pos_offsets, shared_flags, P, lvl::SparseByteMapLevel, coalescent, ranges
)
    ptr = lvl.ptr.data
    srt = lvl.srt.data
    cutoff = 1
    for p in 1:(tid - 1)
        cutoff += length(srt[p]) - (!isempty(srt[p]) && srt[p][end] < 0)
    end
    next_cutoff = cutoff + length(srt[tid]) - (!isempty(srt[tid]) && srt[tid][end] < 0)
    for range in ranges
        isempty(range) && continue
        for pos in first(range):(last(range) + 1)
            coalescent.ptr[pos] = 1 + sum(ptr[p][pos] - 1 for p in 1:P)
        end
        for r in coalesce_child_range(ptr[tid], 0, cutoff, next_cutoff, range)
            q = srt[tid][r - cutoff + 1]
            coalescent.tbl[q] = true
            coalescent.srt[r] = q
        end
    end
    slots = (coalesce_child_range(ptr[tid], 0, cutoff, next_cutoff, r) for r in ranges)
    child_ranges = Iterators.flatten(
        CoalesceRanges(coalescent.srt, first(r), last(r)) for r in slots
    )
    coalesce_dense!(tid, pos_offsets, shared_flags, P, lvl.lvl, coalescent.lvl, child_ranges)
end
