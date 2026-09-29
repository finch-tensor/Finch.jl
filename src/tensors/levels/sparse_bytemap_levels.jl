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

function setup_coalesce!(lvl::SparseByteMapLevel, max_pos, dst, P, shift, overlap)
    srt = lvl.srt.data
    shape = lvl.shape
    shared = zeros(Int, P)
    shared_dst = zeros(Int, P)
    off = zeros(Int, P)
    prev = zeros(Int, P)
    nnz = 0
    last_q = 0
    for p in 1:P
        off[p] = nnz
        prev[p] = last_q
        isempty(srt[p]) && continue
        # A shard shares its first entry if the previous shard ended on it.
        if srt[p][1] + shift[p] * shape == last_q
            shared[p] = srt[p][1]
            shared_dst[p] = last_q
        end
        nnz += length(srt[p]) - (shared[p] != 0)
        last_q = srt[p][end] + shift[p] * shape
    end
    # Unoccupied bounds and bitmap entries retain their cleared values.
    ptr_start = min(length(dst.ptr), max_pos + 1) + 1
    tbl_start = min(length(dst.tbl), max_pos * shape) + 1
    resize!(dst.ptr, max_pos + 1)
    resize!(dst.tbl, max_pos * shape)
    empty!(dst.srt)
    resize!(dst.srt, nnz)
    dst.ptr[1] = 1
    # Both shards' leaves under a shared entry overlap.
    child = setup_coalesce!(
        lvl.lvl, max_pos * shape, dst.lvl, P, shift .* shape, any(!iszero, shared)
    )
    # ptr[1] is the sentinel, not an unoccupied parent bound.
    init = (
        (dst.ptr, max(2, ptr_start), 0),
        (dst.tbl, tbl_start, false), child.init...,
    )
    return (; shift, shared, shared_dst, off, prev, nnz, child, init)
end

# Owned entries fill consecutive slots after earlier shards' entries. ptr keeps
# freeze_level!'s layout, with bounds only around occupied positions, so the merge
# stays O(nnz). The shard owning a position's first entry writes its bounds.
function coalesce_shard!(tid, plan, lvl::SparseByteMapLevel, dst, runs)
    src = lvl.srt.data[tid]
    isempty(src) && return nothing
    shape = lvl.shape
    q_shift = plan.shift[tid] * shape
    slot = plan.off[tid] + 1
    p_prev = fld(plan.prev[tid] - 1, shape) + 1
    for local_q in src
        local_q == plan.shared[tid] && continue
        q = local_q + q_shift
        dst.srt[slot] = q
        dst.tbl[q] = true
        p = fld(q - 1, shape) + 1
        if p != p_prev
            dst.ptr[p_prev + 1] = slot
            dst.ptr[p] = slot
        end
        p_prev = p
        slot += 1
    end
    if slot - 1 == plan.nnz && slot > plan.off[tid] + 1
        dst.ptr[p_prev + 1] = slot
    end

    # Recurse on every entry, owned or not: a shared entry's children are split
    # between both shards.
    leaves = coalesce_leaves(lvl.lvl)
    child_runs = (((q - 1) * leaves + 1):(q * leaves) for q in src)
    coalesce_shard!(tid, plan.child, lvl.lvl, dst.lvl, child_runs)
end
