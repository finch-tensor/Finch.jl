struct SparseVBLLevel{Ti, Tp, Lvl}
    lvl::Lvl
    shape::Ti
    ptr::Vector{Tp}
    idx::Vector{Ti}
    ofs::Vector{Tp}
end

const SparseVBL = SparseVBLLevel
SparseVBLLevel(lvl, ) = SparseVBLLevel{Int}(lvl)
SparseVBLLevel(lvl, shape, args...) = SparseVBLLevel{typeof(shape)}(lvl, shape, args...)
SparseVBLLevel{Ti}(lvl, args...) where {Ti} = SparseVBLLevel{Ti, Int}(lvl, args...)
SparseVBLLevel{Ti, Tp}(lvl, args...) where {Ti, Tp} = SparseVBLLevel{Ti, Tp, typeof(lvl)}(lvl, args...)

SparseVBLLevel{Ti, Tp, Lvl}(lvl) where {Ti, Tp, Lvl} = SparseVBLLevel{Ti, Tp, Lvl}(lvl, zero(Ti))
SparseVBLLevel{Ti, Tp, Lvl}(lvl, shape) where {Ti, Tp, Lvl} = 
    SparseVBLLevel{Ti, Tp, Lvl}(lvl, shape, Tp[1], Ti[], Ti[])

Base.summary(lvl::SparseVBLLevel) = "SparseVBL($(summary(lvl.lvl)))"
similar_level(lvl::SparseVBLLevel) = SparseVBL(similar_level(lvl.lvl))
similar_level(lvl::SparseVBLLevel, dim, tail...) = SparseVBL(similar_level(lvl.lvl, tail...), dim)

pattern!(lvl::SparseVBLLevel{Ti, Tp}) where {Ti, Tp} = 
    SparseVBLLevel{Ti, Tp}(pattern!(lvl.lvl), lvl.shape, lvl.ptr, lvl.idx, lvl.ofs)

function countstored_level(lvl::SparseVBLLevel, pos)
    countstored_level(lvl.lvl, lvl.ofs[lvl.ptr[pos + 1]]-1)
end

redefault!(lvl::SparseVBLLevel{Ti, Tp}, init) where {Ti, Tp} = 
    SparseVBLLevel{Ti, Tp}(redefault!(lvl.lvl, init), lvl.shape, lvl.ptr, lvl.idx, lvl.ofs)

function Base.show(io::IO, lvl::SparseVBLLevel{Ti, Tp}) where {Ti, Tp}
    if get(io, :compact, false)
        print(io, "SparseVBL(")
    else
        print(io, "SparseVBL{$Ti, $Tp}(")
    end
    show(io, lvl.lvl)
    print(io, ", ")
    show(IOContext(io, :typeinfo=>Ti), lvl.shape)
    print(io, ", ")
    if get(io, :compact, false)
        print(io, "…")
    else
        show(IOContext(io, :typeinfo=>Vector{Tp}), lvl.ptr)
        print(io, ", ")
        show(IOContext(io, :typeinfo=>Vector{Ti}), lvl.idx)
        print(io, ", ")
        show(IOContext(io, :typeinfo=>Vector{Tp}), lvl.ofs)
    end
    print(io, ")")
end

function display_fiber(io::IO, mime::MIME"text/plain", fbr::SubFiber{<:SparseVBLLevel}, depth)
    p = fbr.pos
    crds = []
    for r in fbr.lvl.ptr[p]:fbr.lvl.ptr[p + 1] - 1
        i = fbr.lvl.idx[r]
        l = fbr.lvl.ofs[r + 1] - fbr.lvl.ofs[r]
        append!(crds, (i - l + 1):i)
    end

    print_coord(io, crd) = show(io, crd)
    get_fbr(crd) = fbr(crd)

    print(io, "SparseVBL (", default(fbr), ") [", ":,"^(ndims(fbr) - 1), "1:", fbr.lvl.shape, "]")
    display_fiber_data(io, mime, fbr, depth, 1, crds, print_coord, get_fbr)
end

@inline level_ndims(::Type{<:SparseVBLLevel{Ti, Tp, Lvl}}) where {Ti, Tp, Lvl} = 1 + level_ndims(Lvl)
@inline level_size(lvl::SparseVBLLevel) = (lvl.shape, level_size(lvl.lvl)...)
@inline level_axes(lvl::SparseVBLLevel) = (Base.OneTo(lvl.shape), level_axes(lvl.lvl)...)
@inline level_eltype(::Type{<:SparseVBLLevel{Ti, Tp, Lvl}}) where {Ti, Tp, Lvl} = level_eltype(Lvl)
@inline level_default(::Type{<:SparseVBLLevel{Ti, Tp, Lvl}}) where {Ti, Tp, Lvl} = level_default(Lvl)
data_rep_level(::Type{<:SparseVBLLevel{Ti, Tp, Lvl}}) where {Ti, Tp, Lvl} = SparseData(data_rep_level(Lvl))

(fbr::AbstractFiber{<:SparseVBLLevel})() = fbr
function (fbr::SubFiber{<:SparseVBLLevel})(idxs...)
    isempty(idxs) && return fbr
    lvl = fbr.lvl
    p = fbr.pos
    r = lvl.ptr[p] + searchsortedfirst(@view(lvl.idx[lvl.ptr[p]:lvl.ptr[p + 1] - 1]), idxs[end]) - 1
    r < lvl.ptr[p + 1] || return default(fbr)
    q = lvl.ofs[r + 1] - 1 - lvl.idx[r] + idxs[end]
    q >= lvl.ofs[r] || return default(fbr)
    fbr_2 = SubFiber(lvl.lvl, q)
    return fbr_2(idxs[1:end-1]...)
end

mutable struct VirtualSparseVBLLevel <: AbstractVirtualLevel
    lvl
    ex
    Ti
    Tp
    shape
    qos_fill
    qos_stop
    ros_fill
    ros_stop
    dirty
    prev_pos
end

is_level_injective(lvl::VirtualSparseVBLLevel, ctx) = [is_level_injective(lvl.lvl, ctx)..., false]
is_level_concurrent(lvl::VirtualSparseVBLLevel, ctx) = [is_level_concurrent(lvl.lvl, ctx)..., false]
is_level_atomic(lvl::VirtualSparseVBLLevel, ctx) = false

function virtualize(ex, ::Type{SparseVBLLevel{Ti, Tp, Lvl}}, ctx, tag=:lvl) where {Ti, Tp, Lvl}
    sym = freshen(ctx, tag)
    shape = value(:($sym.shape), Int)
    qos_fill = freshen(ctx, sym, :_qos_fill)
    qos_stop = freshen(ctx, sym, :_qos_stop)
    ros_fill = freshen(ctx, sym, :_ros_fill)
    ros_stop = freshen(ctx, sym, :_ros_stop)
    dirty = freshen(ctx, sym, :_dirty)
    push!(ctx.preamble, quote
        $sym = $ex
    end)
    prev_pos = freshen(ctx, sym, :_prev_pos)
    lvl_2 = virtualize(:($sym.lvl), Lvl, ctx, sym)
    VirtualSparseVBLLevel(lvl_2, sym, Ti, Tp, shape, qos_fill, qos_stop, ros_fill, ros_stop, dirty, prev_pos)
end
function lower(lvl::VirtualSparseVBLLevel, ctx::AbstractCompiler, ::DefaultStyle)
    quote
        $SparseVBLLevel{$(lvl.Ti), $(lvl.Tp)}(
            $(ctx(lvl.lvl)),
            $(ctx(lvl.shape)),
            $(lvl.ex).ptr,
            $(lvl.ex).idx,
            $(lvl.ex).ofs,
        )
    end
end

Base.summary(lvl::VirtualSparseVBLLevel) = "SparseVBL($(summary(lvl.lvl)))"

function virtual_level_size(lvl::VirtualSparseVBLLevel, ctx)
    ext = Extent(literal(lvl.Ti(1)), lvl.shape)
    (virtual_level_size(lvl.lvl, ctx)..., ext)
end

function virtual_level_resize!(lvl::VirtualSparseVBLLevel, ctx, dims...)
    lvl.shape = getstop(dims[end])
    lvl.lvl = virtual_level_resize!(lvl.lvl, ctx, dims[1:end-1]...)
    lvl
end

virtual_level_eltype(lvl::VirtualSparseVBLLevel) = virtual_level_eltype(lvl.lvl)
virtual_level_default(lvl::VirtualSparseVBLLevel) = virtual_level_default(lvl.lvl)

function declare_level!(lvl::VirtualSparseVBLLevel, ctx::AbstractCompiler, pos, init)
    Tp = lvl.Tp
    Ti = lvl.Ti
    ros = call(-, call(getindex, :($(lvl.ex).ptr), call(+, pos, 1)), 1)
    qos = call(-, call(getindex, :($(lvl.ex).ofs), call(+, ros, 1)), 1)
    push!(ctx.code.preamble, quote
        $(lvl.qos_fill) = $(Tp(0))
        $(lvl.qos_stop) = $(Tp(0))
        $(lvl.ros_fill) = $(Tp(0))
        $(lvl.ros_stop) = $(Tp(0))
        Finch.resize_if_smaller!($(lvl.ex).ofs, 1)
        $(lvl.ex).ofs[1] = 1
    end)
    if issafe(ctx.mode)
        push!(ctx.code.preamble, quote
            $(lvl.prev_pos) = $(Tp(0))
        end)
    end
    lvl.lvl = declare_level!(lvl.lvl, ctx, qos, init)
    return lvl
end

function trim_level!(lvl::VirtualSparseVBLLevel, ctx::AbstractCompiler, pos)
    Tp = lvl.Tp
    Ti = lvl.Ti
    ros = freshen(ctx.code, :ros)
    qos = freshen(ctx.code, :qos)
    push!(ctx.code.preamble, quote
        resize!($(lvl.ex).ptr, $(ctx(pos)) + 1)
        $ros = $(lvl.ex).ptr[end] - $(lvl.Tp(1))
        resize!($(lvl.ex).idx, $ros)
        resize!($(lvl.ex).ofs, $ros + 1)
        $qos = $(lvl.ex).ofs[end] - $(lvl.Tp(1))
    end)
    lvl.lvl = trim_level!(lvl.lvl, ctx, value(qos, Tp))
    return lvl
end

function assemble_level!(lvl::VirtualSparseVBLLevel, ctx, pos_start, pos_stop)
    pos_start = ctx(cache!(ctx, :p_start, pos_start))
    pos_stop = ctx(cache!(ctx, :p_start, pos_stop))
    return quote
        Finch.resize_if_smaller!($(lvl.ex).ptr, $pos_stop + 1)
        Finch.fill_range!($(lvl.ex).ptr, 0, $pos_start + 1, $pos_stop + 1)
    end
end

function freeze_level!(lvl::VirtualSparseVBLLevel, ctx::AbstractCompiler, pos_stop)
    p = freshen(ctx.code, :p)
    pos_stop = ctx(cache!(ctx, :pos_stop, simplify(pos_stop, ctx)))
    qos_stop = freshen(ctx.code, :qos_stop)
    push!(ctx.code.preamble, quote
        for $p = 2:($pos_stop + 1)
            $(lvl.ex).ptr[$p] += $(lvl.ex).ptr[$p - 1]
        end
        $qos_stop = $(lvl.ex).ptr[$pos_stop + 1] - 1
    end)
    lvl.lvl = freeze_level!(lvl.lvl, ctx, value(qos_stop))
    return lvl
end

function instantiate_reader(fbr::VirtualSubFiber{VirtualSparseVBLLevel}, ctx, subprotos, ::Union{typeof(defaultread), typeof(walk)})
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    Tp = lvl.Tp
    Ti = lvl.Ti
    my_i = freshen(ctx.code, tag, :_i)
    my_i_start = freshen(ctx.code, tag, :_i)
    my_r = freshen(ctx.code, tag, :_r)
    my_r_stop = freshen(ctx.code, tag, :_r_stop)
    my_q = freshen(ctx.code, tag, :_q)
    my_q_stop = freshen(ctx.code, tag, :_q_stop)
    my_q_ofs = freshen(ctx.code, tag, :_q_ofs)
    my_i1 = freshen(ctx.code, tag, :_i1)

    Furlable(
        body = (ctx, ext) -> Thunk(
            preamble = quote
                $my_r = $(lvl.ex).ptr[$(ctx(pos))]
                $my_r_stop = $(lvl.ex).ptr[$(ctx(pos)) + $(Tp(1))]
                if $my_r < $my_r_stop
                    $my_i = $(lvl.ex).idx[$my_r]
                    $my_i1 = $(lvl.ex).idx[$my_r_stop - $(Tp(1))]
                else
                    $my_i = $(Ti(1))
                    $my_i1 = $(Ti(0))
                end
            end,
            body = (ctx) -> Sequence([
                Phase(
                    stop = (ctx, ext) -> value(my_i1),
                    body = (ctx, ext) -> Replay(
                        seek = (ctx, ext) -> quote
                            if $(lvl.ex).idx[$my_r] < $(ctx(getstart(ext)))
                                $my_r = Finch.scansearch($(lvl.ex).idx, $(ctx(getstart(ext))), $my_r, $my_r_stop - 1)
                            end
                        end,
                        body = Step(
                                preamble = quote
                                    $my_i = $(lvl.ex).idx[$my_r]
                                    $my_q_stop = $(lvl.ex).ofs[$my_r + $(Tp(1))]
                                    $my_i_start = $my_i - ($my_q_stop - $(lvl.ex).ofs[$my_r])
                                    $my_q_ofs = $my_q_stop - $my_i - $(Tp(1))
                                end,
                                stop = (ctx, ext) -> value(my_i),
                                body = (ctx, ext) -> Thunk(
                                    body = (ctx) -> Sequence([
                                        Phase(
                                            stop = (ctx, ext) -> value(my_i_start),
                                            body = (ctx, ext) -> Run(Fill(virtual_level_default(lvl))),
                                        ),
                                        Phase(
                                            body = (ctx, ext) -> Lookup(
                                                body = (ctx, i) -> Thunk(
                                                    preamble = quote
                                                        $my_q = $my_q_ofs + $(ctx(i))
                                                    end,
                                                    body = (ctx) -> instantiate_reader(VirtualSubFiber(lvl.lvl, value(my_q, lvl.Tp)), ctx, subprotos),
                                                )
                                            )
                                        )
                                    ]),
                                    epilogue = quote
                                        $my_r += ($(ctx(getstop(ext))) == $my_i)
                                    end
                                )
                            )
                        )
                ),
                Phase(
                    body = (ctx, ext) -> Run(Fill(virtual_level_default(lvl)))
                )
            ])
        )
    )
end

function instantiate_reader(fbr::VirtualSubFiber{VirtualSparseVBLLevel}, ctx, subprotos, ::typeof(gallop))
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    Tp = lvl.Tp
    Ti = lvl.Ti
    my_i = freshen(ctx.code, tag, :_i)
    my_j = freshen(ctx.code, tag, :_j)
    my_i_start = freshen(ctx.code, tag, :_i)
    my_r = freshen(ctx.code, tag, :_r)
    my_r_stop = freshen(ctx.code, tag, :_r_stop)
    my_q = freshen(ctx.code, tag, :_q)
    my_q_stop = freshen(ctx.code, tag, :_q_stop)
    my_q_ofs = freshen(ctx.code, tag, :_q_ofs)
    my_i1 = freshen(ctx.code, tag, :_i1)

    Furlable(
        body = (ctx, ext) -> Thunk(
            preamble = quote
                $my_r = $(lvl.ex).ptr[$(ctx(pos))]
                $my_r_stop = $(lvl.ex).ptr[$(ctx(pos)) + $(Tp(1))]
                if $my_r < $my_r_stop
                    $my_i = $(lvl.ex).idx[$my_r]
                    $my_i1 = $(lvl.ex).idx[$my_r_stop - $(Tp(1))]
                else
                    $my_i = $(Ti(1))
                    $my_i1 = $(Ti(0))
                end
            end,
            body = (ctx) -> Sequence([
                Phase(
                    stop = (ctx, ext) -> value(my_i1),
                    body = (ctx, ext) -> Replay(
                        seek = (ctx, ext) -> quote
                            if $(lvl.ex).idx[$my_r] < $(ctx(getstart(ext)))
                                $my_r = Finch.scansearch($(lvl.ex).idx, $(ctx(getstart(ext))), $my_r, $my_r_stop - 1)
                            end
                        end,
                        body = Jump(
                                preamble = quote
                                    $my_i = $(lvl.ex).idx[$my_r]
                                    $my_q_stop = $(lvl.ex).ofs[$my_r + $(Tp(1))]
                                    $my_i_start = $my_i - ($my_q_stop - $(lvl.ex).ofs[$my_r])
                                    $my_q_ofs = $my_q_stop - $my_i - $(Tp(1))
                                end,
                                stop = (ctx, ext) -> value(my_i),
                                chunk = Sequence([
                                            Phase(
                                                stop = (ctx, ext) -> value(my_i_start),
                                                body = (ctx, ext) -> Run(Fill(virtual_level_default(lvl))),
                                            ),
                                            Phase(
                                                body = (ctx, ext) -> Lookup(
                                                    body = (ctx, i) -> Thunk(
                                                        preamble = :($my_q = $my_q_ofs + $(ctx(i))),
                                                        body = (ctx) -> instantiate_reader(VirtualSubFiber(lvl.lvl, value(my_q, lvl.Tp)), ctx, subprotos),
                                                    )
                                                )
                                            )
                                        ]),
                                next = (ctx, ext) -> :($my_r += $(Tp(1))),
                            )
                        ),
                ),
                Phase(
                    body = (ctx, ext) -> Run(Fill(virtual_level_default(lvl)))
                )
            ])
        )
    )
end

instantiate_updater(fbr::VirtualSubFiber{VirtualSparseVBLLevel}, ctx, protos) =
    instantiate_updater(VirtualTrackedSubFiber(fbr.lvl, fbr.pos, freshen(ctx.code, :null)), ctx, protos)
function instantiate_updater(fbr::VirtualTrackedSubFiber{VirtualSparseVBLLevel}, ctx, subprotos, ::Union{typeof(defaultupdate), typeof(extrude)})
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    Tp = lvl.Tp
    Ti = lvl.Ti
    my_p = freshen(ctx.code, tag, :_p)
    my_q = freshen(ctx.code, tag, :_q)
    my_i_prev = freshen(ctx.code, tag, :_i_prev)
    qos = freshen(ctx.code, tag, :_qos)
    ros = freshen(ctx.code, tag, :_ros)
    qos_fill = lvl.qos_fill
    qos_stop = lvl.qos_stop
    ros_fill = lvl.ros_fill
    ros_stop = lvl.ros_stop
    dirty = freshen(ctx.code, tag, :dirty)

    Furlable(
        body = (ctx, ext) -> Thunk(
            preamble = quote
                $ros = $ros_fill
                $qos = $qos_fill + 1
                $my_i_prev = $(Ti(-1))
                $(if issafe(ctx.mode)
                    quote
                        $(lvl.prev_pos) < $(ctx(pos)) || throw(FinchProtocolError("SparseVBLLevels cannot be updated multiple times"))
                    end
                end)
            end,
            body = (ctx) -> Lookup(
                body = (ctx, idx) -> Thunk(
                    preamble = quote
                        if $qos > $qos_stop
                            $qos_stop = max($qos_stop << 1, 1)
                            $(contain(ctx_2->assemble_level!(lvl.lvl, ctx_2, value(qos, lvl.Tp), value(qos_stop, lvl.Tp)), ctx))
                        end
                        $dirty = false
                    end,
                    body = (ctx) -> instantiate_updater(VirtualTrackedSubFiber(lvl.lvl, value(qos, lvl.Tp), dirty), ctx, subprotos),
                    epilogue = quote
                        if $dirty
                            $(fbr.dirty) = true
                            if $(ctx(idx)) > $my_i_prev + $(Ti(1))
                                $ros += $(Tp(1))
                                if $ros > $ros_stop
                                    $ros_stop = max($ros_stop << 1, 1)
                                    Finch.resize_if_smaller!($(lvl.ex).idx, $ros_stop)
                                    Finch.resize_if_smaller!($(lvl.ex).ofs, $ros_stop + 1)
                                end
                            end
                            $(lvl.ex).idx[$ros] = $my_i_prev = $(ctx(idx))
                            $(qos) += $(Tp(1))
                            $(lvl.ex).ofs[$ros + 1] = $qos
                            $(if issafe(ctx.mode)
                                quote
                                    $(lvl.prev_pos) = $(ctx(pos))
                                end
                            end)
                        end
                    end
                )
            ),
            epilogue = quote
                $(lvl.ex).ptr[$(ctx(pos)) + 1] = $ros - $ros_fill
                $ros_fill = $ros
                $qos_fill = $qos - 1
            end
        )
    )
end
