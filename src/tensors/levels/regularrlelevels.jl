struct RegularRLELevel{Ti, Tp, Vp<:AbstractVector, VLTi<:AbstractVector, VRTi<:AbstractVector, Lvl}
    lvl::Lvl
    shape::Ti
    ptr::Vp
    left::VLTi
    right::VRTi
end

const RegularRLE = RegularRLELevel
RegularRLELevel(lvl:: Lvl) where {Lvl} = RegularRLELevel{Int}(lvl)
RegularRLELevel(lvl, shape, args...) = RegularRLELevel{typeof(shape)}(lvl, shape, args...)
RegularRLELevel{Ti}(lvl, args...) where {Ti} =
    RegularRLELevel{Ti,
        postype(typeof(lvl)),
        (memtype(typeof(lvl))){postype(typeof(lvl)), 1},
        (memtype(typeof(lvl))){Ti, 1},
        (memtype(typeof(lvl))){Ti, 1},
        typeof(lvl)}(lvl, args...)
#RegularRLELevel{Ti, Tp}(lvl, args...) where {Ti, Tp} = RegularRLELevel{Ti, Tp, typeof(lvl)}(lvl, args...)

RegularRLELevel{Ti, Tp, Vp, VLTi, VRTi, Lvl}(lvl) where {Ti, Tp, Vp, VLTi, VRTi, Lvl} = RegularRLELevel{Ti, Tp, Vp, VLTi, VRTi, Lvl}(lvl, zero(Ti))
RegularRLELevel{Ti, Tp, Vp, VLTi, VRTi, Lvl}(lvl, shape) where {Ti, Tp, Vp, VLTi, VRTi, Lvl} = 
    RegularRLELevel{Ti, Tp, Vp, VLTi, VRTi, Lvl}(lvl, shape, Tp[1], Ti[], Ti[])

Base.summary(lvl::RegularRLELevel) = "RegularRLE($(summary(lvl.lvl)))"
similar_level(lvl::RegularRLELevel) = RegularRLE(similar_level(lvl.lvl))
similar_level(lvl::RegularRLELevel, dim, tail...) = RegularRLE(similar_level(lvl.lvl, tail...), dim)

function memtype(::Type{RegularRLELevel{Ti, Tp, Vp, VLTi, VRTi, Lvl}}) where {Ti, Tp, Vp, VLTi, VRTi, Lvl}
    return containertype(Vp)
end

function postype(::Type{RegularRLELevel{Ti, Tp, Vp, VLTi, VRTi, Lvl}}) where {Ti, Tp, Vp, VLTi, VRTi, Lvl}
    return Tp
end

function moveto(lvl::RegularRLELevel{Ti, Tp, Vp, VLTi, VRTi, Lvl}, ::Type{MemType}) where {Ti, Tp, Vp, VLTi, VRTi, Lvl, MemType <: AbstractArray}
    lvl_2 = moveto(lvl.lvl, MemType)
    ptr_2 = MemType{Tp, 1}(lvl.ptr)
    left_2 = MemType{Ti, 1}(lvl.left)
    right_2 = MemType{Ti, 1}(lvl.right)
    return RegularRLELevel{Ti, Tp, MemType{Tp, 1}, MemType{Ti, 1}, MemType{Ti, 1}, typeof(lvl_2)}(lvl_2, lvl.shape, ptr_2, left_2, right_2)
end

pattern!(lvl::RegularRLELevel{Ti, Tp, Vp, VLTi, VRTi, Lvl}) where {Ti, Tp, Vp, VLTi, VRTi, Lvl} = 
    RegularRLELevel{Ti, Tp, Vp, VLTi, VRTi, Lvl}(pattern!(lvl.lvl), lvl.shape, lvl.ptr, lvl.left, lvl.right)

function countstored_level(lvl::RegularRLELevel, pos)
    countstored_level(lvl.lvl, lvl.left[lvl.ptr[pos + 1]]-1)
end

redefault!(lvl::RegularRLELevel{Ti, Tp, Vp, VLTi, VRTi, Lvl}, init) where {Ti, Tp, Vp, VLTi, VRTi, Lvl} = 
    RegularRLELevel{Ti, Tp, Vp, VLTi, VRTi, Lvl}(redefault!(lvl.lvl, init), lvl.shape, lvl.ptr, lvl.left, lvl.right)

function Base.show(io::IO, lvl::RegularRLELevel{Ti, Tp, Vp, VLTi, VRTi, Lvl}) where {Ti, Tp, Vp, VLTi, VRTi, Lvl}
    if get(io, :compact, false)
        print(io, "RegularRLE(")
    else
        print(io, "RegularRLE{$Ti, $Tp, $Vp, $VLTi, $VRTi, $Lvl}(")
    end
    show(io, lvl.lvl)
    print(io, ", ")
    show(IOContext(io, :typeinfo=>Ti), lvl.shape)
    print(io, ", ")
    if get(io, :compact, false)
        print(io, "…")
    else
        show(IOContext(io, :typeinfo=>Vp), lvl.ptr)
        print(io, ", ")
        show(IOContext(io, :typeinfo=>VLTi), lvl.left)
        print(io, ", ")
        show(IOContext(io, :typeinfo=>VRTi), lvl.right)
    end
    print(io, ")")
end

function display_fiber(io::IO, mime::MIME"text/plain", fbr::SubFiber{<:RegularRLELevel}, depth)
    p = fbr.pos
    lvl = fbr.lvl
    left_endpoints = @view(lvl.left[lvl.ptr[p]:lvl.ptr[p + 1] - 1])

    crds = []
    for l in left_endpoints 
        append!(crds, l)
    end

    print_coord(io, crd) = print(io, crd, ":", lvl.right[lvl.ptr[p]-1+searchsortedfirst(left_endpoints, crd)])  
    get_fbr(crd) = fbr(crd)

    print(io, "RegularRLE (", default(fbr), ") [", ":,"^(ndims(fbr) - 1), "1:", fbr.lvl.shape, "]")
    display_fiber_data(io, mime, fbr, depth, 1, crds, print_coord, get_fbr)
end

@inline level_ndims(::Type{<:RegularRLELevel{Ti, Tp, Vp, VLTi, VRTi, Lvl}}) where {Ti, Tp, Vp, VLTi, VRTi, Lvl} = 1 + level_ndims(Lvl)
@inline level_size(lvl::RegularRLELevel) = (lvl.shape, level_size(lvl.lvl)...)
@inline level_axes(lvl::RegularRLELevel) = (Base.OneTo(lvl.shape), level_axes(lvl.lvl)...)
@inline level_eltype(::Type{<:RegularRLELevel{Ti, Tp, Vp, VLTi, VRTi, Lvl}}) where {Ti, Tp, Vp, VLTi, VRTi, Lvl} = level_eltype(Lvl)
@inline level_default(::Type{<:RegularRLELevel{Ti, Tp, Vp, VLTi, VRTi, Lvl}}) where {Ti, Tp, Vp, VLTi, VRTi, Lvl}= level_default(Lvl)
data_rep_level(::Type{<:RegularRLELevel{Ti, Tp, Vp, VLTi, VRTi, Lvl}}) where {Ti, Tp, Vp, VLTi, VRTi, Lvl} = SparseData(data_rep_level(Lvl))

(fbr::AbstractFiber{<:RegularRLELevel})() = fbr
function (fbr::SubFiber{<:RegularRLELevel})(idxs...)
    isempty(idxs) && return fbr
    lvl = fbr.lvl
    p = fbr.pos
    r1 = searchsortedlast(@view(lvl.left[lvl.ptr[p]:lvl.ptr[p + 1] - 1]), idxs[end])
    r2 = searchsortedfirst(@view(lvl.right[lvl.ptr[p]:lvl.ptr[p + 1] - 1]), idxs[end])
    q = lvl.ptr[p] + first(r1) - 1
    fbr_2 = SubFiber(lvl.lvl, q)
    r1 != r2 ? default(fbr_2) : fbr_2(idxs[1:end-1]...)
end


mutable struct VirtualRegularRLELevel <: AbstractVirtualLevel
    lvl
    ex
    Ti
    Tp
    shape
    qos_fill
    qos_stop
    Vp
    VLTi
    VRTi
    Lvl
    prev_pos
end

  is_level_injective(lvl::VirtualRegularRLELevel, ctx) = [false, is_level_injective(lvl.lvl, ctx)...]
is_level_concurrent(lvl::VirtualRegularRLELevel, ctx) = [false, is_level_concurrent(lvl.lvl, ctx)...]
is_level_atomic(lvl::VirtualRegularRLELevel, ctx) = false
  

function virtualize(ex, ::Type{RegularRLELevel{Ti, Tp, Vp, VLTi, VRTi, Lvl}}, ctx, tag=:lvl) where {Ti, Tp, Vp, VLTi, VRTi, Lvl}
    sym = freshen(ctx, tag)
    shape = value(:($sym.shape), Int)
    qos_fill = freshen(ctx, sym, :_qos_fill)
    qos_stop = freshen(ctx, sym, :_qos_stop)
    dirty = freshen(ctx, sym, :_dirty)
    push!(ctx.preamble, quote
        $sym = $ex
    end)
    prev_pos = freshen(ctx, sym, :_prev_pos)
    lvl_2 = virtualize(:($sym.lvl), Lvl, ctx, sym)
    VirtualRegularRLELevel(lvl_2, sym, Ti, Tp, shape, qos_fill, qos_stop, Vp, VLTi, VRTi, Lvl, prev_pos)
end
function lower(lvl::VirtualRegularRLELevel, ctx::AbstractCompiler, ::DefaultStyle)
    quote
        $RegularRLELevel{$(lvl.Ti), $(lvl.Tp), $(lvl.Vp), $(lvl.VLTi), $(lvl.VRTi), $(lvl.Lvl)}(
            $(ctx(lvl.lvl)),
            $(ctx(lvl.shape)),
            $(lvl.ex).ptr,
            $(lvl.ex).left,
            $(lvl.ex).right,
        )
    end
end

Base.summary(lvl::VirtualRegularRLELevel) = "RegularRLE($(summary(lvl.lvl)))"

function virtual_level_size(lvl::VirtualRegularRLELevel, ctx)
    ext = make_extent(lvl.Ti, literal(lvl.Ti(1.0)), lvl.shape)
    ext = similar_extent(ext, getstart(ext), call(-, getstop(ext), getunit(ext)))
    (virtual_level_size(lvl.lvl, ctx)..., ext)
end

function virtual_level_resize!(lvl::VirtualRegularRLELevel, ctx, dims...)
    lvl.shape = getstop(dims[end])
    lvl.lvl = virtual_level_resize!(lvl.lvl, ctx, dims[1:end-1]...)
    lvl
end


virtual_level_eltype(lvl::VirtualRegularRLELevel) = virtual_level_eltype(lvl.lvl)
virtual_level_default(lvl::VirtualRegularRLELevel) = virtual_level_default(lvl.lvl)

function declare_level!(lvl::VirtualRegularRLELevel, ctx::AbstractCompiler, pos, init)
    Tp = lvl.Tp
    Ti = lvl.Ti
    qos = call(-, call(getindex, :($(lvl.ex).ptr), call(+, pos, 1)), 1)
    push!(ctx.code.preamble, quote
        $(lvl.qos_fill) = $(Tp(0))
        $(lvl.qos_stop) = $(Tp(0))
    end)
    if issafe(ctx.mode)
        push!(ctx.code.preamble, quote
            $(lvl.prev_pos) = $(Tp(0))
        end)
    end
    lvl.lvl = declare_level!(lvl.lvl, ctx, qos, init)
    return lvl
end

function trim_level!(lvl::VirtualRegularRLELevel, ctx::AbstractCompiler, pos)
    qos = freshen(ctx.code, :qos)
    push!(ctx.code.preamble, quote
        resize!($(lvl.ex).ptr, $(ctx(pos)) + 1)
        $qos = $(lvl.ex).ptr[end] - $(lvl.Tp(1))
        resize!($(lvl.ex).left, $qos)
        resize!($(lvl.ex).right, $qos)
    end)
    lvl.lvl = trim_level!(lvl.lvl, ctx, value(qos, lvl.Tp))
    return lvl
end

function assemble_level!(lvl::VirtualRegularRLELevel, ctx, pos_start, pos_stop)
    pos_start = ctx(cache!(ctx, :p_start, pos_start))
    pos_stop = ctx(cache!(ctx, :p_start, pos_stop))
    return quote
        Finch.resize_if_smaller!($(lvl.ex).ptr, $pos_stop + 1)
        Finch.fill_range!($(lvl.ex).ptr, 0, $pos_start + 1, $pos_stop + 1)
    end
end

function freeze_level!(lvl::VirtualRegularRLELevel, ctx::AbstractCompiler, pos_stop)
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



function instantiate_reader(fbr::VirtualSubFiber{VirtualRegularRLELevel}, ctx, subprotos, ::Union{typeof(defaultread), typeof(walk)})
    (lvl, pos) = (fbr.lvl, fbr.pos) 
    tag = lvl.ex
    Tp = lvl.Tp
    Ti = lvl.Ti
    my_i_end = freshen(ctx.code, tag, :_i_end)
    my_i_stop = freshen(ctx.code, tag, :_i_stop)
    my_i_start = freshen(ctx.code, tag, :_i_start)
    my_q = freshen(ctx.code, tag, :_q)
    my_q_stop = freshen(ctx.code, tag, :_q_stop)

    Furlable(
        body = (ctx, ext) -> Thunk(
            preamble = quote
                $my_q = $(lvl.ex).ptr[$(ctx(pos))]
                $my_q_stop = $(lvl.ex).ptr[$(ctx(pos)) + $(Tp(1))]
                if $my_q < $my_q_stop
                    $my_i_end = $(lvl.ex).right[$my_q_stop - $(Tp(1))]
                else
                    $my_i_end = $(Ti(0))
                end
                $my_i_start = $(lvl.ex).start
                $my_i_stride = $(lvl.ex).stride
                $my_i_end = $(lvl.ex).length
            end,
            body = (ctx) -> Sequence([
                Phase(
                    start = (ctx, ext) -> literal(lvl.Ti(1)),
                    stop = (ctx, ext) -> call(-, value(my_i_start, lvl.Ti), getunit(ext)),
                    body = (ctx, ext) -> Run(Fill(virtual_level_default(lvl)))
                ),
                Phase(
                    start = (ctx, ext) -> value(my_i_start, lvl.Ti),
                    stop = (ctx, ext) -> call(-, value(my_i_end, lvl.Ti), getunit(ext)),
                    body = (ctx, ext) -> Stepper(
                        seek = (ctx, ext) -> quote
                          $my_q = floor(Int64, ($(ctx(getstart(ext))) - $my_i_start)/$my_i_stride )
                        end,
                        preamble = :($my_i = $my_q * $my_i_stride + $my_i_start),
                        stop = (ctx, ext) -> value(my_i),
                        body = (ctx, ext) -> instantiate_reader(VirtualSubFiber(lvl.lvl, value(my_q)), ctx, subprotos),
                        next = (ctx, ext) -> :($my_q += $(Tp(1))),
                    )
                ),
                Phase(
                    stop = (ctx, ext) -> call(-, lvl.shape, getunit(ext)),
                    body = (ctx, ext) -> Run(Fill(virtual_level_default(lvl)))
                )
            ])
        )
    )
end


instantiate_updater(fbr::VirtualSubFiber{VirtualRegularRLELevel}, ctx, protos) = 
    instantiate_updater(VirtualTrackedSubFiber(fbr.lvl, fbr.pos, freshen(ctx.code, :null)), ctx, protos)

function instantiate_updater(fbr::VirtualTrackedSubFiber{VirtualRegularRLELevel}, ctx, subprotos, ::Union{typeof(defaultupdate), typeof(extrude)})
    (lvl, pos) = (fbr.lvl, fbr.pos) 
    tag = lvl.ex
    Tp = lvl.Tp
    Ti = lvl.Ti
    qos = freshen(ctx.code, tag, :_qos)
    qos_fill = lvl.qos_fill
    qos_stop = lvl.qos_stop
    dirty = freshen(ctx.code, tag, :dirty)
    
    Furlable(
        body = (ctx, ext) -> Thunk(
            preamble = quote
                $qos = $qos_fill + 1
                $(if issafe(ctx.mode)
                    quote
                        $(lvl.prev_pos) < $(ctx(pos)) || throw(FinchProtocolError("RegularRLELevels cannot be updated multiple times"))
                    end
                end)
            end,

            body = (ctx) -> AcceptRun(
                body = (ctx, ext) -> Thunk(
                    preamble = quote
                        if $qos > $qos_stop
                            $qos_stop = max($qos_stop << 1, 1)
                            Finch.resize_if_smaller!($(lvl.ex).left, $qos_stop)
                            Finch.resize_if_smaller!($(lvl.ex).right, $qos_stop)
                            $(contain(ctx_2->assemble_level!(lvl.lvl, ctx_2, value(qos, lvl.Tp), value(qos_stop, lvl.Tp)), ctx))
                        end
                        $dirty = false
                    end,
                    body = (ctx) -> instantiate_updater(VirtualTrackedSubFiber(lvl.lvl, value(qos, lvl.Tp), dirty), ctx, subprotos),
                    epilogue = quote
                        if $dirty
                            $(fbr.dirty) = true
                            $(lvl.ex).left[$qos] = $(ctx(getstart(ext)))
                            $(lvl.ex).right[$qos] = $(ctx(getstop(ext)))
                            $(qos) += $(Tp(1))
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
                $(lvl.ex).ptr[$(ctx(pos)) + 1] = $qos - $qos_fill - 1
                $qos_fill = $qos - 1
            end
        )
    )
end
