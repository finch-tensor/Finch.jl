"""
    SingleListLevel{[Ti=Int], [Tp=Int], [Vp=Vector{Tp}], [Vi=Vector{Ti}]}(lvl, [dim])

A subfiber of a sparse level does not need to represent slices `A[:, ..., :, i]`
which are entirely [`default`](@ref). Instead, only potentially non-default
slices are stored as subfibers in `lvl`.  A sorted list is used to record which
slices are stored. Optionally, `dim` is the size of the last dimension.

`Ti` is the type of the last fiber index, and `Tp` is the type used for
positions in the level. The types `Vp` and `Vi` are the types of the
arrays used to store positions and indicies. 

In the [`Fiber!`](@ref) constructor, `sl` is an alias for `SingleListLevel`.

```jldoctest
julia> Fiber!(Dense(SingleList(Element(0.0))), [10 0 20; 30 0 0; 0 0 40])
Dense [:,1:3]
├─[:,1]: SingleList (0.0) [1:3]
│ ├─[1]: 10.0
│ ├─[2]: 30.0
├─[:,2]: SingleList (0.0) [1:3]
├─[:,3]: SingleList (0.0) [1:3]
│ ├─[1]: 20.0
│ ├─[3]: 40.0

julia> Fiber!(SingleList(SingleList(Element(0.0))), [10 0 20; 30 0 0; 0 0 40])
SingleList (0.0) [:,1:3]
├─[:,1]: SingleList (0.0) [1:3]
│ ├─[1]: 10.0
│ ├─[2]: 30.0
├─[:,3]: SingleList (0.0) [1:3]
│ ├─[1]: 20.0
│ ├─[3]: 40.0

```
"""
struct SingleListLevel{Ti, Tp, Vp, Vi, Lvl}
    lvl::Lvl
    shape::Ti
    ptr::Vp
    idx::Vi
end
const SingleList = SingleListLevel
SingleListLevel(lvl::Lvl) where {Lvl} = SingleListLevel{Int}(lvl)
SingleListLevel(lvl, shape, args...) = SingleListLevel{typeof(shape)}(lvl, shape, args...)
SingleListLevel{Ti}(lvl, shape, ptr, idx) where {Ti} = SingleListLevel{Ti, eltype(ptr)}(lvl, shape, ptr, idx)
SingleListLevel{Ti}(lvl, args...) where {Ti} = SingleListLevel{Ti, postype(typeof(lvl))}(lvl, args...)
SingleListLevel{Ti, Tp}(lvl, args...) where {Ti, Tp} =
    SingleListLevel{Ti, Tp, memtype(typeof(lvl)){Tp, 1}, memtype(typeof(lvl)){Ti, 1}, typeof(lvl)}(lvl, args...)

SingleListLevel{Ti, Tp, Vp, Vi, Lvl}(lvl) where {Ti, Tp, Vp, Vi, Lvl} = SingleListLevel{Ti, Tp, Vp, Vi, Lvl}(lvl, zero(Ti))
SingleListLevel{Ti, Tp, Vp, Vi, Lvl}(lvl, shape) where {Ti, Tp, Vp, Vi, Lvl} = 
    SingleListLevel{Ti, Tp, Vp, Vi, Lvl}(lvl, Ti(shape), Tp[1], Ti[])

SingleListLevel{Ti, Tp, Lvl}(lvl, shape, ptr, idx) where {Ti, Tp, Lvl} = 
    SingleListLevel{Ti, Tp, typeof(ptr), typeof(idx), Lvl}(lvl, Ti(shape), ptr, idx)
    
Base.summary(lvl::SingleListLevel) = "SingleList($(summary(lvl.lvl)))"
similar_level(lvl::SingleListLevel) = SingleList(similar_level(lvl.lvl))
similar_level(lvl::SingleListLevel, dim, tail...) = SingleList(similar_level(lvl.lvl, tail...), dim)

function memtype(::Type{SingleListLevel{Ti, Tp,  Vp, Vi, Lvl}}) where {Ti, Tp, Lvl, Vi, Vp}
    return containertype(Vp)
end

function postype(::Type{SingleListLevel{Ti, Tp,  Vp, Vi, Lvl}}) where {Ti, Tp, Lvl, Vi, Vp}
    return Tp
end

function moveto(lvl::SingleListLevel{Ti, Tp, Vp, Vi, Lvl}, ::Type{MemType}) where {Ti, Tp, Lvl, Vi, Vp, MemType <: AbstractArray}
    lvl_2 = moveto(lvl.lvl, MemType)
    ptr_2 = MemType{Tp, 1}(lvl.ptr)
    idx_2 = MemType{Ti, 1}(lvl.idx)
    return SingleListLevel{Ti, Tp, typeof(ptr_2), typeof(idx_2), typeof(lvl_2)}(lvl_2, lvl.shape, ptr_2, idx_2)
end



function countstored_level(lvl::SingleListLevel, pos)
    countstored_level(lvl.lvl, lvl.ptr[pos + 1] - 1)
end

pattern!(lvl::SingleListLevel{Ti, Tp}) where {Ti, Tp} = 
    SingleListLevel{Ti, Tp}(pattern!(lvl.lvl), lvl.shape, lvl.ptr, lvl.idx)

redefault!(lvl::SingleListLevel{Ti, Tp}, init) where {Ti, Tp} = 
    SingleListLevel{Ti, Tp}(redefault!(lvl.lvl, init), lvl.shape, lvl.ptr, lvl.idx)

function Base.show(io::IO, lvl::SingleListLevel{Ti, Tp,  Vp, Vi, Lvl}) where {Ti, Tp, Lvl, Vi, Vp}
    if get(io, :compact, false)
        print(io, "SingleList(")
    else
        print(io, "SingleList{$Ti, $Tp}(")
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
        show(IOContext(io, :typeinfo=>Vi), lvl.idx)
    end
    print(io, ")")
end

function display_fiber(io::IO, mime::MIME"text/plain", fbr::SubFiber{<:SingleListLevel}, depth)
    p = fbr.pos
    crds = @view(fbr.lvl.idx[fbr.lvl.ptr[p]:fbr.lvl.ptr[p + 1] - 1])

    print_coord(io, crd) = show(io, crd)
    get_fbr(crd) = fbr(crd)

    print(io, "SingleList (", default(fbr), ") [", ":,"^(ndims(fbr) - 1), "1:", fbr.lvl.shape, "]")
    display_fiber_data(io, mime, fbr, depth, 1, crds, print_coord, get_fbr)
end

@inline level_ndims(::Type{<:SingleListLevel{Ti, Tp,  Vp, Vi, Lvl}}) where {Ti, Tp,  Vp, Vi, Lvl} = 1 + level_ndims(Lvl)
@inline level_size(lvl::SingleListLevel) = (level_size(lvl.lvl)..., lvl.shape)
@inline level_axes(lvl::SingleListLevel) = (level_axes(lvl.lvl)..., Base.OneTo(lvl.shape))
@inline level_eltype(::Type{<:SingleListLevel{Ti, Tp,  Vp, Vi, Lvl}}) where {Ti, Tp,  Vp, Vi, Lvl} = level_eltype(Lvl)
@inline level_default(::Type{<:SingleListLevel{Ti, Tp,  Vp, Vi, Lvl}}) where {Ti, Tp,  Vp, Vi, Lvl} = level_default(Lvl)
data_rep_level(::Type{<:SingleListLevel{Ti, Tp,  Vp, Vi, Lvl}}) where {Ti, Tp,  Vp, Vi, Lvl} = SparseData(data_rep_level(Lvl))

(fbr::AbstractFiber{<:SingleListLevel})() = fbr
function (fbr::SubFiber{<:SingleListLevel{Ti}})(idxs...) where {Ti}
    isempty(idxs) && return fbr
    lvl = fbr.lvl
    p = fbr.pos
    r = searchsorted(@view(lvl.idx[lvl.ptr[p]:lvl.ptr[p + 1] - 1]), idxs[end])
    q = lvl.ptr[p] + first(r) - 1
    fbr_2 = SubFiber(lvl.lvl, q)
    length(r) == 0 ? default(fbr_2) : fbr_2(idxs[1:end-1]...)
end

mutable struct VirtualSingleListLevel <: AbstractVirtualLevel
    lvl
    ex
    Ti
    Tp
    shape
    qos_fill
    qos_stop
    prev_pos
end
  
is_level_injective(lvl::VirtualSingleListLevel, ctx) = [is_level_injective(lvl.lvl, ctx)..., false]
is_level_concurrent(lvl::VirtualSingleListLevel, ctx) = [is_level_concurrent(lvl.lvl, ctx)..., false]
is_level_atomic(lvl::VirtualSingleListLevel, ctx) = false

  

function virtualize(ex, ::Type{SingleListLevel{Ti, Tp,  Vp, Vi, Lvl}}, ctx, tag=:lvl) where {Ti, Tp, Lvl, Vi, Vp}
    sym = freshen(ctx, tag)
    shape = value(:($sym.shape), Int)
    qos_fill = freshen(ctx, sym, :_qos_fill)
    qos_stop = freshen(ctx, sym, :_qos_stop)
    push!(ctx.preamble, quote
        $sym = $ex
    end)
    prev_pos = freshen(ctx, sym, :_prev_pos)
    lvl_2 = virtualize(:($sym.lvl), Lvl, ctx, sym)
    VirtualSingleListLevel(lvl_2, sym, Ti, Tp, shape, qos_fill, qos_stop, prev_pos)
end
function lower(lvl::VirtualSingleListLevel, ctx::AbstractCompiler, ::DefaultStyle)
    quote
        $SingleListLevel{$(lvl.Ti), $(lvl.Tp)}(
            $(ctx(lvl.lvl)),
            $(ctx(lvl.shape)),
            $(lvl.ex).ptr,
            $(lvl.ex).idx,
        )
    end
end

Base.summary(lvl::VirtualSingleListLevel) = "SingleList($(summary(lvl.lvl)))"

function virtual_level_size(lvl::VirtualSingleListLevel, ctx)
    ext = make_extent(lvl.Ti, literal(lvl.Ti(1)), lvl.shape)
    (virtual_level_size(lvl.lvl, ctx)..., ext)
end

function virtual_level_resize!(lvl::VirtualSingleListLevel, ctx, dims...)
    lvl.shape = getstop(dims[end])
    lvl.lvl = virtual_level_resize!(lvl.lvl, ctx, dims[1:end-1]...)
    lvl
end

virtual_level_eltype(lvl::VirtualSingleListLevel) = virtual_level_eltype(lvl.lvl)
virtual_level_default(lvl::VirtualSingleListLevel) = virtual_level_default(lvl.lvl)

function declare_level!(lvl::VirtualSingleListLevel, ctx::AbstractCompiler, pos, init)
    #TODO check that init == default
    Ti = lvl.Ti
    Tp = lvl.Tp
    qos = call(-, call(getindex, :($(lvl.ex).ptr), call(+, pos, 1)),  1)
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

function trim_level!(lvl::VirtualSingleListLevel, ctx::AbstractCompiler, pos)
    qos = freshen(ctx.code, :qos)
    push!(ctx.code.preamble, quote
        resize!($(lvl.ex).ptr, $(ctx(pos)) + 1)
        $qos = $(lvl.ex).ptr[end] - $(lvl.Tp(1))
        resize!($(lvl.ex).idx, $qos)
    end)
    lvl.lvl = trim_level!(lvl.lvl, ctx, value(qos, lvl.Tp))
    return lvl
end

function assemble_level!(lvl::VirtualSingleListLevel, ctx, pos_start, pos_stop)
    pos_start = ctx(cache!(ctx, :p_start, pos_start))
    pos_stop = ctx(cache!(ctx, :p_start, pos_stop))
    return quote
        Finch.resize_if_smaller!($(lvl.ex).ptr, $pos_stop + 1)
        Finch.fill_range!($(lvl.ex).ptr, 0, $pos_start + 1, $pos_stop + 1)
    end
end

function freeze_level!(lvl::VirtualSingleListLevel, ctx::AbstractCompiler, pos_stop)
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

function instantiate_reader(fbr::VirtualSubFiber{VirtualSingleListLevel}, ctx, subprotos, ::Union{typeof(defaultread), typeof(walk)})
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    Tp = lvl.Tp
    Ti = lvl.Ti
    my_i = freshen(ctx.code, tag, :_i)
    my_q = freshen(ctx.code, tag, :_q)
    my_q_stop = freshen(ctx.code, tag, :_q_stop)
    my_i1 = freshen(ctx.code, tag, :_i1)

    Furlable(
        body = (ctx, ext) -> Thunk(
            preamble = quote
                $my_q = $(lvl.ex).ptr[$(ctx(pos))]
                $my_q_stop = $(lvl.ex).ptr[$(ctx(pos)) + $(Tp(1))]
                if $my_q < $my_q_stop
                    $my_i = $(lvl.ex).idx[$my_q]
                else
                    $my_i = $(Ti(0))
                end
            end,
            body = (ctx) -> Sequence([
                Phase(
                    start = (ctx, ext) -> literal(lvl.Ti(1)),
                    stop = (ctx, ext) -> call(-, value(my_i, lvl.Ti), getunit(ext)),
                    body = (ctx, ext) -> Run(Fill(virtual_level_default(lvl)))
                ),
                Phase(
                    stop = (ctx, ext) -> value(my_i),
                    body = (ctx, ext) -> Run(Simplify(instantiate_reader(VirtualSubFiber(lvl.lvl, value(my_q)), ctx, subprotos))),
                ),
                Phase(
                    stop = (ctx, ext) -> lvl.shape,
                    body = (ctx, ext) -> Run(Fill(virtual_level_default(lvl)))
                )
            ])
        )
    )
end

function instantiate_reader(fbr::VirtualSubFiber{VirtualSingleListLevel}, ctx, subprotos, ::typeof(gallop))
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    Tp = lvl.Tp
    Ti = lvl.Ti
    my_i = freshen(ctx.code, tag, :_i)
    my_q = freshen(ctx.code, tag, :_q)
    my_q_stop = freshen(ctx.code, tag, :_q_stop)
    my_i1 = freshen(ctx.code, tag, :_i1)
    my_i2 = freshen(ctx.code, tag, :_i2)
    my_i3 = freshen(ctx.code, tag, :_i3)
    my_i4 = freshen(ctx.code, tag, :_i4)

    Furlable(
        body = (ctx, ext) -> Thunk(
            preamble = quote
                $my_q = $(lvl.ex).ptr[$(ctx(pos))]
                $my_q_stop = $(lvl.ex).ptr[$(ctx(pos)) + 1]
                if $my_q < $my_q_stop
                    $my_i = $(lvl.ex).idx[$my_q]
                    $my_i1 = $(lvl.ex).idx[$my_q_stop - $(Tp(1))]
                else
                    $my_i = $(Ti(1))
                    $my_i1 = $(Ti(0))
                end
            end,
            body = (ctx) -> Sequence([
                Phase(
                    start = (ctx, ext) -> literal(lvl.Ti(1)),
                    stop = (ctx, ext) -> value(my_i1),
                    body = (ctx, ext) -> Jumper(
                        seek = (ctx, ext) -> quote
                            if $(lvl.ex).idx[$my_q] < $(ctx(getstart(ext)))
                                $my_q = Finch.scansearch($(lvl.ex).idx, $(ctx(getstart(ext))), $my_q, $my_q_stop - 1)
                            end
                        end,                        
                        preamble = :($my_i2 = $(lvl.ex).idx[$my_q]),
                        stop = (ctx, ext) -> value(my_i2),
                        chunk =  Spike(
                            body = Fill(virtual_level_default(lvl)),
                            tail = instantiate_reader(VirtualSubFiber(lvl.lvl, value(my_q, Ti)), ctx, subprotos),
                        ),
                        next = (ctx, ext) -> :($my_q += $(Tp(1))),
                    )  
                ),
                Phase(
                    body = (ctx, ext) -> Run(Fill(virtual_level_default(lvl)))
                )
            ])
        )
    )
end

instantiate_updater(fbr::VirtualSubFiber{VirtualSingleListLevel}, ctx, protos) = begin
    instantiate_updater(VirtualTrackedSubFiber(fbr.lvl, fbr.pos, freshen(ctx.code, :null)), ctx, protos)
end
function instantiate_updater(fbr::VirtualTrackedSubFiber{VirtualSingleListLevel}, ctx, subprotos, ::Union{typeof(defaultupdate), typeof(extrude)})
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    Tp = lvl.Tp
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
                        $(lvl.prev_pos) < $(ctx(pos)) || throw(FinchProtocolError("SingleListLevels cannot be updated multiple times"))
                    end
                end)
            end,
            body = (ctx) -> Lookup(
                body = (ctx, idx) -> Thunk(
                    preamble = quote
                        if $qos > $qos_stop
                            $qos_stop = max($qos_stop << 1, 1)
                            Finch.resize_if_smaller!($(lvl.ex).idx, $qos_stop)
                            $(contain(ctx_2->assemble_level!(lvl.lvl, ctx_2, value(qos, lvl.Tp), value(qos_stop, lvl.Tp)), ctx))
                        end
                        $dirty = false
                    end,
                    body = (ctx) -> instantiate_updater(VirtualTrackedSubFiber(lvl.lvl, value(qos, lvl.Tp), dirty), ctx, subprotos),
                    epilogue = quote
                        if $dirty
                            $(fbr.dirty) = true
                            $(lvl.ex).idx[$qos] = $(ctx(idx))
                            $qos += $(Tp(1))
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
