"""
    SparseListLevel{[Ti=Int], [Tp=Int]}(lvl, [dim])

A subfiber of a sparse level does not need to represent slices `A[:, ..., :, i]`
which are entirely [`default`](@ref). Instead, only potentially non-default
slices are stored as subfibers in `lvl`.  A sorted list is used to record which
slices are stored. Optionally, `dim` is the size of the last dimension.

`Ti` is the type of the last fiber index, and `Tp` is the type used for
positions in the level.

In the [@fiber](@ref) constructor, `sl` is an alias for `SparseListLevel`.

```jldoctest
julia> @fiber(d(sl(e(0.0))), [10 0 20; 30 0 0; 0 0 40])
Dense [:,1:3]
├─[:,1]: SparseList (0.0) [1:3]
│ ├─[1]: 10.0
│ ├─[2]: 30.0
├─[:,2]: SparseList (0.0) [1:3]
├─[:,3]: SparseList (0.0) [1:3]
│ ├─[1]: 20.0
│ ├─[3]: 40.0

julia> @fiber(sl(sl(e(0.0))), [10 0 20; 30 0 0; 0 0 40])
SparseList (0.0) [:,1:3]
├─[:,1]: SparseList (0.0) [1:3]
│ ├─[1]: 10.0
│ ├─[2]: 30.0
├─[:,3]: SparseList (0.0) [1:3]
│ ├─[1]: 20.0
│ ├─[3]: 40.0

```
"""
struct SparseListLevel{Ti, Tp, Lvl}
    lvl::Lvl
    I::Ti
    ptr::Vector{Tp}
    idx::Vector{Ti}
end
const SparseList = SparseListLevel
SparseListLevel(lvl) = SparseListLevel{Int}(lvl)
SparseListLevel(lvl, I, args...) = SparseListLevel{typeof(I)}(lvl, I, args...)
SparseListLevel{Ti}(lvl, args...) where {Ti} = SparseListLevel{Ti, Int}(lvl, args...)
SparseListLevel{Ti, Tp}(lvl, args...) where {Ti, Tp} = SparseListLevel{Ti, Tp, typeof(lvl)}(lvl, args...)

SparseListLevel{Ti, Tp, Lvl}(lvl) where {Ti, Tp, Lvl} = SparseListLevel{Ti, Tp, Lvl}(lvl, zero(Ti))
SparseListLevel{Ti, Tp, Lvl}(lvl, I) where {Ti, Tp, Lvl} = 
    SparseListLevel{Ti, Tp, Lvl}(lvl, Ti(I), Tp[1], Ti[])

"""
`f_code(l)` = [SparseListLevel](@ref).
"""
f_code(::Val{:sl}) = SparseList
summary_f_code(lvl::SparseListLevel) = "sl($(summary_f_code(lvl.lvl)))"
similar_level(lvl::SparseListLevel) = SparseList(similar_level(lvl.lvl))
similar_level(lvl::SparseListLevel, dim, tail...) = SparseList(similar_level(lvl.lvl, tail...), dim)

pattern!(lvl::SparseListLevel{Ti}) where {Ti} = 
    SparseListLevel{Ti}(pattern!(lvl.lvl), lvl.I, lvl.ptr, lvl.idx)

function Base.show(io::IO, lvl::SparseListLevel{Ti, Tp}) where {Ti, Tp}
    if get(io, :compact, false)
        print(io, "SparseList(")
    else
        print(io, "SparseList{$Ti, $Tp}(")
    end
    show(io, lvl.lvl)
    print(io, ", ")
    show(IOContext(io, :typeinfo=>Ti), lvl.I)
    print(io, ", ")
    if get(io, :compact, false)
        print(io, "…")
    else
        show(IOContext(io, :typeinfo=>Vector{Tp}), lvl.ptr)
        print(io, ", ")
        show(IOContext(io, :typeinfo=>Vector{Ti}), lvl.idx)
    end
    print(io, ")")
end

function display_fiber(io::IO, mime::MIME"text/plain", fbr::SubFiber{<:SparseListLevel}, depth)
    p = fbr.pos
    crds = @view(fbr.lvl.idx[fbr.lvl.ptr[p]:fbr.lvl.ptr[p + 1] - 1])

    print_coord(io, crd) = show(io, crd)
    get_fbr(crd) = fbr(crd)

    print(io, "SparseList (", default(fbr), ") [", ":,"^(ndims(fbr) - 1), "1:", fbr.lvl.I, "]")
    display_fiber_data(io, mime, fbr, depth, 1, crds, print_coord, get_fbr)
end

@inline level_ndims(::Type{<:SparseListLevel{Ti, Tp, Lvl}}) where {Ti, Tp, Lvl} = 1 + level_ndims(Lvl)
@inline level_size(lvl::SparseListLevel) = (level_size(lvl.lvl)..., lvl.I)
@inline level_axes(lvl::SparseListLevel) = (level_axes(lvl.lvl)..., Base.OneTo(lvl.I))
@inline level_eltype(::Type{<:SparseListLevel{Ti, Tp, Lvl}}) where {Ti, Tp, Lvl} = level_eltype(Lvl)
@inline level_default(::Type{<:SparseListLevel{Ti, Tp, Lvl}}) where {Ti, Tp, Lvl} = level_default(Lvl)
data_rep_level(::Type{<:SparseListLevel{Ti, Tp, Lvl}}) where {Ti, Tp, Lvl} = SparseData(data_rep_level(Lvl))

(fbr::AbstractFiber{<:SparseListLevel})() = fbr
function (fbr::SubFiber{<:SparseListLevel{Ti}})(idxs...) where {Ti}
    isempty(idxs) && return fbr
    lvl = fbr.lvl
    p = fbr.pos
    r = searchsorted(@view(lvl.idx[lvl.ptr[p]:lvl.ptr[p + 1] - 1]), idxs[end])
    q = lvl.ptr[p] + first(r) - 1
    fbr_2 = SubFiber(lvl.lvl, q)
    length(r) == 0 ? default(fbr_2) : fbr_2(idxs[1:end-1]...)
end

mutable struct VirtualSparseListLevel
    lvl
    ex
    Ti
    Tp
    I
    qos_fill
    qos_stop
end
function virtualize(ex, ::Type{SparseListLevel{Ti, Tp, Lvl}}, ctx, tag=:lvl) where {Ti, Tp, Lvl}
    sym = ctx.freshen(tag)
    I = value(:($sym.I), Int)
    qos_fill = ctx.freshen(sym, :_qos_fill)
    qos_stop = ctx.freshen(sym, :_qos_stop)
    push!(ctx.preamble, quote
        $sym = $ex
    end)
    lvl_2 = virtualize(:($sym.lvl), Lvl, ctx, sym)
    VirtualSparseListLevel(lvl_2, sym, Ti, Tp, I, qos_fill, qos_stop)
end
function (ctx::Finch.LowerJulia)(lvl::VirtualSparseListLevel)
    quote
        $SparseListLevel{$(lvl.Ti)}(
            $(ctx(lvl.lvl)),
            $(ctx(lvl.I)),
            $(lvl.ex).ptr,
            $(lvl.ex).idx,
        )
    end
end

summary_f_code(lvl::VirtualSparseListLevel) = "sl($(summary_f_code(lvl.lvl)))"

function virtual_level_size(lvl::VirtualSparseListLevel, ctx)
    ext = Extent(literal(lvl.Ti(1)), lvl.I)
    (virtual_level_size(lvl.lvl, ctx)..., ext)
end

function virtual_level_resize!(lvl::VirtualSparseListLevel, ctx, dims...)
    lvl.I = getstop(dims[end])
    lvl.lvl = virtual_level_resize!(lvl.lvl, ctx, dims[1:end-1]...)
    lvl
end

virtual_level_eltype(lvl::VirtualSparseListLevel) = virtual_level_eltype(lvl.lvl)
virtual_level_default(lvl::VirtualSparseListLevel) = virtual_level_default(lvl.lvl)

function initialize_level!(lvl::VirtualSparseListLevel, ctx::LowerJulia, pos)
    Ti = lvl.Ti
    Tp = lvl.Tp
    qos = call(-, call(getindex, :($(lvl.ex).ptr), call(+, pos, 1)),  1)
    push!(ctx.preamble, quote
        $(lvl.qos_fill) = $(Tp(0))
        $(lvl.qos_stop) = $(Tp(0))
    end)
    lvl.lvl = initialize_level!(lvl.lvl, ctx, qos)
    return lvl
end

function trim_level!(lvl::VirtualSparseListLevel, ctx::LowerJulia, pos)
    qos = ctx.freshen(:qos)
    push!(ctx.preamble, quote
        resize!($(lvl.ex).ptr, $(ctx(pos)) + 1)
        $qos = $(lvl.ex).ptr[end] - $(lvl.Tp(1))
        resize!($(lvl.ex).idx, $qos)
    end)
    lvl.lvl = trim_level!(lvl.lvl, ctx, value(qos, lvl.Tp))
    return lvl
end

function assemble_level!(lvl::VirtualSparseListLevel, ctx, pos_start, pos_stop)
    pos_start = ctx(cache!(ctx, :p_start, pos_start))
    pos_stop = ctx(cache!(ctx, :p_start, pos_stop))
    return quote
        $resize_if_smaller!($(lvl.ex).ptr, $pos_stop + 1)
        $fill_range!($(lvl.ex).ptr, 0, $pos_start + 1, $pos_stop + 1)
    end
end

function freeze_level!(lvl::VirtualSparseListLevel, ctx::LowerJulia, pos_stop)
    p = ctx.freshen(:p)
    pos_stop = ctx(cache!(ctx, :pos_stop, simplify(pos_stop, ctx)))
    qos_stop = ctx.freshen(:qos_stop)
    push!(ctx.preamble, quote
        for $p = 2:($pos_stop + 1)
            $(lvl.ex).ptr[$p] += $(lvl.ex).ptr[$p - 1]
        end
        $qos_stop = $(lvl.ex).ptr[$pos_stop + 1] - 1
    end)
    lvl.lvl = freeze_level!(lvl.lvl, ctx, value(qos_stop))
    return lvl
end

function get_reader(fbr::VirtualSubFiber{VirtualSparseListLevel}, ctx, ::Union{Nothing, Walk}, protos...)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    Tp = lvl.Tp
    Ti = lvl.Ti
    my_i = ctx.freshen(tag, :_i)
    my_q = ctx.freshen(tag, :_q)
    my_q_stop = ctx.freshen(tag, :_q_stop)
    my_i1 = ctx.freshen(tag, :_i1)

    Furlable(
        size = virtual_level_size(lvl, ctx),
        body = (ctx, idx, ext) -> Thunk(
            preamble = quote
                $my_q = $(lvl.ex).ptr[$(ctx(pos))]
                $my_q_stop = $(lvl.ex).ptr[$(ctx(pos)) + $(Tp(1))]
                if $my_q < $my_q_stop
                    $my_i = $(lvl.ex).idx[$my_q]
                    $my_i1 = $(lvl.ex).idx[$my_q_stop - $(Tp(1))]
                else
                    $my_i = $(Ti(1))
                    $my_i1 = $(Ti(0))
                end
            end,
            body = Pipeline([
                Phase(
                    stride = (ctx, idx, ext) -> value(my_i1),
                    body = (start, step) -> Stepper(
                        seek = (ctx, ext) -> quote
                            while $my_q + $(Tp(1)) < $my_q_stop && $(lvl.ex).idx[$my_q] < $(ctx(getstart(ext)))
                                $my_q += $(Tp(1))
                            end
                        end,
                        body = Thunk(
                            preamble = quote
                                $my_i = $(lvl.ex).idx[$my_q]
                            end,
                            body = Step(
                                stride = (ctx, idx, ext) -> value(my_i),
                                chunk = Spike(
                                    body = Simplify(Fill(virtual_level_default(lvl))),
                                    tail = get_reader(VirtualSubFiber(lvl.lvl, value(my_q, Ti)), ctx, protos...)
                                ),
                                next = (ctx, idx, ext) -> quote
                                    $my_q += $(Tp(1))
                                end
                            )
                        )
                    )
                ),
                Phase(
                    body = (start, step) -> Run(Simplify(Fill(virtual_level_default(lvl))))
                )
            ])
        )
    )
end

function get_reader(fbr::VirtualSubFiber{VirtualSparseListLevel}, ctx, ::FastWalk, protos...)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    Tp = lvl.Tp
    Ti = lvl.Ti
    my_i = ctx.freshen(tag, :_i)
    my_q = ctx.freshen(tag, :_q)
    my_q_stop = ctx.freshen(tag, :_q_stop)
    my_i1 = ctx.freshen(tag, :_i1)

    Furlable(
        size = virtual_level_size(lvl, ctx),
        body = (ctx, idx, ext) -> Thunk(
            preamble = quote
                $my_q = $(lvl.ex).ptr[$(ctx(pos))]
                $my_q_stop = $(lvl.ex).ptr[$(ctx(pos)) + $(Tp(1))]
                if $my_q < $my_q_stop
                    $my_i = $(lvl.ex).idx[$my_q]
                    $my_i1 = $(lvl.ex).idx[$my_q_stop - $(Tp(1))]
                else
                    $my_i = $(Ti(1))
                    $my_i1 = $(Ti(0))
                end
            end,
            body = Pipeline([
                Phase(
                    stride = (ctx, idx, ext) -> value(my_i1),
                    body = (start, step) -> Stepper(
                        seek = (ctx, ext) -> quote
                            $my_q = $Tp(searchsortedfirst($(lvl.ex).idx, Int($(ctx(getstart(ext)))), Int($my_q), Int($my_q_stop - 1), Base.Forward))
                        end,
                        body = Thunk(
                            preamble = :(
                                $my_i = $(lvl.ex).idx[$my_q]
                            ),
                            body = Step(
                                stride = (ctx, idx, ext) -> value(my_i),
                                chunk = Spike(
                                    body = Simplify(Fill(virtual_level_default(lvl))),
                                    tail = get_reader(VirtualSubFiber(lvl.lvl, value(my_q, Ti)), ctx, protos...),
                                ),
                                next = (ctx, idx, ext) -> quote
                                    $my_q += $(Tp(1))
                                end
                            )
                        )
                    )
                ),
                Phase(
                    body = (start, step) -> Run(Simplify(Fill(virtual_level_default(lvl))))
                )
            ])
        )
    )
end

function get_reader(fbr::VirtualSubFiber{VirtualSparseListLevel}, ctx, ::Gallop, protos...)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    Tp = lvl.Tp
    Ti = lvl.Ti
    my_i = ctx.freshen(tag, :_i)
    my_q = ctx.freshen(tag, :_q)
    my_q_stop = ctx.freshen(tag, :_q_stop)
    my_i1 = ctx.freshen(tag, :_i1)

    Furlable(
        size = virtual_level_size(lvl, ctx),
        body = (ctx, idx, ext) -> Thunk(
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
            body = Pipeline([
                Phase(
                    stride = (ctx, idx, ext) -> value(my_i1),
                    body = (start, step) -> Jumper(
                        body = Thunk(
                            body = Jump(
                                seek = (ctx, ext) -> quote
                                    while $my_q + $(Tp(1)) < $my_q_stop && $(lvl.ex).idx[$my_q] < $(ctx(getstart(ext)))
                                        $my_q += $(Tp(1))
                                    end
                                    $my_i = $(lvl.ex).idx[$my_q]
                                end,
                                stride = (ctx, ext) -> value(my_i),
                                body = (ctx, ext, ext_2) -> Switch([
                                    value(:($(ctx(getstop(ext_2))) == $my_i)) => Thunk(
                                        body = Spike(
                                            body = Simplify(Fill(virtual_level_default(lvl))),
                                            tail = get_reader(VirtualSubFiber(lvl.lvl, value(my_q, Ti)), ctx, protos...),
                                        ),
                                        epilogue = quote
                                            $my_q += $(Tp(1))
                                        end
                                    ),
                                    literal(true) => Stepper(
                                        seek = (ctx, ext) -> quote
                                            while $my_q + $(Tp(1)) < $my_q_stop && $(lvl.ex).idx[$my_q] < $(ctx(getstart(ext)))
                                                $my_q += $(Tp(1))
                                            end
                                        end,
                                        body = Thunk(
                                            preamble = :(
                                                $my_i = $(lvl.ex).idx[$my_q]
                                            ),
                                            body = Step(
                                                stride = (ctx, idx, ext) -> value(my_i),
                                                chunk = Spike(
                                                    body = Simplify(Fill(virtual_level_default(lvl))),
                                                    tail =  get_reader(VirtualSubFiber(lvl.lvl, value(my_q, Ti)), ctx, protos...),
                                                ),
                                                next = (ctx, idx, ext) -> quote
                                                    $my_q += $(Tp(1))
                                                end
                                            )
                                        )
                                    ),
                                ])
                            )
                        )
                    )
                ),
                Phase(
                    body = (start, step) -> Run(Simplify(Fill(virtual_level_default(lvl))))
                )
            ])
        )
    )
end

get_updater(fbr::VirtualSubFiber{VirtualSparseListLevel}, ctx, protos...) =
    get_updater(VirtualTrackedSubFiber(fbr.lvl, fbr.pos, ctx.freshen(:null)), ctx, protos...)
function get_updater(fbr::VirtualTrackedSubFiber{VirtualSparseListLevel}, ctx, ::Union{Nothing, Extrude}, protos...)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    Tp = lvl.Tp
    qos = ctx.freshen(tag, :_qos)
    qos_fill = lvl.qos_fill
    qos_stop = lvl.qos_stop
    dirty = ctx.freshen(tag, :dirty)

    Furlable(
        val = virtual_level_default(lvl),
        size = virtual_level_size(lvl, ctx),
        body = (ctx, idx, ext) -> Thunk(
            preamble = quote
                $qos = $qos_fill + 1
            end,
            body = AcceptSpike(
                val = virtual_level_default(lvl),
                tail = (ctx, idx) -> Thunk(
                    preamble = quote
                        if $qos > $qos_stop
                            $qos_stop = max($qos_stop << 1, 1)
                            $resize_if_smaller!($(lvl.ex).idx, $qos_stop)
                            $(contain(ctx_2->assemble_level!(lvl.lvl, ctx_2, value(qos, lvl.Tp), value(qos_stop, lvl.Tp)), ctx))
                        end
                        $dirty = false
                    end,
                    body = get_updater(VirtualTrackedSubFiber(lvl.lvl, value(qos, lvl.Tp), dirty), ctx, protos...),
                    epilogue = quote
                        if $dirty
                            $(fbr.dirty) = true
                            $(lvl.ex).idx[$qos] = $(ctx(idx))
                            $qos += $(Tp(1))
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