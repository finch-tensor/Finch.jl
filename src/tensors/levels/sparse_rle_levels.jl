"""
    SparseRunListLevel{[Ti=Int], [Ptr, Left, Right]}(lvl, [dim]; [merge = true])

The SparseRunListLevel represent runs of equivalent slices `A[:, ..., :, i]`
which are not entirely [`fill_value`](@ref). A sorted list is used to record the
left and right endpoints of each run. Optionally, `dim` is the size of the last dimension.

`Ti` is the type of the last tensor index, and `Tp` is the type used for
positions in the level. The types `Ptr`, `Left`, and `Right` are the types of the
arrays used to store positions and endpoints.

The `merge` keyword argument is used to specify whether the level should merge
duplicate consecutive runs.

```jldoctest
julia> tensor_tree(Tensor(Dense(SparseRunListLevel(Element(0.0))), [10 0 20; 30 0 0; 0 0 40]))
3×3-Tensor
└─ Dense [:,1:3]
   ├─ [:, 1]: SparseRunList (0.0) [1:3]
   │  ├─ [1:1]: 10.0
   │  └─ [2:2]: 30.0
   ├─ [:, 2]: SparseRunList (0.0) [1:3]
   └─ [:, 3]: SparseRunList (0.0) [1:3]
      ├─ [1:1]: 20.0
      └─ [3:3]: 40.0
```
"""
struct SparseRunListLevel{
    Ti,Ptr<:AbstractVector,Left<:AbstractVector,Right<:AbstractVector,merge,Lvl
} <: AbstractLevel
    lvl::Lvl
    shape::Ti
    ptr::Ptr
    left::Left
    right::Right
    buf::Lvl
end

const SparseRunList = SparseRunListLevel
SparseRunListLevel(lvl; kwargs...) = SparseRunListLevel{Int}(lvl; kwargs...)
function SparseRunListLevel(lvl, shape, args...; kwargs...)
    SparseRunListLevel{typeof(shape)}(lvl, shape, args...; kwargs...)
end
function SparseRunListLevel{Ti}(lvl; kwargs...) where {Ti}
    SparseRunListLevel(lvl, zero(Ti); kwargs...)
end
function SparseRunListLevel{Ti}(lvl, shape; kwargs...) where {Ti}
    SparseRunListLevel{Ti}(
        lvl, shape, postype(lvl)[1], Ti[], Ti[], deepcopy(lvl); kwargs...
    )
end #TODO if similar_level could return the same type, we could use it here
function SparseRunListLevel{Ti}(
    lvl::Lvl, shape, ptr::Ptr, left::Left, right::Right, buf::Lvl; merge=true
) where {Ti,Lvl,Ptr,Left,Right}
    SparseRunListLevel{Ti,Ptr,Left,Right,merge,Lvl}(lvl, Ti(shape), ptr, left, right, buf)
end

function getmerge(
    lvl::SparseRunListLevel{Ti,Ptr,Left,Right,merge}
) where {Ti,Ptr,Left,Right,merge}
    merge
end

Base.summary(lvl::SparseRunListLevel) = "SparseRunList($(summary(lvl.lvl)))"
function similar_level(lvl::SparseRunListLevel, fill_value, eltype::Type, dim, tail...)
    SparseRunList(similar_level(lvl.lvl, fill_value, eltype, tail...), dim)
end

function postype(
    ::Type{SparseRunListLevel{Ti,Ptr,Left,Right,merge,Lvl}}
) where {Ti,Ptr,Left,Right,merge,Lvl}
    return postype(Lvl)
end

function transfer(lvl::SparseRunListLevel{Ti}, device, style) where {Ti}
    lvl_2 = transfer(lvl.lvl, device, style)
    ptr = transfer(lvl.ptr, device, style)
    left = transfer(lvl.left, device, style)
    right = transfer(lvl.right, device, style)
    buf = transfer(lvl.buf, device, style)
    return SparseRunListLevel{Ti}(
        lvl_2, lvl.shape, lvl.ptr, lvl.left, lvl.right, lvl.buf; merge=getmerge(lvl)
    )
end

function pattern!(lvl::SparseRunListLevel{Ti}) where {Ti}
    SparseRunListLevel{Ti}(
        pattern!(lvl.lvl),
        lvl.shape,
        lvl.ptr,
        lvl.left,
        lvl.right,
        pattern!(lvl.buf);
        merge=getmerge(lvl),
    )
end

function countstored_level(lvl::SparseRunListLevel, pos)
    countstored_level(lvl.lvl, lvl.ptr[pos + 1] - 1)
end

function set_fill_value!(lvl::SparseRunListLevel{Ti}, init) where {Ti}
    SparseRunListLevel{Ti}(
        set_fill_value!(lvl.lvl, init),
        lvl.shape,
        lvl.ptr,
        lvl.left,
        lvl.right,
        set_fill_value!(lvl.buf, init);
        merge=getmerge(lvl),
    )
end

function Base.resize!(lvl::SparseRunListLevel{Ti}, dims...) where {Ti}
    SparseRunListLevel{Ti}(
        resize!(lvl.lvl, dims[1:(end - 1)]...),
        dims[end],
        lvl.ptr,
        lvl.left,
        lvl.right,
        resize!(lvl.buf, dims[1:(end - 1)]...);
        merge=getmerge(lvl),
    )
end

function Base.show(
    io::IO, lvl::SparseRunListLevel{Ti,Ptr,Left,Right,merge,Lvl}
) where {Ti,Ptr,Left,Right,merge,Lvl}
    if get(io, :compact, false)
        print(io, "SparseRunList(")
    else
        print(io, "SparseRunList{$Ti}(")
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
        show(io, lvl.left)
        print(io, ", ")
        show(io, lvl.right)
        print(io, ", ")
        show(io, lvl.buf)
        print(io, "; merge =")
        show(io, merge)
    end
    print(io, ")")
end

function labelled_show(io::IO, fbr::SubFiber{<:SparseRunListLevel})
    print(
        io,
        "SparseRunList (",
        fill_value(fbr),
        ") [",
        ":,"^(ndims(fbr) - 1),
        "1:",
        size(fbr)[end],
        "]",
    )
end

function labelled_children(fbr::SubFiber{<:SparseRunListLevel})
    lvl = fbr.lvl
    pos = fbr.pos
    pos + 1 > length(lvl.ptr) && return []
    map(lvl.ptr[pos]:(lvl.ptr[pos + 1] - 1)) do qos
        LabelledTree(
            cartesian_label(
                [range_label() for _ in 1:(ndims(fbr) - 1)]...,
                range_label(lvl.left[qos], lvl.right[qos]),
            ),
            SubFiber(lvl.lvl, qos),
        )
    end
end

@inline level_ndims(
    ::Type{<:SparseRunListLevel{Ti,Ptr,Left,Right,merge,Lvl}}
) where {Ti,Ptr,Left,Right,merge,Lvl} = 1 + level_ndims(Lvl)
@inline level_size(lvl::SparseRunListLevel) = (level_size(lvl.lvl)..., lvl.shape)
@inline level_axes(lvl::SparseRunListLevel) =
    (level_axes(lvl.lvl)..., Base.OneTo(lvl.shape))
@inline level_eltype(
    ::Type{<:SparseRunListLevel{Ti,Ptr,Left,Right,merge,Lvl}}
) where {Ti,Ptr,Left,Right,merge,Lvl} = level_eltype(Lvl)
@inline level_fill_value(
    ::Type{<:SparseRunListLevel{Ti,Ptr,Left,Right,merge,Lvl}}
) where {Ti,Ptr,Left,Right,merge,Lvl} = level_fill_value(Lvl)
function data_rep_level(
    ::Type{<:SparseRunListLevel{Ti,Ptr,Left,Right,merge,Lvl}}
) where {Ti,Ptr,Left,Right,merge,Lvl}
    SparseData(data_rep_level(Lvl))
end

function isstructequal(a::T, b::T) where {T<:SparseRunList}
    a.shape == b.shape &&
        a.ptr == b.ptr &&
        a.left == b.left &&
        a.right == b.right &&
        isstructequal(a.lvl, b.lvl)
end

(fbr::AbstractFiber{<:SparseRunListLevel})() = fbr
function (fbr::SubFiber{<:SparseRunListLevel})(idxs...)
    isempty(idxs) && return fbr
    lvl = fbr.lvl
    p = fbr.pos
    r1 = searchsortedlast(@view(lvl.left[lvl.ptr[p]:(lvl.ptr[p + 1] - 1)]), idxs[end])
    r2 = searchsortedfirst(@view(lvl.right[lvl.ptr[p]:(lvl.ptr[p + 1] - 1)]), idxs[end])
    q = lvl.ptr[p] + first(r1) - 1
    fbr_2 = SubFiber(lvl.lvl, q)
    r1 != r2 ? fill_value(fbr_2) : fbr_2(idxs[1:(end - 1)]...)
end

mutable struct VirtualSparseRunListLevel <: AbstractVirtualLevel
    tag
    lvl
    Ti
    shape
    qos_fill
    qos_stop
    ptr
    left
    right
    buf
    merge
    prev_pos
end

function reroot_set!(ctx::AbstractCompiler, lvl::VirtualSparseRunListLevel, diff)
    diff[lvl.tag] = lvl
    reroot_set!(ctx, lvl.lvl, diff)
    reroot_set!(ctx, lvl.buf, diff)
end

reroot_get(ctx::AbstractCompiler, lvl::VirtualSparseRunListLevel, diff) =
    get(diff, lvl.tag, VirtualSparseRunListLevel(
        lvl.tag,
        reroot_get(ctx, lvl.lvl, diff),
        lvl.Ti,
        lvl.qos_fill,
        lvl.qos_stop,
        lvl.ptr,
        lvl.left,
        lvl.right,
        reroot_get(ctx, lvl.buf, diff),
        lvl.merge,
        lvl.prev_pos
    ))

function is_level_injective(ctx, lvl::VirtualSparseRunListLevel)
    [false, is_level_injective(ctx, lvl.lvl)...]
end
function is_level_atomic(ctx, lvl::VirtualSparseRunListLevel)
    (below, atomic) = is_level_atomic(ctx, lvl.lvl)
    return ([below; [atomic]], atomic)
end
function is_level_concurrent(ctx, lvl::VirtualSparseRunListLevel)
    (data, _) = is_level_concurrent(ctx, lvl.lvl)
    return ([data; [false]], false)
end

postype(lvl::VirtualSparseRunListLevel) = postype(lvl.lvl)

function virtualize(
    ctx, ex, ::Type{SparseRunListLevel{Ti,Ptr,Left,Right,merge,Lvl}}, tag=:lvl
) where {Ti,Ptr,Left,Right,merge,Lvl}
    tag = freshen(ctx, tag)
    qos_fill = freshen(ctx, tag, :_qos_fill)
    qos_stop = freshen(ctx, tag, :_qos_stop)
    dirty = freshen(ctx, tag, :_dirty)
    ptr = freshen(ctx, tag, :_ptr)
    left = freshen(ctx, tag, :_left)
    right = freshen(ctx, tag, :_right)
    buf = freshen(ctx, tag, :_buf)
    stop = freshen(ctx, tag, :_stop)
    push_preamble!(
        ctx,
        quote
            $tag = $ex
            $ptr = $tag.ptr
            $left = $tag.left
            $right = $tag.right
            $buf = $tag.buf
            $stop = $tag.shape
        end,
    )
    shape = value(stop, Int)
    prev_pos = freshen(ctx, tag, :_prev_pos)
    lvl_2 = virtualize(ctx, :($tag.lvl), Lvl, tag)
    buf = virtualize(ctx, :($tag.buf), Lvl, tag)
    VirtualSparseRunListLevel(
        tag, lvl_2, Ti, shape, qos_fill, qos_stop, ptr, left, right, buf, merge,
        prev_pos,
    )
end
function lower(ctx::AbstractCompiler, lvl::VirtualSparseRunListLevel, ::DefaultStyle)
    quote
        $SparseRunListLevel{$(lvl.Ti)}(
            $(ctx(lvl.lvl)),
            $(ctx(lvl.shape)),
            $(lvl.ptr),
            $(lvl.left),
            $(lvl.right),
            $(ctx(lvl.buf));
            merge=$(lvl.merge),
        )
    end
end

Base.summary(lvl::VirtualSparseRunListLevel) = "SparseRunList($(summary(lvl.lvl)))"

function virtual_level_size(ctx, lvl::VirtualSparseRunListLevel)
    ext = make_extent(lvl.Ti, literal(lvl.Ti(1.0)), lvl.shape)
    (virtual_level_size(ctx, lvl.lvl)..., ext)
end

function virtual_level_resize!(ctx, lvl::VirtualSparseRunListLevel, dims...)
    lvl.shape = getstop(dims[end])
    lvl.lvl = virtual_level_resize!(ctx, lvl.lvl, dims[1:(end - 1)]...)
    lvl.buf = virtual_level_resize!(ctx, lvl.buf, dims[1:(end - 1)]...)
    lvl
end

function virtual_transfer_level(
    ctx::AbstractCompiler, lvl::VirtualSparseRunListLevel, arch, style
)
    ptr_2 = freshen(ctx, lvl.ptr)
    left_2 = freshen(ctx, lvl.left)
    right_2 = freshen(ctx, lvl.right)
    push_preamble!(
        ctx,
        quote
            $ptr_2 = $transfer($(lvl.ptr), $(ctx(arch)), $style)
            $left_2 = $transfer($(lvl.left), $(ctx(arch)), $style)
            $right_2 = $transfer($(lvl.right), $(ctx(arch)), $style)
        end,
    )
    lvl_2 = virtual_transfer_level(ctx, lvl.lvl, arch, style)
    buf_2 = virtual_transfer_level(ctx, lvl.buf, arch, style)
    return VirtualSparseRunListLevel(
        lvl.tag,
        lvl_2,
        lvl.Ti,
        lvl.shape,
        lvl.qos_fill,
        lvl.qos_stop,
        ptr_2,
        left_2,
        right_2,
        buf_2,
        lvl.merge,
        lvl.prev_pos,
    )
end

virtual_level_eltype(lvl::VirtualSparseRunListLevel) = virtual_level_eltype(lvl.lvl)
virtual_level_fill_value(lvl::VirtualSparseRunListLevel) = virtual_level_fill_value(lvl.lvl)

function declare_level!(ctx::AbstractCompiler, lvl::VirtualSparseRunListLevel, pos, init)
    Tp = postype(lvl)
    Ti = lvl.Ti
    qos = call(-, call(getindex, :($(lvl.ptr)), call(+, pos, 1)), 1)
    push_preamble!(
        ctx,
        quote
            $(lvl.qos_fill) = $(Tp(0))
            $(lvl.qos_stop) = $(Tp(0))
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
    lvl.buf = declare_level!(ctx, lvl.buf, qos, init)
    return lvl
end

function assemble_level!(ctx, lvl::VirtualSparseRunListLevel, pos_start, pos_stop)
    pos_start = ctx(cache!(ctx, :p_start, pos_start))
    pos_stop = ctx(cache!(ctx, :p_start, pos_stop))
    return quote
        Finch.resize_if_smaller!($(lvl.ptr), $pos_stop + 1)
        Finch.fill_range!($(lvl.ptr), 0, $pos_start + 1, $pos_stop + 1)
    end
end

#=
function freeze_level!(ctx::AbstractCompiler, lvl::VirtualSparseRunListLevel, pos_stop)
    (lvl.buf, lvl.lvl) = (lvl.lvl, lvl.buf)
    p = freshen(ctx, :p)
    pos_stop = ctx(cache!(ctx, :pos_stop, simplify(ctx, pos_stop)))
    qos_stop = freshen(ctx, :qos_stop)
    push_preamble!(ctx, quote
        resize!($(lvl.ptr), $pos_stop + 1)
        for $p = 1:$pos_stop
            $(lvl.ptr)[$p + 1] += $(lvl.ptr)[$p]
        end
        $qos_stop = $(lvl.ptr)[$pos_stop + 1] - 1
        resize!($(lvl.left), $qos_stop)
        resize!($(lvl.right), $qos_stop)
    end)
    lvl.lvl = freeze_level!(ctx, lvl.lvl, value(qos_stop))
    return lvl
end
=#

function freeze_level!(ctx::AbstractCompiler, lvl::VirtualSparseRunListLevel, pos_stop)
    Tp = postype(lvl)
    p = freshen(ctx, :p)
    pos_stop = ctx(cache!(ctx, :pos_stop, simplify(ctx, pos_stop)))
    qos_stop = freshen(ctx, :qos_stop)
    push_preamble!(
        ctx,
        quote
            resize!($(lvl.ptr), $pos_stop + 1)
            for $p in 1:($pos_stop)
                $(lvl.ptr)[$p + 1] += $(lvl.ptr)[$p]
            end
            $qos_stop = $(lvl.ptr)[$pos_stop + 1] - 1
        end,
    )
    if lvl.merge
        lvl.buf = freeze_level!(ctx, lvl.buf, value(qos_stop))
        lvl.lvl = declare_level!(
            ctx, lvl.lvl, literal(1), literal(virtual_level_fill_value(lvl.buf))
        )
        unit = ctx(get_smallest_measure(virtual_level_size(ctx, lvl)[end]))
        p = freshen(ctx, :p)
        q = freshen(ctx, :q)
        q_head = freshen(ctx, :q_head)
        q_stop = freshen(ctx, :q_stop)
        q_2 = freshen(ctx, :q_2)
        checkval = freshen(ctx, :check)
        push_preamble!(
            ctx,
            quote
                $(contain(
                    ctx_2 ->
                        assemble_level!(ctx_2, lvl.lvl, value(1, Tp), value(qos_stop, Tp)),
                    ctx,
                ))
                $q = 1
                $q_2 = 1
                for $p in 1:($pos_stop)
                    $q_stop = $(lvl.ptr)[$p + 1]
                    while $q < $q_stop
                        $q_head = $q
                        while $q + 1 < $q_stop &&
                            $(lvl.right)[$q] == $(lvl.left)[$q + 1] - $(unit)
                            $checkval = true
                            $(
                                contain(ctx) do ctx_2
                                    left = variable(freshen(ctx, :left))
                                    set_binding!(
                                        ctx_2,
                                        left,
                                        virtual(
                                            VirtualSubFiber(lvl.buf, value(q_head, Tp))
                                        ),
                                    )
                                    right = variable(freshen(ctx, :right))
                                    set_binding!(
                                        ctx_2,
                                        right,
                                        virtual(
                                            VirtualSubFiber(
                                                lvl.buf, call(+, value(q, Tp), Tp(1))
                                            ),
                                        ),
                                    )
                                    check = VirtualScalar(
                                        nothing, :UNREACHABLE, Bool, false, :check, checkval
                                    )
                                    exts = virtual_level_size(ctx_2, lvl.buf)
                                    inds = [
                                        index(freshen(ctx_2, :i, n)) for n in 1:length(exts)
                                    ]
                                    prgm = assign(
                                        access(check, updater(and)),
                                        and,
                                        call(
                                            isequal,
                                            access(left, reader(), inds...),
                                            access(right, reader(), inds...),
                                        ),
                                    )
                                    for (ind, ext) in zip(inds, exts)
                                        prgm = loop(ind, ext, prgm)
                                    end
                                    prgm = instantiate!(ctx_2, prgm)
                                    ctx_2(prgm)
                                end
                            )
                            if !$checkval
                                break
                            else
                                $q += 1
                            end
                        end
                        $(lvl.left)[$q_2] = $(lvl.left)[$q_head]
                        $(lvl.right)[$q_2] = $(lvl.right)[$q]
                        $(
                            contain(ctx) do ctx_2
                                src = variable(freshen(ctx, :src))
                                set_binding!(
                                    ctx_2,
                                    src,
                                    virtual(VirtualSubFiber(lvl.buf, value(q_head, Tp))),
                                )
                                dst = variable(freshen(ctx, :dst))
                                set_binding!(
                                    ctx_2,
                                    dst,
                                    virtual(VirtualSubFiber(lvl.lvl, value(q_2, Tp))),
                                )
                                exts = virtual_level_size(ctx_2, lvl.buf)
                                inds = [
                                    index(freshen(ctx_2, :i, n)) for n in 1:length(exts)
                                ]
                                op = initwrite(virtual_level_fill_value(lvl.lvl))
                                prgm = assign(
                                    access(dst, updater(op), inds...),
                                    op,
                                    access(src, reader(), inds...),
                                )
                                for (ind, ext) in zip(inds, exts)
                                    prgm = loop(ind, ext, prgm)
                                end
                                prgm = instantiate!(ctx_2, prgm)
                                ctx_2(prgm)
                            end
                        )
                        $q_2 += 1
                        $q += 1
                    end
                    $(lvl.ptr)[$p + 1] = $q_2
                end
                resize!($(lvl.left), $q_2 - 1)
                resize!($(lvl.right), $q_2 - 1)
                $qos_stop = $q_2 - 1
            end,
        )
        lvl.lvl = freeze_level!(ctx, lvl.lvl, value(qos_stop))
        lvl.buf = declare_level!(
            ctx, lvl.buf, literal(1), literal(virtual_level_fill_value(lvl.buf))
        )
        lvl.buf = freeze_level!(ctx, lvl.buf, literal(0))
        return lvl
    else
        push_preamble!(
            ctx,
            quote
                resize!($(lvl.left), $qos_stop)
                resize!($(lvl.right), $qos_stop)
            end,
        )
        (lvl.lvl, lvl.buf) = (lvl.buf, lvl.lvl)
        lvl.lvl = freeze_level!(ctx, lvl.lvl, value(qos_stop))
        return lvl
    end
end

function thaw_level!(ctx::AbstractCompiler, lvl::VirtualSparseRunListLevel, pos_stop)
    p = freshen(ctx, :p)
    pos_stop = ctx(cache!(ctx, :pos_stop, simplify(ctx, pos_stop)))
    qos_stop = freshen(ctx, :qos_stop)
    push_preamble!(
        ctx,
        quote
            $(lvl.qos_fill) = $(lvl.ptr)[$pos_stop + 1] - 1
            $(lvl.qos_stop) = $(lvl.qos_fill)
            $qos_stop = $(lvl.qos_fill)
            $(
                if issafe(get_mode_flag(ctx))
                    quote
                        $(lvl.prev_pos) =
                            Finch.scansearch(
                                $(lvl.ptr), $(lvl.qos_stop) + 1, 1, $pos_stop
                            ) - 1
                    end
                end
            )
            for $p in ($pos_stop):-1:1
                $(lvl.ptr)[$p + 1] -= $(lvl.ptr)[$p]
            end
        end,
    )
    (lvl.lvl, lvl.buf) = (lvl.buf, lvl.lvl)
    lvl.buf = thaw_level!(ctx, lvl.buf, value(qos_stop))
    return lvl
end

function unfurl(
    ctx,
    fbr::VirtualSubFiber{VirtualSparseRunListLevel},
    ext,
    mode,
    ::Union{typeof(defaultread),typeof(walk)},
)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.tag
    Tp = postype(lvl)
    Ti = lvl.Ti
    my_i_end = freshen(ctx, tag, :_i_end)
    my_i_stop = freshen(ctx, tag, :_i_stop)
    my_i_start = freshen(ctx, tag, :_i_start)
    my_q = freshen(ctx, tag, :_q)
    my_q_stop = freshen(ctx, tag, :_q_stop)

    Thunk(;
        preamble=quote
            $my_q = $(lvl.ptr)[$(ctx(pos))]
            $my_q_stop = $(lvl.ptr)[$(ctx(pos)) + $(Tp(1))]
            if $my_q < $my_q_stop
                $my_i_end = $(lvl.right)[$my_q_stop - $(Tp(1))]
            else
                $my_i_end = $(Ti(0))
            end
        end,
        body=(ctx) -> Sequence([
            Phase(;
                stop = (ctx, ext) -> value(my_i_end),
                body = (ctx, ext) -> Stepper(;
                seek=(ctx, ext) -> quote
                    if $(lvl.right)[$my_q] < $(ctx(getstart(ext)))
                        $my_q = Finch.scansearch($(lvl.right), $(ctx(getstart(ext))), $my_q, $my_q_stop - 1)
                    end
                end,
                preamble=quote
                    $my_i_start = $(lvl.left)[$my_q]
                    $my_i_stop = $(lvl.right)[$my_q]
                end,
                stop=(ctx, ext) -> value(my_i_stop),
                body=(ctx, ext) -> Thunk(;
                body=(ctx) -> Sequence([
                Phase(;
                stop = (ctx, ext) -> call(-, value(my_i_start), getunit(ext)),
                body = (ctx, ext) -> Run(FillLeaf(virtual_level_fill_value(lvl)))
            ),
                Phase(;
                body=(ctx, ext) -> Run(;
                body=Simplify(instantiate(ctx, VirtualSubFiber(lvl.lvl, value(my_q)), mode))
            )
            )
            ]),
                epilogue=quote
                    $my_q += ($(ctx(getstop(ext))) == $my_i_stop)
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
    ctx,
    fbr::VirtualSubFiber{VirtualSparseRunListLevel},
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
    fbr::VirtualHollowSubFiber{VirtualSparseRunListLevel},
    ext,
    mode,
    ::Union{typeof(defaultupdate),typeof(extrude)},
)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.tag
    Tp = postype(lvl)
    Ti = lvl.Ti
    qos = freshen(ctx, tag, :_qos)
    qos_fill = lvl.qos_fill
    qos_stop = lvl.qos_stop
    dirty = freshen(ctx, tag, :dirty)

    Thunk(;
        preamble=quote
            $qos = $qos_fill + 1
            $(
                if issafe(get_mode_flag(ctx))
                    quote
                        $(lvl.prev_pos) < $(ctx(pos)) || throw(
                            FinchProtocolError(
                                "SparseRunListLevels cannot be updated multiple times"
                            ),
                        )
                    end
                end
            )
        end,
        body=(ctx) -> AcceptRun(;
            body=(ctx, ext) -> Thunk(;
                preamble = quote
                    if $qos > $qos_stop
                        $qos_stop = max($qos_stop << 1, 1)
                        Finch.resize_if_smaller!($(lvl.left), $qos_stop)
                        Finch.resize_if_smaller!($(lvl.right), $qos_stop)
                        $(contain(ctx_2 -> assemble_level!(ctx_2, lvl.buf, value(qos, Tp), value(qos_stop, Tp)), ctx))
                    end
                    $dirty = false
                end,
                body     = (ctx) -> instantiate(ctx, VirtualHollowSubFiber(lvl.buf, value(qos, Tp), dirty), mode),
                epilogue = quote
                    if $dirty
                        $(fbr.dirty) = true
                        $(lvl.left)[$qos] = $(ctx(getstart(ext)))
                        $(lvl.right)[$qos] = $(ctx(getstop(ext)))
                        $(qos) += $(Tp(1))
                        $(if issafe(get_mode_flag(ctx))
                            quote
                                $(lvl.prev_pos) = $(ctx(pos))
                            end
                        end)
                    end
                end,
            ),
        ),
        epilogue=quote
            $(lvl.ptr)[$(ctx(pos)) + 1] += $qos - $qos_fill - 1
            $qos_fill = $qos - 1
        end,
    )
end
