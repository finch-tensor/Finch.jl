"""
    RunListLevel{[Ti=Int], [Ptr, Right]}(lvl, [dim], [merge = true])

The RunListLevel represent runs of equivalent slices `A[:, ..., :, i]`. A
sorted list is used to record the right endpoint of each run. Optionally, `dim`
is the size of the last dimension.

`Ti` is the type of the last tensor index, and `Tp` is the type used for
positions in the level. The types `Ptr` and `Right` are the types of the
arrays used to store positions and endpoints.

The `merge` keyword argument is used to specify whether the level should merge
duplicate consecutive runs.

```jldoctest
julia> tensor_tree(Tensor(Dense(RunListLevel(Element(0.0))), [10 0 20; 30 0 0; 0 0 40]))
3×3-Tensor
└─ Dense [:,1:3]
   ├─ [:, 1]: RunList (0.0) [1:3]
   │  ├─ [1:1]: 10.0
   │  ├─ [2:2]: 30.0
   │  └─ [3:3]: 0.0
   ├─ [:, 2]: RunList (0.0) [1:3]
   │  └─ [1:3]: 0.0
   └─ [:, 3]: RunList (0.0) [1:3]
      ├─ [1:1]: 20.0
      ├─ [2:2]: 0.0
      └─ [3:3]: 40.0
```
"""
struct RunListLevel{Ti,Ptr<:AbstractVector,Right<:AbstractVector,merge,Lvl} <: AbstractLevel
    lvl::Lvl
    shape::Ti
    ptr::Ptr
    right::Right
    buf::Lvl
end

const RunList = RunListLevel
RunListLevel(lvl::Lvl; kwargs...) where {Lvl} = RunListLevel{Int}(lvl; kwargs...)
function RunListLevel(lvl, shape, args...; kwargs...)
    RunListLevel{typeof(shape)}(lvl, shape, args...; kwargs...)
end
RunListLevel{Ti}(lvl; kwargs...) where {Ti} = RunListLevel(lvl, zero(Ti); kwargs...)
function RunListLevel{Ti}(lvl, shape; kwargs...) where {Ti}
    RunListLevel{Ti}(lvl, shape, postype(lvl)[1], Ti[], deepcopy(lvl); kwargs...)
end #TODO if similar_level could return the same type, we could use it here
function RunListLevel{Ti}(
    lvl::Lvl, shape, ptr::Ptr, right::Right, buf::Lvl; merge=true
) where {Ti,Lvl,Ptr,Right}
    RunListLevel{Ti,Ptr,Right,merge,Lvl}(lvl, Ti(shape), ptr, right, buf)
end

getmerge(lvl::RunListLevel{Ti,Ptr,Right,merge}) where {Ti,Ptr,Right,merge} = merge

Base.summary(lvl::RunListLevel) = "RunList($(summary(lvl.lvl)))"
function similar_level(lvl::RunListLevel, fill_value, eltype::Type, dim, tail...)
    RunList(similar_level(lvl.lvl, fill_value, eltype, tail...), dim; merge=getmerge(lvl))
end

function postype(
    ::Type{RunListLevel{Ti,Ptr,Right,merge,Lvl}}
) where {Ti,Ptr,Right,merge,Lvl}
    return postype(Lvl)
end

function transfer(lvl::RunListLevel{Ti}, device, style) where {Ti}
    lvl_2 = transfer(lvl.lvl, device, style)
    ptr = transfer(lvl.ptr, device, style)
    right = transfer(lvl.right, device, style)
    buf = transfer(lvl.buf, device, style)
    return RunListLevel{Ti}(
        lvl_2, lvl.shape, lvl.ptr, lvl.right, lvl.buf; merge=getmerge(lvl)
    )
end

function pattern!(lvl::RunListLevel{Ti}) where {Ti}
    RunListLevel{Ti}(
        pattern!(lvl.lvl),
        lvl.shape,
        lvl.ptr,
        lvl.right,
        pattern!(lvl.buf);
        merge=getmerge(lvl),
    )
end

function countstored_level(lvl::RunListLevel, pos)
    countstored_level(lvl.lvl, lvl.ptr[pos + 1] - 1)
end

function set_fill_value!(lvl::RunListLevel{Ti}, init) where {Ti}
    RunListLevel{Ti}(
        set_fill_value!(lvl.lvl, init),
        lvl.shape,
        lvl.ptr,
        lvl.right,
        set_fill_value!(lvl.buf, init);
        merge=getmerge(lvl),
    )
end

function Base.resize!(lvl::RunListLevel{Ti}, dims...) where {Ti}
    RunListLevel{Ti}(
        resize!(lvl.lvl, dims[1:(end - 1)]...),
        dims[end],
        lvl.ptr,
        lvl.right,
        resize!(lvl.buf, dims[1:(end - 1)]...);
        merge=getmerge(lvl),
    )
end

function Base.show(
    io::IO, lvl::RunListLevel{Ti,Ptr,Right,merge,Lvl}
) where {Ti,Ptr,Right,merge,Lvl}
    if get(io, :compact, false)
        print(io, "RunList(")
    else
        print(io, "RunList{$Ti}(")
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
        show(io, lvl.right)
        print(io, ", ")
        show(io, lvl.buf)
        print(io, "; merge = ")
        show(io, merge)
    end
    print(io, ")")
end

function labelled_show(io::IO, fbr::SubFiber{<:RunListLevel})
    print(
        io,
        "RunList (",
        fill_value(fbr),
        ") [",
        ":,"^(ndims(fbr) - 1),
        "1:",
        size(fbr)[end],
        "]",
    )
end

function labelled_children(fbr::SubFiber{<:RunListLevel})
    lvl = fbr.lvl
    pos = fbr.pos
    pos + 1 > length(lvl.ptr) && return []
    map(lvl.ptr[pos]:(lvl.ptr[pos + 1] - 1)) do qos
        left = qos == lvl.ptr[pos] ? 1 : lvl.right[qos - 1] + 1
        LabelledTree(
            cartesian_label(
                [range_label() for _ in 1:(ndims(fbr) - 1)]...,
                range_label(left, lvl.right[qos]),
            ),
            SubFiber(lvl.lvl, qos),
        )
    end
end

@inline level_ndims(
    ::Type{<:RunListLevel{Ti,Ptr,Right,merge,Lvl}}
) where {Ti,Ptr,Right,merge,Lvl} = 1 + level_ndims(Lvl)
@inline level_size(lvl::RunListLevel) = (level_size(lvl.lvl)..., lvl.shape)
@inline level_axes(lvl::RunListLevel) = (level_axes(lvl.lvl)..., Base.OneTo(lvl.shape))
@inline level_eltype(
    ::Type{<:RunListLevel{Ti,Ptr,Right,merge,Lvl}}
) where {Ti,Ptr,Right,merge,Lvl} = level_eltype(Lvl)
@inline level_fill_value(
    ::Type{<:RunListLevel{Ti,Ptr,Right,merge,Lvl}}
) where {Ti,Ptr,Right,merge,Lvl} = level_fill_value(Lvl)
function data_rep_level(
    ::Type{<:RunListLevel{Ti,Ptr,Right,merge,Lvl}}
) where {Ti,Ptr,Right,merge,Lvl}
    DenseData(data_rep_level(Lvl))
end

function isstructequal(a::T, b::T) where {T<:RunList}
    a.shape == b.shape &&
        a.ptr == b.ptr &&
        a.right == b.right &&
        isstructequal(a.lvl, b.lvl)
end

(fbr::AbstractFiber{<:RunListLevel})() = fbr
function (fbr::SubFiber{<:RunListLevel})(idxs...)
    isempty(idxs) && return fbr
    lvl = fbr.lvl
    p = fbr.pos
    r2 = searchsortedfirst(@view(lvl.right[lvl.ptr[p]:(lvl.ptr[p + 1] - 1)]), idxs[end])
    q = lvl.ptr[p] + r2 - 1
    fbr_2 = SubFiber(lvl.lvl, q)
    fbr_2(idxs[1:(end - 1)]...)
end

mutable struct VirtualRunListLevel <: AbstractVirtualLevel
    tag
    lvl
    Ti
    shape
    qos_fill
    qos_stop
    ptr
    right
    buf
    prev_pos
    i_prev
    merge
end

function reroot_set!(ctx::AbstractCompiler, lvl::VirtualRunListLevel, diff) 
    diff[lvl.tag] = lvl
    reroot_set!(ctx, lvl.lvl, diff)
    reroot_set!(ctx, lvl.buf, diff)
end

reroot_get(ctx::AbstractCompiler, lvl::VirtualRunListLevel, diff) =
    get(diff, lvl.tag, VirtualRunListLevel(
        lvl.tag,
        reroot_get(ctx, lvl.lvl, diff),
        lvl.Ti,
        lvl.shape,
        lvl.qos_fill,
        lvl.qos_stop,
        lvl.ptr,
        lvl.right,
        reroot_get(ctx, lvl.buf, diff),
        lvl.prev_pos,
        lvl.i_prev,
        lvl.merge
    ))

function is_level_injective(ctx, lvl::VirtualRunListLevel)
    [false, is_level_injective(ctx, lvl.lvl)...]
end
function is_level_atomic(ctx, lvl::VirtualRunListLevel)
    (below, atomic) = is_level_atomic(ctx, lvl.lvl)
    return ([below; [atomic]], atomic)
end
function is_level_concurrent(ctx, lvl::VirtualRunListLevel)
    (data, _) = is_level_concurrent(ctx, lvl.lvl)
    return ([data; [false]], false)
end

postype(lvl::VirtualRunListLevel) = postype(lvl.lvl)

function virtualize(
    ctx, ex, ::Type{RunListLevel{Ti,Ptr,Right,merge,Lvl}}, tag=:lvl
) where {Ti,Ptr,Right,merge,Lvl}
    #Invariants of the level (Read Mode):
    # 1. right[ptr[p]:ptr[p + 1] - 1] is the sorted list of right endpoints of the runs
    #
    #Invariants of the level (Write Mode):
    # 1. prevpos is the last position written (initially 0)
    # 2. i_prev is the last index written (initially shape)
    # 3. for all p in 1:prevpos-1, ptr[p] is the number of runs in that position
    # 4. qos_fill is the position of the last index written

    tag = freshen(ctx, tag)
    stop = freshen(ctx, tag, :_stop)
    qos_fill = freshen(ctx, tag, :_qos_fill)
    qos_stop = freshen(ctx, tag, :_qos_stop)
    dirty = freshen(ctx, tag, :_dirty)
    ptr = freshen(ctx, tag, :_ptr)
    right = freshen(ctx, tag, :_right)
    buf = freshen(ctx, tag, :_buf)
    push_preamble!(
        ctx,
        quote
            $tag = $ex
            $ptr = $tag.ptr
            $right = $tag.right
            $buf = $tag.buf
            $stop = $tag.shape
        end,
    )
    shape = value(stop, Int)
    i_prev = freshen(ctx, tag, :_i_prev)
    prev_pos = freshen(ctx, tag, :_prev_pos)
    lvl_2 = virtualize(ctx, :($tag.lvl), Lvl, tag)
    buf = virtualize(ctx, :($tag.buf), Lvl, tag)
    VirtualRunListLevel(
        tag, lvl_2, Ti, shape, qos_fill, qos_stop, ptr, right, buf, prev_pos, i_prev,
        merge,
    )
end

function lower(ctx::AbstractCompiler, lvl::VirtualRunListLevel, ::DefaultStyle)
    quote
        $RunListLevel{$(lvl.Ti)}(
            $(ctx(lvl.lvl)),
            $(ctx(lvl.shape)),
            $(lvl.ptr),
            $(lvl.right),
            $(ctx(lvl.buf));
            merge=$(lvl.merge),
        )
    end
end

Base.summary(lvl::VirtualRunListLevel) = "RunList($(summary(lvl.lvl)))"

function virtual_level_size(ctx, lvl::VirtualRunListLevel)
    ext = make_extent(lvl.Ti, literal(lvl.Ti(1.0)), lvl.shape)
    (virtual_level_size(ctx, lvl.lvl)..., ext)
end

function virtual_level_resize!(ctx, lvl::VirtualRunListLevel, dims...)
    lvl.shape = getstop(dims[end])
    lvl.lvl = virtual_level_resize!(ctx, lvl.lvl, dims[1:(end - 1)]...)
    lvl.buf = virtual_level_resize!(ctx, lvl.buf, dims[1:(end - 1)]...)
    lvl
end

function virtual_transfer_level(
    ctx::AbstractCompiler, lvl::VirtualRunListLevel, arch, style
)
    ptr_2 = freshen(ctx, lvl.ptr)
    right_2 = freshen(ctx, lvl.right)
    push_preamble!(
        ctx,
        quote
            $ptr_2 = $transfer($(lvl.ptr), $(ctx(arch)), $style)
            $right_2 = $transfer($(lvl.right), $(ctx(arch)), $style)
        end,
    )
    lvl_2 = virtual_transfer_level(ctx, lvl.lvl, arch, style)
    buf_2 = virtual_transfer_level(ctx, lvl.buf, arch, style)
    VirtualRunListLevel(
        lvl.tag,
        lvl_2,
        lvl.Ti,
        lvl.shape,
        lvl.qos_fill,
        lvl.qos_stop,
        ptr_2,
        right_2,
        buf_2,
        lvl.prev_pos,
        lvl.i_prev,
        lvl.merge,
    )
end

virtual_level_eltype(lvl::VirtualRunListLevel) = virtual_level_eltype(lvl.lvl)
virtual_level_fill_value(lvl::VirtualRunListLevel) = virtual_level_fill_value(lvl.lvl)

function declare_level!(ctx::AbstractCompiler, lvl::VirtualRunListLevel, pos, init)
    Tp = postype(lvl)
    Ti = lvl.Ti
    qos = call(-, call(getindex, :($(lvl.ptr)), call(+, pos, 1)), 1)
    unit = ctx(get_smallest_measure(virtual_level_size(ctx, lvl)[end]))
    push_preamble!(
        ctx,
        quote
            $(lvl.qos_fill) = $(Tp(0))
            $(lvl.qos_stop) = $(Tp(0))
            $(lvl.i_prev) = $(Ti(1)) - $unit
            $(lvl.prev_pos) = $(Tp(1))
        end,
    )
    lvl.buf = declare_level!(ctx, lvl.buf, qos, init)
    return lvl
end

function assemble_level!(ctx, lvl::VirtualRunListLevel, pos_start, pos_stop)
    pos_start = ctx(cache!(ctx, :p_start, pos_start))
    pos_stop = ctx(cache!(ctx, :p_start, pos_stop))
    return quote
        Finch.resize_if_smaller!($(lvl.ptr), $pos_stop + 1)
        Finch.fill_range!($(lvl.ptr), 1, $pos_start + 1, $pos_stop + 1)
    end
end

#=
function freeze_level!(ctx::AbstractCompiler, lvl::VirtualRunListLevel, pos_stop)
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
        resize!($(lvl.right), $qos_stop)
    end)
    lvl.lvl = freeze_level!(ctx, lvl.lvl, value(qos_stop))
    return lvl
end
=#

function freeze_level!(ctx::AbstractCompiler, lvl::VirtualRunListLevel, pos_stop)
    Tp = postype(lvl)
    p = freshen(ctx, :p)
    pos_stop = ctx(cache!(ctx, :pos_stop, simplify(ctx, pos_stop)))
    Ti = lvl.Ti
    pos_2 = freshen(ctx, tag, :_pos)
    qos_stop = lvl.qos_stop
    qos_fill = lvl.qos_fill
    qos = freshen(ctx, :qos)
    unit = ctx(get_smallest_measure(virtual_level_size(ctx, lvl)[end]))
    push_preamble!(
        ctx,
        quote
            $qos = $(lvl.qos_fill)
            #if we did not write something to finish out the last run, we need to fill that in
            $qos += $(lvl.i_prev) < $(ctx(lvl.shape))
            #and all the runs after that
            $qos += $(pos_stop) - $(lvl.prev_pos)
            if $qos > $qos_stop
                $qos_stop = $qos
                Finch.resize_if_smaller!($(lvl.right), $qos_stop)
                Finch.fill_range!(
                    $(lvl.right), $(ctx(lvl.shape)), $qos_fill + 1, $qos_stop
                )
                $(contain(
                    ctx_2 -> assemble_level!(
                        ctx_2,
                        lvl.buf,
                        call(+, value(qos_fill, Tp), Tp(1)),
                        value(qos_stop, Tp),
                    ),
                    ctx,
                ))
            end
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
                            $(lvl.right)[$q] == $(lvl.right)[$q + 1] - $(unit)
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
                resize!($(lvl.right), $qos_stop)
            end,
        )
        (lvl.buf, lvl.lvl) = (lvl.lvl, lvl.buf)
        lvl.lvl = freeze_level!(ctx, lvl.lvl, value(qos_stop))
        return lvl
    end
end

function thaw_level!(ctx::AbstractCompiler, lvl::VirtualRunListLevel, pos_stop)
    error(
        "Thaw is not yet implemented for RunList level. To implement, we need to cache the last written qos as a Ref{Int}, then reconstruct prev_pos and i_prev from the ptr and right arrays"
    )
    #=
    p = freshen(ctx, :p)
    pos_stop = ctx(cache!(ctx, :pos_stop, simplify(ctx, pos_stop)))
    qos_stop = freshen(ctx, :qos_stop)
    unit = ctx(get_smallest_measure(virtual_level_size(ctx, lvl)[end]))
    push_preamble!(ctx, quote
        $(lvl.qos_fill) = $(lvl.ptr)[$pos_stop + 1] - 1
        $(lvl.qos_stop) = $(lvl.qos_fill)
        $(lvl.i_prev) = $(lvl.right)[$(lvl.qos_fill)]
        $qos_stop = $(lvl.qos_fill)
        $(lvl.prev_pos) = Finch.scansearch($(lvl.ptr), $(lvl.qos_stop) + 1, 1, $pos_stop) - 1
        for $p = $pos_stop:-1:1
            $(lvl.ptr)[$p + 1] -= $(lvl.ptr)[$p]
        end
    end)
    (lvl.lvl, lvl.buf) = (lvl.buf, lvl.lvl)
    lvl.buf = thaw_level!(ctx, lvl.buf, value(qos_stop))
    return lvl
    =#
end

function unfurl(
    ctx,
    fbr::VirtualSubFiber{VirtualRunListLevel},
    ext,
    mode,
    ::Union{typeof(defaultread),typeof(walk)},
)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.tag
    Tp = postype(lvl)
    Ti = lvl.Ti
    my_i = freshen(ctx, tag, :_i)
    my_q = freshen(ctx, tag, :_q)
    my_q_stop = freshen(ctx, tag, :_q_stop)
    my_i1 = freshen(ctx, tag, :_i1)

    Unfurled(;
        arr=fbr,
        body=Thunk(;
            preamble=(
                quote
                    $my_q = $(lvl.ptr)[$(ctx(pos))]
                    $my_q_stop = $(lvl.ptr)[$(ctx(pos)) + $(Tp(1))]
                    #TODO I think this if is only ever true
                    if $my_q < $my_q_stop
                        $my_i = $(lvl.right)[$my_q]
                        $my_i1 = $(lvl.right)[$my_q_stop - $(Tp(1))]
                    else
                        $my_i = $(Ti(1))
                        $my_i1 = $(Ti(0))
                    end
                end
            ),
            body=(ctx) -> Stepper(;
                seek=(ctx, ext) -> quote
                    if $(lvl.right)[$my_q] < $(ctx(getstart(ext)))
                        $my_q = Finch.scansearch(
                            $(lvl.right),
                            $(ctx(getstart(ext))),
                            $my_q,
                            $my_q_stop - 1,
                        )
                    end
                end,
                preamble=:($my_i = $(lvl.right)[$my_q]),
                stop=(ctx, ext) -> value(my_i),
                chunk=Run(;
                    body=Simplify(
                        instantiate(ctx, VirtualSubFiber(lvl.lvl, value(my_q)), mode)
                    ),
                ),
                next=(ctx, ext) -> :($my_q += $(Tp(1))),
            ),
        ),
    )
end

function unfurl(
    ctx,
    fbr::VirtualSubFiber{VirtualRunListLevel},
    ext,
    mode,
    proto::Union{typeof(defaultupdate),typeof(extrude)},
)
    unfurl(
        ctx, VirtualHollowSubFiber(fbr.lvl, fbr.pos, freshen(ctx, :null)), ext, mode, proto
    )
end

#Invariants of the level (Write Mode):
# 1. prevpos is the last position written (initially 0)
# 2. i_prev is the last index written (initially shape)
# 3. for all p in 1:prevpos-1, ptr[p] is the number of runs in that position
# 4. qos_fill is the position of the last index written

function unfurl(
    ctx,
    fbr::VirtualHollowSubFiber{VirtualRunListLevel},
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
    pos_2 = freshen(ctx, tag, :_pos)
    unit = ctx(get_smallest_measure(virtual_level_size(ctx, lvl)[end]))
    qos_2 = freshen(ctx, tag, :_qos_2)
    qos_set = freshen(ctx, tag, :_qos_set)
    qos_3 = freshen(ctx, tag, :_qos_3)
    local_i_prev = freshen(ctx, tag, :_i_prev)

    Unfurled(;
        arr=fbr,
        body=Thunk(;
            preamble=quote
                $qos = $qos_fill + 1
                $(
                    if issafe(get_mode_flag(ctx))
                        quote
                            $(lvl.prev_pos) <= $(ctx(pos)) || throw(
                                FinchProtocolError(
                                    "RunListLevels cannot be updated multiple times"
                                ),
                            )
                        end
                    end
                )
                $local_i_prev = $(lvl.i_prev)
                #if the previous position is not the same as the current position, we will eventually need to fill in the gap
                if $(lvl.prev_pos) < $(ctx(pos))
                    $qos += $(ctx(pos)) - $(lvl.prev_pos) - 1
                    #only if we did not write something to finish out the last run do we eventually need to fill that in too
                    $qos += $(lvl.i_prev) < $(ctx(lvl.shape))
                    $local_i_prev = $(Ti(1)) - $unit
                end
                $qos_set = $qos
            end,
            body=(ctx) -> AcceptRun(;
                body=(ctx, ext) -> Thunk(;
                    preamble = quote
                        $qos_3 = $qos + ($(local_i_prev) < ($(ctx(getstart(ext))) - $unit))
                        if $qos_3 > $qos_stop
                            $qos_2 = $qos_stop + 1
                            while $qos_3 > $qos_stop
                                $qos_stop = max($qos_stop << 1, 1)
                            end
                            Finch.resize_if_smaller!($(lvl.right), $qos_stop)
                            Finch.fill_range!($(lvl.right), $(ctx(lvl.shape)), $qos_2, $qos_stop)
                            $(contain(ctx_2 -> assemble_level!(ctx_2, lvl.buf, value(qos_2, Tp), value(qos_stop, Tp)), ctx))
                        end
                        $dirty = false
                    end,
                    body     = (ctx) -> instantiate(ctx, VirtualHollowSubFiber(lvl.buf, value(qos_3, Tp), dirty), mode),
                    epilogue = quote
                        if $dirty
                            $(lvl.right)[$qos] = $(ctx(getstart(ext))) - $unit
                            $(lvl.right)[$qos_3] = $(ctx(getstop(ext)))
                            $(qos) = $qos_3 + $(Tp(1))
                            $(local_i_prev) = $(ctx(getstop(ext)))
                        end
                    end,
                ),
            ),
            epilogue=quote
                if $qos - $qos_set > 0
                    $(fbr.dirty) = true
                    $(lvl.ptr)[$(ctx(pos)) + 1] +=
                        $qos - $qos_set - ($(local_i_prev) == $(ctx(lvl.shape))) #the last run is accounted for already because ptr starts out at 1
                    $(lvl.prev_pos) = $(ctx(pos))
                    $(lvl.i_prev) = $(local_i_prev)
                    $qos_fill = $qos - 1
                end
            end,
        ),
    )
end
