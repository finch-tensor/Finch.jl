module SparseArraysExt

using Finch
using Finch: AbstractCompiler, DefaultStyle, Extent
using Finch: Unfurled, Stepper, Jumper, Run, FillLeaf, Lookup, Simplify, Sequence, Phase,
    Thunk, Spike
using Finch: virtual_size, virtual_fill_value, getstart, getstop, freshen, push_preamble!,
    push_epilogue!, SwizzleArray
using Finch: get_mode_flag, issafe, contain
using Finch: FinchProtocolError
using Finch.FinchNotation

using Base: @kwdef

isdefined(Base, :get_extension) ? (using SparseArrays) : (using ..SparseArrays)

function Finch.Tensor(arr::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    (m, n) = size(arr)
    return Tensor(
        Dense(SparseList{Ti}(Element{zero(Tv)}(arr.nzval), m, arr.colptr, arr.rowval), n)
    )
end

function Finch.Tensor(arr::SparseVector{Tv,Ti}) where {Tv,Ti}
    (n,) = size(arr)
    return Tensor(
        SparseList{Ti}(
            Element{zero(Tv)}(arr.nzval), n, [1, length(arr.nzind) + 1], arr.nzind
        ),
    )
end

"""
    SparseMatrixCSC(arr::Union{Tensor, SwizzleArray})

Construct a sparse matrix from a tensor or swizzle. May reuse the underlying storage if possible.
"""
function SparseArrays.SparseMatrixCSC(arr::Union{Tensor,SwizzleArray})
    fill_value(arr) === zero(eltype(arr)) || throw(
        ArgumentError(
            "SparseArrays, a Julia stdlib, only supports zero fill values, was given $(fill_value(arr)) as fill_value"
        ),
    )
    return SparseMatrixCSC(Tensor(Dense(SparseList(Element(0.0))), arr))
end

function SparseArrays.SparseMatrixCSC(
    arr::Tensor{<:Dense{Ti,<:SparseList{Ti,Ptr,Idx,<:Element{Vf,Tv}}}}
) where {Vf,Ti,Ptr,Idx,Tv}
    Vf === zero(Tv) || throw(
        ArgumentError(
            "SparseArrays, a Julia stdlib, only supports zero fill values, was given $Vf as fill_value"
        ),
    )
    return SparseMatrixCSC{Tv,Ti}(
        size(arr)..., arr.lvl.lvl.ptr, arr.lvl.lvl.idx, arr.lvl.lvl.lvl.val
    )
end

"""
    sparse(arr::Union{Tensor, SwizzleArray})

Construct a SparseArray from a Tensor or Swizzle. May reuse the underlying storage if possible.
"""
function SparseArrays.sparse(fbr::Union{Tensor,SwizzleArray})
    if ndims(fbr) == 1
        return SparseVector(fbr)
    elseif ndims(fbr) == 2
        return SparseMatrixCSC(fbr)
    else
        throw(
            ArgumentError(
                "SparseArrays, a Julia stdlib, only supports 1-D and 2-D arrays, was given a $(ndims(fbr))-Vf array"
            ),
        )
    end
end

@kwdef mutable struct VirtualSparseMatrixCSC
    tag
    Tv
    Ti
    shape
    ptr
    idx
    val
    qos_fill
    qos_stop
    prev_pos
end

function Finch.virtual_size(ctx::AbstractCompiler, arr::VirtualSparseMatrixCSC)
    return arr.shape
end

function Finch.virtual_resize!(ctx::AbstractCompiler, arr::VirtualSparseMatrixCSC, m, n)
    arr.shape = [m, n]
end

function Finch.lower(ctx::AbstractCompiler, arr::VirtualSparseMatrixCSC, ::DefaultStyle)
    return quote
        $SparseMatrixCSC(
            $(ctx(getstop(arr.shape[1]))),
            $(ctx(getstop(arr.shape[2]))),
            $(arr.ptr),
            $(arr.idx),
            $(arr.val),
        )
    end
end

function Finch.virtualize(ctx, ex, ::Type{<:SparseMatrixCSC{Tv,Ti}}, tag=:tns) where {Tv,Ti}
    tag = freshen(ctx, tag)
    m = freshen(ctx, tag, :_m)
    n = freshen(ctx, tag, :_n)
    ptr = freshen(ctx, tag, :_ptr)
    idx = freshen(ctx, tag, :_idx)
    val = freshen(ctx, tag, :_val)
    push_preamble!(
        ctx,
        quote
            $tag = $ex
            $m = $tag.m
            $n = $tag.n
            $ptr = $tag.colptr
            $idx = $tag.rowval
            $val = $tag.nzval
        end,
    )
    qos_fill = freshen(ctx, tag, :_qos_fill)
    qos_stop = freshen(ctx, tag, :_qos_stop)
    prev_pos = freshen(ctx, tag, :_prev_pos)
    shape = [
        Extent(literal(1), value(m, Ti)), Extent(literal(1), value(n, Ti))
    ]
    VirtualSparseMatrixCSC(
        tag, Tv, Ti, shape, ptr, idx, val, qos_fill, qos_stop, prev_pos
    )
end

function distribute(
    ctx::AbstractCompiler, arr::VirtualSparseMatrixCSC, arch, diff, style
)
    return diff[arr.tag] = VirtualSparseMatrixCSC(
        arr.tag,
        arr.Tv,
        arr.Ti,
        arr.shape,
        distribute_buffer(ctx, arr.ptr, arch, style),
        distribute_buffer(ctx, arr.idx, arch, style),
        distribute_buffer(ctx, arr.val, arch, style),
        arr.qos_fill,
        arr.qos_stop,
        arr.prev_pos,
    )
end

function Finch.reroot_get(ctx::AbstractCompiler, arr::VirtualSparseMatrixCSC, diff)
    get(diff, arr.tag, arr)
end

function Finch.declare!(ctx::AbstractCompiler, arr::VirtualSparseMatrixCSC, init)
    #TODO check that init == fill_value
    Tp = Ti = arr.Ti
    pos_stop = ctx(getstop(virtual_size(ctx, arr)[2]))
    push_preamble!(
        ctx,
        quote
            $(arr.qos_fill) = $(Tp(0))
            $(arr.qos_stop) = $(Tp(0))
            resize!($(arr.ptr), $pos_stop + 1)
            fill_range!($(arr.ptr), $(Tp(0)), 1, $pos_stop + 1)
            $(arr.ptr)[1] = $(Tp(1))
        end,
    )
    if issafe(get_mode_flag(ctx))
        push_preamble!(
            ctx,
            quote
                $(arr.prev_pos) = $(Tp(0))
            end,
        )
    end
    return arr
end

function Finch.freeze!(ctx::AbstractCompiler, arr::VirtualSparseMatrixCSC)
    p = freshen(ctx, :p)
    pos_stop = ctx(getstop(virtual_size(ctx, arr)[2]))
    qos_stop = freshen(ctx, :qos_stop)
    push_preamble!(
        ctx,
        quote
            resize!($(arr.ptr), $pos_stop + 1)
            for $p in 1:($pos_stop)
                $(arr.ptr)[$p + 1] += $(arr.ptr)[$p]
            end
            $qos_stop = $(arr.ptr)[$pos_stop + 1] - 1
            resize!($(arr.idx), $qos_stop)
            resize!($(arr.val), $qos_stop)
        end,
    )
    return arr
end

function Finch.thaw!(ctx::AbstractCompiler, arr::VirtualSparseMatrixCSC)
    p = freshen(ctx, :p)
    pos_stop = ctx(getstop(virtual_size(ctx, arr)[2]))
    qos_stop = freshen(ctx, :qos_stop)
    push_preamble!(
        ctx,
        quote
            $(arr.qos_fill) = $(arr.ptr)[$pos_stop + 1] - 1
            $(arr.qos_stop) = $(arr.qos_fill)
            $qos_stop = $(arr.qos_fill)
            $(
                if issafe(get_mode_flag(ctx))
                    quote
                        $(arr.prev_pos) =
                            Finch.scansearch(
                                $(arr.ptr), $(arr.qos_stop) + 1, 1, $pos_stop
                            ) - 1
                    end
                end
            )
            for $p in ($pos_stop):-1:1
                $(arr.ptr)[$p + 1] -= $(arr.ptr)[$p]
            end
        end,
    )
    return arr
end

@kwdef struct VirtualSparseMatrixCSCColumn
    mtx::VirtualSparseMatrixCSC
    j
end

function Finch.reroot_get(ctx::AbstractCompiler, arr::VirtualSparseMatrixCSCColumn, diff)
    VirtualSparseMatrixCSCColumn(Finch.reroot_get(ctx, arr.mtx, diff), arr.j)
end

FinchNotation.finch_leaf(x::VirtualSparseMatrixCSCColumn) = virtual(x)

function Finch.unfurl(
    ctx,
    tns::VirtualSparseMatrixCSCColumn,
    ext,
    mode,
    ::Union{typeof(defaultread),typeof(walk)},
)
    arr = tns.mtx
    tag = arr.tag
    j = tns.j
    Ti = arr.Ti
    my_i = freshen(ctx, tag, :_i)
    my_q = freshen(ctx, tag, :_q)
    my_q_stop = freshen(ctx, tag, :_q_stop)
    my_i1 = freshen(ctx, tag, :_i1)
    my_val = freshen(ctx, tag, :_val)
    Thunk(;
        preamble=quote
            $my_q = $(arr.ptr)[$(ctx(j))]
            $my_q_stop = $(arr.ptr)[$(ctx(j)) + $(Ti(1))]
            if $my_q < $my_q_stop
                $my_i = $(arr.idx)[$my_q]
                $my_i1 = $(arr.idx)[$my_q_stop - $(Ti(1))]
            else
                $my_i = $(Ti(1))
                $my_i1 = $(Ti(0))
            end
        end,
        body=(ctx) -> Sequence([
            Phase(;
                stop = (ctx, ext) -> value(my_i1),
                body = (ctx, ext) -> Stepper(;
                seek=(ctx, ext) -> quote
                    if $(arr.idx)[$my_q] < $(ctx(getstart(ext)))
                        $my_q = Finch.scansearch($(arr.idx), $(ctx(getstart(ext))), $my_q, $my_q_stop - 1)
                    end
                end,
                preamble=:($my_i = $(arr.idx)[$my_q]),
                stop=(ctx, ext) -> value(my_i),
                chunk=Spike(;
                body = FillLeaf(zero(arr.Tv)),
                tail = Thunk(;
                preamble=quote
                    $my_val = $(arr.val)[$my_q]
                end,
                body=(ctx) -> FillLeaf(value(my_val, arr.Tv))
            )
            ),
                next=(ctx, ext) -> quote
                    $my_q += $(Ti(1))
                end
            ),
            ),
            Phase(;
                body=(ctx, ext) -> Run(FillLeaf(zero(arr.Tv)))
            ),
        ]),
    )
end

#Finch.is_injective(ctx, tns::VirtualSparseMatrixCSCColumn) = is_injective(ctx, tns.mtx)[1]
#Finch.is_atomic(ctx, tns::VirtualSparseMatrixCSCColumn) = is_atomic(ctx, tns.mtx)[1]
#Finch.is_concurrent(ctx, tns::VirtualSparseMatrixCSCColumn) = is_concurrent(ctx, tns.mtx)[1]

function Finch.unfurl(
    ctx::AbstractCompiler,
    arr::VirtualSparseMatrixCSC,
    ext,
    mode,
    ::Union{typeof(defaultread),typeof(walk)},
)
    tag = arr.tag
    Unfurled(;
        arr=arr,
        body=Lookup(;
            body=(ctx, j) -> VirtualSparseMatrixCSCColumn(arr, j)
        ),
    )
end

function Finch.unfurl(
    ctx::AbstractCompiler,
    arr::VirtualSparseMatrixCSC,
    ext,
    mode,
    ::Union{typeof(defaultupdate),typeof(extrude)},
)
    tag = arr.tag
    Unfurled(;
        arr=arr,
        body=Lookup(;
            body=(ctx, j) -> VirtualSparseMatrixCSCColumn(arr, j)
        ),
    )
end

function Finch.unfurl(
    ctx,
    tns::VirtualSparseMatrixCSCColumn,
    ext,
    mode,
    ::Union{typeof(defaultupdate),typeof(extrude)},
)
    arr = tns.mtx
    tag = arr.tag
    j = tns.j
    Tp = arr.Ti
    qos = freshen(ctx, tag, :_qos)
    qos_fill = arr.qos_fill
    qos_stop = arr.qos_stop
    dirty = freshen(ctx, tag, :dirty)
    Thunk(;
        preamble = quote
            $qos = $qos_fill + 1
            $(if issafe(get_mode_flag(ctx))
                quote
                    $(arr.prev_pos) < $(ctx(j)) || throw(FinchProtocolError("SparseMatrixCSCs cannot be updated multiple times"))
                end
            end)
        end,
        body     = (ctx) -> Lookup(;
        body=(ctx, idx) -> Thunk(;
        preamble = quote
            if $qos > $qos_stop
                $qos_stop = max($qos_stop << 1, 1)
                Finch.resize_if_smaller!($(arr.idx), $qos_stop)
                Finch.resize_if_smaller!($(arr.val), $qos_stop)
            end
            $dirty = false
        end,
        body     = (ctx) -> Finch.instantiate(ctx, Finch.VirtualSparseScalar(nothing, nothing, arr.Tv, zero(arr.Tv), gensym(), :($(arr.val)[$(ctx(qos))]), dirty), mode),
        epilogue = quote
            if $dirty
                $(arr.idx)[$qos] = $(ctx(idx))
                $qos += $(Tp(1))
                $(if issafe(get_mode_flag(ctx))
                    quote
                        $(arr.prev_pos) = $(ctx(j))
                    end
                end)
            end
        end
    )
    ),
        epilogue = quote
            $(arr.ptr)[$(ctx(j)) + 1] += $qos - $qos_fill - 1
            $qos_fill = $qos - 1
        end,
    )
end

Finch.FinchNotation.finch_leaf(x::VirtualSparseMatrixCSC) = virtual(x)

Finch.virtual_fill_value(ctx, arr::VirtualSparseMatrixCSC) = zero(arr.Tv)
Finch.virtual_eltype(ctx, tns::VirtualSparseMatrixCSC) = tns.Tv

"""
    SparseVector(arr::Union{Tensor, SwizzleArray})

Construct a sparse matrix from a tensor or swizzle. May reuse the underlying storage if possible.
"""
function SparseArrays.SparseVector(arr::Union{Tensor,SwizzleArray})
    fill_value(arr) === zero(eltype(arr)) || throw(
        ArgumentError(
            "SparseArrays, a Julia stdlib, only supports zero fill values, was given $(fill_value(arr)) as fill_value"
        ),
    )
    return SparseVector(Tensor(SparseList(Element(0.0)), arr))
end

function SparseArrays.SparseVector(
    arr::Tensor{<:SparseList{Ti,Ptr,Idx,<:Element{Vf,Tv}}}
) where {Ti,Ptr,Idx,Tv,Vf}
    Vf === zero(Tv) || throw(
        ArgumentError(
            "SparseArrays, a Julia stdlib, only supports zero fill values, was given $Vf as fill_value"
        ),
    )
    return SparseVector{Tv,Ti}(size(arr)..., arr.lvl.idx, arr.lvl.lvl.val)
end
@kwdef mutable struct VirtualSparseVector
    tag
    Tv
    Ti
    shape
    idx
    val
    qos_fill
    qos_stop
end

function Finch.virtual_size(ctx::AbstractCompiler, arr::VirtualSparseVector)
    return arr.shape
end

function Finch.virtual_resize!(ctx::AbstractCompiler, arr::VirtualSparseVector, n)
    arr.shape = [n]
end

function Finch.lower(ctx::AbstractCompiler, arr::VirtualSparseVector, ::DefaultStyle)
    return quote
        $SparseVector($(ctx(getstop(arr.shape[1]))), $(arr.idx), $(arr.val))
    end
end

function Finch.virtualize(ctx, ex, ::Type{<:SparseVector{Tv,Ti}}, tag=:tns) where {Tv,Ti}
    tag = freshen(ctx, tag)
    shape = [Extent(literal(1), value(:($ex.n), Ti))]
    idx = freshen(ctx, tag, :_idx)
    val = freshen(ctx, tag, :_val)
    push_preamble!(
        ctx,
        quote
            $tag = $ex
            $idx = $tag.nzind
            $val = $tag.nzval
        end,
    )
    qos_fill = freshen(ctx, tag, :_qos_fill)
    qos_stop = freshen(ctx, tag, :_qos_stop)
    VirtualSparseVector(tag, Tv, Ti, shape, idx, val, qos_fill, qos_stop)
end

function distribute(
    ctx::AbstractCompiler, arr::VirtualSparseVector, arch, diff, style
)
    return diff[arr.tag] = VirtualSparseVector(
        arr.tag,
        arr.Tv,
        arr.Ti,
        arr.shape,
        distribute_buffer(ctx, arr.idx, arch, style),
        distribute_buffer(ctx, arr.val, arch, style),
        arr.qos_fill,
        arr.qos_stop,
    )
end

function Finch.reroot_get(ctx::AbstractCompiler, arr::VirtualSparseVector, diff)
    get(diff, arr.tag, arr)
end

function Finch.declare!(ctx::AbstractCompiler, arr::VirtualSparseVector, init)
    #TODO check that init == fill_value
    Tp = Ti = arr.Ti
    push_preamble!(
        ctx,
        quote
            $(arr.qos_fill) = $(Tp(0))
            $(arr.qos_stop) = $(Tp(0))
        end,
    )
    return arr
end

function Finch.freeze!(ctx::AbstractCompiler, arr::VirtualSparseVector)
    p = freshen(ctx, :p)
    qos_stop = freshen(ctx, :qos_stop)
    push_preamble!(
        ctx,
        quote
            $qos_stop = $(ctx(arr.qos_fill))
            resize!($(arr.idx), $qos_stop)
            resize!($(arr.val), $qos_stop)
        end,
    )
    return arr
end

function Finch.thaw!(ctx::AbstractCompiler, arr::VirtualSparseVector)
    p = freshen(ctx, :p)
    qos_stop = freshen(ctx, :qos_stop)
    push_preamble!(
        ctx,
        quote
            $(arr.qos_fill) = length($(arr.idx))
            $(arr.qos_stop) = $(arr.qos_fill)
            $qos_stop = $(arr.qos_fill)
        end,
    )
    return arr
end

function Finch.unfurl(
    ctx::AbstractCompiler,
    arr::VirtualSparseVector,
    ext,
    mode,
    ::Union{typeof(defaultread),typeof(walk)},
)
    tag = arr.tag
    Ti = arr.Ti
    my_i = freshen(ctx, tag, :_i)
    my_q = freshen(ctx, tag, :_q)
    my_q_stop = freshen(ctx, tag, :_q_stop)
    my_i1 = freshen(ctx, tag, :_i1)
    my_val = freshen(ctx, tag, :_val)

    Unfurled(;
        arr=arr,
        body=Thunk(;
            preamble=quote
                $my_q = 1
                $my_q_stop = length($(arr.idx)) + 1
                if $my_q < $my_q_stop
                    $my_i = $(arr.idx)[$my_q]
                    $my_i1 = $(arr.idx)[$my_q_stop - $(Ti(1))]
                else
                    $my_i = $(Ti(1))
                    $my_i1 = $(Ti(0))
                end
            end,
            body=(ctx) -> Sequence([
                Phase(;
                    stop = (ctx, ext) -> value(my_i1),
                    body = (ctx, ext) -> Stepper(;
                    seek=(ctx, ext) -> quote
                        if $(arr.idx)[$my_q] < $(ctx(getstart(ext)))
                            $my_q = Finch.scansearch($(arr.idx), $(ctx(getstart(ext))), $my_q, $my_q_stop - 1)
                        end
                    end,
                    preamble=:($my_i = $(arr.idx)[$my_q]),
                    stop=(ctx, ext) -> value(my_i),
                    chunk=Spike(;
                    body = FillLeaf(zero(arr.Tv)),
                    tail = Thunk(;
                    preamble=quote
                        $my_val = $(arr.val)[$my_q]
                    end,
                    body=(ctx) -> FillLeaf(value(my_val, arr.Tv))
                )
                ),
                    next=(ctx, ext) -> quote
                        $my_q += $(Ti(1))
                    end
                ),
                ),
                Phase(;
                    body=(ctx, ext) -> Run(FillLeaf(zero(arr.Tv)))
                ),
            ]),
        ),
    )
end

function Finch.unfurl(
    ctx, arr::VirtualSparseVector, ext, mode, ::Union{typeof(defaultupdate),typeof(extrude)}
)
    tag = arr.tag
    Tp = arr.Ti
    qos = freshen(ctx, tag, :_qos)
    qos_fill = arr.qos_fill
    qos_stop = arr.qos_stop
    dirty = freshen(ctx, tag, :dirty)

    Unfurled(;
        arr=arr,
        body=Thunk(;
            preamble = quote
                $qos = $qos_fill + 1
            end,
            body     = (ctx) -> Lookup(;
            body=(ctx, idx) -> Thunk(;
            preamble = quote
                if $qos > $qos_stop
                    $qos_stop = max($qos_stop << 1, 1)
                    Finch.resize_if_smaller!($(arr.idx), $qos_stop)
                    Finch.resize_if_smaller!($(arr.val), $qos_stop)
                end
                $dirty = false
            end,
            body     = (ctx) -> Finch.instantiate(ctx, Finch.VirtualSparseScalar(nothing, nothing, arr.Tv, zero(arr.Tv), gensym(), :($(arr.val)[$(ctx(qos))]), dirty), mode),
            epilogue = quote
                if $dirty
                    $(arr.idx)[$qos] = $(ctx(idx))
                    $qos += $(Tp(1))
                end
            end
        )
        ),
            epilogue = quote
                $qos_fill = $qos - 1
            end,
        ),
    )
end

Finch.FinchNotation.finch_leaf(x::VirtualSparseVector) = virtual(x)

Finch.virtual_fill_value(ctx, arr::VirtualSparseVector) = zero(arr.Tv)
Finch.virtual_eltype(ctx, tns::VirtualSparseVector) = tns.Tv

SparseArrays.nnz(fbr::Tensor) = countstored(fbr)

end
