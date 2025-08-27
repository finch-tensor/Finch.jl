abstract type AbstractExtent end
abstract type AbstractVirtualExtent end

FinchNotation.finch_leaf(x::Auto) = virtual(x)
FinchNotation.finch_leaf_instance(x::Auto) = value_instance(x)
virtualize(ctx, ex, ::Type{Auto}) = auto

getstart(::Auto) = error("asked for start of dimensionless range")
getstop(::Auto) = error("asked for stop of dimensionless range")

struct UnknownDimension <: AbstractExtent end

resolvedim(ext) = ext

resultdim(ctx, a, b, c, tail...) = resultdim(ctx, a, resultdim(ctx, b, c, tail...))
function resultdim(ctx, a, b)
    c = combinedim(ctx, a, b)
    d = combinedim(ctx, b, a)
    return _resultdim(ctx, a, b, c, d)
end
function _resultdim(ctx, a, b, c::UnknownDimension, d::UnknownDimension)
    throw(MethodError(combinedim, (ctx, a, b)))
end
_resultdim(ctx, a, b, c, d::UnknownDimension) = c
_resultdim(ctx, a, b, c::UnknownDimension, d) = d
_resultdim(ctx, a, b, c, d) = c #TODO assert same lattice type here.
#_resultdim(a, b, c::T, d::T) where {T} = (c == d) ? c : @assert false "TODO combinedim_ambiguity_error"

"""
    combinedim(ctx, a, b)

Combine the two dimensions `a` and `b`.  To avoid ambiguity, only define one of

```
combinedim(ctx, ::A, ::B)
combinedim(ctx, ::B, ::A)
```
"""
combinedim(ctx, a, b) = UnknownDimension()

combinedim(ctx, a::Auto, b) = b

@kwdef struct Extent{Start,Stop} <: AbstractExtent
    start::Start
    stop::Stop
end

FinchNotation.extent(start::Integer, stop::Integer) = Extent(start, stop)

@kwdef struct VirtualExtent <: AbstractVirtualExtent
    start
    stop
end

function virtualize(ctx, ex, ::Type{Extent{Start,Stop}}) where {Start,Stop}
    VirtualExtent(
        virtualize(ctx, :($ex.start), Start),
        virtualize(ctx, :($ex.stop), Stop),
    )
end

function lower(ctx, ex::VirtualExtent)
    :($Extent($(ctx(ex.start)), $(ctx(ex.stop))))
end

FinchNotation.finch_leaf(x::VirtualExtent) = virtual(x)

function virtual_call_def(
    ctx, alg, ::typeof(extent), ::Type{<:Tuple{<:Integer,<:Integer}}, start, stop
)
    if isfoldable(start) && isfoldable(stop)
        VirtualExtent(start, stop)
    end
end

@kwdef struct ContinuousExtent{Start,Stop} <: AbstractExtent
    start::Start
    stop::Stop
end

function FinchNotation.extent(start::Union{Real,Limit}, stop::Union{Real,Limit})
    ContinuousExtent(start, stop)
end

@kwdef struct VirtualContinuousExtent <: AbstractVirtualExtent
    start
    stop
end

function virtual_call_def(
    ctx,
    alg,
    ::typeof(extent),
    ::Type{<:Tuple{<:Union{<:Real,<:Limit},<:Union{<:Real,<:Limit}}},
    start,
    stop,
)
    if isfoldable(start) && isfoldable(stop)
        VirtualContinuousExtent(start, stop)
    end
end

function virtualize(ctx, ex, ::Type{ContinuousExtent{Start,Stop}}) where {Start,Stop}
    VirtualExtent(
        virtualize(ctx, :($ex.start), Start),
        virtualize(ctx, :($ex.stop), Stop),
    )
end

function lower(ctx, ex::VirtualContinuousExtent)
    :($ContinuousExtent($(ctx(ex.start)), $(ctx(ex.stop))))
end

FinchNotation.finch_leaf(x::VirtualContinuousExtent) = virtual(x)

Base.:(==)(a::VirtualExtent, b::VirtualExtent) =
    a.start == b.start &&
    a.stop == b.stop

bound_below!(val, below) = cached(val, literal(call(max, val, below)))

bound_above!(val, above) = cached(val, literal(call(min, val, above)))

function bound_measure_below!(ext::VirtualExtent, m)
    VirtualExtent(ext.start, bound_below!(ext.stop, call(+, ext.start, m)))
end
function bound_measure_above!(ext::VirtualExtent, m)
    VirtualExtent(ext.start, bound_above!(ext.stop, call(+, ext.start, m)))
end

function cache_dim!(ctx, var, ext::VirtualExtent)
    VirtualExtent(;
        start=cache!(ctx, Symbol(var, :_start), ext.start),
        stop=cache!(ctx, Symbol(var, :_stop), ext.stop),
    )
end

getstart(ext::VirtualExtent) = ext.start
getstop(ext::VirtualExtent) = ext.stop
measure(ext::VirtualExtent) = call(+, call(-, ext.stop, ext.start), 1)

function combinedim(ctx, a::VirtualExtent, b::VirtualExtent)
    VirtualExtent(;
        start=checklim(ctx, a.start, b.start),
        stop=checklim(ctx, a.stop, b.stop),
    )
end

combinedim(ctx, a::Auto, b::VirtualExtent) = b

struct SuggestedExtent{Ext} <: AbstractExtent
    ext::Ext
end

struct VirtualSuggestedExtent <: AbstractVirtualExtent
    ext
end

FinchNotation.finch_leaf(x::VirtualSuggestedExtent) = virtual(x)

function virtualize(ctx, ex, ::Type{SuggestedExtent{Ext}}) where {Ext}
    VirtualSuggestedExtent(virtualize(ctx, :($ex.ext), Ext))
end

function lower(ctx, ex::VirtualSuggestedExtent)
    :($SuggestedExtent($(ctx(ex.ext))))
end

Base.:(==)(a::VirtualSuggestedExtent, b::VirtualSuggestedExtent) = a.ext == b.ext

suggest(ext) = VirtualSuggestedExtent(ext)
suggest(ext::VirtualSuggestedExtent) = ext
suggest(ext::Auto) = auto

resolvedim(ext::Symbol) = error()
resolvedim(ext::VirtualSuggestedExtent) = resolvedim(ext.ext)
function cache_dim!(ctx, tag, ext::VirtualSuggestedExtent)
    VirtualSuggestedExtent(cache_dim!(ctx, tag, ext.ext))
end

combinedim(ctx, a::VirtualSuggestedExtent, b::VirtualExtent) = b

combinedim(ctx, a::VirtualSuggestedExtent, b::Auto) = a

function combinedim(ctx, a::VirtualSuggestedExtent, b::VirtualSuggestedExtent)
    VirtualSuggestedExtent(combinedim(ctx, a.ext, b.ext))
end

function checklim(ctx::AbstractCompiler, a::FinchNode, b::FinchNode)
    shash = get_static_hash(ctx)
    if isliteral(a) && isliteral(b)
        a == b || throw(DimensionMismatch("mismatched dimension limits ($a != $b)"))
    end
    if shash(a) < shash(b) #TODO instead of this, we should introduce a lazy operator to assert equality
        push_preamble!(
            ctx,
            quote
                $(ctx(a)) == $(ctx(b)) || throw(
                    DimensionMismatch(
                        "mismatched dimension limits ($($(ctx(a))) != $($(ctx(b))))"
                    ),
                )
            end,
        )
        a
    else
        b
    end
end

@kwdef struct ParallelDimension{Ext,Device,Schedule} <: AbstractExtent
    ext::Ext
    device::Device
    schedule::Schedule
end

@kwdef struct VirtualParallelDimension <: AbstractVirtualExtent
    ext
    device
    schedule
end

FinchNotation.finch_leaf(x::VirtualParallelDimension) = virtual(x)
function virtualize(
    ctx, ex, ::Type{ParallelDimension{Ext,Device,Schedule}}
) where {Ext,Device,Schedule}
    VirtualParallelDimension(
        virtualize(ctx, :($ex.ext), Ext),
        virtualize(ctx, :($ex.device), Device),
        virtualize(ctx, :($ex.schedule), Schedule),
    )
end
function lower(ctx, ex::VirtualParallelDimension)
    :($ParallelDimension($(ctx(ex.ext)), $(ctx(ex.device))))
end

"""
    parallel(ext, device=CPU(nthreads()), schedule=static_schedule())

A dimension `ext` that is parallelized over `device` using the `schedule`. The `ext` field is usually
`_`, or dimensionless, but can be any standard dimension argument.
"""
function parallel(dim, device=cpu(Threads.nthreads()), schedule=static_schedule())
    ParallelDimension(dim, device, schedule)
end

function virtual_call_def(
    ctx,
    alg,
    ::typeof(parallel),
    ::Any,
    ext,
    device=finch_leaf(virtual_call(ctx, cpu)),
    schedule=finch_leaf(VirtualFinchStaticSchedule(:dynamic)),
)
    ext = resolve(ctx, ext)
    device = resolve(ctx, device)
    schedule = resolve(ctx, schedule)
    VirtualParallelDimension(ext, device, schedule)
end

Base.:(==)(a::VirtualParallelDimension, b::VirtualParallelDimension) = a.ext == b.ext

getstart(ext::VirtualParallelDimension) = getstart(ext.ext)
getstop(ext::VirtualParallelDimension) = getstop(ext.ext)

function combinedim(ctx, a::VirtualParallelDimension, b::VirtualExtent)
    VirtualParallelDimension(resultdim(ctx, a.ext, b), a.device, a.schedule)
end
combinedim(ctx, a::VirtualParallelDimension, b::VirtualSuggestedExtent) = a
function combinedim(ctx, a::VirtualParallelDimension, b::VirtualParallelDimension)
    @assert a.device == b.device
    @assert a.schedule == b.schedule
    VirtualParallelDimension(combinedim(ctx, a.ext, b.ext), a.device, a.schedule)
end

function resolvedim(ext::VirtualParallelDimension)
    VirtualParallelDimension(resolvedim(ext.ext), ext.device, ext.schedule)
end
function cache_dim!(ctx, tag, ext::VirtualParallelDimension)
    VirtualParallelDimension(cache_dim!(ctx, tag, ext.ext), ext.device, ext.schedule)
end

promote_rule(::Type{VirtualExtent}, ::Type{VirtualExtent}) = VirtualExtent

function shiftdim(ext::VirtualExtent, delta)
    VirtualExtent(;
        start=call(+, ext.start, delta),
        stop=call(+, ext.stop, delta),
    )
end
function shiftdim(ext::VirtualContinuousExtent, delta)
    VirtualContinuousExtent(;
        start=call(+, ext.start, delta),
        stop=call(+, ext.stop, delta),
    )
end

shiftdim(ext::Auto, delta) = auto
function shiftdim(ext::VirtualParallelDimension, delta)
    VirtualParallelDimension(ext, shiftdim(ext.ext, delta), ext.device, ext.schedule)
end

function shiftdim(ext::FinchNode, body)
    if ext.kind === virtual
        shiftdim(ext.val, body)
    else
        error("unimplemented")
    end
end

function scaledim(ext::VirtualExtent, scale)
    VirtualExtent(;
        start=call(*, ext.start, scale),
        stop=call(*, ext.stop, scale),
    )
end
function scaledim(ext::VirtualContinuousExtent, scale)
    VirtualContinuousExtent(;
        start=call(*, ext.start, scale),
        stop=call(*, ext.stop, scale),
    )
end

scaledim(ext::Auto, scale) = auto
function scaledim(ext::VirtualParallelDimension, scale)
    VirtualParallelDimension(ext, scaledim(ext.ext, scale), ext.device, ext.schedule)
end

function scaledim(ext::FinchNode, body)
    if ext.kind === virtual
        scaledim(ext.val, body)
    else
        error("unimplemented")
    end
end

#virtual_intersect(ctx, a, b) = virtual_intersect(ctx, promote(a, b)...)
function virtual_intersect(ctx, a, b)
    println(a, b)
    println("problem!")
    error()
end

virtual_intersect(ctx, a::Auto, b) = b
virtual_intersect(ctx, a, b::Auto) = a
virtual_intersect(ctx, a::Auto, b::Auto) = b

function virtual_intersect(ctx, a::VirtualExtent, b::VirtualExtent)
    VirtualExtent(;
        start=call(max, getstart(a), getstart(b)),
        stop=call(min, getstop(a), getstop(b)),
    )
end

virtual_union(ctx, a::Auto, b) = b
virtual_union(ctx, a, b::Auto) = a
virtual_union(ctx, a::Auto, b::Auto) = b

#virtual_union(ctx, a, b) = virtual_union(ctx, promote(a, b)...)
function virtual_union(ctx, a::VirtualExtent, b::VirtualExtent)
    VirtualExtent(;
        start=call(min, getstart(a), getstart(b)),
        stop=call(max, getstop(a), getstop(b)),
    )
end

similar_extent(ext::VirtualExtent, start, stop) = VirtualExtent(start, stop)
function similar_extent(ext::VirtualContinuousExtent, start, stop)
    VirtualContinuousExtent(start, stop)
end
function similar_extent(ext::FinchNode, start, stop)
    if ext.kind === virtual
        similar_extent(ext.val, start, stop)
    else
        similar_extent(ext, start, stop)
    end
end

is_continuous_extent(x) = false # generic
is_continuous_extent(x::VirtualContinuousExtent) = true
function is_continuous_extent(x::FinchNode)
    x.kind === virtual ? is_continuous_extent(x.val) : is_continuous_extent(x)
end

function Base.:(==)(a::VirtualContinuousExtent, b::VirtualContinuousExtent)
    a.start == b.start && a.stop == b.stop
end
function Base.:(==)(a::VirtualExtent, b::VirtualContinuousExtent)
    throw(ArgumentError("VirtualExtent and VirtualContinuousExtent cannot interact ...yet"))
end

function bound_measure_below!(ext::VirtualContinuousExtent, m)
    VirtualContinuousExtent(ext.start, bound_below!(ext.stop, call(+, ext.start, m)))
end
function bound_measure_above!(ext::VirtualContinuousExtent, m)
    VirtualContinuousExtent(ext.start, bound_above!(ext.stop, call(+, ext.start, m)))
end

function cache_dim!(ctx, var, ext::VirtualContinuousExtent)
    VirtualContinuousExtent(;
        start=cache!(ctx, Symbol(var, :_start), ext.start),
        stop=cache!(ctx, Symbol(var, :_stop), ext.stop),
    )
end

getunit(ext::VirtualExtent) = literal(1)
getunit(ext::VirtualContinuousExtent) = Eps
getunit(ext::FinchNode) = ext.kind === virtual ? getunit(ext.val) : ext

get_smallest_measure(ext::VirtualExtent) = literal(1)
get_smallest_measure(ext::VirtualContinuousExtent) = literal(0)
function get_smallest_measure(ext::FinchNode)
    ext.kind === virtual ? get_smallest_measure(ext.val) : ext
end

getstart(ext::VirtualContinuousExtent) = ext.start
getstart(ext::FinchNode) = ext.kind === virtual ? getstart(ext.val) : ext

getstop(ext::VirtualContinuousExtent) = ext.stop
getstop(ext::FinchNode) = ext.kind === virtual ? getstop(ext.val) : ext

measure(ext::VirtualContinuousExtent) = call(-, ext.stop, ext.start) # TODO: Think carefully, Not quite sure!

function combinedim(ctx, a::VirtualContinuousExtent, b::VirtualContinuousExtent)
    VirtualContinuousExtent(checklim(ctx, a.start, b.start), checklim(ctx, a.stop, b.stop))
end
combinedim(ctx, a::Auto, b::VirtualContinuousExtent) = b
function combinedim(ctx, a::VirtualExtent, b::VirtualContinuousExtent)
    throw(ArgumentError("VirtualExtent and VirtualContinuousExtent cannot interact ...yet"))
end

combinedim(ctx, a::VirtualSuggestedExtent, b::VirtualContinuousExtent) = b

is_continuous_extent(x::VirtualParallelDimension) = is_continuous_extent(x.ext)

function virtual_intersect(ctx, a::VirtualContinuousExtent, b::VirtualContinuousExtent)
    VirtualContinuousExtent(;
        start=call(max, getstart(a), getstart(b)),
        stop=call(min, getstop(a), getstop(b)),
    )
end

function virtual_union(ctx, a::VirtualContinuousExtent, b::VirtualContinuousExtent)
    VirtualContinuousExtent(;
        start=call(min, getstart(a), getstart(b)),
        stop=call(max, getstop(a), getstop(b)),
    )
end
