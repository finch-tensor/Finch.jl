FinchNotation.finch_leaf(x::Auto) = virtual(x)
FinchNotation.finch_leaf_instance(x::Auto) = value_instance(x)
virtualize(ctx, ex, ::Type{Auto}) = auto

getstart(::Auto) = error("asked for start of dimensionless range")
getstop(::Auto) = error("asked for stop of dimensionless range")

struct UnknownDimension end

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

@kwdef struct Extent
    start
    stop
end

@kwdef struct ContinuousExtent
    start
    stop
end

function virtual_call(ctx, ::typeof(extent), start, stop)
    if isfoldable(start) && isfoldable(stop)
        Extent(start, stop)
    end
end
function virtual_call(ctx, ::typeof(realextent), start, stop)
    if isfoldable(start) && isfoldable(stop)
        ContinuousExtent(start, stop)
    end
end

FinchNotation.finch_leaf(x::Extent) = virtual(x)
FinchNotation.finch_leaf(x::ContinuousExtent) = virtual(x)

Base.:(==)(a::Extent, b::Extent) =
    a.start == b.start &&
    a.stop == b.stop

bound_below!(val, below) = cached(val, literal(call(max, val, below)))

bound_above!(val, above) = cached(val, literal(call(min, val, above)))

function bound_measure_below!(ext::Extent, m)
    Extent(ext.start, bound_below!(ext.stop, call(+, ext.start, m)))
end
function bound_measure_above!(ext::Extent, m)
    Extent(ext.start, bound_above!(ext.stop, call(+, ext.start, m)))
end

function cache_dim!(ctx, var, ext::Extent)
    Extent(;
        start=cache!(ctx, Symbol(var, :_start), ext.start),
        stop=cache!(ctx, Symbol(var, :_stop), ext.stop),
    )
end

getstart(ext::Extent) = ext.start
getstop(ext::Extent) = ext.stop
measure(ext::Extent) = call(+, call(-, ext.stop, ext.start), 1)

function combinedim(ctx, a::Extent, b::Extent)
    Extent(;
        start=checklim(ctx, a.start, b.start),
        stop=checklim(ctx, a.stop, b.stop),
    )
end

combinedim(ctx, a::Auto, b::Extent) = b

struct SuggestedExtent
    ext
end

FinchNotation.finch_leaf(x::SuggestedExtent) = virtual(x)

Base.:(==)(a::SuggestedExtent, b::SuggestedExtent) = a.ext == b.ext

suggest(ext) = SuggestedExtent(ext)
suggest(ext::SuggestedExtent) = ext
suggest(ext::Auto) = auto

resolvedim(ext::Symbol) = error()
resolvedim(ext::SuggestedExtent) = resolvedim(ext.ext)
cache_dim!(ctx, tag, ext::SuggestedExtent) = SuggestedExtent(cache_dim!(ctx, tag, ext.ext))

combinedim(ctx, a::SuggestedExtent, b::Extent) = b

combinedim(ctx, a::SuggestedExtent, b::Auto) = a

function combinedim(ctx, a::SuggestedExtent, b::SuggestedExtent)
    SuggestedExtent(combinedim(ctx, a.ext, b.ext))
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

struct ParallelDimension
    ext
    device
end

"""
    parallel(ext, device=CPU(nthreads()))

A dimension `ext` that is parallelized over `device`. The `ext` field is usually
`_`, or dimensionless, but can be any standard dimension argument.
"""
parallel(dim, device=CPU(nthreads())) = ParallelDimension(dim, device)

function virtual_call(ctx, ::typeof(parallel), ext)
    if ext.kind === virtual
        n = cache!(ctx, :n, value(:(Threads.nthreads()), Int))
        virtual_call(ctx, parallel, ext, finch_leaf(VirtualCPU(nothing, n)))
    end
end

function virtual_call(ctx, ::typeof(parallel), ext, device)
    device = resolve(ctx, device) #TODO this feels broken
    if ext.kind === virtual
        ParallelDimension(ext.val, device)
    end
end

FinchNotation.finch_leaf(x::ParallelDimension) = virtual(x)

Base.:(==)(a::ParallelDimension, b::ParallelDimension) = a.ext == b.ext

getstart(ext::ParallelDimension) = getstart(ext.ext)
getstop(ext::ParallelDimension) = getstop(ext.ext)

function combinedim(ctx, a::ParallelDimension, b::Extent)
    ParallelDimension(resultdim(ctx, a.ext, b), a.device)
end
combinedim(ctx, a::ParallelDimension, b::SuggestedExtent) = a
function combinedim(ctx, a::ParallelDimension, b::ParallelDimension)
    @assert a.device == b.device
    ParallelDimension(combinedim(ctx, a.ext, b.ext), a.device)
end

resolvedim(ext::ParallelDimension) = ParallelDimension(resolvedim(ext.ext), ext.device)
function cache_dim!(ctx, tag, ext::ParallelDimension)
    ParallelDimension(cache_dim!(ctx, tag, ext.ext), ext.device)
end

promote_rule(::Type{Extent}, ::Type{Extent}) = Extent

function shiftdim(ext::Extent, delta)
    Extent(;
        start=call(+, ext.start, delta),
        stop=call(+, ext.stop, delta),
    )
end
function shiftdim(ext::ContinuousExtent, delta)
    ContinuousExtent(;
        start=call(+, ext.start, delta),
        stop=call(+, ext.stop, delta),
    )
end

shiftdim(ext::Auto, delta) = auto
function shiftdim(ext::ParallelDimension, delta)
    ParallelDimension(ext, shiftdim(ext.ext, delta), ext.device)
end

function shiftdim(ext::FinchNode, body)
    if ext.kind === virtual
        shiftdim(ext.val, body)
    else
        error("unimplemented")
    end
end

function scaledim(ext::Extent, scale)
    Extent(;
        start=call(*, ext.start, scale),
        stop=call(*, ext.stop, scale),
    )
end
function scaledim(ext::ContinuousExtent, scale)
    ContinuousExtent(;
        start=call(*, ext.start, scale),
        stop=call(*, ext.stop, scale),
    )
end

scaledim(ext::Auto, scale) = auto
function scaledim(ext::ParallelDimension, scale)
    ParallelDimension(ext, scaledim(ext.ext, scale), ext.device)
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

function virtual_intersect(ctx, a::Extent, b::Extent)
    Extent(;
        start=call(max, getstart(a), getstart(b)),
        stop=call(min, getstop(a), getstop(b)),
    )
end

virtual_union(ctx, a::Auto, b) = b
virtual_union(ctx, a, b::Auto) = a
virtual_union(ctx, a::Auto, b::Auto) = b

#virtual_union(ctx, a, b) = virtual_union(ctx, promote(a, b)...)
function virtual_union(ctx, a::Extent, b::Extent)
    Extent(;
        start=call(min, getstart(a), getstart(b)),
        stop=call(max, getstop(a), getstop(b)),
    )
end

make_extent(::Type, start, stop) = throw(ArgumentError("Unsupported type"))
make_extent(::Type{T}, start, stop) where {T<:Integer} = Extent(start, stop)
make_extent(::Type{T}, start, stop) where {T<:Real} = ContinuousExtent(start, stop)
make_extent(::Type{T}, start, stop) where {T<:Limit} = ContinuousExtent(start, stop)

similar_extent(ext::Extent, start, stop) = Extent(start, stop)
similar_extent(ext::ContinuousExtent, start, stop) = ContinuousExtent(start, stop)
function similar_extent(ext::FinchNode, start, stop)
    if ext.kind === virtual
        similar_extent(ext.val, start, stop)
    else
        similar_extent(ext, start, stop)
    end
end

is_continuous_extent(x) = false # generic
is_continuous_extent(x::ContinuousExtent) = true
function is_continuous_extent(x::FinchNode)
    x.kind === virtual ? is_continuous_extent(x.val) : is_continuous_extent(x)
end

function Base.:(==)(a::ContinuousExtent, b::ContinuousExtent)
    a.start == b.start && a.stop == b.stop
end
function Base.:(==)(a::Extent, b::ContinuousExtent)
    throw(ArgumentError("Extent and ContinuousExtent cannot interact ...yet"))
end

function bound_measure_below!(ext::ContinuousExtent, m)
    ContinuousExtent(ext.start, bound_below!(ext.stop, call(+, ext.start, m)))
end
function bound_measure_above!(ext::ContinuousExtent, m)
    ContinuousExtent(ext.start, bound_above!(ext.stop, call(+, ext.start, m)))
end

function cache_dim!(ctx, var, ext::ContinuousExtent)
    ContinuousExtent(;
        start=cache!(ctx, Symbol(var, :_start), ext.start),
        stop=cache!(ctx, Symbol(var, :_stop), ext.stop),
    )
end

getunit(ext::Extent) = literal(1)
getunit(ext::ContinuousExtent) = Eps
getunit(ext::FinchNode) = ext.kind === virtual ? getunit(ext.val) : ext

get_smallest_measure(ext::Extent) = literal(1)
get_smallest_measure(ext::ContinuousExtent) = literal(0)
function get_smallest_measure(ext::FinchNode)
    ext.kind === virtual ? get_smallest_measure(ext.val) : ext
end

getstart(ext::ContinuousExtent) = ext.start
getstart(ext::FinchNode) = ext.kind === virtual ? getstart(ext.val) : ext

getstop(ext::ContinuousExtent) = ext.stop
getstop(ext::FinchNode) = ext.kind === virtual ? getstop(ext.val) : ext

measure(ext::ContinuousExtent) = call(-, ext.stop, ext.start) # TODO: Think carefully, Not quite sure!

function combinedim(ctx, a::ContinuousExtent, b::ContinuousExtent)
    ContinuousExtent(checklim(ctx, a.start, b.start), checklim(ctx, a.stop, b.stop))
end
combinedim(ctx, a::Auto, b::ContinuousExtent) = b
function combinedim(ctx, a::Extent, b::ContinuousExtent)
    throw(ArgumentError("Extent and ContinuousExtent cannot interact ...yet"))
end

combinedim(ctx, a::SuggestedExtent, b::ContinuousExtent) = b

is_continuous_extent(x::ParallelDimension) = is_continuous_extent(x.dim)

function virtual_intersect(ctx, a::ContinuousExtent, b::ContinuousExtent)
    ContinuousExtent(;
        start=call(max, getstart(a), getstart(b)),
        stop=call(min, getstop(a), getstop(b)),
    )
end

function virtual_union(ctx, a::ContinuousExtent, b::ContinuousExtent)
    ContinuousExtent(;
        start=call(min, getstart(a), getstart(b)),
        stop=call(max, getstop(a), getstop(b)),
    )
end
