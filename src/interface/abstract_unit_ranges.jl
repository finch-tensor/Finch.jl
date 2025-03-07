@kwdef mutable struct VirtualAbstractUnitRange
    id
    ex
    target
    arrtype
    eltype
end

function lower(ctx::AbstractCompiler, arr::VirtualAbstractUnitRange, ::DefaultStyle)
    return arr.ex
end

function virtualize(ctx, ex, arrtype::Type{<:AbstractUnitRange{T}}, tag=:tns) where {T}
    sym = freshen(ctx, tag)
    push_preamble!(ctx, :($sym = $ex))
    target = Extent(value(:(first($sym)), T), value(:(last($sym)), T))
    VirtualAbstractUnitRange(sym, sym, target, arrtype, T)
end

function virtual_size(ctx::AbstractCompiler, arr::VirtualAbstractUnitRange)
    return [Extent(literal(1), value(:(length($(arr.ex))), Int))]
end

virtual_resize!(ctx::AbstractCompiler, arr::VirtualAbstractUnitRange, idx_dim) = arr

function declare!(ctx::AbstractCompiler, arr::VirtualAbstractUnitRange, init)
    throw(FinchProtocolError("$(arr.arrtype) is not writeable"))
end

function unfurl(ctx::AbstractCompiler, arr::VirtualAbstractUnitRange, ext, mode, proto)
    if mode.kind === reader
        Unfurled(;
            arr=arr,
            body=Lookup(;
                body=(ctx, i) -> FillLeaf(value(:($(arr.ex)[$(ctx(i))])))
            ),
        )
    else
        throw(FinchProtocolError("$(arr.arrtype) is not writeable"))
    end
end

FinchNotation.finch_leaf(x::VirtualAbstractUnitRange) = virtual(x)

virtual_fill_value(ctx, ::VirtualAbstractUnitRange) = 0
virtual_eltype(ctx, tns::VirtualAbstractUnitRange) = tns.eltype
