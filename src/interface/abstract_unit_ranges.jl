@kwdef mutable struct VirtualAbstractUnitRange
    tag
    data
    target
    arrtype
    eltype
end

function lower(ctx::AbstractCompiler, arr::VirtualAbstractUnitRange, ::DefaultStyle)
    return arr.ex
end

function virtualize(ctx, ex, arrtype::Type{<:AbstractUnitRange{T}}, tag=:tns) where {T}
    tag = freshen(ctx, tag)
    data = freshen(ctx, tag, :_data)
    push_preamble!(ctx, :($data = $ex))
    target = Extent(value(:(first($data)), T), value(:(last($data)), T))
    VirtualAbstractUnitRange(tag, data, target, arrtype, T)
end

function virtual_size(ctx::AbstractCompiler, arr::VirtualAbstractUnitRange)
    return [Extent(literal(1), value(:(length($(arr.data))), Int))]
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
                body=(ctx, i) -> FillLeaf(value(:($(arr.data)[$(ctx(i))])))
            ),
        )
    else
        throw(FinchProtocolError("$(arr.arrtype) is not writeable"))
    end
end

FinchNotation.finch_leaf(x::VirtualAbstractUnitRange) = virtual(x)

virtual_fill_value(ctx, ::VirtualAbstractUnitRange) = 0
virtual_eltype(ctx, tns::VirtualAbstractUnitRange) = tns.eltype
