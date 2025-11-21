struct DiagMask <: AbstractTensor end

"""
    diagmask

A mask for a diagonal tensor, `diagmask[i, j] = i == j`. Note that this
specializes each column for the cases where `i < j`, `i == j`, and `i > j`.
"""
const diagmask = DiagMask()

Base.show(io::IO, ex::DiagMask) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::DiagMask)
    print(io, "diagmask")
end

struct VirtualDiagMask <: AbstractVirtualTensor end

virtualize(ctx, ex, ::Type{DiagMask}) = VirtualDiagMask()
FinchNotation.finch_leaf(x::VirtualDiagMask) = virtual(x)
Finch.virtual_size(ctx, ::VirtualDiagMask) = (auto, auto)

struct VirtualDiagMaskColumn
    j
end

FinchNotation.finch_leaf(x::VirtualDiagMaskColumn) = virtual(x)

function unfurl(ctx, arr::VirtualDiagMask, ext, mode, proto::typeof(defaultread))
    Unfurled(;
        arr=arr,
        body=Lookup(;
            body=(ctx, j) -> VirtualDiagMaskColumn(j)
        ),
    )
end

function unfurl(ctx, arr::VirtualDiagMaskColumn, ext, mode, proto::typeof(defaultread))
    j = arr.j
    Sequence([
        Phase(;
            stop=(ctx, ext) -> value(:($(ctx(j)) - 1)),
            body=(ctx, ext) -> Run(; body=FillLeaf(false)),
        ),
        Phase(;
            stop=(ctx, ext) -> j,
            body=(ctx, ext) -> Run(; body=FillLeaf(true)),
        ),
        Phase(; body=(ctx, ext) -> Run(; body=FillLeaf(false))),
    ])
end

struct UpTriMask <: AbstractTensor end

"""
    uptrimask

A mask for an upper triangular tensor, `uptrimask[i, j] = i <= j`. Note that this
specializes each column for the cases where `i <= j` and `i > j`.
"""
const uptrimask = UpTriMask()

Base.show(io::IO, ex::UpTriMask) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::UpTriMask)
    print(io, "uptrimask")
end

struct VirtualUpTriMask <: AbstractVirtualTensor end

virtualize(ctx, ex, ::Type{UpTriMask}) = VirtualUpTriMask()
FinchNotation.finch_leaf(x::VirtualUpTriMask) = virtual(x)
Finch.virtual_size(ctx, ::VirtualUpTriMask) = (auto, auto)

struct VirtualUpTriMaskColumn
    j
end

FinchNotation.finch_leaf(x::VirtualUpTriMaskColumn) = virtual(x)

function unfurl(ctx, arr::VirtualUpTriMask, ext, mode, proto::typeof(defaultread))
    Unfurled(;
        arr=arr,
        body=Lookup(;
            body=(ctx, j) -> VirtualUpTriMaskColumn(j)
        ),
    )
end

function unfurl(ctx, arr::VirtualUpTriMaskColumn, ext, mode, proto::typeof(defaultread))
    j = arr.j
    Sequence([
        Phase(;
            stop=(ctx, ext) -> value(:($(ctx(j)))),
            body=(ctx, ext) -> Run(; body=FillLeaf(true)),
        ),
        Phase(;
            body=(ctx, ext) -> Run(; body=FillLeaf(false))
        ),
    ])
end

struct LoTriMask <: AbstractTensor end

"""
    lotrimask

A mask for an upper triangular tensor, `lotrimask[i, j] = i >= j`. Note that this
specializes each column for the cases where `i < j` and `i >= j`.
"""
const lotrimask = LoTriMask()

Base.show(io::IO, ex::LoTriMask) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::LoTriMask)
    print(io, "lotrimask")
end

struct VirtualLoTriMask <: AbstractVirtualTensor end

virtualize(ctx, ex, ::Type{LoTriMask}) = VirtualLoTriMask()
FinchNotation.finch_leaf(x::VirtualLoTriMask) = virtual(x)
Finch.virtual_size(ctx, ::VirtualLoTriMask) = (auto, auto)

struct VirtualLoTriMaskColumn
    j
end

FinchNotation.finch_leaf(x::VirtualLoTriMaskColumn) = virtual(x)

function unfurl(ctx, arr::VirtualLoTriMask, ext, mode, proto::typeof(defaultread))
    Unfurled(;
        arr=arr,
        body=Lookup(;
            body=(ctx, j) -> VirtualLoTriMaskColumn(j)
        ),
    )
end

function unfurl(ctx, arr::VirtualLoTriMaskColumn, ext, mode, proto::typeof(defaultread))
    j = arr.j
    Sequence([
        Phase(;
            stop=(ctx, ext) -> value(:($(ctx(j)) - 1)),
            body=(ctx, ext) -> Run(; body=FillLeaf(false)),
        ),
        Phase(;
            body=(ctx, ext) -> Run(; body=FillLeaf(true))
        ),
    ])
end

struct BandMask <: AbstractTensor end

"""
    bandmask

A mask for a banded tensor, `bandmask[i, j, k] = j <= i <= k`. Note that this
specializes each column for the cases where `i < j`, `j <= i <= k`, and `k < i`.
"""
const bandmask = BandMask()

Base.show(io::IO, ex::BandMask) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::BandMask)
    print(io, "bandmask")
end

struct VirtualBandMask <: AbstractVirtualTensor end

virtualize(ctx, ex, ::Type{BandMask}) = VirtualBandMask()
FinchNotation.finch_leaf(x::VirtualBandMask) = virtual(x)
Finch.virtual_size(ctx, ::VirtualBandMask) = (auto, auto, auto)

struct VirtualBandMaskSlice
    j_lo
end

FinchNotation.finch_leaf(x::VirtualBandMaskSlice) = virtual(x)

struct VirtualBandMaskColumn
    j_lo
    j_hi
end

FinchNotation.finch_leaf(x::VirtualBandMaskColumn) = virtual(x)
Finch.virtual_size(ctx, ::VirtualBandMaskColumn) = (auto,)

function unfurl(ctx, arr::VirtualBandMask, ext, mode, proto::typeof(defaultread))
    Unfurled(;
        arr=arr,
        body=Lookup(;
            body=(ctx, j_lo) -> VirtualBandMaskSlice(j_lo)
        ),
    )
end

function unfurl(ctx, arr::VirtualBandMaskSlice, ext, mode, proto::typeof(defaultread))
    Lookup(;
        body=(ctx, j_hi) -> VirtualBandMaskColumn(arr.j_lo, j_hi)
    )
end

function unfurl(ctx, arr::VirtualBandMaskColumn, ext, mode, proto::typeof(defaultread))
    Sequence([
        Phase(;
            stop=(ctx, ext) -> call(-, arr.j_lo, 1),
            body=(ctx, ext) -> Run(; body=FillLeaf(false)),
        ),
        Phase(;
            stop=(ctx, ext) -> arr.j_hi,
            body=(ctx, ext) -> Run(; body=FillLeaf(true)),
        ),
        Phase(;
            body=(ctx, ext) -> Run(; body=FillLeaf(false))
        ),
    ])
end

struct SplitMask{Ti} <: AbstractTensor
    stop::Ti
    P::Int
end

Base.ndims(::SplitMask) = 2
Base.ndims(::Type{SplitMask{Ti}}) where {Ti} = 2
Base.eltype(::SplitMask) = Bool
Base.eltype(::Type{SplitMask{Ti}}) where {Ti} = Bool
Base.size(tns::SplitMask) = (tns.stop, tns.P)
Base.axes(tns::SplitMask) = (1:(tns.stop), 1:(tns.P))
fill_value(::SplitMask) = false
fill_value(::Type{SplitMask{Ti}}) where {Ti} = false

"""
    splitmask(n, P)

A mask to evenly divide `n` indices into P regions. If `M = splitmask(P, n)`,
then `M[i, j] = fld(n * (j - 1), P) <= i < fld(n * j, P)`.
```jldoctest setup=:(using Finch)
julia> splitmask(10, 3)
10×3 Finch.SplitMask{Int64}:
 1  0  0
 1  0  0
 1  0  0
 0  1  0
 0  1  0
 0  1  0
 0  0  1
 0  0  1
 0  0  1
 0  0  1

```
"""
splitmask(stop, P) = SplitMask(stop, P)

function Base.summary(io::IO, ex::SplitMask)
    print(io, "splitmask(", ex.stop, ", ", ex.P, ")")
end

struct VirtualSplitMask
    stop
    P
end

function virtualize(ctx, ex, ::Type{SplitMask{Ti}}) where {Ti}
    P = freshen(ctx, :P)
    stop = freshen(ctx, :stop)
    push_preamble!(
        ctx,
        quote
            $P = $ex.P
            $stop = $ex.stop
        end,
    )
    return VirtualSplitMask(value(stop, Ti), value(P, Int))
end

FinchNotation.finch_leaf(x::VirtualSplitMask) = virtual(x)
function virtual_size(ctx, arr::VirtualSplitMask)
    (VirtualExtent(literal(1), arr.stop), VirtualExtent(literal(1), arr.P))
end
virtual_fill_value(ctx, arr::VirtualSplitMask) = false
virtual_eltype(ctx, arr::VirtualSplitMask) = Bool

struct VirtualSplitMaskColumn
    arr
    j
end

FinchNotation.finch_leaf(x::VirtualSplitMaskColumn) = virtual(x)

function unfurl(ctx, arr::VirtualSplitMask, ext, mode, proto::typeof(defaultread))
    Unfurled(;
        arr=arr,
        body=Lookup(;
            body=(ctx, j) -> VirtualSplitMaskColumn(arr, j)
        ),
    )
end

function unfurl(ctx, arr::VirtualSplitMaskColumn, ext_2, mode, proto::typeof(defaultread))
    j = arr.j
    P = arr.arr.P
    Sequence([
        Phase(;
            stop=(ctx, ext) -> call(fld, call(*, arr.arr.stop, call(-, j, 1)), P),
            body=(ctx, ext) -> Run(; body=FillLeaf(false)),
        ),
        Phase(;
            stop=(ctx, ext) -> call(fld, call(*, arr.arr.stop, j), P),
            body=(ctx, ext) -> Run(; body=FillLeaf(true)),
        ),
        Phase(; body=(ctx, ext) -> Run(; body=FillLeaf(false))),
    ])
end

struct ChunkMask{Ti} <: AbstractTensor
    stop::Ti
    b::Int
end

Base.ndims(::ChunkMask) = 2
Base.ndims(::Type{ChunkMask{Ti}}) where {Ti} = 2
Base.eltype(::ChunkMask) = Bool
Base.eltype(::Type{ChunkMask{Ti}}) where {Ti} = Bool
Base.size(tns::ChunkMask) = (tns.stop, cld(tns.stop, tns.b))
Base.axes(tns::ChunkMask) = (1:(tns.stop), cld(tns.stop, tns.b))
fill_value(::ChunkMask) = false
fill_value(::Type{ChunkMask{Ti}}) where {Ti} = false

function Base.summary(io::IO, ex::ChunkMask)
    print(io, "chunkmask(", ex.stop, ", ", ex.b, ")")
end

struct VirtualChunkMask
    stop
    b
end

function virtualize(ctx, ex, ::Type{ChunkMask{Ti}}) where {Ti}
    b = freshen(ctx, :b)
    stop = freshen(ctx, :stop)
    push_preamble!(
        ctx,
        quote
            $b = $ex.b
            $stop = $ex.stop
        end,
    )
    return VirtualChunkMask(value(stop, Ti), value(b, Int))
end

"""
    chunkmask(n, b)

A mask to evenly divide `n` indices into regions of size `b`. If `m =
chunkmask(b, n)`, then `m[i, j] = b * (j - 1) < i <= b * j`. Note that this
specializes for the cleanup case at the end of the range.
```jldoctest setup=:(using Finch)
julia> chunkmask(10, 3)
10×4 Finch.ChunkMask{Int64}:
 1  0  0  0
 1  0  0  0
 1  0  0  0
 0  1  0  0
 0  1  0  0
 0  1  0  0
 0  0  1  0
 0  0  1  0
 0  0  1  0
 0  0  0  1

```
"""
chunkmask(stop, b) = ChunkMask(stop, b)

FinchNotation.finch_leaf(x::VirtualChunkMask) = virtual(x)
function virtual_size(ctx, arr::VirtualChunkMask)
    (
        VirtualExtent(literal(1), arr.stop),
        VirtualExtent(literal(1), call(cld, arr.stop, arr.b)),
    )
end
virtual_fill_value(ctx, arr::VirtualChunkMask) = false
virtual_eltype(ctx, arr::VirtualChunkMask) = Bool

struct VirtualChunkMaskColumn
    arr::VirtualChunkMask
    j
end

struct VirtualChunkMaskCleanupColumn
    arr::VirtualChunkMask
end

FinchNotation.finch_leaf(x::VirtualChunkMaskColumn) = virtual(x)
FinchNotation.finch_leaf(x::VirtualChunkMaskCleanupColumn) = virtual(x)

function unfurl(ctx, arr::VirtualChunkMask, ext, mode, proto::typeof(defaultread))
    Unfurled(;
        arr=arr,
        body=Sequence([
            Phase(;
                stop=(ctx, ext) -> call(cld, arr.stop, arr.b),
                body=(ctx, ext) -> Lookup(;
                    body=(ctx, j) -> VirtualChunkMaskColumn(arr, j)
                ),
            ),
            Phase(;
                body=(ctx, ext) -> Run(;
                    body=VirtualChunkMaskCleanupColumn(arr)
                ),
            ),
        ]),
    )
end

function unfurl(ctx, arr::VirtualChunkMaskColumn, ext, mode, proto::typeof(defaultread))
    j = arr.j
    Sequence([
        Phase(;
            stop=(ctx, ext) -> call(*, arr.arr.b, call(-, j, 1)),
            body=(ctx, ext) -> Run(; body=FillLeaf(false)),
        ),
        Phase(;
            stop=(ctx, ext) -> call(*, arr.arr.b, j),
            body=(ctx, ext) -> Run(; body=FillLeaf(true)),
        ),
        Phase(; body=(ctx, ext) -> Run(; body=FillLeaf(false))),
    ])
end

function unfurl(
    ctx, arr::VirtualChunkMaskCleanupColumn, ext, mode, proto::typeof(defaultread)
)
    Sequence([
        Phase(;
            stop=(ctx, ext) -> call(*, call(fld, arr.arr.stop, arr.arr.b), arr.arr.b),
            body=(ctx, ext) -> Run(; body=FillLeaf(false)),
        ),
        Phase(;
            body=(ctx, ext) -> Run(; body=FillLeaf(true))
        ),
    ])
end
