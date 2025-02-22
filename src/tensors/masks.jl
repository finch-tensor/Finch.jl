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
    Lookup(;
        body=(ctx, j) -> Provenance(
            path = SubSliceOf(j),
            body = VirtualDiagMaskColumn(j)
        ),
    )
end

function unfurl(ctx, arr::VirtualDiagMaskColumn, ext, mode, proto::typeof(defaultread))
    j = arr.j
    Sequence([
        Phase(;
            stop = (ctx, ext) -> value(:($(ctx(j)) - 1)),
            body = (ctx, ext) -> Run(; body=FillLeaf(false)),
        ),
        Phase(;
            stop = (ctx, ext) -> j,
            body = (ctx, ext) -> Run(; body=FillLeaf(true)),
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
    body=Lookup(;
        body=(ctx, j) -> Provenance(;
            path=SubSliceOf(j),
            body=VirtualUpTriMaskColumn(j)
        ),
    )
end

function unfurl(ctx, arr::VirtualUpTriMaskColumn, ext, mode, proto::typeof(defaultread))
    j = arr.j
    Sequence([
        Phase(;
            stop = (ctx, ext) -> value(:($(ctx(j)))),
            body = (ctx, ext) -> Run(; body=FillLeaf(true)),
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
    Lookup(;
        body=(ctx, j) -> Provenance(;
            path=SubSliceOf(j),
            body=VirtualLoTriMaskColumn(j)
        ),
    )
end

function unfurl(ctx, arr::VirtualLoTriMaskColumn, ext, mode, proto::typeof(defaultread))
    j = arr.j
    Sequence([
        Phase(;
            stop = (ctx, ext) -> value(:($(ctx(j)) - 1)),
            body = (ctx, ext) -> Run(; body=FillLeaf(false)),
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

function unfurl(ctx, arr::VirtualBandMask, ext, mode, proto::typeof(defaultread))
    Provenance(;
        arr=arr,
        body=Lookup(;
            body=(ctx, j_lo) -> Provenance(;
                path = SubSliceOf(j_lo),
                body = VirtualBandMaskSlice(j_lo)
            )
        ),
    )
end

function unfurl(ctx, arr::VirtualBandMaskSlice, ext, mode, proto::typeof(defaultread))
    Lookup(;
        body=(ctx, j_hi) -> Provenance(;
            path = SubSliceOf(j_hi),
            body = VirtualBandMaskColumn(arr.j_lo, j_hi)
        )
    )
end

function unfurl(ctx, arr::VirtualBandMaskColumn, ext, mode, proto::typeof(defaultread))
    Sequence([
        Phase(;
            stop = (ctx, ext) -> value(:($(ctx(j)) - 1)),
            body = (ctx, ext) -> Run(; body=FillLeaf(false)),
        ),
        Phase(;
            stop = (ctx, ext) -> k,
            body = (ctx, ext) -> Run(; body=FillLeaf(true)),
        ),
        Phase(;
            body=(ctx, ext) -> Run(; body=FillLeaf(false))
        ),
    ])
end

struct SplitMask <: AbstractTensor
    P::Int
end

Base.show(io::IO, ex::SplitMask) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::SplitMask)
    print(io, "splitmask(", ex.P, ")")
end

struct VirtualSplitMask
    P
end

function virtualize(ctx, ex, ::Type{SplitMask})
    return VirtualSplitMask(value(:($ex.P), Int))
end

FinchNotation.finch_leaf(x::VirtualSplitMask) = virtual(x)
Finch.virtual_size(ctx, arr::VirtualSplitMask) = (auto, Extent(literal(1), arr.P))

struct VirtualSplitMaskColumn
    P
    j
end

FinchNotation.finch_leaf(x::VirtualSplitMaskColumn) = virtual(x)

function unfurl(ctx, arr::VirtualSplitMask, ext, mode, proto::typeof(defaultread))
    Lookup(;
        body=(ctx, j) -> Provenance(
            path = SubSliceOf(j),
            body = VirtualSplitMaskColumn(arr.P, j)
        ),
    )
end

function unfurl(ctx, arr::VirtualSplitMaskColumn, ext_2, mode, proto::typeof(defaultread))
    j = arr.j
    P = arr.P
    Sequence([
        Phase(;
            stop = (ctx, ext) -> call(+, call(-, getstart(ext_2), 1), call(fld, call(*, measure(ext_2), call(-, j, 1)), P)),
            body = (ctx, ext) -> Run(; body=FillLeaf(false)),
        ),
        Phase(;
            stop = (ctx, ext) -> call(+, call(-, getstart(ext_2), 1), call(fld, call(*, measure(ext_2), j), P)),
            body = (ctx, ext) -> Run(; body=FillLeaf(true)),
        ),
        Phase(; body=(ctx, ext) -> Run(; body=FillLeaf(false))),
    ])
end

struct ChunkMask{Dim} <: AbstractTensor
    b::Int
    dim::Dim
end

Base.show(io::IO, ex::ChunkMask) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::ChunkMask)
    print(io, "chunkmask(", ex.b, ex.dim, ")")
end

struct VirtualChunkMask
    b
    dim
end

function virtualize(ctx, ex, ::Type{ChunkMask{Dim}}) where {Dim}
    return VirtualChunkMask(
        value(:($ex.b), Int),
        virtualize(ctx, :($ex.dim), Dim))
end

"""
    chunkmask(b)

A mask for a chunked tensor, `chunkmask[i, j] = b * (j - 1) < i <= b * j`. Note
that this specializes each column for the cases where `i < b * (j - 1)`, `b * (j
- 1) < i <= b * j`, and `b * j < i`.
"""
function chunkmask end

function Finch.virtual_call(ctx, ::typeof(chunkmask), b, dim)
    if dim.kind === virtual
        return VirtualChunkMask(b, dim.val)
    end
end

FinchNotation.finch_leaf(x::VirtualChunkMask) = virtual(x)
function Finch.virtual_size(ctx, arr::VirtualChunkMask)
    (arr.dim, Extent(literal(1), call(cld, measure(arr.dim), arr.b)))
end

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
        body=Sequence([
            Phase(;
                stop = (ctx, ext) -> call(cld, measure(arr.dim), arr.b),
                body = (ctx, ext) -> Lookup(;
                body=(ctx, j) -> Provenance(;
                    path=SubSliceOf(j),
                    VirtualChunkMaskColumn(arr, j),
                )
            ),
            ),
            Phase(;
                body=(ctx, ext) -> Run(;
                    body=Provenance(;
                        path=SubSliceOf(j),
                        body = VirtualChunkMaskCleanupColumn(arr)
                    )
                ),
            ),
        ]),
    )
end

function unfurl(ctx, arr::VirtualChunkMaskColumn, ext, mode, proto::typeof(defaultread))
    j = arr.j
    Sequence([
        Phase(;
            stop = (ctx, ext) -> call(*, arr.arr.b, call(-, j, 1)),
            body = (ctx, ext) -> Run(; body=FillLeaf(false)),
        ),
        Phase(;
            stop = (ctx, ext) -> call(*, arr.arr.b, j),
            body = (ctx, ext) -> Run(; body=FillLeaf(true)),
        ),
        Phase(; body=(ctx, ext) -> Run(; body=FillLeaf(false))),
    ])
end

function unfurl(
    ctx, arr::VirtualChunkMaskCleanupColumn, ext, mode, proto::typeof(defaultread)
)
    Sequence([
        Phase(;
            stop = (ctx, ext) -> call(*, call(fld, measure(arr.arr.dim), arr.arr.b), arr.arr.b),
            body = (ctx, ext) -> Run(; body=FillLeaf(false)),
        ),
        Phase(;
            body=(ctx, ext) -> Run(; body=FillLeaf(true))
        ),
    ])
end
