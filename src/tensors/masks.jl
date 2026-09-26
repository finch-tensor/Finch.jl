struct DiagMask <: AbstractTensor end

"""
    diagmask

A mask for a diagonal tensor, `diagmask[i, j] = i == j`. Note that this
specializes each column for the cases where `i < j`, `i == j`, and `i > j`.
For a diagonal with offset `k`, use `diagmask[i, j - k]`.
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
            stop=(ctx, ext) -> j,
            body=(ctx, ext) -> truncate(
                ctx,
                Spike(; body=FillLeaf(false), tail=FillLeaf(true)),
                similar_extent(ext, getstart(ext), j),
                ext,
            ),
        ),
        Phase(; body=(ctx, ext) -> Run(; body=FillLeaf(false))),
    ])
end

struct UpTriMask <: AbstractTensor end

"""
    uptrimask

A mask for an upper triangular tensor, `uptrimask[i, j] = i <= j`. Note that this
specializes each column for the cases where `i <= j` and `i > j`.
For an upper triangle with offset `k`, use `uptrimask[i, j - k]`.
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

A mask for a lower triangular tensor, `lotrimask[i, j] = i >= j`. Note that this
specializes each column for the cases where `i < j` and `i >= j`.
For a lower triangle with offset `k`, use `lotrimask[i, j - k]`.
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
```jldoctest; setup=:(using Finch)
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
```jldoctest; setup=:(using Finch)
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

struct PairSumMask <: AbstractTensor end

"""
    pairsummask

A mask for summing adjacent pairs, `pairsummask[i, j] = 2i - 1 <= j <= 2i`.
Each column contains a single true entry at `i = cld(j, 2)`.
"""
const pairsummask = PairSumMask()

Base.show(io::IO, ex::PairSumMask) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::PairSumMask)
    print(io, "pairsummask")
end

struct VirtualPairSumMask <: AbstractVirtualTensor end

virtualize(ctx, ex, ::Type{PairSumMask}) = VirtualPairSumMask()
FinchNotation.finch_leaf(x::VirtualPairSumMask) = virtual(x)
Finch.virtual_size(ctx, ::VirtualPairSumMask) = (auto, auto)

function unfurl(ctx, arr::VirtualPairSumMask, ext, mode, proto::typeof(defaultread))
    Unfurled(;
        arr=arr,
        body=Lookup(;
            body=(ctx, j) -> VirtualDiagMaskColumn(call(cld, j, 2))
        ),
    )
end

struct PairCarryMask <: AbstractTensor end

"""
    paircarrymask

A mask for carrying partial pair sums, `paircarrymask[i, j] = 2j <= i <= 2j + 1`.
The first row is false. Each column specializes the interval containing its two
true entries.
"""
const paircarrymask = PairCarryMask()

Base.show(io::IO, ex::PairCarryMask) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::PairCarryMask)
    print(io, "paircarrymask")
end

struct VirtualPairCarryMask <: AbstractVirtualTensor end

virtualize(ctx, ex, ::Type{PairCarryMask}) = VirtualPairCarryMask()
FinchNotation.finch_leaf(x::VirtualPairCarryMask) = virtual(x)
Finch.virtual_size(ctx, ::VirtualPairCarryMask) = (auto, auto)

struct VirtualPairCarryMaskColumn
    j
end

FinchNotation.finch_leaf(x::VirtualPairCarryMaskColumn) = virtual(x)

function unfurl(ctx, arr::VirtualPairCarryMask, ext, mode, proto::typeof(defaultread))
    Unfurled(;
        arr=arr,
        body=Lookup(;
            body=(ctx, j) -> VirtualPairCarryMaskColumn(j)
        ),
    )
end

function unfurl(ctx, arr::VirtualPairCarryMaskColumn, ext, mode, proto::typeof(defaultread))
    Sequence([
        Phase(;
            stop=(ctx, ext) -> call(-, call(*, 2, arr.j), 1),
            body=(ctx, ext) -> Run(; body=FillLeaf(false)),
        ),
        Phase(;
            stop=(ctx, ext) -> call(+, call(*, 2, arr.j), 1),
            body=(ctx, ext) -> Run(; body=FillLeaf(true)),
        ),
        Phase(; body=(ctx, ext) -> Run(; body=FillLeaf(false))),
    ])
end

struct ReverseMask{Ti} <: AbstractTensor
    stop::Ti
end

"""
    reversemask(n)

A mask for reversing an axis of length `n`, `reversemask(n)[i, j] = j == n - i + 1`.
Each column specializes its single true entry. The row extent is inferred from
other tensors or the loop bounds.
"""
reversemask(stop) = ReverseMask(stop)

Base.show(io::IO, ex::ReverseMask) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::ReverseMask)
    print(io, "reversemask(", ex.stop, ")")
end

struct VirtualReverseMask <: AbstractVirtualTensor
    stop
end

function virtualize(ctx, ex, ::Type{ReverseMask{Ti}}) where {Ti}
    stop = freshen(ctx, :stop)
    push_preamble!(ctx, :($stop = $ex.stop))
    VirtualReverseMask(value(stop, Ti))
end

FinchNotation.finch_leaf(x::VirtualReverseMask) = virtual(x)
function virtual_size(ctx, arr::VirtualReverseMask)
    (auto, VirtualExtent(literal(1), arr.stop))
end

function unfurl(ctx, arr::VirtualReverseMask, ext, mode, proto::typeof(defaultread))
    Unfurled(;
        arr=arr,
        body=Lookup(;
            body=(ctx, j) -> VirtualDiagMaskColumn(call(+, call(-, arr.stop, j), 1))
        ),
    )
end

struct RollMask{Ti} <: AbstractTensor
    stop::Ti
    k::Int
end

"""
    rollmask(n, k=0)

A mask for rolling an axis of length `n` by `k`,
`rollmask(n, k)[i, j] = n > 0 && j == mod(i - k - 1, n) + 1`.
The row extent is inferred. Each column steps through its true entries with
period `n`, so rectangular uses may contain multiple true entries per column.
"""
rollmask(stop, k=0) = RollMask(stop, k)

Base.show(io::IO, ex::RollMask) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::RollMask)
    print(io, "rollmask(", ex.stop, ", ", ex.k, ")")
end

struct VirtualRollMask <: AbstractVirtualTensor
    stop
    k
end

function virtualize(ctx, ex, ::Type{RollMask{Ti}}) where {Ti}
    stop = freshen(ctx, :stop)
    k = freshen(ctx, :k)
    push_preamble!(
        ctx,
        quote
            $stop = $ex.stop
            $k = $ex.k
        end,
    )
    VirtualRollMask(value(stop, Ti), value(k, Int))
end

FinchNotation.finch_leaf(x::VirtualRollMask) = virtual(x)
function virtual_size(ctx, arr::VirtualRollMask)
    (auto, VirtualExtent(literal(1), arr.stop))
end

struct VirtualRollMaskColumn
    arr::VirtualRollMask
    j
end

FinchNotation.finch_leaf(x::VirtualRollMaskColumn) = virtual(x)

function unfurl(ctx, arr::VirtualRollMask, ext, mode, proto::typeof(defaultread))
    Unfurled(;
        arr=arr,
        body=Lookup(;
            body=(ctx, j) -> VirtualRollMaskColumn(arr, j)
        ),
    )
end

function unfurl(ctx, arr::VirtualRollMaskColumn, ext, mode, proto::typeof(defaultread))
    i = freshen(ctx, :roll_i)
    n = arr.arr.stop
    k = arr.arr.k
    j = arr.j
    Switch([
        call(>, n, 0) => Stepper(;
            seek=(ctx, ext) -> quote
                $i =
                    $(ctx(getstart(ext))) + mod(
                        $(ctx(j)) + $(ctx(k)) - $(ctx(getstart(ext))), $(ctx(n))
                    )
            end,
            stop=(ctx, ext) -> value(i),
            chunk=Spike(; body=FillLeaf(false), tail=FillLeaf(true)),
            next=(ctx, ext) -> :($i += $(ctx(n))),
        ),
        literal(true) => Run(; body=FillLeaf(false)),
    ])
end

struct RepeatMask <: AbstractTensor
    k::Int
end

"""
    repeatmask(k=0)

A mask for repeating each entry `k` times,
`repeatmask(k)[i, j] = k > 0 && j == fld(i - 1, k) + 1`.
Each column specializes the interval `k * (j - 1) < i <= k * j`.
For `k <= 0`, all entries are false. Both extents are inferred.
"""
repeatmask(k=0) = RepeatMask(k)

Base.show(io::IO, ex::RepeatMask) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::RepeatMask)
    print(io, "repeatmask(", ex.k, ")")
end

struct VirtualRepeatMask <: AbstractVirtualTensor
    k
end

function virtualize(ctx, ex, ::Type{RepeatMask})
    k = freshen(ctx, :k)
    push_preamble!(ctx, :($k = $ex.k))
    VirtualRepeatMask(value(k, Int))
end

FinchNotation.finch_leaf(x::VirtualRepeatMask) = virtual(x)
Finch.virtual_size(ctx, ::VirtualRepeatMask) = (auto, auto)

struct VirtualRepeatMaskColumn
    arr::VirtualRepeatMask
    j
end

FinchNotation.finch_leaf(x::VirtualRepeatMaskColumn) = virtual(x)

function unfurl(ctx, arr::VirtualRepeatMask, ext, mode, proto::typeof(defaultread))
    Unfurled(;
        arr=arr,
        body=Lookup(;
            body=(ctx, j) -> VirtualRepeatMaskColumn(arr, j)
        ),
    )
end

function unfurl(ctx, arr::VirtualRepeatMaskColumn, ext, mode, proto::typeof(defaultread))
    k = arr.arr.k
    j = arr.j
    Switch([
        call(>, k, 0) => Sequence([
            Phase(;
                stop=(ctx, ext) -> call(*, k, call(-, j, 1)),
                body=(ctx, ext) -> Run(; body=FillLeaf(false)),
            ),
            Phase(;
                stop=(ctx, ext) -> call(*, k, j),
                body=(ctx, ext) -> Run(; body=FillLeaf(true)),
            ),
            Phase(; body=(ctx, ext) -> Run(; body=FillLeaf(false))),
        ]),
        literal(true) => Run(; body=FillLeaf(false)),
    ])
end

struct OneHotMask{Ti} <: AbstractTensor
    index::Ti
end

"""
    onehotmask(index)

A vector mask with a single true entry, `onehotmask(index)[i] = i == index`.
The index is one-based and the extent is inferred from other tensors or loop
bounds. An index outside that extent produces an all-false mask.
"""
onehotmask(index) = OneHotMask(index)

Base.show(io::IO, ex::OneHotMask) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::OneHotMask)
    print(io, "onehotmask(", ex.index, ")")
end

struct VirtualOneHotMask <: AbstractVirtualTensor
    index
end

function virtualize(ctx, ex, ::Type{OneHotMask{Ti}}) where {Ti}
    index = freshen(ctx, :index)
    push_preamble!(ctx, :($index = $ex.index))
    VirtualOneHotMask(value(index, Ti))
end

FinchNotation.finch_leaf(x::VirtualOneHotMask) = virtual(x)
Finch.virtual_size(ctx, ::VirtualOneHotMask) = (auto,)

function unfurl(ctx, arr::VirtualOneHotMask, ext, mode, proto::typeof(defaultread))
    Unfurled(;
        arr=arr,
        body=unfurl(ctx, VirtualDiagMaskColumn(arr.index), ext, mode, proto),
    )
end

struct ParityMask <: AbstractTensor
    parity::Int
end

"""
    paritymask(parity=0)

A vector mask selecting alternating entries,
`paritymask(parity)[i] = mod(i - 1, 2) == parity`.
Parity refers to the zero-based position: `0` selects Julia indices `1, 3, 5, …`
and `1` selects `2, 4, 6, …`. Other values produce an all-false mask.
The extent is inferred.
"""
paritymask(parity=0) = ParityMask(parity)

Base.show(io::IO, ex::ParityMask) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::ParityMask)
    print(io, "paritymask(", ex.parity, ")")
end

struct VirtualParityMask <: AbstractVirtualTensor
    parity
end

function virtualize(ctx, ex, ::Type{ParityMask})
    parity = freshen(ctx, :parity)
    push_preamble!(ctx, :($parity = $ex.parity))
    VirtualParityMask(value(parity, Int))
end

FinchNotation.finch_leaf(x::VirtualParityMask) = virtual(x)
Finch.virtual_size(ctx, ::VirtualParityMask) = (auto,)

function unfurl(ctx, arr::VirtualParityMask, ext, mode, proto::typeof(defaultread))
    i = freshen(ctx, :parity_i)
    parity = arr.parity
    Unfurled(;
        arr=arr,
        body=Switch([
            call(and, call(<=, 0, parity), call(<=, parity, 1)) => Stepper(;
                seek=(ctx, ext) -> quote
                    $i =
                        $(ctx(getstart(ext))) +
                        mod($(ctx(parity)) + 1 - $(ctx(getstart(ext))), 2)
                end,
                stop=(ctx, ext) -> value(i),
                chunk=Spike(; body=FillLeaf(false), tail=FillLeaf(true)),
                next=(ctx, ext) -> :($i += 2),
            ),
            literal(true) => Run(; body=FillLeaf(false)),
        ]),
    )
end

function odd_even_merge_sort_is_left(i, n, p, k)
    i -= 1
    offset = mod(k, p)
    i + k < n && i >= offset && mod(i - offset, 2k) < k &&
        fld(i, 2p) == fld(i + k, 2p)
end

function odd_even_merge_sort_partner(i, n, p, k)
    if odd_even_merge_sort_is_left(i, n, p, k)
        i + k
    elseif i > k && odd_even_merge_sort_is_left(i - k, n, p, k)
        i - k
    else
        i
    end
end

struct OddEvenMergeSortPartnerMask{Ti} <: AbstractTensor
    stop::Ti
    p::Int
    k::Int
end

"""
    oddevenmergesortpartnermask(n, p, k)

A mask mapping each index to its compare-exchange partner in an odd-even merge
sort stage of length `n`. With zero-based position `r = i - 1` and
`offset = mod(k, p)`, `i` is a left endpoint when `r + k < n`, `r >= offset`,
`mod(r - offset, 2k) < k`, and `fld(r, 2p) == fld(r + k, 2p)`.
Partners exchange `i` and `i + k`; unpaired indices map to themselves.
Both `p` and `k` must be positive. The row extent is inferred.
"""
function oddevenmergesortpartnermask(stop, p, k)
    p > 0 && k > 0 || throw(ArgumentError("p and k must be positive"))
    OddEvenMergeSortPartnerMask(stop, p, k)
end

Base.show(io::IO, ex::OddEvenMergeSortPartnerMask) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::OddEvenMergeSortPartnerMask)
    print(io, "oddevenmergesortpartnermask(", ex.stop, ", ", ex.p, ", ", ex.k, ")")
end

struct VirtualOddEvenMergeSortPartnerMask <: AbstractVirtualTensor
    stop
    p
    k
end

function virtualize(ctx, ex, ::Type{OddEvenMergeSortPartnerMask{Ti}}) where {Ti}
    stop = freshen(ctx, :stop)
    p = freshen(ctx, :p)
    k = freshen(ctx, :k)
    push_preamble!(
        ctx,
        quote
            $stop = $ex.stop
            $p = $ex.p
            $k = $ex.k
        end,
    )
    VirtualOddEvenMergeSortPartnerMask(value(stop, Ti), value(p, Int), value(k, Int))
end

FinchNotation.finch_leaf(x::VirtualOddEvenMergeSortPartnerMask) = virtual(x)
function virtual_size(ctx, arr::VirtualOddEvenMergeSortPartnerMask)
    (auto, VirtualExtent(literal(1), arr.stop))
end

function unfurl(
    ctx, arr::VirtualOddEvenMergeSortPartnerMask, ext, mode, proto::typeof(defaultread)
)
    Unfurled(;
        arr=arr,
        body=Lookup(;
            body=(ctx, j) -> VirtualDiagMaskColumn(
                call(odd_even_merge_sort_partner, j, arr.stop, arr.p, arr.k)
            ),
        ),
    )
end

struct OddEvenMergeSortLowerMask{Ti} <: AbstractTensor
    stop::Ti
    p::Int
    k::Int
end

"""
    oddevenmergesortlowermask(n, p, k)

A vector mask selecting the left endpoints of the compare-exchanges in
`oddevenmergesortpartnermask(n, p, k)`. These endpoints receive the smaller value.
Both `p` and `k` must be positive.
"""
function oddevenmergesortlowermask(stop, p, k)
    p > 0 && k > 0 || throw(ArgumentError("p and k must be positive"))
    OddEvenMergeSortLowerMask(stop, p, k)
end

Base.show(io::IO, ex::OddEvenMergeSortLowerMask) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::OddEvenMergeSortLowerMask)
    print(io, "oddevenmergesortlowermask(", ex.stop, ", ", ex.p, ", ", ex.k, ")")
end

struct VirtualOddEvenMergeSortLowerMask <: AbstractVirtualTensor
    stop
    p
    k
end

function virtualize(ctx, ex, ::Type{OddEvenMergeSortLowerMask{Ti}}) where {Ti}
    stop = freshen(ctx, :stop)
    p = freshen(ctx, :p)
    k = freshen(ctx, :k)
    push_preamble!(
        ctx,
        quote
            $stop = $ex.stop
            $p = $ex.p
            $k = $ex.k
        end,
    )
    VirtualOddEvenMergeSortLowerMask(value(stop, Ti), value(p, Int), value(k, Int))
end

FinchNotation.finch_leaf(x::VirtualOddEvenMergeSortLowerMask) = virtual(x)
function virtual_size(ctx, arr::VirtualOddEvenMergeSortLowerMask)
    (VirtualExtent(literal(1), arr.stop),)
end

function unfurl(
    ctx, arr::VirtualOddEvenMergeSortLowerMask, ext, mode, proto::typeof(defaultread)
)
    Unfurled(;
        arr=arr,
        body=Lookup(;
            body=(ctx, i) -> FillLeaf(
                call(odd_even_merge_sort_is_left, i, arr.stop, arr.p, arr.k)
            ),
        ),
    )
end

struct ReshapeMask{M,N} <: AbstractTensor
    old_shape::NTuple{M,Int}
    new_shape::NTuple{N,Int}
end

Base.ndims(::ReshapeMask{M,N}) where {M,N} = M + N
Base.ndims(::Type{ReshapeMask{M,N}}) where {M,N} = M + N
Base.eltype(::ReshapeMask) = Bool
Base.eltype(::Type{ReshapeMask{M,N}}) where {M,N} = Bool
Base.size(tns::ReshapeMask) = (tns.old_shape..., tns.new_shape...)
Base.axes(tns::ReshapeMask) = map(n -> 1:n, size(tns))
fill_value(::ReshapeMask) = false
fill_value(::Type{ReshapeMask{M,N}}) where {M,N} = false

"""
    reshapemask(old_shape, new_shape)

A mask relating coordinates with equal row-major linear positions in two shapes.
The axes are `(old_shape..., new_shape...)`; the last axis of each shape varies
fastest, matching the Python pattern. Coordinates are one-based.
The shapes must have equal products. Empty tuples represent scalars.

For example, `reshapemask((2, 3), (3, 2))[i, j, k, l]` is true when
`3(i - 1) + j - 1 == 2(k - 1) + l - 1`.
"""
function reshapemask(old_shape, new_shape)
    old_shape = Tuple(Int(n) for n in old_shape)
    new_shape = Tuple(Int(n) for n in new_shape)
    all(n -> n >= 0, (old_shape..., new_shape...)) ||
        throw(ArgumentError("shape dimensions must be nonnegative"))
    prod(old_shape) == prod(new_shape) ||
        throw(DimensionMismatch("reshape shapes must have equal sizes"))
    ReshapeMask(old_shape, new_shape)
end

function Base.summary(io::IO, ex::ReshapeMask)
    print(io, "reshapemask(", ex.old_shape, ", ", ex.new_shape, ")")
end

struct VirtualReshapeMask <: AbstractVirtualTensor
    old_shape
    new_shape
end

function virtualize(ctx, ex, ::Type{ReshapeMask{M,N}}) where {M,N}
    old_shape = map(1:M) do d
        dim = freshen(ctx, :old_dim)
        push_preamble!(ctx, :($dim = $ex.old_shape[$d]))
        value(dim, Int)
    end
    new_shape = map(1:N) do d
        dim = freshen(ctx, :new_dim)
        push_preamble!(ctx, :($dim = $ex.new_shape[$d]))
        value(dim, Int)
    end
    VirtualReshapeMask(Tuple(old_shape), Tuple(new_shape))
end

FinchNotation.finch_leaf(x::VirtualReshapeMask) = virtual(x)
function virtual_size(ctx, arr::VirtualReshapeMask)
    map(n -> VirtualExtent(literal(1), n), (arr.old_shape..., arr.new_shape...))
end
virtual_fill_value(ctx, arr::VirtualReshapeMask) = false
virtual_eltype(ctx, arr::VirtualReshapeMask) = Bool

function instantiate(ctx, arr::VirtualReshapeMask, mode)
    isempty(arr.old_shape) && isempty(arr.new_shape) ? FillLeaf(true) : arr
end

struct VirtualReshapeMaskSlice
    strides
    offset
end

FinchNotation.finch_leaf(x::VirtualReshapeMaskSlice) = virtual(x)

function unfurl(ctx, arr::VirtualReshapeMask, ext, mode, proto::typeof(defaultread))
    old_strides = map(eachindex(arr.old_shape)) do d
        foldl((a, b) -> call(*, a, b), arr.old_shape[(d + 1):end]; init=literal(1))
    end
    new_strides = map(eachindex(arr.new_shape)) do d
        foldl((a, b) -> call(*, a, b), arr.new_shape[(d + 1):end]; init=literal(-1))
    end
    slice = VirtualReshapeMaskSlice((old_strides..., new_strides...), literal(0))
    Unfurled(; arr=arr, body=unfurl(ctx, slice, ext, mode, proto))
end

function unfurl(ctx, arr::VirtualReshapeMaskSlice, ext, mode, proto::typeof(defaultread))
    stride = arr.strides[end]
    if length(arr.strides) > 1
        Lookup(;
            body=(ctx, i) -> VirtualReshapeMaskSlice(
                arr.strides[1:(end - 1)],
                call(+, arr.offset, call(*, stride, call(-, i, 1))),
            ),
        )
    else
        Switch([
            call(!=, stride, 0) => Switch([
                call(==, call(mod, arr.offset, stride), 0) => unfurl(
                    ctx,
                    VirtualDiagMaskColumn(call(-, 1, call(fld, arr.offset, stride))),
                    ext,
                    mode,
                    proto,
                ),
                literal(true) => Run(; body=FillLeaf(false)),
            ]),
            literal(true) => Run(; body=FillLeaf(call(==, arr.offset, 0))),
        ])
    end
end

# SplitMix64's fixed-width mixing step, applied without mutable RNG state.
# Sebastiano Vigna's public-domain implementation: https://prng.di.unimi.it/splitmix64.c
function randommask_mix(x::UInt64)
    x += 0x9e3779b97f4a7c15
    x = xor(x, x >> 30) * 0xbf58476d1ce4e5b9
    x = xor(x, x >> 27) * 0x94d049bb133111eb
    xor(x, x >> 31)
end

randommask_uniform(h::UInt64) = Float64(h >> 11) * 0x1.0p-53

struct RandomMask{D} <: AbstractTensor
    shape::NTuple{D,Int}
    p::Float64
    seed::UInt64
end

Base.ndims(::RandomMask{D}) where {D} = D
Base.ndims(::Type{RandomMask{D}}) where {D} = D
Base.eltype(::RandomMask) = Bool
Base.eltype(::Type{RandomMask{D}}) where {D} = Bool
Base.size(tns::RandomMask) = tns.shape
Base.axes(tns::RandomMask) = map(n -> 1:n, tns.shape)
fill_value(::RandomMask) = false
fill_value(::Type{RandomMask{D}}) where {D} = false

"""
    randommask([rng], shape, p; seed=nothing)

A Boolean mask with reproducible random-looking entries and an approximate true
fraction of `p`. `shape` is a tuple of nonnegative dimensions, or an integer for
a vector. An empty tuple gives a scalar. The probability is stored as `Float64`
and must lie in `[0, 1]`; `p = 0` and `p = 1` give constant masks.

An indexed mask starts with a mixed 64-bit seed, then XORs in each one-based
coordinate and mixes again, from the last axis to the first. The high 53 bits of the result
are converted to a value in `[0, 1)` and compared with `p`. This is a pseudorandom
hash construction; it does not promise n-wise independence.
Scalar masks convert the seed directly to a value in `[0, 1)` and compare with `p`.
Each read specializes into separate true and false branches.

Reads consume no randomness and are independent of traversal order. The same
seed and coordinates give the same entry even when the shape changes. There is
no limit on the product of the dimensions, since coordinates are never flattened.
Only the shape, probability, and one seed are stored.

Pass `seed` as an integer in `[0, typemax(UInt64)]` to reproduce a mask directly.
If omitted, one seed is drawn from `rng` (the default RNG if omitted).

For example, `mask = randommask((100, 200), 0.1; seed=42)` creates a matrix mask
that can be read as `mask[i, j]` inside `@finch`.
"""
randommask(shape, p; seed=nothing) = randommask(default_rng(), shape, p; seed=seed)
function randommask(rng::AbstractRNG, shape::Integer, p::Real; seed=nothing)
    randommask(rng, (shape,), p; seed=seed)
end
function randommask(rng::AbstractRNG, shape::Tuple{Vararg{Integer}}, p::Real; seed=nothing)
    0 <= p <= 1 || throw(ArgumentError("probability must lie in [0, 1]"))
    shape = map(Int, shape)
    all(d -> d >= 0, shape) || throw(ArgumentError("shape dimensions must be nonnegative"))
    if seed === nothing
        seed = rand(rng, UInt64)
    else
        seed isa Integer && 0 <= seed <= typemax(UInt64) ||
            throw(ArgumentError("seed must be an integer in [0, typemax(UInt64)]"))
        seed = UInt64(seed)
    end
    RandomMask(shape, Float64(p), seed)
end

function Base.summary(io::IO, ex::RandomMask)
    print(io, "randommask(", ex.shape, ", ", ex.p, "; seed=", ex.seed, ")")
end

struct VirtualRandomMask <: AbstractVirtualTensor
    shape
    p
    seed
end

function virtualize(ctx, ex, ::Type{RandomMask{D}}) where {D}
    shape = ntuple(D) do d
        dim = freshen(ctx, :random_dim)
        push_preamble!(ctx, :($dim = $ex.shape[$d]))
        value(dim, Int)
    end
    p = freshen(ctx, :p)
    seed = freshen(ctx, :seed)
    push_preamble!(
        ctx,
        quote
            $p = $ex.p
            $seed = $ex.seed
        end,
    )
    VirtualRandomMask(shape, value(p, Float64), value(seed, UInt64))
end

FinchNotation.finch_leaf(x::VirtualRandomMask) = virtual(x)
function virtual_size(ctx, arr::VirtualRandomMask)
    map(n -> VirtualExtent(literal(1), n), arr.shape)
end
virtual_fill_value(ctx, ::VirtualRandomMask) = false
virtual_eltype(ctx, ::VirtualRandomMask) = Bool

function instantiate(ctx, arr::VirtualRandomMask, mode)
    if isempty(arr.shape)
        Switch([
            call(<, call(randommask_uniform, arr.seed), arr.p) =>
                FillLeaf(true),
            literal(true) => FillLeaf(false),
        ])
    else
        arr
    end
end

struct VirtualRandomMaskSlice
    ndims::Int
    p
    state
end

FinchNotation.finch_leaf(x::VirtualRandomMaskSlice) = virtual(x)

function unfurl(ctx, arr::VirtualRandomMask, ext, mode, proto::typeof(defaultread))
    slice = VirtualRandomMaskSlice(length(arr.shape), arr.p, call(randommask_mix, arr.seed))
    Unfurled(; arr=arr, body=unfurl(ctx, slice, ext, mode, proto))
end

function unfurl(ctx, arr::VirtualRandomMaskSlice, ext, mode, proto::typeof(defaultread))
    Lookup(;
        body=(ctx, i) -> begin
            state = call(randommask_mix, call(xor, arr.state, call(UInt64, i)))
            if arr.ndims == 1
                Switch([
                    call(<, call(randommask_uniform, state), arr.p) => FillLeaf(true),
                    literal(true) => FillLeaf(false),
                ])
            else
                VirtualRandomMaskSlice(arr.ndims - 1, arr.p, state)
            end
        end,
    )
end
