"""
    PatternLevel{[Tp=Int]}()

A subfiber of a pattern level is the Boolean value true, but it's `fill_value` is
false. PatternLevels are used to create tensors that represent which values
are stored by other fibers. See [`pattern!`](@ref) for usage examples.

```jldoctest
julia> tensor_tree(Tensor(Dense(Pattern()), 3))
3-Tensor
└─ Dense [1:3]
   ├─ [1]: true
   ├─ [2]: true
   └─ [3]: true
```
"""
struct PatternLevel{Tp} <: AbstractLevel end
const Pattern = PatternLevel

PatternLevel() = PatternLevel{Int}()

Base.summary(::Pattern) = "Pattern()"
similar_level(::PatternLevel, ::Any, ::Type, ::Vararg) = PatternLevel()

countstored_level(lvl::PatternLevel, pos) = pos

labelled_show(io::IO, ::SubFiber{<:PatternLevel}) =
    print(io, true)

Base.resize!(lvl::PatternLevel) = lvl

pattern!(::PatternLevel{Tp}) where {Tp} = Pattern{Tp}()

function Base.show(io::IO, lvl::PatternLevel)
    print(io, "Pattern()")
end

@inline level_ndims(::Type{<:PatternLevel}) = 0
@inline level_size(::PatternLevel) = ()
@inline level_axes(::PatternLevel) = ()
@inline level_eltype(::Type{<:PatternLevel}) = Bool
@inline level_fill_value(::Type{<:PatternLevel}) = false
(fbr::AbstractFiber{<:PatternLevel})() = true
data_rep_level(::Type{<:PatternLevel}) = ElementData(false, Bool)

isstructequal(a::T, b::T) where {T<:Pattern} = true

postype(::Type{<:PatternLevel{Tp}}) where {Tp} = Tp

function transfer(lvl::PatternLevel{Tp}, device, style) where {Tp}
    return PatternLevel{Tp}()
end

"""
    pattern!(fbr)

Return the pattern of `fbr`. That is, return a tensor which is true wherever
`fbr` is structurally unequal to its fill_value. May reuse memory and render the
original tensor unusable when modified.

```jldoctest
julia> A = Tensor(SparseList(Element(0.0), 10), [2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0, 0.0])
10 Tensor{SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}:
 2.0
 0.0
 3.0
 0.0
 4.0
 0.0
 5.0
 0.0
 6.0
 0.0

julia> pattern!(A)
10 Tensor{SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, PatternLevel{Int64}}}:
 1
 0
 1
 0
 1
 0
 1
 0
 1
 0
```
"""
pattern!(fbr::Tensor) = Tensor(pattern!(fbr.lvl))
pattern!(fbr::SubFiber) = SubFiber(pattern!(fbr.lvl), fbr.pos)

struct VirtualPatternLevel <: AbstractVirtualLevel
    Tp
end

reroot_set!(ctx::AbstractCompiler, lvl::VirtualPatternLevel, diff) = nothing

reroot_get(ctx::AbstractCompiler, lvl::VirtualPatternLevel, diff) = lvl

function distribute_level(
    ctx::AbstractCompiler, lvl::VirtualPatternLevel, arch, style
)
end

is_level_injective(ctx, ::VirtualPatternLevel) = []
is_level_atomic(ctx, lvl::VirtualPatternLevel) = ([], false)
is_level_concurrent(ctx, lvl::VirtualPatternLevel) = ([], true)

lower(ctx::AbstractCompiler, lvl::VirtualPatternLevel, ::DefaultStyle) = :(PatternLevel())
virtualize(ctx, ex, ::Type{PatternLevel{Tp}}) where {Tp} = VirtualPatternLevel(Tp)

virtual_level_resize!(ctx, lvl::VirtualPatternLevel) = lvl
virtual_level_size(ctx, ::VirtualPatternLevel) = ()
virtual_level_fill_value(::VirtualPatternLevel) = false
virtual_level_eltype(::VirtualPatternLevel) = Bool

postype(lvl::VirtualPatternLevel) = lvl.Tp

function declare_level!(ctx, lvl::VirtualPatternLevel, pos, init)
    init == literal(false) ||
        throw(FinchProtocolError("Must initialize Pattern Levels to false"))
    lvl
end

freeze_level!(ctx, lvl::VirtualPatternLevel, pos) = lvl

thaw_level!(ctx, lvl::VirtualPatternLevel, pos) = lvl

assemble_level!(ctx, lvl::VirtualPatternLevel, pos_start, pos_stop) = quote end
reassemble_level!(ctx, lvl::VirtualPatternLevel, pos_start, pos_stop) = quote end

function instantiate(ctx, ::VirtualSubFiber{VirtualPatternLevel}, mode)
    if mode.kind === reader
        FillLeaf(true)
    else
        val = freshen(ctx, :null)
        push_preamble!(ctx, :($val = false))
        VirtualScalar(nothing, nothing, Bool, false, gensym(), val)
    end
end

function instantiate(ctx, fbr::VirtualHollowSubFiber{VirtualPatternLevel}, mode)
    @assert mode.kind === updater
    VirtualScalar(nothing, nothing, Bool, false, gensym(), fbr.dirty)
end
