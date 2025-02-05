abstract type AbstractLevel end
abstract type AbstractVirtualLevel end

virtual_level_ndims(lvl::AbstractVirtualLevel, ctx) = length(virtual_level_size(lvl, ctx))

#is_laminable_updater(lvl::AbstractVirtualLevel, ctx, ::Union{::typeof(defaultread), ::typeof(walk), ::typeof(gallop), ::typeof(follow), typeof(defaultupdate), typeof(laminate), typeof(extrude)}, protos...) = false

#is_laminable_updater(lvl::AbstractVirtualLevel, ctx) = false

#is_level_injective(ctx, lvl::AbstractVirtualLevel) = false

# supports_reassembly(lvl::AbstractVirtualLevel) = false
"""
    level_ndims(::Type{Lvl})

The result of `level_ndims(Lvl)` defines [ndims](https://docs.julialang.org/en/v1/base/arrays/#Base.ndims) for all subfibers
in a level of type `Lvl`.
"""
function level_ndims end

"""
    level_size(lvl)

The result of `level_size(lvl)` defines the [size](https://docs.julialang.org/en/v1/base/arrays/#Base.size) of all subfibers in the
level `lvl`.
"""
function level_size end

"""
    level_axes(lvl)

The result of `level_axes(lvl)` defines the [axes](https://docs.julialang.org/en/v1/base/arrays/#Base.axes-Tuple{Any}) of all subfibers in the
level `lvl`.
"""
function level_axes end

"""
    level_eltype(::Type{Lvl})

The result of `level_eltype(Lvl)` defines
[`eltype`](https://docs.julialang.org/en/v1/base/collections/#Base.eltype) for
all subfibers in a level of type `Lvl`.
"""
function level_eltype end

"""
    level_fill_value(::Type{Lvl})

The result of `level_fill_value(Lvl)` defines [`fill_value`](@ref) for all subfibers in a
level of type `Lvl`.
"""
function level_fill_value end

"""
    declare_level!(ctx, lvl, pos, init)

Initialize and thaw all fibers within `lvl`, assuming positions `1:pos` were
previously assembled and frozen. The resulting level has no assembled positions.
"""
function declare_level! end

"""
    freeze_level!(ctx, lvl, pos, init)

Given the last reference position, `pos`, freeze all fibers within `lvl` assuming
that we have potentially updated `1:pos`.
"""
function freeze_level! end

"""
    thaw_level!(ctx, lvl, pos, init)

Given the last reference position, `pos`, thaw all fibers within `lvl` assuming
that we have previously assembled and frozen `1:pos`.
"""
function thaw_level! end

"""
    assemble_level!(ctx, lvl, pos, new_pos)

Assemble and positions `pos+1:new_pos` in `lvl`, assuming positions `1:pos` were
previously assembled.
"""
function assemble_level! end

"""
    reassemble_level!(lvl, ctx, pos_start, pos_end)

Set the previously assempled positions from `pos_start` to `pos_end` to
`level_fill_value(lvl)`.  Not avaliable on all level types as this presumes updating.
"""
function reassemble_level! end

"""
    freeze_level!(ctx, lvl, pos)

Freeze all fibers in `lvl`. Positions `1:pos` need freezing.
"""
freeze_level!(ctx, fbr, mode) = fbr.lvl

"""
    postype(lvl)
Return a position type with the same flavor as those used to store the positions of the fibers contained in `lvl`.
The name position descends from the pos or position or pointer arrays found in many definitions of CSR or CSC.
In Finch, positions should be data used to access either a subfiber or some other similar auxiliary data. Thus,
we often end up iterating over positions.
"""
function postype end

postype(arg) = postype(typeof(arg))
postype(T::Type) = throw(MethodError(postype, Tuple{T}))

#postype(::Type{Vector{S}}) where {S} = Int
