@kwdef mutable struct VirtualAbstractArray
    ex
    eltype
    ndims
end

function virtual_size(arr::VirtualAbstractArray, ctx::LowerJulia)
    dims = map(i -> Symbol(arr.ex, :_mode, i, :_stop), 1:arr.ndims)
    push!(ctx.preamble, quote
        ($(dims...),) = size($(arr.ex))
    end)
    return map(i->Extent(literal(1), value(dims[i], Int)), 1:arr.ndims)
end

function (ctx::LowerJulia)(arr::VirtualAbstractArray, ::DefaultStyle)
    return arr.ex
end

function virtualize(ex, ::Type{<:AbstractArray{T, N}}, ctx, tag=:tns) where {T, N}
    sym = ctx.freshen(tag)
    push!(ctx.preamble, :($sym = $ex))
    VirtualAbstractArray(sym, T, N)
end

function declare!(arr::VirtualAbstractArray, ctx::LowerJulia, init)
    push!(ctx.preamble, quote
        fill!($(arr.ex), $(ctx(init)))
    end)
    arr
end

freeze!(arr::VirtualAbstractArray, ctx::LowerJulia) = arr
thaw!(arr::VirtualAbstractArray, ctx::LowerJulia) = arr

get_reader(arr::VirtualAbstractArray, ctx::LowerJulia, protos...) = arr
get_updater(arr::VirtualAbstractArray, ctx::LowerJulia, protos...) = arr

FinchNotation.finch_leaf(x::VirtualAbstractArray) = virtual(x)

virtual_default(::VirtualAbstractArray) = 0
virtual_eltype(tns::VirtualAbstractArray) = tns.eltype

default(a::AbstractArray) = default(typeof(a))
default(T::Type{<:AbstractArray}) = zero(eltype(T))

struct AsArray{T, N, Fbr} <: AbstractArray{T, N}
    fbr::Fbr
    function AsArray{T, N, Fbr}(fbr::Fbr) where {T, N, Fbr}
        @assert T == eltype(fbr)
        @assert N == ndims(fbr)
        new{T, N, Fbr}(fbr)
    end
end

AsArray(fbr::Fbr) where {Fbr} = AsArray{eltype(Fbr), ndims(Fbr), Fbr}(fbr)

Base.size(arr::AsArray) = size(arr.fbr)
Base.getindex(arr::AsArray{T, N}, i::Vararg{Int, N}) where {T, N} = arr.fbr[i...]
Base.getindex(arr::AsArray{T, N}, i::Vararg{Any, N}) where {T, N} = arr.fbr[i...]
Base.setindex!(arr::AsArray{T, N}, v, i::Vararg{Int, N}) where {T, N} = arr.fbr[i...] = v
Base.setindex!(arr::AsArray{T, N}, v, i::Vararg{Any, N}) where {T, N} = arr.fbr[i...] = v