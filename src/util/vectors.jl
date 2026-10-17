using Base: @propagate_inbounds

"""
    PlusOneVector(data, [fields])

A vector whose entries are the entries of `data` plus one, e.g. to view 0-based
indices as 1-based ones. When the entries are tuples, only the fields listed in
`fields` are offset; by default, every field is.
"""
struct PlusOneVector{T,A<:AbstractVector{T},F} <: AbstractVector{T}
    data::A
end

function PlusOneVector(data::AbstractVector{T}, fields=nothing) where {T}
    PlusOneVector{T,typeof(data),fields}(data)
end

@inline plus_one_offset(x) = x + 0x01
@inline plus_one_offset(x::Tuple{Vararg{Any,N}}) where {N} =
    ntuple(i -> plus_one_offset(x[i]), Val(N))
@inline plus_one_offset(x, ::Nothing) = plus_one_offset(x)
@inline plus_one_offset(x::Tuple{Vararg{Any,N}}, fields::Tuple) where {N} =
    ntuple(i -> i in fields ? plus_one_offset(x[i]) : x[i], Val(N))
@inline minus_one_offset(x) = x - 0x01
@inline minus_one_offset(x::Tuple{Vararg{Any,N}}) where {N} =
    ntuple(i -> minus_one_offset(x[i]), Val(N))
@inline minus_one_offset(x, ::Nothing) = minus_one_offset(x)
@inline minus_one_offset(x::Tuple{Vararg{Any,N}}, fields::Tuple) where {N} =
    ntuple(i -> i in fields ? minus_one_offset(x[i]) : x[i], Val(N))

@propagate_inbounds function Base.getindex(vec::PlusOneVector{T,A,F},
    index::Int) where {T,A,F}
    return plus_one_offset(vec.data[index], F)
end

@propagate_inbounds function Base.getindex(vec::PlusOneVector{T,A,F},
    index::Vararg{Int}) where {T,A,F}
    return plus_one_offset(vec.data[index...], F)
end

@propagate_inbounds function Base.setindex!(vec::PlusOneVector{T,A,F},
    val,
    index::Int) where {T,A,F}
    vec.data[index] = minus_one_offset(convert(T, val), F)
end

@propagate_inbounds function Base.setindex!(vec::PlusOneVector{T,A,F},
    val,
    index::Vararg{Int}) where {T,A,F}
    vec.data[index...] = minus_one_offset(convert(T, val), F)
end

Base.parent(vec::PlusOneVector{T}) where {T} = vec.data
Base.size(vec::PlusOneVector{T}) where {T} = size(vec.data)
Base.axes(vec::PlusOneVector{T}) where {T} = axes(vec.data)
Base.resize!(vec::PlusOneVector{T}, dim) where {T} = (resize!(vec.data, dim); vec)
Base.empty!(vec::PlusOneVector{T}) where {T} = (empty!(vec.data); vec)
Base.copy(vec::PlusOneVector{T,A,F}) where {T,A,F} = PlusOneVector(copy(vec.data), F)
Base.dataids(vec::PlusOneVector{T}) where {T} = Base.dataids(vec.data)

function transfer(device, vec::PlusOneVector{T,A,F}) where {T,A,F}
    data = transfer(device, vec.data)
    return PlusOneVector(data, F)
end

"""
    MinusOneVector(data, [fields])

A vector whose entries are the entries of `data` minus one, e.g. to view 1-based
indices as 0-based ones. When the entries are tuples, only the fields listed in
`fields` are offset; by default, every field is.
"""
struct MinusOneVector{T,A<:AbstractVector{T},F} <: AbstractVector{T}
    data::A
end

function MinusOneVector(data::AbstractVector{T}, fields=nothing) where {T}
    MinusOneVector{T,typeof(data),fields}(data)
end

@propagate_inbounds function Base.getindex(vec::MinusOneVector{T,A,F},
    index::Int) where {T,A,F}
    return minus_one_offset(vec.data[index], F)
end

@propagate_inbounds function Base.getindex(vec::MinusOneVector{T,A,F},
    index::Vararg{Int}) where {T,A,F}
    return minus_one_offset(vec.data[index...], F)
end

@propagate_inbounds function Base.setindex!(vec::MinusOneVector{T,A,F},
    val,
    index::Int) where {T,A,F}
    vec.data[index] = plus_one_offset(convert(T, val), F)
end

@propagate_inbounds function Base.setindex!(vec::MinusOneVector{T,A,F},
    val,
    index::Vararg{Int}) where {T,A,F}
    vec.data[index...] = plus_one_offset(convert(T, val), F)
end

Base.parent(vec::MinusOneVector{T}) where {T} = vec.data
Base.size(vec::MinusOneVector{T}) where {T} = size(vec.data)
Base.axes(vec::MinusOneVector{T}) where {T} = axes(vec.data)
Base.resize!(vec::MinusOneVector{T}, dim) where {T} = (resize!(vec.data, dim); vec)
Base.empty!(vec::MinusOneVector{T}) where {T} = (empty!(vec.data); vec)
Base.copy(vec::MinusOneVector{T,A,F}) where {T,A,F} = MinusOneVector(copy(vec.data), F)
Base.dataids(vec::MinusOneVector{T}) where {T} = Base.dataids(vec.data)

function transfer(device, vec::MinusOneVector{T,A,F}) where {T,A,F}
    data = transfer(device, vec.data)
    return MinusOneVector(data, F)
end

struct MinusEpsVector{T,S,A<:AbstractVector{S}} <: AbstractVector{T}
    data::A
end

function MinusEpsVector(data::AbstractVector{T}) where {T}
    MinusEpsVector{Limit{T},T,typeof(data)}(data)
end

@propagate_inbounds function Base.getindex(vec::MinusEpsVector{T},
    index::Int) where {T}
    return minus_eps(vec.data[index])
end

@propagate_inbounds function Base.getindex(vec::MinusEpsVector{T},
    index::Vararg{Int}) where {T}
    return minus_eps(vec.data[index...])
end

@propagate_inbounds function Base.setindex!(vec::MinusEpsVector{Limit{T}},
    val::Limit{T},
    index::Int) where {T}
    Base.@boundscheck begin
        @assert val.sign == tiny_negative()
    end
    vec.data[index] = val.val
end

@propagate_inbounds function Base.setindex!(vec::MinusEpsVector{Limit{T}},
    val::Limit{T},
    index::Vararg{Int}) where {T}
    Base.@boundscheck begin
        @assert val.sign == tiny_negative()
    end
    vec.data[index...] = val.val
end

Base.parent(vec::MinusEpsVector{T}) where {T} = vec.data
Base.size(vec::MinusEpsVector{T}) where {T} = size(vec.data)
Base.axes(vec::MinusEpsVector{T}) where {T} = axes(vec.data)
Base.resize!(vec::MinusEpsVector{T}, dim) where {T} = resize!(vec.data, dim)

function transfer(device, vec::MinusEpsVector{T}) where {T}
    data = transfer(device, vec.data)
    return MinusEpsVector{T}(data)
end

struct PlusEpsVector{T,S,A<:AbstractVector{S}} <: AbstractVector{T}
    data::A
end

function PlusEpsVector(data::AbstractVector{T}) where {T}
    PlusEpsVector{Limit{T},T,typeof(data)}(data)
end

@propagate_inbounds function Base.getindex(vec::PlusEpsVector{T},
    index::Int) where {T}
    return plus_eps(vec.data[index])
end

@propagate_inbounds function Base.getindex(vec::PlusEpsVector{T},
    index::Vararg{Int}) where {T}
    return plus_eps(vec.data[index...])
end

@propagate_inbounds function Base.setindex!(vec::PlusEpsVector{Limit{T}},
    val::Limit{T},
    index::Int) where {T}
    Base.@boundscheck begin
        @assert val.sign == tiny_positive()
    end
    vec.data[index] = val.val
end

@propagate_inbounds function Base.setindex!(vec::PlusEpsVector{Limit{T}},
    val::Limit{T},
    index::Vararg{Int}) where {T}
    Base.@boundscheck begin
        @assert val.sign == tiny_positive()
    end
    vec.data[index...] = val.val
end

Base.parent(vec::PlusEpsVector{T}) where {T} = vec.data
Base.size(vec::PlusEpsVector{T}) where {T} = size(vec.data)
Base.axes(vec::PlusEpsVector{T}) where {T} = axes(vec.data)
Base.resize!(vec::PlusEpsVector{T}, dim) where {T} = resize!(vec.data, dim)

function transfer(device, vec::PlusEpsVector{T}) where {T}
    data = transfer(device, vec.data)
    return PlusEpsVector{T}(data)
end
