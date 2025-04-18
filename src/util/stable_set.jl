struct StableSet{T} <: AbstractSet{T}
    data::OrderedSet{T}
    StableSet{T}(arg) where {T} = StableSet{T}(OrderedSet(arg))
end

StableSet(arg) = StableSet(OrderedSet(arg))
StableSet(args...) = StableSet(OrderedSet(args...))
StableSet{T}(args...) where {T} = StableSet{T}(OrderedSet(args...))

Base.push!(s::StableSet, x) = push!(s.data, x)
Base.pop!(s::StableSet) = pop!(s.data)
Base.iterate(s::StableSet) = iterate(s.data)
Base.iterate(s::StableSet, i) = iterate(s.data, i)
Base.intersect!(s::StableSet, x...) = intersect!(s.data, x...)
Base.union!(s::StableSet, x...) = union!(s.data, x...)
Base.setdiff!(s::StableSet, x...) = setdiff!(s.data, x...)
Base.intersect(s::StableSet, x...) = intersect(s.data, x...)
Base.union(s::StableSet, x...) = union(s.data, x...)
Base.setdiff(s::StableSet, x...) = setdiff(s.data, x...)
Base.length(s::StableSet) = length(s.data)
Base.in(s::StableSet, x) = in(s.data, x)
Base.delete!(s::StableSet, x) = delete!(s.data, x)
Base.empty!(s::StableSet) = empty!(s.data)
function Base.hash(s::StableSet{T}, h::UInt) where {T}
    h = hash(hash(StableSet{T}, h), h)
    h_2 = UInt(0)
    for k in s.data
        h_2 ⊻= hash(k, h)
    end
    h_2
end
