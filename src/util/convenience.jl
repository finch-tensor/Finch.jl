struct All{F}
    f::F
end

@inline (f::All{F})(args) where {F} = all(f.f, args)

struct Or{Fs}
    fs::Fs
end

Or(fs...) = Or{typeof(fs)}(fs)

@inline (f::Or{Fs})(arg) where {Fs} = any(g -> g(arg), f.fs)

struct And{Fs}
    fs::Fs
end

And(fs...) = And{typeof(fs)}(fs)

@inline (f::And{Fs})(arg) where {Fs} = all(g -> g(arg), f.fs)

kwfields(x::T) where {T} = Dict((k => getfield(x, k) for k in fieldnames(T))...)

(Base.:^)(T::Type, i::Int) = ∘(repeated(T, i)..., identity)
(Base.:^)(f::Function, i::Int) = ∘(repeated(f, i)..., identity)

pass_nothing(f, val) = val === nothing ? nothing : f(val)

struct StableSet{T} <: AbstractSet{T}
    data::OrderedSet{T}
    StableSet(arg) = StableSet(OrderedSet(arg))
    StableSet(arg::OrderedSet{T}) where {T} = StableSet{T}(arg)
    StableSet{T}(arg) where {T} = new{T}(OrderedSet{T}(arg))
    StableSet{T}(arg::OrderedSet{T}) where {T} = new{T}(arg)
end

StableSet(args...) = StableSet(OrderedSet(args...))
StableSet{T}(args...) where {T} = StableSet{T}(OrderedSet{T}(args...))

Base.push!(s::StableSet, x) = StableSet(push!(s.data, x))
Base.pop!(s::StableSet) = pop!(s.data)
Base.iterate(s::StableSet) = iterate(s.data)
Base.iterate(s::StableSet, i) = iterate(s.data, i)
Base.intersect!(s::StableSet, x...) = StableSet(intersect!(s.data, x...))
Base.union!(s::StableSet, x...) = StableSet(union!(s.data, x...))
Base.setdiff!(s::StableSet, x...) = StableSet(setdiff!(s.data, x...))
Base.intersect(s::StableSet, x...) = StableSet(intersect(s.data, x...))
Base.union(s::StableSet, x...) = StableSet(union(s.data, x...))
Base.setdiff(s::StableSet, x...) = StableSet(setdiff(s.data, x...))
Base.length(s::StableSet) = length(s.data)
Base.in(x, s::StableSet) = in(x, s.data)
Base.delete!(s::StableSet, x) = StableSet(delete!(s.data, x))
Base.empty!(s::StableSet) = StableSet(empty!(s.data))
Base.copy(s::StableSet) = StableSet(copy(s.data))
Base.:(==)(s::StableSet, x::StableSet) = s.data == x.data
Base.isequal(s::StableSet, x::StableSet) = isequal(s.data, x.data)
Base.hash(s::StableSet, h::UInt) = hash(Set(s.data), h)
