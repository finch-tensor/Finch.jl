"""
    Infintesimal(s)

The Infintesimal type represents an infinitestimal number.  The sign field is
used to represent positive, negative, or zero in this number system.


```jl-doctest
julia> tiny()
+0

julia> positive_tiny()
+ϵ

julia> negative_tiny()
-ϵ

julia> positive_tiny() + negative_tiny()
+0

julia> positive_tiny() * 2
+ϵ

julia> positive_tiny() * negative_tiny()
-ϵ
"""
struct Infinitesimal <: Number
    sign::Int8
end

tiny(x) = Infinitesimal(x)
tiny_zero() = tiny(Int8(0))
tiny_positive() = tiny(Int8(1))
tiny_negative() = tiny(Int8(-1))

function Base.show(io::IO, x::Infinitesimal)
    if x.sign > 0
        print(io, "+ϵ")
    elseif x.sign < 0
        print(io, "-ϵ")
    elseif x.sign == 0
        print(io, "")
    else
        error(io, "unimplemented")
    end
end

function Base.show(io::IO, mime::MIME"text/plain", x::Infinitesimal)
    if x.sign > 0
        print(io, "+ϵ")
    elseif x.sign < 0
        print(io, "-ϵ")
    elseif x.sign == 0
        print(io, "+0")
    else
        error(io, "unimplemented")
    end
end

#Core definitions for limit type
function Base.:(+)(x::Infinitesimal, y::Infinitesimal)
    tiny(min(max(x.sign + y.sign, Int8(-1)), Int8(1)))
end # only operation that needs to be fast
function Base.:(-)(x::Infinitesimal, y::Infinitesimal)
    tiny(min(max(x.sign - y.sign, Int8(-1)), Int8(1)))
end # only operation that needs to be fast
Base.:(*)(x::Infinitesimal, y::Infinitesimal) = tiny(x.sign * y.sign)
Base.:(<)(x::Infinitesimal, y::Infinitesimal) = x.sign < y.sign
Base.:(<=)(x::Infinitesimal, y::Infinitesimal) = x.sign <= y.sign
Base.:(==)(x::Infinitesimal, y::Infinitesimal) = x.sign == y.sign
Base.isless(x::Infinitesimal, y::Infinitesimal) = x < y
Base.isinf(x::Infinitesimal) = false
Base.zero(::Infinitesimal) = tiny(0)
Base.min(x::Infinitesimal, y::Infinitesimal) = tiny(min(x.sign, y.sign))
Base.max(x::Infinitesimal, y::Infinitesimal) = tiny(max(x.sign, y.sign))
Base.:(+)(x::Infinitesimal) = x
Base.:(-)(x::Infinitesimal) = tiny(-x.sign)

Base.convert(::Type{Infinitesimal}, i::Number) = tiny(Int8(sign(i)))
Base.convert(::Type{Infinitesimal}, i::Infinitesimal) = i

#Crazy julia multiple dispatch stuff don't worry about it
limit_types = [
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    UInt128,
    BigInt,
    Float32,
    Float64,
]
for S in limit_types
    @eval begin
        (::Type{$S})(i::Infinitesimal) = zero($S)
        @inline Base.promote_rule(::Type{Infinitesimal}, ::Type{$S}) = Infinitesimal
        Base.:(*)(x::$S, y::Infinitesimal) = tiny(Int8(sign(x)) * y.sign)
        Base.:(*)(y::Infinitesimal, x::$S) = tiny(x.sign * Int8(sign(y)))
    end
end

Base.hash(x::Infinitesimal, h::UInt) = hash(typeof(x), hash(x.sign, h))

"""
    Limit{T}(x, s)

The Limit type represents endpoints of closed and open intervals.  The val field
is the value of the endpoint.  The sign field is used to represent the
openness/closedness of the interval endpoint, using an Infinitesmal.

```jl-doctest
julia> limit(1.0)
1.0+0

julia> plus_eps(1.0)
1.0+ϵ

julia> minus_eps(1.0)
1.0-ϵ

julia> plus_eps(1.0) + minus_eps(1.0)
2.0+0.0

julia> plus_eps(1.0) * 2
2.0+2.0ϵ

julia> plus_eps(1.0) * minus_eps(1.0)
1.0-1.0ϵ

julia> plus_eps(-1.0) * minus_eps(1.0)
-1.0+2.0ϵ

julia> 1.0 < plus_eps(1.0)
true

julia> 1.0 < minus_eps(1.0)
false
"""
struct Limit{T} <: Number
    val::T
    sign::Infinitesimal
    Limit{T}(x::T, y) where {T} = new{T}(x, y)
    Limit{T}(x::T, y) where {T<:Limit} = error()
end

limit(x::T, s) where {T} = Limit{T}(x, s)
plus_eps(x)::Limit = limit(x, tiny_positive())
minus_eps(x)::Limit = limit(x, tiny_negative())
limit(x) = limit(x, tiny_zero())
limit(x::Limit) = x
Limit{T}(x::Number) where {T} = limit(T(x))
drop_eps(x::Limit) = x.val
drop_eps(x::Number) = x

const Eps = Finch.plus_eps(Int8(0))

function Base.show(io::IO, x::Limit)
    print(io, "limit(", x.val, x.sign, ")")
end

function Base.show(io::IO, mime::MIME"text/plain", x::Limit)
    show(io, mime, x.val)
    show(io, mime, x.sign)
end

#Core definitions for limit type
Base.:(+)(x::Limit, y::Limit)::Limit = limit(x.val + y.val, x.sign + y.sign)
Base.:(*)(x::Limit, y::Limit)::Limit = limit(x.val * y.val, x.val * y.sign + y.val * x.sign)
Base.:(-)(x::Limit, y::Limit)::Limit = limit(x.val - y.val, x.sign - y.sign)
Base.:(<)(x::Limit, y::Limit)::Bool = x.val < y.val || (x.val == y.val && x.sign < y.sign)
Base.:(<=)(x::Limit, y::Limit)::Bool = x.val < y.val || (x.val == y.val && x.sign <= y.sign)
Base.:(==)(x::Limit, y::Limit)::Bool = x.val == y.val && x.sign == y.sign
Base.isless(x::Limit, y::Limit)::Bool = x < y
Base.isinf(x::Limit) = isinf(x.val)
Base.zero(x::Limit{T}) where {T} = limit(convert(T, 0))
Base.min(x::Limit) = x
Base.max(x::Limit) = x
Base.:(+)(x::Limit)::Limit = x
Base.:(-)(x::Limit)::Limit = limit(-x.val, -x.sign)

#Crazy julia multiple dispatch stuff don't worry about it
limit_types = [
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    UInt128,
    BigInt,
    Float32,
    Float64,
]
for S in limit_types
    @eval begin
        @inline Base.promote_rule(::Type{Limit{T}}, ::Type{$S}) where {T} =
            Limit{promote_type(T, $S)}
        Base.convert(::Type{Limit{T}}, i::$S) where {T} = limit(convert(T, i))
        Limit(i::$S) = Limit{$S}(i, tiny_zero())
        (::Type{$S})(i::Limit{T}) where {T} = convert($S, i.val)
        Base.convert(::Type{$S}, i::Limit) = convert($S, i.val)
        Base.:(+)(x::Limit, y::$S)::Limit = x + limit(y)
        Base.:(+)(x::$S, y::Limit)::Limit = limit(x) + y
        Base.:(-)(x::Limit, y::$S)::Limit = x - limit(y)
        Base.:(-)(x::$S, y::Limit)::Limit = limit(x) - y
        Base.:(<)(x::Limit, y::$S) = x < limit(y)
        Base.:(<)(x::$S, y::Limit) = limit(x) < y
        Base.:(<=)(x::Limit, y::$S) = x <= limit(y)
        Base.:(<=)(x::$S, y::Limit) = limit(x) <= y
        Base.:(==)(x::Limit, y::$S) = x == limit(y)
        Base.:(==)(x::$S, y::Limit) = limit(x) == y
        Base.isless(x::Limit, y::$S) = x < limit(y)
        Base.isless(x::$S, y::Limit) = limit(x) < y
        Base.max(x::$S, y::Limit)::Limit = max(limit(x), y)
        Base.max(x::Limit, y::$S)::Limit = max(x, limit(y))
        Base.min(x::$S, y::Limit)::Limit = min(limit(x), y)
        Base.min(x::Limit, y::$S)::Limit = min(x, limit(y))
    end
end

function Base.promote_rule(::Type{Limit{T}}, ::Type{Limit{S}}) where {T,S}
    Limit{promote_type(T, S)}
end
Base.convert(::Type{Limit{T}}, i::Limit) where {T} = Limit{T}(convert(T, i.val), i.sign)
Base.hash(x::Limit, h::UInt) = hash(typeof(x), hash(x.val, hash(x.sign, h)))
