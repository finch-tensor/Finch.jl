using Base: Broadcast
using Base.Broadcast: Broadcasted, BroadcastStyle, AbstractArrayStyle
using Base: broadcasted
using LinearAlgebra
const AbstractArrayOrBroadcasted = Union{AbstractArray,Broadcasted}

"""
    LazyTensor{Vf, [Tv=typeof(Vf)], N, TI<:Tuple}()
"""
mutable struct LazyTensor{Vf,Tv,N,TI<:Tuple}
    data
    shape::TI
end
function LazyTensor{Vf}(data, shape::TI
) where {Vf,TI<:Tuple}
    LazyTensor{Vf,typeof(Vf),length(shape),TI}(data, shape)
end

function LazyTensor{Vf,Tv}(data, shape::TI
) where {Vf,Tv,TI<:Tuple}
    LazyTensor{Vf,Tv,length(shape),TI}(data, shape)
end

function Base.show(io::IO, tns::LazyTensor)
    join(io, tns.shape, "×")
    print(io, "-LazyTensor{", eltype(tns), "}")
end

Base.ndims(::Type{<:LazyTensor{Vf,Tv,N}}) where {Vf,Tv,N} = N
Base.ndims(tns::LazyTensor) = ndims(typeof(tns))
Base.eltype(::Type{<:LazyTensor{Vf,Tv}}) where {Vf,Tv} = Tv
Base.eltype(tns::LazyTensor) = eltype(typeof(tns))
fill_value(::Type{<:LazyTensor{Vf}}) where {Vf} = Vf
fill_value(tns::LazyTensor) = fill_value(typeof(tns))

Base.size(tns::LazyTensor) = tns.shape

function Base.getindex(::LazyTensor, i...)
    throw(
        ErrorException(
            "Lazy indexing with named indices is not supported. Call `compute()` first."
        ),
    )
end

function Base.getindex(arr::LazyTensor, idxs::Vararg{Union{Nothing,Colon}})
    if length(idxs) - count(isnothing, idxs) != ndims(arr)
        throw(
            ArgumentError(
                "Cannot index a lazy tensor with more or fewer `:` dims than it had original dims."
            ),
        )
    end
    return expanddims(arr; dims=findall(isnothing, idxs))
end

function expanddims(arr::LazyTensor{Vf,Tv}; dims) where {Vf,Tv}
    dims = [dims...]
    @assert allunique(dims)
    @assert issubset(dims, 1:(ndims(arr) + length(dims)))
    offset = zeros(Int, ndims(arr) + length(dims))
    offset[dims] .= 1
    offset = cumsum(offset)
    idxs_1 = [field(gensym(:i)) for _ in 1:ndims(arr)]
    idxs_2 = [
        n in dims ? field(gensym(:i)) : idxs_1[n - offset[n]] for
        n in 1:(ndims(arr) + length(dims))
    ]
    data_2 = reorder(relabel(arr.data, idxs_1...), idxs_2...)
    shape_2 = ntuple(
        n -> n in dims ? 1 : arr.shape[n - offset[n]], ndims(arr) + length(dims)
    )

    return LazyTensor{Vf,Tv}(data_2, shape_2)
end

function Base.dropdims(arr::LazyTensor{Vf,Tv}; dims) where {Vf,Tv}
    dims = dims === Colon() ? [1:ndims(arr)...] : [dims...]
    @assert allunique(dims)
    @assert issubset(dims, 1:ndims(arr))
    @assert all(isone, arr.shape[dims])
    newdims = setdiff(1:ndims(arr), dims)
    idxs_1 = [field(gensym(:i)) for _ in 1:ndims(arr)]
    idxs_2 = idxs_1[newdims]
    data_2 = reorder(relabel(arr.data, idxs_1...), idxs_2...)
    shape_2 = arr.shape[newdims]
    return LazyTensor{Vf,Tv}(
        data_2, tuple(shape_2...)
    )
end

function identify(data)
    lhs = alias(gensym(:A))
    subquery(lhs, data)
end

LazyTensor(data::Number) = LazyTensor{data,typeof(data)}(immediate(data), ())
LazyTensor{Vf}(data::Number) where {Vf} = LazyTensor{data,typeof(data)}(immediate(data), ())
LazyTensor{Vf,Tv}(data::Number) where {Vf,Tv} = LazyTensor{Vf,Tv}(immediate(data), ())
function LazyTensor(arr::Base.AbstractArrayOrBroadcasted)
    LazyTensor{fill_value(arr),eltype(arr)}(arr)
end
function LazyTensor{Vf,Tv}(arr::Base.AbstractArrayOrBroadcasted) where {Vf,Tv}
    name = alias(gensym(:A))
    idxs = [field(gensym(:i)) for _ in 1:ndims(arr)]
    shape = size(arr)
    tns = subquery(name, table(immediate(arr), idxs...))
    LazyTensor{Vf,Tv}(tns, shape)
end
LazyTensor(arr::AbstractTensor) = LazyTensor{fill_value(arr),eltype(arr)}(arr)
function LazyTensor(swizzle_arr::SwizzleArray{dims,<:Tensor}) where {dims}
    permutedims(LazyTensor(swizzle_arr.body), dims)
end
function LazyTensor{Vf,Tv}(arr::AbstractTensor) where {Vf,Tv}
    name = alias(gensym(:A))
    idxs = [field(gensym(:i)) for _ in 1:ndims(arr)]
    shape = size(arr)
    tns = subquery(name, table(immediate(arr), idxs...))
    LazyTensor{Vf,Tv}(tns, shape)
end
LazyTensor(data::LazyTensor) = data

swizzle(arr::LazyTensor, dims...) = permutedims(arr, dims)

Base.sum(arr::LazyTensor; kwargs...) = reduce(+, arr; kwargs...)
Base.prod(arr::LazyTensor; kwargs...) = reduce(*, arr; kwargs...)
Base.any(arr::LazyTensor; kwargs...) = reduce(or, arr; init=false, kwargs...)
Base.all(arr::LazyTensor; kwargs...) = reduce(and, arr; init=true, kwargs...)
function Base.minimum(arr::LazyTensor; kwargs...)
    reduce(min, arr; init=typemax(eltype(arr)), kwargs...)
end
function Base.maximum(arr::LazyTensor; kwargs...)
    reduce(max, arr; init=typemin(eltype(arr)), kwargs...)
end

function Base.mapreduce(f, op, src::LazyTensor, args...; kw...)
    reduce(op, map(f, src, args...); kw...)
end

function Base.map(f, src::LazyTensor, args...)
    largs = map(LazyTensor, (src, args...))
    shape = largs[something(findfirst(arg -> length(arg.shape) > 0, largs), 1)].shape
    idxs = [field(gensym(:i)) for _ in shape]
    ldatas = map(largs) do larg
        if larg.shape == shape
            return relabel(larg.data, idxs...)
        elseif larg.shape == ()
            return relabel(larg.data)
        else
            throw(DimensionMismatch("Cannot map across arrays with different sizes."))
        end
    end
    Tv_2 = return_type(DefaultAlgebra(), f, eltype(src), map(eltype, args)...)
    Vf_2 = f(map(fill_value, largs)...)
    data = mapjoin(immediate(f), ldatas...)
    return LazyTensor{Vf_2,Tv_2}(identify(data), shape)
end

function Base.map!(dst, f, src::LazyTensor, args...)
    res = map(f, src, args...)
    return LazyTensor(identify(reformat(dst, res.data)), res.shape)
end

function initial_value(op, T)
    try
        return reduce(op, Vector{T}())
    catch
    end
    throw(ArgumentError("Please supply initial value for reduction of $T with $op."))
end

initial_value(::typeof(max), T) = typemin(T)
initial_value(::typeof(min), T) = typemax(T)

function fixpoint_type(op, z, T)
    S = Union{}
    R = typeof(z)
    while R != S
        S = R
        R = Union{R,return_type(DefaultAlgebra(), op, R, T)}
    end
    R
end

function Base.reduce(
    op, arg::LazyTensor{Vf,Tv,N}; dims=:, init=initial_value(op, Tv)
) where {Vf,Tv,N}
    dims_2 = dims === Colon() ? [1:N...] : [dims...]
    shape = ((n in dims_2 ? one(arg.shape[n]) : arg.shape[n] for n in 1:N)...,)
    fields = [field(gensym(:i)) for _ in 1:N]
    fields2 = copy(fields)
    for i in dims_2
        fields2[i] = field(gensym(:i))
    end
    Sv = fixpoint_type(op, init, Tv)
    data = aggregate(
        immediate(op), immediate(init), relabel(arg.data, fields), fields[dims_2]...
    )
    if dims === Colon()
        shape = ()
    else
        data = reorder(data, fields2...)
    end
    LazyTensor{init,Sv}(identify(data), shape)
end

function tensordot(A::LazyTensor, B::Union{AbstractTensor,AbstractArray}, idxs; kwargs...)
    tensordot(A, LazyTensor(B), idxs; kwargs...)
end
function tensordot(A::Union{AbstractTensor,AbstractArray}, B::LazyTensor, idxs; kwargs...)
    tensordot(LazyTensor(A), B, idxs; kwargs...)
end

# tensordot takes in two tensors `A` and `B` and performs a product and contraction
function tensordot(
    A::LazyTensor{Vf1,Tv1,N1},
    B::LazyTensor{Vf2,Tv2,N2},
    idxs;
    mult_op=*,
    add_op=+,
    init=initial_value(add_op, return_type(DefaultAlgebra(), mult_op, Tv1, Tv2)),
) where {Vf1,Vf2,Tv1,Tv2,N1,N2}
    if idxs isa Number
        idxs = ([i for i in 1:idxs], [i for i in 1:idxs])
    end
    A_idxs = idxs[1]
    B_idxs = idxs[2]
    if length(A_idxs) != length(B_idxs)
        throw(
            ArgumentError(
                "lists of contraction indices must be the same length for both inputs"
            ),
        )
    end
    if any([i > N1 for i in A_idxs]) || any([i > N2 for i in B_idxs])
        throw(
            ArgumentError(
                "contraction indices cannot be greater than the number of dimensions"
            ),
        )
    end

    shape = ((A.shape[n] for n in 1:N1 if !(n in A_idxs))...,
        (B.shape[n] for n in 1:N2 if !(n in B_idxs))...)
    A_fields = [field(gensym(:i)) for _ in 1:N1]
    B_fields = [field(gensym(:i)) for _ in 1:N2]
    reduce_fields = []
    for i in eachindex(A_idxs)
        B_fields[B_idxs[i]] = A_fields[A_idxs[i]]
        push!(reduce_fields, A_fields[A_idxs[i]])
    end
    AB = mapjoin(immediate(mult_op), relabel(A.data, A_fields), relabel(B.data, B_fields))
    AB_reduce = aggregate(immediate(add_op), immediate(init), AB, reduce_fields...)
    Tv = return_type(DefaultAlgebra(), mult_op, Tv1, Tv2)
    Sv = fixpoint_type(add_op, init, Tv)
    return LazyTensor{init,Sv}(identify(AB_reduce), shape)
end

struct LazyStyle{N} <: BroadcastStyle end
function Base.Broadcast.BroadcastStyle(F::Type{<:LazyTensor{Vf,Tv,N}}) where {Vf,Tv,N}
    LazyStyle{N}()
end
Base.Broadcast.broadcastable(tns::LazyTensor) = tns
function Base.Broadcast.BroadcastStyle(a::LazyStyle{M}, b::LazyStyle{N}) where {M,N}
    LazyStyle{max(M, N)}()
end
function Base.Broadcast.BroadcastStyle(
    a::LazyStyle{M}, b::Broadcast.AbstractArrayStyle{N}
) where {M,N}
    LazyStyle{max(M, N)}()
end

function broadcast_to_logic(bc::Broadcast.Broadcasted)
    broadcasted(bc.f, map(broadcast_to_logic, bc.args)...)
end

function broadcast_to_logic(tns::LazyTensor)
    tns
end

function broadcast_to_logic(tns)
    LazyTensor(tns)
end

function broadcast_to_query(bc::Broadcast.Broadcasted, idxs)
    mapjoin(immediate(bc.f), map(arg -> broadcast_to_query(arg, idxs), bc.args)...)
end

function broadcast_to_query(tns::LazyTensor{Vf,Tv,N}, idxs) where {Vf,Tv,N}
    idxs_2 = [isone(tns.shape[i]) ? field(gensym(:idx)) : idxs[i] for i in 1:N]
    data_2 = relabel(tns.data, idxs_2...)
    reorder(data_2, idxs[[i for i in 1:N if !isone(tns.shape[i])]]...)
end

function broadcast_to_shape(bc::Broadcast.Broadcasted, n)
    maximum(map(arg -> broadcast_to_shape(arg, n), bc.args))
end

function broadcast_to_shape(tns::LazyTensor, n)
    get(tns.shape, n, 0)
end

function broadcast_to_fill_value(bc::Broadcast.Broadcasted)
    bc.f(map(arg -> broadcast_to_fill_value(arg), bc.args)...)
end

function broadcast_to_fill_value(tns::LazyTensor)
    fill_value(tns)
end

function broadcast_to_eltype(bc::Broadcast.Broadcasted)
    return_type(DefaultAlgebra(), bc.f, map(arg -> broadcast_to_eltype(arg), bc.args)...)
end

function broadcast_to_eltype(arg)
    eltype(arg)
end

Base.Broadcast.instantiate(bc::Broadcasted{LazyStyle{N}}) where {N} = bc

Base.copyto!(out, bc::Broadcasted{LazyStyle{N}}) where {N} = copyto!(out, copy(bc))

function Base.copy(bc::Broadcasted{LazyStyle{N}}) where {N}
    bc_lgc = broadcast_to_logic(bc)
    idxs = [field(gensym(:i)) for _ in 1:N]
    data = reorder(broadcast_to_query(bc_lgc, idxs), idxs)
    shape = ntuple(n -> broadcast_to_shape(bc_lgc, n), N)
    return LazyTensor{broadcast_to_fill_value(bc_lgc),broadcast_to_eltype(bc)}(
        identify(data), shape
    )
end

function Base.copyto!(::LazyTensor, ::Any)
    throw(ArgumentError("cannot materialize into a LazyTensor"))
end

function Base.copyto!(dst::AbstractArray, src::LazyTensor{Vf,Tv}) where {Vf,Tv}
    return LazyTensor{Vf,Tv}(reformat(immediate(dst), src.data), src.shape)
end

Base.permutedims(arg::LazyTensor{Vf,Tv,2}) where {Vf,Tv} = permutedims(arg, [2, 1])
function Base.permutedims(arg::LazyTensor{Vf,Tv,N}, perm) where {Vf,Tv,N}
    length(perm) == N ||
        throw(ArgumentError("permutedims given wrong number of dimensions"))
    isperm(perm) || throw(ArgumentError("permutedims given invalid permutation"))
    perm = [perm...]
    idxs = [field(gensym(:i)) for _ in 1:N]
    return LazyTensor{Vf,Tv}(
        reorder(relabel(arg.data, idxs...), idxs[perm]...),
        (arg.shape[perm]...,),
    )
end
Base.permutedims(arr::SwizzleArray, perm) = swizzle(arr, perm...)

function Base.:+(
    x::LazyTensor,
    y::Union{LazyTensor,AbstractTensor,Base.AbstractArrayOrBroadcasted,Number},
    z::Union{LazyTensor,AbstractTensor,Base.AbstractArrayOrBroadcasted,Number}...,
)
    map(+, x, y, z...)
end
function Base.:+(
    x::Union{LazyTensor,AbstractTensor,Base.AbstractArrayOrBroadcasted,Number},
    y::LazyTensor,
    z::Union{LazyTensor,AbstractTensor,Base.AbstractArrayOrBroadcasted,Number}...,
)
    map(+, y, x, z...)
end
function Base.:+(
    x::LazyTensor,
    y::LazyTensor,
    z::Union{LazyTensor,AbstractTensor,Base.AbstractArrayOrBroadcasted,Number}...,
)
    map(+, x, y, z...)
end
Base.:*(
    x::LazyTensor,
    y::Number,
    z::Number...,
) = map(*, x, y, z...)
Base.:*(
    x::Number,
    y::LazyTensor,
    z::Number...,
) = map(*, y, x, z...)

function Base.:*(
    A::LazyTensor,
    B::Union{LazyTensor,AbstractTensor,AbstractArray},
)
    tensordot(A, B, (2, 1))
end
function Base.:*(
    A::Union{LazyTensor,AbstractTensor,AbstractArray},
    B::LazyTensor,
)
    tensordot(A, B, (2, 1))
end
Base.:*(
    A::LazyTensor,
    B::LazyTensor,
) = tensordot(A, B, (2, 1))

Base.:-(x::LazyTensor) = map(-, x)

function Base.:-(
    x::LazyTensor,
    y::Union{LazyTensor,AbstractTensor,Base.AbstractArrayOrBroadcasted,Number},
)
    map(-, x, y)
end
function Base.:-(
    x::Union{LazyTensor,AbstractTensor,Base.AbstractArrayOrBroadcasted,Number},
    y::LazyTensor,
)
    map(-, x, y)
end
Base.:-(x::LazyTensor, y::LazyTensor) = map(-, x, y)

Base.:/(x::LazyTensor, y::Number) = map(/, x, y)
Base.:/(x::Number, y::LazyTensor) = map(\, y, x)

min1max2((a, b), (c, d)) = (min(a, c), max(b, d))
plex(a) = (a, a)
isassociative(::AbstractAlgebra, ::typeof(min1max2)) = true
iscommutative(::AbstractAlgebra, ::typeof(min1max2)) = true
isidempotent(::AbstractAlgebra, ::typeof(min1max2)) = true
function isidentity(alg::AbstractAlgebra, ::typeof(min1max2), x::Tuple)
    !ismissing(x) && isinf(x[1]) && x[1] > 0 && isinf(x[2]) && x[2] < 0
end
function isannihilator(alg::AbstractAlgebra, ::typeof(min1max2), x::Tuple)
    !ismissing(x) && isinf(x[1]) && x[1] < 0 && isinf(x[2]) && x[2] > 0
end
function Base.extrema(arr::LazyTensor; kwargs...)
    mapreduce(
        plex, min1max2, arr; init=(typemax(eltype(arr)), typemin(eltype(arr))), kwargs...
    )
end

struct Square{T,S}
    arg::T
    scale::S
end

@inline square(x) = Square(sign(x)^2 / one(x), norm(x))

@inline root(x::Square) = sqrt(x.arg) * x.scale

@inline Base.zero(::Type{Square{T,S}}) where {T,S} = Square{T,S}(zero(T), zero(S))
@inline Base.zero(::Square{T,S}) where {T,S} = Square{T,S}(zero(T), zero(S))
@inline Base.isone(x::Square) = isone(root(x))

@inline Base.isinf(x::Finch.Square) = isinf(x.arg) || isinf(x.scale)

function Base.promote_rule(::Type{Square{T1,S1}}, ::Type{Square{T2,S2}}) where {T1,S1,T2,S2}
    return Square{promote_type(T1, T2),promote_type(S1, S2)}
end

function Base.convert(::Type{Square{T,S}}, x::Square) where {T,S}
    return Square(convert(T, x.arg), convert(S, x.scale))
end

function Base.promote_rule(::Type{Square{T1,S1}}, ::Type{T2}) where {T1,S1,T2<:Number}
    return promote_type(T1, T2)
end

function Base.convert(T::Type{<:Number}, x::Square)
    return convert(T, root(x))
end

@inline function Base.:+(x::T, y::T) where {T<:Square}
    if x.scale < y.scale
        (x, y) = (y, x)
    end
    if x.scale > y.scale
        if iszero(y.scale)
            return Square(x.arg + zero(y.arg) * (one(y.scale) / one(x.scale))^1, x.scale)
        else
            return Square(x.arg + y.arg * (y.scale / x.scale)^2, x.scale)
        end
    else
        return Square(x.arg + y.arg * (one(y.scale) / one(x.scale))^1, x.scale)
    end
end

@inline function Base.:*(x::Square, y::Integer)
    return Square(x.arg * y, x.scale)
end

@inline function Base.:*(x::Integer, y::Square)
    return Square(y.arg * x, y.scale)
end

struct Power{T,S,E}
    arg::T
    scale::S
    exponent::E
end

@inline power(x, p) = Power(sign(x)^p, norm(x), p)

@inline root(x::Power) = x.arg^inv(x.exponent) * x.scale

@inline Base.zero(::Type{Power{T,S,E}}) where {T,S,E} =
    Power{T,S,E}(zero(T), zero(S), one(E))
@inline Base.zero(x::Power) = Power(zero(x.arg), zero(x.scale), x.exponent)
@inline Base.isinf(x::Finch.Power) = isinf(x.arg) || isinf(x.scale) || isinf(x.exponent)
@inline Base.isone(x::Power) = isone(root(x))

function Base.promote_rule(
    ::Type{Power{T1,S1,E1}}, ::Type{Power{T2,S2,E2}}
) where {T1,S1,E1,T2,S2,E2}
    return Power{promote_type(T1, T2),promote_type(S1, S2),promote_type(E1, E2)}
end

function Base.convert(::Type{Power{T,S,E}}, x::Power) where {T,S,E}
    return Power(convert(T, x.arg), convert(S, x.scale), convert(E, x.exponent))
end

function Base.promote_rule(::Type{Power{T1,S1,E1}}, ::Type{T2}) where {T1,S1,E1,T2<:Number}
    return promote_type(T1, T2)
end

function Base.convert(T::Type{<:Number}, x::Power)
    return convert(T, root(x))
end

@inline function Base.:+(x::T, y::T) where {T<:Power}
    if x.exponent != y.exponent
        if iszero(x.arg) && iszero(x.scale)
            (x, y) = (y, x)
        end
        if iszero(y.arg) && iszero(y.scale)
            y = Power(y.arg, y.scale, x.exponent)
        else
            throw(ArgumentError("Cannot accurately add Powers with different exponents"))
        end
    end
    #TODO handle negative exponent
    if x.scale < y.scale
        (x, y) = (y, x)
    end
    if x.scale > y.scale
        if iszero(y.scale)
            return Power(
                x.arg + zero(y.arg) * (one(y.scale) / one(x.scale))^one(y.exponent),
                x.scale,
                x.exponent,
            )
        else
            return Power(
                x.arg + y.arg * (y.scale / x.scale)^y.exponent, x.scale, x.exponent
            )
        end
    else
        return Power(
            x.arg + y.arg * (one(y.scale) / one(x.scale))^one(y.exponent),
            x.scale,
            x.exponent,
        )
    end
end

@inline function Base.:*(x::Power, y::Integer)
    return Power(x.arg * y, x.scale, x.exponent)
end

@inline function Base.:*(x::Integer, y::Power)
    return Power(y.arg * x, y.scale, y.exponent)
end

function LinearAlgebra.norm(arr::LazyTensor, p::Real=2)
    if p == 2
        return map(root, sum(map(square, arr)))
    elseif p == 1
        return sum(map(abs, arr))
    elseif p == Inf
        return maximum(map(abs, arr))
    elseif p == 0
        return sum(map(!, map(iszero, arr)))
    elseif p == -Inf
        return minimum(map(abs, arr))
    else
        return map(root, sum(map(power, map(norm, arr, p), p)))
    end
end

function Statistics.mean(tns::LazyTensor; dims=:)
    dims_2 = dims === Colon() ? [1:ndims(tns)...] : [dims...]
    n = prod(collect(size(tns))[dims_2])
    return sum(tns; dims=dims) ./ n
end

function Statistics.mean(f, tns::LazyTensor; dims=:)
    dims_2 = dims === Colon() ? [1:ndims(tns)...] : [dims...]
    n = prod(collect(size(tns))[dims_2])
    return sum(map(f, tns); dims=dims) ./ n
end

function Statistics.var(tns::LazyTensor; mean=nothing, corrected=true, dims=:)
    dims_2 = dims === Colon() ? [1:ndims(tns)...] : [dims...]
    if mean === nothing
        mean = Statistics.mean(tns; dims=dims)
    end
    n = prod(collect(size(tns))[dims_2])
    return sum(abs2.(tns .- mean); dims=dims) ./ (n - corrected)
end

function Statistics.varm(tns::LazyTensor, mean; corrected=true, dims=:)
    var(tns; mean=mean, corrected=corrected, dims=dims)
end

function Statistics.std(
    tns::LazyTensor; corrected=true, mean=nothing, dims=:
)
    sqrt.(var(tns; corrected=corrected, mean=mean, dims=dims))
end

function Statistics.stdm(tns::LazyTensor, m; corrected=true, dims=:)
    std(tns; corrected=corrected, mean=m, dims=dims)
end

"""
    lazy(arg)

Create a lazy tensor from an argument. All operations on lazy tensors are
lazy, and will not be executed until `compute` is called on their result.

for example,
```julia
x = lazy(rand(10))
y = lazy(rand(10))
z = x + y
z = z + 1
z = compute(z)
```
will not actually compute `z` until `compute(z)` is called, so the execution of `x + y`
is fused with the execution of `z + 1`.
"""
lazy(arg) = LazyTensor(arg)

"""
    default_scheduler(;verbose=false)

The default scheduler used by `compute` to execute lazy tensor programs.
Fuses all pointwise expresions into reductions. Only fuses reductions
into pointwise expressions when they are the only usage of the reduction.
"""
default_scheduler(; verbose=false) =
    LogicExecutor(DefaultLogicOptimizer(LogicCompiler()); verbose=verbose)

"""
    fused(f, args...; kwargs...)

This function decorator modifies `f` to fuse the contained array operations and
optimize the resulting program. The function must return a single array or tuple
of arrays.  Some keyword arguments can be passed to control the execution of the
program:
    - `verbose=false`: Print the generated code before execution
    - `tag=:global`: A tag to distinguish between different classes of inputs for the same program.
"""
function fused(f, args...; kwargs...)
    compute(f(map(LazyTensor, args)...); kwargs...)
end

current_scheduler = Ref{Any}(default_scheduler())

"""
    set_scheduler!(scheduler)

Set the current scheduler to `scheduler`. The scheduler is used by `compute` to
execute lazy tensor programs.
"""
set_scheduler!(scheduler) = current_scheduler[] = scheduler

"""
    get_scheduler()

Get the current Finch scheduler used by `compute` to execute lazy tensor programs.
"""
get_scheduler() = current_scheduler[]

"""
    with_scheduler(f, scheduler)

Execute `f` with the current scheduler set to `scheduler`.
"""
function with_scheduler(f, scheduler)
    old_scheduler = get_scheduler()
    set_scheduler!(scheduler)
    try
        return f()
    finally
        set_scheduler!(old_scheduler)
    end
end

"""
    compute(args...; ctx=default_scheduler(), kwargs...) -> Any

Compute the value of a lazy tensor. The result is the argument itself, or a
tuple of arguments if multiple arguments are passed. Some keyword arguments
can be passed to control the execution of the program:
    - `verbose=false`: Print the generated code before execution
    - `tag=:global`: A tag to distinguish between different classes of inputs for the same program.
"""
compute(args...; ctx=get_scheduler(), kwargs...) =
    compute_parse(set_options(ctx; kwargs...), map(lazy, args))
function compute(arg; ctx=get_scheduler(), kwargs...)
    compute_parse(set_options(ctx; kwargs...), (lazy(arg),))[1]
end
function compute(args::Tuple; ctx=get_scheduler(), kwargs...)
    compute_parse(set_options(ctx; kwargs...), map(lazy, args))
end
function compute_parse(ctx, args::Tuple)
    args = [args...]
    vars = map(arg -> alias(gensym(:A)), args)
    bodies = map((arg, var) -> query(var, arg.data), args, vars)
    prgm = plan(bodies, produces(vars))

    ress = ctx(prgm)

    @debug begin
        for (arg, res) in zip(args, ress)
            @assert size(arg) == size(res)
        end
    end

    return ress
end

function Base.argmin(A::LazyTensor; dims=:)
    if (ndims(A) >= 2)
        return map(
            last,
            reduce(
                minby,
                map(
                    Pair,
                    A,
                    CartesianIndices(size(A)),
                );
                dims=dims,
                init=Inf => CartesianIndex(fill(0, length(size(A)))...),
            ),
        )
    else
        return map(
            last, reduce(minby, map(Pair, A, 1:size(A)[1]); dims=dims, init=Inf => 0)
        )
    end
end

function Base.argmax(A::LazyTensor; dims=:)
    if (ndims(A) >= 2)
        return map(
            last,
            reduce(
                maxby,
                map(
                    Pair,
                    A,
                    CartesianIndices(size(A)),
                );
                dims=dims,
                init=-Inf => CartesianIndex(fill(0, length(size(A)))...),
            ),
        )
    else
        return map(
            last, reduce(maxby, map(Pair, A, 1:size(A)[1]); dims=dims, init=-Inf => 0)
        )
    end
end

function argmin_python(A::LazyTensor; dims=:)
    dims_2 = dims === Colon() ? [1:ndims(A)...] : [dims...]
    if length(dims_2) == 1
        dim = first(dims_2)

        return map(
            last,
            reduce(
                minby,
                broadcast(
                    Pair,
                    A,
                    expanddims(lazy(1:size(A)[dim]); dims=setdiff(1:ndims(A), dim)),
                );
                dims=dims,
                init=Inf => 0,
            ),
        )
    elseif dims === Colon() || length(dims_2) == ndims(A)
        return map(
            last,
            reduce(
                minby,
                map(
                    Pair,
                    A,
                    LinearIndices(size(A)),
                );
                dims=dims,
                init=Inf => 0,
            ),
        )
    end
end

function argmax_python(A::LazyTensor; dims=:)
    dims_2 = dims === Colon() ? [1:ndims(A)...] : [dims...]
    if length(dims_2) == 1
        dim = first(dims_2)

        return map(
            last,
            reduce(
                maxby,
                broadcast(
                    Pair,
                    A,
                    expanddims(lazy(1:size(A)[dim]); dims=setdiff(1:ndims(A), dim)),
                );
                dims=dims,
                init=-Inf => 0,
            ),
        )
    elseif dims === Colon() || length(dims_2) === ndims(A)
        return map(
            last,
            reduce(
                maxby,
                map(
                    Pair,
                    A,
                    LinearIndices(size(A)),
                );
                dims=dims,
                init=-Inf => 0,
            ),
        )
    end
end
