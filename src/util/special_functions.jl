struct Chooser{Vf} end

(f::Chooser{Vf})(x) where {Vf} = x
function (f::Chooser{Vf})(x, y, tail...) where {Vf}
    if isequal(x, Vf)
        return f(y, tail...)
    else
        return x
    end
end

"""
    choose(z)(a, b)

`choose(z)` is a function which returns whichever of `a` or `b` is not
[isequal](https://docs.julialang.org/en/v1/base/base/#Base.isequal) to `z`. If
neither are `z`, then return `a`. Useful for getting the first nonfill value in
a sparse array.
```jldoctest; setup=:(using Finch)
julia> a = Tensor(SparseList(Element(0.0)), [0, 1.1, 0, 4.4, 0])
5 Tensor{SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}:
 0.0
 1.1
 0.0
 4.4
 0.0

julia> x = Scalar(0.0); @finch for i=_; x[] <<choose(0.0)>>= a[i] end;

julia> x[]
1.1
```
"""
choose(d) = Chooser{d}()

struct FilterOp{Vf} end

(f::FilterOp{Vf})(cond, arg) where {Vf} = ifelse(cond, arg, Vf)

"""
    filterop(z)(cond, arg)

`filterop(z)` is a function which returns `ifelse(cond, arg, z)`. This operation
is handy for filtering out values based on a mask or a predicate.
`map(filterop(0), cond, arg)` is analogous to `filter(x -> cond ? x: z, arg)`.

```jldoctest; setup=:(using Finch)
julia> a = Tensor(SparseList(Element(0.0)), [0, 1.1, 0, 4.4, 0])
5 Tensor{SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}:
 0.0
 1.1
 0.0
 4.4
 0.0

julia> x = Tensor(SparseList(Element(0.0)));

julia> c = Tensor(SparseList(Element(false)), [false, false, false, true, false]);

julia> @finch (x .= 0; for i=_; x[i] = filterop(0)(c[i], a[i]) end)
(x = Tensor(SparseList{Int64}(Element{0.0, Float64, Int64}([4.4]), 5, [1, 2], [4])),)

julia> x
5 Tensor{SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}:
 0.0
 0.0
 0.0
 4.4
 0.0
```
"""
filterop(d) = FilterOp{d}()

"""
    minby(a, b)

Return the min of `a` or `b`, comparing them by `a[1]` and `b[1]`, and breaking
ties to the left. Useful for implementing argmin operations:
```jldoctest; setup=:(using Finch)
julia> a = [7.7, 3.3, 9.9, 3.3, 9.9]; x = Scalar(Inf => 0);

julia> @finch for i=_; x[] <<minby>>= a[i] => i end;

julia> x[]
3.3 => 2
```
"""
minby(a, b) = a[1] > b[1] ? b : a

"""
    maxby(a, b)

Return the max of `a` or `b`, comparing them by `a[1]` and `b[1]`, and breaking
ties to the left. Useful for implementing argmax operations:
```jldoctest; setup=:(using Finch)
julia> a = [7.7, 3.3, 9.9, 3.3, 9.9]; x = Scalar(-Inf => 0);

julia> @finch for i=_; x[] <<maxby>>= a[i] => i end;

julia> x[]
9.9 => 3
```
"""
maxby(a, b) = a[1] < b[1] ? b : a

"""
    logical_and(args...)

Return `true` when every argument is nonzero, like NumPy's `logical_and`.
Unlike `&` and `&&`, the arguments need not be `Bool`s.
```jldoctest; setup=:(using Finch)
julia> Finch.logical_and(2, 0.5, true)
true

julia> Finch.logical_and(2, 0)
false
```
"""
function logical_and end
logical_and() = true
logical_and(x) = !iszero(x)
logical_and(x, y, tail...) = logical_and(x) & logical_and(y, tail...)

"""
    logical_or(args...)

Return `true` when any argument is nonzero, like NumPy's `logical_or`.
Unlike `|` and `||`, the arguments need not be `Bool`s.
```jldoctest; setup=:(using Finch)
julia> a = Tensor(SparseList(Element(0)), [0, 2, 0, 3]); x = Scalar(false);

julia> @finch for i=_; x[] <<Finch.logical_or>>= a[i] end;

julia> x[]
true
```
"""
function logical_or end
logical_or() = false
logical_or(x) = !iszero(x)
logical_or(x, y, tail...) = logical_or(x) | logical_or(y, tail...)

"""
    logical_xor(args...)

Return `true` when an odd number of arguments are nonzero, like NumPy's
`logical_xor`. Unlike `xor`, the arguments need not be `Bool`s.
```jldoctest; setup=:(using Finch)
julia> Finch.logical_xor(2, 0.5)
false

julia> Finch.logical_xor(2, 0)
true
```
"""
function logical_xor end
logical_xor() = false
logical_xor(x) = !iszero(x)
logical_xor(x, y, tail...) = xor(logical_xor(x), logical_xor(y, tail...))

"""
    scaled_square(x)

Return the term of a scaled sum of squares for a nonnegative `x`. A scaled sum
is a tuple `(arg, scale)` which represents `arg * scale^2`, where `scale` is the
largest term so far, so that the sum neither overflows nor underflows. Terms are
summed with `add_scaled_square`, and `root_scaled_square` returns the 2-norm.
```jldoctest; setup=:(using Finch)
julia> a = Tensor(SparseList(Element(0.0)), [0, 3e200, 0, 4e200]); x = Scalar((0.0, 0.0));

julia> @finch for i=_; x[] <<Finch.add_scaled_square>>= Finch.scaled_square(abs(a[i])) end;

julia> Finch.root_scaled_square(x[]) ≈ 5e200
true
```
"""
scaled_square(x) = (ifelse(iszero(x), zero(x), one(x)) / one(x), x)

@inline function add_scaled(rescale, x, y)
    (x_arg, y_arg), (x_scale, y_scale) = promote(x[1], y[1]), promote(x[2], y[2])
    if isnan(x_arg) || isnan(x_scale) || isnan(y_arg) || isnan(y_scale)
        return (oftype(x_arg, NaN), oftype(x_scale, NaN))
    end
    if x_scale < y_scale
        (x_arg, x_scale, y_arg, y_scale) = (y_arg, y_scale, x_arg, x_scale)
    end
    if x_scale > y_scale
        return (x_arg + y_arg * rescale(y_scale / x_scale), x_scale)
    end
    return (x_arg + y_arg, x_scale)
end

add_scaled_square(x) = x
add_scaled_square(x, y) = add_scaled(r -> r * r, x, y)
function add_scaled_square(x, y, z, tail...)
    add_scaled_square(add_scaled_square(x, y), z, tail...)
end

root_scaled_square((arg, scale)) = sqrt(arg) * scale

struct ScaledPower{P} end
struct AddScaledPower{P} end
struct RootScaledPower{P} end

"""
    scaled_power(p)(x)

Like [`scaled_square`](@ref), for scaled sums `(arg, scale)` which represent
`arg * scale^p`, where `p` is positive. Terms are summed with
`add_scaled_power(p)`, and `root_scaled_power(p)` returns the `p`-norm.
"""
scaled_power(p) = ScaledPower{p}()
add_scaled_power(p) = AddScaledPower{p}()
root_scaled_power(p) = RootScaledPower{p}()

(f::ScaledPower)(x) = scaled_square(x)

(f::AddScaledPower)(x) = x
(f::AddScaledPower{P})(x, y) where {P} = add_scaled(r -> r^P, x, y)
(f::AddScaledPower)(x, y, z, tail...) = f(f(x, y), z, tail...)

(f::RootScaledPower{P})((arg, scale)) where {P} = arg^inv(P) * scale

struct ScaledNegativePower{P} end
struct AddScaledNegativePower{P} end
struct RootScaledNegativePower{P} end

"""
    scaled_negative_power(p)(x)

Like [`scaled_power`](@ref), for negative `p`. Here `scale` is the smallest term
so far, a zero term makes the sum infinite, and the empty sum is `(0, Inf)`.
Terms are summed with `add_scaled_negative_power(p)`, and
`root_scaled_negative_power(p)` returns the `p`-norm.
"""
scaled_negative_power(p) = ScaledNegativePower{p}()
add_scaled_negative_power(p) = AddScaledNegativePower{p}()
root_scaled_negative_power(p) = RootScaledNegativePower{p}()

function (f::ScaledNegativePower)(x)
    T = typeof(one(x) / one(x))
    return (iszero(x) ? T(Inf) : one(T), T(x))
end

(f::AddScaledNegativePower)(x) = x
function (f::AddScaledNegativePower{P})(x, y) where {P}
    (x_arg, y_arg), (x_scale, y_scale) = promote(x[1], y[1]), promote(x[2], y[2])
    if isnan(x_arg) || isnan(x_scale) || isnan(y_arg) || isnan(y_scale)
        return (oftype(x_arg, NaN), oftype(x_scale, NaN))
    end
    if iszero(x_scale) || iszero(y_scale)
        return (oftype(x_arg, Inf), zero(x_scale))
    end
    if x_scale > y_scale
        (x_arg, x_scale, y_arg, y_scale) = (y_arg, y_scale, x_arg, x_scale)
    end
    if x_scale < y_scale
        return (x_arg + y_arg * (x_scale / y_scale)^-P, x_scale)
    end
    return (x_arg + y_arg, x_scale)
end
(f::AddScaledNegativePower)(x, y, z, tail...) = f(f(x, y), z, tail...)

function (f::RootScaledNegativePower{P})((arg, scale)) where {P}
    root = arg^inv(P) * scale
    if iszero(scale) || iszero(arg) && isinf(scale)
        return oftype(root, scale)
    end
    return root
end

"""
    rem_nothrow(x, y)

Returns `rem(x, y)` normally, returns zero and issues a warning if `y` is zero.
"""
function rem_nothrow(x, y)
    if iszero(y)
        (@warn("Division by zero in rem"); zero(promote_type(typeof(x), typeof(y))))
    else
        rem(x, y)
    end
end

"""
    mod_nothrow(x, y)

Returns `mod(x, y)` normally, returns zero and issues a warning if `y` is zero.
"""
function mod_nothrow(x, y)
    if iszero(y)
        (@warn("Division by zero in mod"); zero(promote_type(typeof(x), typeof(y))))
    else
        mod(x, y)
    end
end

"""
    mod1_nothrow(x, y)

Returns `mod1(x, y)` normally, returns one and issues a warning if `y` is zero.
"""
function mod1_nothrow(x, y)
    if iszero(y)
        (@warn("Division by zero in mod1"); one(promote_type(typeof(x), typeof(y))))
    else
        mod1(x, y)
    end
end

"""
    fld_nothrow(x, y)

Returns `fld(x, y)` normally, returns zero and issues a warning if `y` is zero.
"""
function fld_nothrow(x, y)
    if iszero(y)
        (@warn("Division by zero in fld"); zero(promote_type(typeof(x), typeof(y))))
    else
        fld(x, y)
    end
end

"""
    fld1_nothrow(x, y)

Returns `fld1(x, y)` normally, returns one and issues a warning if `y` is zero.
"""
function fld1_nothrow(x, y)
    if iszero(y)
        (@warn("Division by zero in fld1"); one(promote_type(typeof(x), typeof(y))))
    else
        fld1(x, y)
    end
end

"""
    cld_nothrow(x, y)

Returns `cld(x, y)` normally, returns zero and issues a warning if `y` is zero.
"""
function cld_nothrow(x, y)
    if iszero(y)
        (@warn("Division by zero in cld"); zero(promote_type(typeof(x), typeof(y))))
    else
        cld(x, y)
    end
end

struct InitMin{D} end
(f::InitMin{D})(x) where {D} = x

"""
    InitMin{D}(x, y)

Returns min(x, y, D). Used to add an init parameter to a minimum operation.
"""
@inline function (f::InitMin{D})(x, y) where {D}
    min(x, y, D)
end
initmin(z) = InitMin{z}()

struct InitMax{D} end
(f::InitMax{D})(x) where {D} = x
"""
    InitMax{D}(x, y)

Returns max(x, y, D). Used to add an init parameter to a maximum operation.
"""
@inline function (f::InitMax{D})(x, y) where {D}
    max(x, y, D)
end
initmax(z) = InitMax{z}()
