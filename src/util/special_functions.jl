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
```jldoctest setup=:(using Finch)
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

```jldoctest setup=:(using Finch)
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
```jldoctest setup=:(using Finch)
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
```jldoctest setup=:(using Finch)
julia> a = [7.7, 3.3, 9.9, 3.3, 9.9]; x = Scalar(-Inf => 0);

julia> @finch for i=_; x[] <<maxby>>= a[i] => i end;

julia> x[]
9.9 => 3
```
"""
maxby(a, b) = a[1] < b[1] ? b : a

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
