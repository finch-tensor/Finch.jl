```@meta
CurrentModule = Finch
```

# User-Defined Functions

## User Functions

Finch supports arbitrary Julia Base functions over
[`isbits`](https://docs.julialang.org/en/v1/base/base/#Base.isbits) types.  You
can also use your own functions and use them in Finch! Just remember to define
any special algebraic properties of your functions so that Finch can optimize
them better. You must declare the properties of your functions before you call
any Finch functions on them.

Finch only supports incrementing assignments to arrays such as `+=` or `*=`. If
you would like to increment `A[i...]` by the value of `ex` with a custom
reduction operator `op`, you may use the following syntax: `A[i...] <<op>>= ex`.

Consider the greatest common divisor function `gcd`. This function is
associative and commutative, and the greatest common divisor of 1 and anything
else is 1, so 1 is an annihilator.  We declare these properties by overloading
trait functions on Finch's default algebra as follows:

```
Finch.isassociative(::Finch.DefaultAlgebra, ::typeof(gcd)) = true
Finch.iscommutative(::Finch.DefaultAlgebra, ::typeof(gcd)) = true
Finch.isannihilator(::Finch.DefaultAlgebra, ::typeof(gcd), x) = x == 1
```

Then, the following code will only call gcd when neither `u[i]` nor `v[i]` are 1
(just once!).

```
u = Tensor(SparseList(Element(1)), [3, 1, 6, 1, 9, 1, 4, 1, 8, 1])
v = Tensor(SparseList(Element(1)), [1, 2, 3, 1, 1, 1, 1, 4, 1, 1])
w = Tensor(SparseList(Element(1)))

@finch MyAlgebra() (w .= 1; for i=_; w[i] = gcd(u[i], v[i]) end)
```

## A Few Convenient Functions

For your convenience, Finch defines a few useful functions that help express common array operations inside Finch:

```@docs
choose
minby
maxby
```

## Properties

The full list of properties recognized by Finch is as follows (use these to declare the properties of your own functions):

```@docs
isassociative
iscommutative
isdistributive
isidempotent
isidentity
isannihilator
isinverse
isinvolution
Finch.return_type
```

## Finch Kernel Caching

Finch code is cached when you first run it. Thus, if you run a Finch
function once, then make changes to the Finch compiler (like defining new
properties), the cached code will be used and the changes will not be reflected.

It's best to design your code so that modifications to the Finch compiler occur
before any Finch functions are called. However, if you really need to modify a
precompiled Finch kernel, you can call `Finch.refresh()` to invalidate the
code cache.

```@docs
refresh
```

### (Advanced) On World-Age and Generated Functions

Julia uses a "world age" to describe the set of defined functions at a point in time. Generated functions run in the same world age in which they were defined, so they can't call functions defined after the generated function. This means that if Finch used normal generated functions, users can't define their own functions without first redefining all of Finch's generated functions.

Finch uses special generators that run in the current world age, but do not
update with subsequent compiler function invalidations. If two packages modify
the behavior of Finch in different ways, and call those Finch functions during
precompilation, the resulting behavior is undefined.

There are several packages that take similar, but different, approaches to
allow user participation in staged Julia programming (not to mention Base `eval`
or `@generated`):
[StagedFunctions.jl](https://github.com/NHDaly/StagedFunctions.jl),
[GeneralizedGenerated.jl](https://github.com/JuliaStaging/GeneralizedGenerated.jl),
[RuntimeGeneratedFunctions.jl](https://github.com/SciML/RuntimeGeneratedFunctions.jl),
or [Zygote.jl](https://github.com/FluxML/Zygote.jl).

Our approach is most similar to that of StagedFunctions.jl or Zygote.jl. We
chose our approach to be the simple and flexible while keeping the kernel call
overhead low.

## (Advanced) Separate Algebras

If you want to define non-standard properties or custom rewrite rules for some
functions in a separate context, you can represent these changes with your own
algebra type.  We express this by subtyping `AbstractAlgebra` and defining
properties as follows:

```
struct MyAlgebra <: AbstractAlgebra end

Finch.isassociative(::MyAlgebra, ::typeof(gcd)) = true
Finch.iscommutative(::MyAlgebra, ::typeof(gcd)) = true
Finch.isannihilator(::MyAlgebra, ::typeof(gcd), x) = x == 1
```

We pass the algebra to Finch as an optional first argument:

```
@finch MyAlgebra() (w .= 1; for i=_; w[i] = gcd(u[i], v[i]) end; return w)
```

### Rewriting

Define custom rewrite rules by overloading the `get_simplify_rules` function
on your algebra.  Unless you want to write the full rule set from scratch, be
sure to append your new rules to the old rules, which can be obtained by calling
`get_simplify_rules` with another algebra. Rules can be specified directly on Finch IR using
[RewriteTools.jl](https://github.com/willow-ahrens/RewriteTools.jl).

```@docs
get_simplify_rules
get_prove_rules
```
