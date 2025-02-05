```@meta
CurrentModule = Finch
```

## Finch Notation

Finch programs are written in Julia, but they are not Julia programs.
Instead, they are an abstraction description of a tensor computation.

Several examples are given in the [examples directory](https://github.com/finch-tensor/Finch.jl/blob/main/docs/examples).

Finch programs are blocks of tensor operations, joined by control flow. Finch
is an imperative language. The AST is separated into statements and expressions,
where statements can modify the state of the program but expressions cannot.

The core Finch expressions are:

  - [`literal`](@ref) e.g. `1`, `1.0`, `nothing`
  - [`value`](@ref) e.g. `x`, `y`
  - [`index`](@ref) e.g. `i`, inside of `for i = _; ... end`
  - [`variable`](@ref) e.g. `x`, inside of `(x = y; ...)`
  - [`call`](@ref) e.g. `op(args...)`
  - [`access`](@ref) e.g. `tns[idxs...]`

And the core Finch statements are:

  - [`declare`](@ref) e.g. `tns .= init`
  - [`assign`](@ref) e.g. `lhs[idxs...] <<op>>= rhs`
  - [`loop`](@ref) e.g. `for i = _; ... end`
  - [`define`](@ref) e.g. `let var = val; ... end`
  - [`sieve`](@ref) e.g. `if cond; ... end`
  - [`block`](@ref) e.g. `begin ... end`

```@docs
literal
value
index
variable
call
access
define
assign
loop
sieve
block
```

# Scoping

Finch programs are scoped. Scopes contain variable definitions and tensor
declarations.  Loops and sieves introduce new scopes. The following program
has four scopes, each of which is numbered to the left of the statements it contains.

```
@finch begin
1   y .= 0
1   for j = _
1   2   t .= 0
1   2   for i = _
1   2   3   t[] += A[i, j] * x[i]
1   2   end
1   2   for i = _
1   2   4   y[i] += A[i, j] * t[]
1   2   end
1   end
end
```

Variables refer to their defined values in the innermost containing scope. If variables are undefined, they are assumed to have global scope (they may come from the surrounding program).

# Tensor Lifecycle

Tensors have two modes: Read and Update. Tensors in read mode may be read, but not updated. Tensors in update mode may be updated, but not read. A tensor declaration initializes and possibly resizes the tensor, setting it to update mode. Also, Finch will automatically change the mode of tensors as they are used. However, tensors may only change their mode within scopes that contain their declaration. If a tensor has not been declared, it is assumed to have global scope.

Tensor declaration is different than variable definition. Declaring a tensor initializes the memory (usually to zero) and sets the tensor to update mode. Defining a tensor simply gives a name to that memory. A tensor may be declared multiple times, but it may only be defined once.

Tensors are assumed to be in read mode when they are defined.
Tensors must enter and exit scope in read mode. Finch inserts
`freeze` and `thaw` statements to ensure that tensors are in the correct mode. Freezing a tensor prevents further updates and allows reads. Thawing a tensor allows further updates and prevents reads.

Tensor lifecycle statements consist of:

```@docs
reader
updater
declare
freeze
thaw
```

# Dimensionalization

Finch loops have dimensions. Accessing a tensor with an unmodified loop index
"hints" that the loop should have the same dimension as the corresponding axis
of the tensor. Finch will automatically dimensionalize loops that are hinted by
tensor accesses. One may refer to the automatically determined dimension using a
variable named `_` or `:`.

Similarly, tensor declarations also set the dimensions of a tensor. Accessing a tensor with an unmodified loop index
"hints" that the tensor axis should have the same dimension as the corresponding loop. Finch will automatically dimensionalize declarations based on all updates up to the first read.

# Array Combinators

Finch includes several array combinators that modify the behavior of arrays. For
example, the `OffsetArray` type wraps an existing array, but shifts its
indices. The `PermissiveArray` type wraps an existing array, but allows
out-of-bounds reads and writes. When an array is accessed out of bounds, it
produces `Missing`.

Array combinators introduce some complexity to the tensor lifecycle, as wrappers
may contain multiple or different arrays that could potentially be in different
modes. Any array combinators used in a tensor access must reference a single
global variable which holds the root array. The root array is the single array
that gets declared, and changes modes from read to update, or vice versa.

# Fancy Indexing

Finch supports arbitrary indexing of arrays, but certain
indexing operations have first class support through
array combinators. Before dimensionalization, the following transformations are performed:

```
    A[i + c] =>        OffsetArray(A, c)[i]
    A[i + j] =>      ToeplitzArray(A, 1)[i, j]
       A[~i] => PermissiveArray(A, true)[i]
```

Note that these transformations may change the behavior of dimensionalization, since they often result in unmodified loop indices (the index `i` will participate in dimensionalization, but an index expression like `i + 1` will not).
