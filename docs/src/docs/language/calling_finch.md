```@meta
CurrentModule = Finch
```

# Finch Language

Writing Finch language directly is a powerful way to express complex array
operations. Finch supports complex optimizations under the hood, including
conditionals, multiple outputs, and even user-defined types and functions.

### Supported Syntax and Structures

| Feature/Structure                          | Example Usage                                      |
|:------------------------------------------ |:-------------------------------------------------- |
| Major Sparse Formats and Structured Arrays | `A = Tensor(Dense(SparseList(Element(0.0)), 3, 4)` |
| Background Values Other Than Zero          | `B = Tensor(SparseList(Element(1.0)), 9)`          |
| Broadcasts and Reductions                  | `sum(A .* B)`                                      |
| User-Defined Functions                     | `x[] <<min>>= y[i] + z[i]`                         |
| Multiple Outputs                           | `x[] <<min>>= y[i]; z[] <<max>>= y[i]`             |
| Multicore Parallelism                      | `for i = parallel(1:100)`                          |
| Conditionals                               | `if dist[] < best_dist[]`                          |
| Affine Indexing (e.g. Convolution)         | `A[i + j]`                                         |

## Quick Start: Examples

To begin, the following program sums the rows of a sparse matrix:

```julia
using Finch
A = sprand(5, 5, 0.5)
y = zeros(5)
@finch begin
    y .= 0
    for i in _, j in _
        y[i] += A[i, j]
    end
end
```

The [`@finch`](@ref) macro takes a block of code, and compiles it using the sparsity
attributes of the arguments. In this case, `A` is a sparse matrix, so the
compiler generates a sparse loop nest. The compiler takes care of applying rules
like `x * 0 => 0` during compilation to make the code more efficient.

You can call [`@finch`](@ref) on any loop program, but it will only generate sparse code
if the arguments are sparse. For example, the following program calculates the
sum of the elements of a dense matrix:

```julia
using Finch
A = rand(5, 5)
s = Scalar(0.0)
@finch begin
    s .= 0
    for i in _, j in _
        s[] += A[i, j]
    end
end
```

You can call [`@finch_code`](@ref) to see the generated code (since `A` is dense, the
code is dense):

```jldoctest example1; setup=:(using Finch; A = rand(5, 5); s = Scalar(0))
julia> @finch_code for i in _, j in _
           s[] += A[i, j]
       end
quote
    s = (ex.bodies[1]).body.body.lhs.tns.bind
    s_val = s.val
    A_data = (ex.bodies[1]).body.body.rhs.tns.bind
    sugar_1 = size((ex.bodies[1]).body.body.rhs.tns.bind)
    A_mode1_stop = sugar_1[1]
    A_mode2_stop = sugar_1[2]
    @warn "Performance Warning: non-concordant traversal of A[i, j] (hint: most arrays prefer column major or first index fast, run in fast mode to ignore this warning)"
    for i_3 = 1:A_mode1_stop
        for j_3 = 1:A_mode2_stop
            val = A_data[i_3, j_3]
            s_val = val + s_val
        end
    end
    result = ()
    s.val = s_val
    result
end
```

### Calculating Sparse Vector Statistics

Here, we write a Julia program using Finch to compute the minimum, maximum, sum, and variance of a sparse vector. This program efficiently reads the vector once, focusing only on nonzero values.

```julia
using Finch

X = Tensor(SparseList(Element(0.0)), fsprand(10, 0.5))
x_min = Scalar(Inf)
x_max = Scalar(-Inf)
x_sum = Scalar(0.0)
x_var = Scalar(0.0)

@finch begin
    for i in _
        let x = X[i]
            x_min[] << min >>= x
            x_max[] << max >>= x
            x_sum[] += x
            x_var[] += x * x
        end
    end
end;
```

### Sparse Matrix-Vector Multiplication

As a more traditional example, what follows is a sparse matrix-vector multiplication using a column-major approach.

```julia
x = Tensor(Dense(Element(0.0)), rand(42));
A = Tensor(Dense(SparseList(Element(0.0))), fsprand(42, 42, 0.1));
y = Tensor(Dense(Element(0.0)));

@finch begin
    y .= 0
    for j in _, i in _
        y[i] += A[i, j] * x[j]
    end
end
```

More examples are given in the [examples directory](https://github.com/finch-tensor/Finch.jl/blob/main/docs/examples).

# Usage

## General Purpose (`@finch`)

Most users will want to use the [`@finch`](@ref) macro, which executes the given
program immediately in the given scope. The program will be JIT-compiled on the
first call to `@finch` with the given array argument types. If the array
arguments to `@finch` are [type
stable](https://docs.julialang.org/en/v1/manual/faq/#man-type-stability), the
program will be JIT-compiled when the surrounding function is compiled.

Very often, the best way to inspect Finch compiler behavior is through the
[`@finch_code`](@ref) macro, which prints the generated code instead of
executing it.

```@docs
@finch
@finch_code
```

## Ahead Of Time (`@finch_kernel`)

While [`@finch`](@ref) is the recommended way to use Finch, it is also possible
to run finch ahead-of-time. The [`@finch_kernel`](@ref) macro generates a
function definition ahead-of-time, which can be evaluated and then called later.

There are several reasons one might want to do this:

 1. If we want to make tweaks to the Finch implementation, we can directly modify the source code of the resulting function.
 2. When benchmarking Finch functions, we can easily and reliably ensure the benchmarked code is [inferrable](https://docs.julialang.org/en/v1/devdocs/inference/).
 3. If we want to use Finch to generate code but don't want to include Finch as a dependency in our project, we can use [`@finch_kernel`](@ref) to generate the functions ahead of time and copy and paste the generated code into our project.  Consider automating this workflow to keep the kernels up to date!

```@docs
    @finch_kernel
```

As an example, the following code generates an spmv kernel definition, evaluates
the definition, and then calls the kernel several times.

```julia
let
    A = Tensor(Dense(SparseList(Element(0.0))))
    x = Tensor(Dense(Element(0.0)))
    y = Tensor(Dense(Element(0.0)))
    def = @finch_kernel function spmv(y, A, x)
        y .= 0.0
        for j in _, i in _
            y[i] += A[i, j] * x[j]
        end
        return y
    end
    eval(def)
end

function main()
    for i in 1:10
        A2 = Tensor(Dense(SparseList(Element(0.0))), fsprand(10, 10, 0.1))
        x2 = Tensor(Dense(Element(0.0)), rand(10))
        y2 = Tensor(Dense(Element(0.0)))
        spmv(y2, A2, x2)
    end
end

main()
```
