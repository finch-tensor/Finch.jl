```@meta
CurrentModule = Finch
```

# Getting Started

## Tensor Formats

### Creating Tensors

You can create Finch tensors using the [`Tensor`](@ref) constructor, which closely follows the `Array` constructor syntax. The first argument specifies the storage format.

```jldoctest tensorformats; setup = :(using Finch)
julia> A = Tensor(CSCFormat(), 4, 3);

julia> B = Tensor(COOFormat(2), A);

```

Some pre-defined formats include:

| **Signature**                                        | **Description**                                     |
|:---------------------------------------------------- |:--------------------------------------------------- |
| [`DenseFormat`](@ref)`(N, z = 0.0, T = typeof(z))`   | A dense format with a fill value of `z`.            |
| [`CSFFormat`](@ref)`(N, z = 0.0, T = typeof(z))`     | An `N`-dimensional CSC format for sparse tensors.   |
| [`CSCFormat`](@ref)`(z = 0.0, T = typeof(z))`        | A 2D CSC format storing matrices as dense lists.    |
| [`DCSFFormat`](@ref)`(N, z = 0.0, T = typeof(z))`    | A DCSF format storing tensors as nested lists.      |
| [`HashFormat`](@ref)`(N, z = 0.0, T = typeof(z))`    | A hash-table-based format for sparse data.          |
| [`ByteMapFormat`](@ref)`(N, z = 0.0, T = typeof(z))` | A byte-map-based format for compact storage.        |
| [`DCSCFormat`](@ref)`(z = 0.0, T = typeof(z))`       | A 2D DCSC format storing matrices as lists.         |
| [`COOFormat`](@ref)`(N, T = Float64, z = zero(T))`   | An `N`-dimensional COO format for coordinate lists. |

It is also possible to build custom formats using the interface, as described in the [Tensor Formats](#tensor-formats) section.

## High-Level Array API

### Basic Array Operations

Finch tensors support indexing, slicing, mapping, broadcasting, and reducing.
Many functions in the Julia standard array library are supported.

```jldoctest arrayapi; setup = :(using Finch)
julia> A = Tensor(CSCFormat(), [0 1; 2 3]);

julia> B = A .+ 1
2×2 Tensor{DenseLevel{Int64, DenseLevel{Int64, ElementLevel{1.0, Float64, Int64, Vector{Float64}}}}}:
 1.0  2.0
 3.0  4.0

julia> C = max.(A, B)
2×2 Tensor{DenseLevel{Int64, DenseLevel{Int64, ElementLevel{1.0, Float64, Int64, Vector{Float64}}}}}:
 1.0  2.0
 3.0  4.0

julia> D = sum(C; dims=2)
2×1 Tensor{DenseLevel{Int64, DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}:
 3.0
 7.0

julia> E = B[1, :]
2 Tensor{DenseLevel{Int64, ElementLevel{1.0, Float64, Int64, Vector{Float64}}}}:
 1.0
 2.0
```

For situations which are difficult to express in the julia standard library, Finch also supports an `@einsum` syntax:

```jldoctest arrayapi; setup = :(using Finch)
julia> @einsum F[i, j, k] *= A[i, j] * B[j, k]
2×2×2 Tensor{DenseLevel{Int64, DenseLevel{Int64, SparseDictLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, Dict{Tuple{Int64, Int64}, Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}}:
[:, :, 1] =
 0.0  3.0
 2.0  9.0

[:, :, 2] =
 0.0   4.0
 4.0  12.0

julia> @einsum G[j, k] << max >>= A[i, j] + B[j, k]
2×2 Tensor{DenseLevel{Int64, DenseLevel{Int64, ElementLevel{-Inf, Float64, Int64, Vector{Float64}}}}}:
 3.0  4.0
 6.0  7.0
```

The `@einsum` macro is a powerful tool for expressing complex array operations concisely.

## Array Fusion

To get the full benefits of a sparse compiler, it is critical to fuse certain operations together. For this, Finch exposes two functions, [`lazy`](@ref) and [`compute`](@ref).
The `lazy` function creates a lazy tensor, which is a symbolic representation of the computation. The `compute` function evaluates the computation. For convenience, you may wish to use the [`fused`](@ref) function, which automatically fuses the computations it contains.

```jldoctest fusion; setup = :(using Finch)
julia> A = fsparse([1, 1, 2, 3], [2, 4, 5, 6], [1.0, 2.0, 3.0])
3×6 Tensor{SparseCOOLevel{2, Tuple{Int64, Int64}, Vector{Int64}, Tuple{Vector{Int64}, Vector{Int64}}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}:
 0.0  1.0  0.0  2.0  0.0  0.0
 0.0  0.0  0.0  0.0  3.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0

julia> B = A .* 2
3×6 Tensor{SparseDictLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, Dict{Tuple{Int64, Int64}, Int64}, Vector{Int64}, SparseDictLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, Dict{Tuple{Int64, Int64}, Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}:
 0.0  2.0  0.0  4.0  0.0  0.0
 0.0  0.0  0.0  0.0  6.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0

julia> C = lazy(A)
3×6-LazyTensor{Float64}

julia> D = lazy(B)
3×6-LazyTensor{Float64}

julia> E = (C .+ D) ./ 2
3×6-LazyTensor{Float64}

julia> compute(E)
3×6 Tensor{SparseDictLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, Dict{Tuple{Int64, Int64}, Int64}, Vector{Int64}, SparseDictLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, Dict{Tuple{Int64, Int64}, Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}:
 0.0  1.5  0.0  3.0  0.0  0.0
 0.0  0.0  0.0  0.0  4.5  0.0
 0.0  0.0  0.0  0.0  0.0  0.0
```

The `lazy` and `compute` functions allow the compiler to fuse operations together, resulting in asymptotically more efficient code.

```julia
julia> using BenchmarkTools

julia> A = fsprand(1000, 1000, 100);
       B = Tensor(rand(1000, 1000));
       C = Tensor(rand(1000, 1000));

julia> @btime A .* (B * C);
  145.940 ms (859 allocations: 7.69 MiB)

julia> @btime compute(lazy(A) .* (lazy(B) * lazy(C)));
  694.666 μs (712 allocations: 60.86 KiB)
```

Different optimizers can be used with `compute`, such as the state-of-the-art `Galley` optimizer, which can adapt to the
sparsity patterns of the inputs.

```julia
julia> A = fsprand(1000, 1000, 0.1);
       B = fsprand(1000, 1000, 0.1);
       C = fsprand(1000, 1000, 0.0001);

julia> A = lazy(A);
       B = lazy(B);
       C = lazy(C);

julia> @btime compute(sum(A * B * C));
  282.503 ms (1018 allocations: 184.43 MiB)

julia> @btime compute(sum(A * B * C), ctx=galley_scheduler());
  152.792 μs (672 allocations: 28.81 KiB)
```

## Sparse and Structured Utilities

### Sparse Constructors

[`fsparse`](@ref) constructs a tensor from lists of nonzero coordinates. For example,

```jldoctest sparseutils; setup = :(using Finch)
julia> A = fsparse([1, 2, 3], [2, 3, 4], [1.0, 2.0, 3.0])
3×4 Tensor{SparseCOOLevel{2, Tuple{Int64, Int64}, Vector{Int64}, Tuple{Vector{Int64}, Vector{Int64}}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}:
 0.0  1.0  0.0  0.0
 0.0  0.0  2.0  0.0
 0.0  0.0  0.0  3.0
```

The inverse of [`fsparse`](@ref) is [`ffindnz`](@ref), which returns a list of nonzero coordinates in a tensor.

```jldoctest sparseutils; setup = :(using Finch)
julia> ffindnz(A)
([1, 2, 3], [2, 3, 4], [1.0, 2.0, 3.0])
```

### Random Sparse Tensors

The [`fsprand`](@ref) constructs a random sparse tensor with a specified sparsity or number of nonzeros:

```julia
julia> A = fsprand(5, 5, 0.1)

```

### Fill Values

Fill values represent the background value of a sparse tensor. Usually, this value is zero, but some applications may choose to use other fill values as fits their application. Only values which are not equal to the fill value are stored

  - **[`fill_value`](@ref)**: Retrieve the fill value.
  - **[`set_fill_value!`](@ref)**: Set a new fill value.
  - **[`dropfills`](@ref)** or **[`dropfills!`](@ref)**: Remove elements matching the fill value.
  - **[`countstored`](@ref)**: Return the number of stored values in a tensor

```jldoctest sparseutils; setup = :(using Finch)
julia> t = Tensor(Dense(SparseList(Element(0.0))), 3, 3)
3×3 Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}:
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

julia> fill_value(t)
0.0

julia> set_fill_value!(t, -1.0)
3×3 Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{-1.0, Float64, Int64, Vector{Float64}}}}}:
 -1.0  -1.0  -1.0
 -1.0  -1.0  -1.0
 -1.0  -1.0  -1.0

julia> countstored(t)
0

julia> countstored(dropfills(t))
0
```

### Empty Tensors

The Tensor constructor initializes tensors to their fill value when given a list of dimensions, but you can also use [`fspzeros`](@ref) for an empty COO Tensor, for consistency with MATLAB.

```jldoctest sparseutils; setup = :(using Finch)
julia> A = fspzeros(3, 3)
3×3 Tensor{SparseCOOLevel{2, Tuple{Int64, Int64}, Vector{Int64}, Tuple{Vector{Int64}, Vector{Int64}}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}:
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

julia> B = Tensor(CSCFormat(1.0), 3, 3)
3×3 Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{1.0, Float64, Int64, Vector{Float64}}}}}:
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0
```

### Converting Between Formats

You can convert between tensor formats with the `Tensor` constructor. Simply construct a new Tensor in the desired format and

```jldoctest tensorformats; setup = :(using Finch)
julia> A = Tensor(CSCFormat(), [0 0 2 1; 0 0 1 0; 1 0 0 0])
3×4 Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}:
 0.0  0.0  2.0  1.0
 0.0  0.0  1.0  0.0
 1.0  0.0  0.0  0.0

julia> B = Tensor(DCSCFormat(), A)
3×4 Tensor{SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}:
 0.0  0.0  2.0  1.0
 0.0  0.0  1.0  0.0
 1.0  0.0  0.0  0.0
```

### Storage Order

By default, tensors in Finch are column-major. However, you can use the
`swizzle` function to transpose them lazily. To convert to a transposed format,
use the `dropfills!` function.

```jldoctest tensorformats; setup = :(using Finch)
julia> A = Tensor(CSCFormat(), [0 0 2 1; 0 0 1 0; 1 0 0 0])
3×4 Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}:
 0.0  0.0  2.0  1.0
 0.0  0.0  1.0  0.0
 1.0  0.0  0.0  0.0

julia> swizzle(A, 2, 1)
4×3 Finch.SwizzleArray{(2, 1), Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}}:
 0.0  0.0  1.0
 0.0  0.0  0.0
 2.0  1.0  0.0
 1.0  0.0  0.0

julia> dropfills!(swizzle(Tensor(CSCFormat()), 2, 1), A)
3×4 Finch.SwizzleArray{(2, 1), Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}}:
 0.0  0.0  2.0  1.0
 0.0  0.0  1.0  0.0
 1.0  0.0  0.0  0.0
```

## File I/O

### Reading and Writing Files

Finch supports multiple formats, such as `.bsp`, `.mtx`, and `.tns`. Use `fread` and `fwrite` to read and write tensors.

```julia
julia> fwrite("tensor.bsp", A)

julia> B = fread("tensor.bsp")

```
