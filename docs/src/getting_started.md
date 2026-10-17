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

## Writing Finch Programs

### Indexing

Finch tensors support indexing and slicing, as well as standard array functions
such as `copy`, `permutedims`, and `reshape`.

```jldoctest arrayapi; setup = :(using Finch)
julia> A = Tensor(CSCFormat(), [0 1; 2 3]);

julia> A[2, 1]
2.0

julia> A[1, :]
2 Tensor{SparseDictLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, Dict{Tuple{Int64, Int64}, Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}:
 0.0
 1.0

julia> permutedims(A)
2×2 Tensor{SparseDictLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, Dict{Tuple{Int64, Int64}, Int64}, Vector{Int64}, SparseDictLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, Dict{Tuple{Int64, Int64}, Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}:
 0.0  2.0
 1.0  3.0
```

### Loops

To compute with tensors, write loops with the [`@finch`](@ref) macro. Finch
compiles each program for the formats of the tensors it touches, so every
operation in the loop body is fused into a single kernel that skips the zeros of
the sparse inputs.

```jldoctest arrayapi; setup = :(using Finch)
julia> B = Tensor(CSCFormat(), [1 0; 0 4]);

julia> C = Tensor(CSCFormat());

julia> D = Tensor(Dense(Element(0.0)));

julia> @finch begin
           C .= 0
           D .= 0
           for j = _, i = _
               C[i, j] = A[i, j] * B[i, j]
               D[i] += A[i, j] + B[i, j]
           end
       end;

julia> C
2×2 Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}:
 0.0   0.0
 0.0  12.0

julia> D
2 Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}:
 2.0
 9.0
```

To generate a reusable function ahead of time instead, use the
[`@finch_kernel`](@ref) macro. The [Finch Language](docs/language/finch_language.md)
section describes the full language.

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
