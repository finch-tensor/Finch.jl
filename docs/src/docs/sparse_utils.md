# Sparse Array Utilities

## Sparse Constructors

In addition to the `Tensor` constructor, Finch provides a number of convenience
constructors for common tensor types. For example, the `spzeros` and `sprand` functions
have `fspzeros` and `fsprand` counterparts that return Finch tensors. We can also construct
a sparse COO `Tensor` from a list of indices and values using the `fsparse` function.

```@docs
fsparse
fsparse!
fsprand
fspzeros
ffindnz
```

## Fill Values

Finch tensors support an arbitrary "background" value for sparse arrays. While most arrays use `0` as the background value, this is not always the case. For example, a sparse array of `Int` might use `typemin(Int)` as the background value. The `fill_value` function returns the background value of a tensor. If you ever want to change the background value of an existing array, you can use the `set_fill_value!` function. The `countstored` function returns the number of stored elements in a tensor, and calling `pattern!` on a tensor returns tensor which is true whereever the original tensor stores a value. Note that countstored doesn't always return the number of non-zero elements in a tensor, as it counts the number of stored elements, and stored elements may include the background value. You can call `dropfills!` to remove explicitly stored background values from a tensor.

```jldoctest example1; setup = :(using Finch)
julia> A = fsparse([1, 1, 2, 3], [2, 4, 5, 6], [1.0, 2.0, 3.0])
3×6 Tensor{SparseCOOLevel{2, Tuple{Int64, Int64}, Vector{Int64}, Tuple{Vector{Int64}, Vector{Int64}}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}:
 0.0  1.0  0.0  2.0  0.0  0.0
 0.0  0.0  0.0  0.0  3.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0

julia> min.(A, -1)
3×6 Tensor{DenseLevel{Int64, DenseLevel{Int64, ElementLevel{-1.0, Float64, Int64, Vector{Float64}}}}}:
 -1.0  -1.0  -1.0  -1.0  -1.0  -1.0
 -1.0  -1.0  -1.0  -1.0  -1.0  -1.0
 -1.0  -1.0  -1.0  -1.0  -1.0  -1.0

julia> fill_value(A)
0.0

julia> B = set_fill_value!(A, -Inf)
3×6 Tensor{SparseCOOLevel{2, Tuple{Int64, Int64}, Vector{Int64}, Tuple{Vector{Int64}, Vector{Int64}}, ElementLevel{-Inf, Float64, Int64, Vector{Float64}}}}:
 -Inf    1.0  -Inf    2.0  -Inf   -Inf
 -Inf  -Inf   -Inf  -Inf     3.0  -Inf
 -Inf  -Inf   -Inf  -Inf   -Inf   -Inf

julia> min.(B, -1)
3×6 Tensor{SparseDictLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, Dict{Tuple{Int64, Int64}, Int64}, Vector{Int64}, SparseDictLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, Dict{Tuple{Int64, Int64}, Int64}, Vector{Int64}, ElementLevel{-Inf, Float64, Int64, Vector{Float64}}}}}:
 -Inf   -1.0  -Inf   -1.0  -Inf   -Inf
 -Inf  -Inf   -Inf  -Inf    -1.0  -Inf
 -Inf  -Inf   -Inf  -Inf   -Inf   -Inf

julia> countstored(A)
3

julia> pattern!(A)
3×6 Tensor{SparseCOOLevel{2, Tuple{Int64, Int64}, Vector{Int64}, Tuple{Vector{Int64}, Vector{Int64}}, PatternLevel{Int64}}}:
 0  1  0  1  0  0
 0  0  0  0  1  0
 0  0  0  0  0  0
```

```@docs
fill_value
set_fill_value!
pattern!
countstored
dropfills
dropfills!
```

### How to tell whether an entry is "fill"

In the sparse world, a semantic distinction is sometimes made between
"explicitly stored" values and "implicit" or "fill" values (usually zero).
However, the formats in the Finch compiler represent a diverse set of structures
beyond sparsity, and it is often unclear whether any of the values in the tensor
are "explicit" (consider a mask matrix, which can be represented with a constant
number of bits). Thus, Finch makes no semantic distinction between values which
are stored explicitly or not. If users wish to make this distinction, they should
instead store a tensor of tuples of the form `(value, is_fill)`. For example,

```jldoctest example3; setup = :(using Finch)
julia> A = fsparse(
           [1, 1, 2, 3],
           [2, 4, 5, 6],
           [(1.0, false), (0.0, true), (3.0, false)];
           fill_value=(0.0, true),
       )
3×6 Tensor{SparseCOOLevel{2, Tuple{Int64, Int64}, Vector{Int64}, Tuple{Vector{Int64}, Vector{Int64}}, ElementLevel{(0.0, true), Tuple{Float64, Bool}, Int64, Vector{Tuple{Float64, Bool}}}}}:
 (0.0, 1)  (1.0, 0)  (0.0, 1)  (0.0, 1)  (0.0, 1)  (0.0, 1)
 (0.0, 1)  (0.0, 1)  (0.0, 1)  (0.0, 1)  (3.0, 0)  (0.0, 1)
 (0.0, 1)  (0.0, 1)  (0.0, 1)  (0.0, 1)  (0.0, 1)  (0.0, 1)

julia> B = Tensor(Dense(SparseList(Element((0.0, true)))), A)
3×6 Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{(0.0, true), Tuple{Float64, Bool}, Int64, Vector{Tuple{Float64, Bool}}}}}}:
 (0.0, 1)  (1.0, 0)  (0.0, 1)  (0.0, 1)  (0.0, 1)  (0.0, 1)
 (0.0, 1)  (0.0, 1)  (0.0, 1)  (0.0, 1)  (3.0, 0)  (0.0, 1)
 (0.0, 1)  (0.0, 1)  (0.0, 1)  (0.0, 1)  (0.0, 1)  (0.0, 1)

julia> sum(map(last, B))
16

julia> sum(map(first, B))
4.0
```

## Format Conversion and Storage Order

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
use the `dropfills!` function. Note that the `permutedims` function transposes eagerly.

```@docs
swizzle
```

```jldoctest tensorformats; setup = :(using Finch)
julia> A = Tensor(CSCFormat(), [0 0 2 1; 0 0 1 0; 1 0 0 0])
3×4 Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}:
 0.0  0.0  2.0  1.0
 0.0  0.0  1.0  0.0
 1.0  0.0  0.0  0.0

julia> tensor_tree(swizzle(A, 2, 1))
SwizzleArray (2, 1)
└─ 3×4-Tensor
   └─ Dense [:,1:4]
      ├─ [:, 1]: SparseList (0.0) [1:3]
      │  └─ [3]: 1.0
      ├─ [:, 2]: SparseList (0.0) [1:3]
      ├─ [:, 3]: SparseList (0.0) [1:3]
      │  ├─ [1]: 2.0
      │  └─ [2]: 1.0
      └─ [:, 4]: SparseList (0.0) [1:3]
         └─ [1]: 1.0

julia> tensor_tree(permutedims(A, (2, 1)))
4×3-Tensor
└─ SparseDict (0.0) [:,1:3]
   ├─ [:, 1]: SparseDict (0.0) [1:4]
   │  ├─ [3]: 2.0
   │  └─ [4]: 1.0
   ├─ [:, 2]: SparseDict (0.0) [1:4]
   │  └─ [3]: 1.0
   └─ [:, 3]: SparseDict (0.0) [1:4]
      └─ [1]: 1.0

julia> dropfills!(swizzle(Tensor(CSCFormat()), 2, 1), A)
3×4 Finch.SwizzleArray{(2, 1), Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}}:
 0.0  0.0  2.0  1.0
 0.0  0.0  1.0  0.0
 1.0  0.0  0.0  0.0
```
