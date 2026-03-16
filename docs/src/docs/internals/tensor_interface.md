```@meta
CurrentModule = Finch
```

# Tensor Interface

The `AbstractTensor` interface (defined in `src/abstract_tensor.jl`) is the interface through which Finch understands tensors. It is a high-level interace which allows tensors to interact with the rest of the Finch system. The interface is designed to be extensible, allowing users to define their own tensor types and behaviors. For a minimal example, read the definitions in [`/ext/SparseArraysExt.jl`](https://github.com/finch-tensor/Finch.jl/blob/main/ext/SparseArraysExt.jl) and in [`/src/interface/abstractarray.jl`](https://github.com/finch-tensor/Finch.jl/blob/main/src/interface/abstractarray.jl). Once these methods are defined that tell Finch how to generate code for an array, the `AbstractTensor` interface will also use Finch to generate code for several Julia `AbstractArray` methods, such as `getindex`, `setindex!`, `map`, and `reduce`. An important note: `getindex` and `setindex!` are not a source of truth for Finch tensors. Search the codebase for `::AbstractTensor` for a full list of methods that are implemented for `AbstractTensor`. Note that most `AbstractTensor` implement `labelled_show` and `labelled_children` methods instead of `show(::IO, ::MIME"text/plain", t::AbstractTensor)` for pretty printed display.

## Tensor Methods

```@docs
declare!
freeze!
thaw!
unfurl
instantiate
virtual_eltype
virtual_fill_value
virtual_size
virtual_resize!
labelled_show
labelled_children
is_injective
is_atomic
is_concurrent
```

# Level Interface

```jldoctest example1; setup=:(using Finch)
julia> A = [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0]
4×3 Matrix{Float64}:
 0.0  0.0  4.4
 1.1  0.0  0.0
 2.2  0.0  5.5
 3.3  0.0  0.0

julia> A_fbr = Tensor(Dense(Dense(Element(0.0))), A)
4×3 Tensor{DenseLevel{Int64, DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}:
 0.0  0.0  4.4
 1.1  0.0  0.0
 2.2  0.0  5.5
 3.3  0.0  0.0

julia> tensor_tree(A_fbr)
4×3-Tensor
└─ Dense [:,1:3]
   ├─ [:, 1]: Dense [1:4]
   │  ├─ [1]: 0.0
   │  ├─ [2]: 1.1
   │  ├─ [3]: 2.2
   │  └─ [4]: 3.3
   ├─ [:, 2]: Dense [1:4]
   │  ├─ [1]: 0.0
   │  ├─ [2]: 0.0
   │  ├─ [3]: 0.0
   │  └─ [4]: 0.0
   └─ [:, 3]: Dense [1:4]
      ├─ [1]: 4.4
      ├─ [2]: 0.0
      ├─ [3]: 5.5
      └─ [4]: 0.0
```

We refer to a node in the tree as a subfiber. All of the nodes at the same level
are stored in the same datastructure, and disambiguated by an integer
`position`.  in the above example, there are three levels: the rootmost level
contains only one subfiber, the root. The middle level has 3 subfibers, one for
each column. The leafmost level has 12 subfibers, one for each element of the
array.  For example, the first level is `A_fbr.lvl`, and we can represent it's
third position as `SubFiber(A_fbr.lvl.lvl, 3)`. The second level is `A_fbr.lvl.lvl`,
and we can access it's 9th position as `SubFiber(A_fbr.lvl.lvl.lvl, 9)`. For
instructional purposes, you can use parentheses to call a subfiber on an index to
select among children of a subfiber.

```jldoctest example1
julia> tensor_tree(Finch.SubFiber(A_fbr.lvl.lvl, 3))
Dense [1:4]
├─ [1]: 4.4
├─ [2]: 0.0
├─ [3]: 5.5
└─ [4]: 0.0

julia> tensor_tree(A_fbr[:, 3])
4-Tensor
└─ Dense [1:4]
   ├─ [1]: 4.4
   ├─ [2]: 0.0
   ├─ [3]: 5.5
   └─ [4]: 0.0

julia> tensor_tree(A_fbr(3))
Dense [1:4]
├─ [1]: 4.4
├─ [2]: 0.0
├─ [3]: 5.5
└─ [4]: 0.0

julia> Finch.SubFiber(A_fbr.lvl.lvl.lvl, 9)
 Finch.SubFiber{ElementLevel{0.0, Float64, Int64, Vector{Float64}}, Int64}:
4.4

julia> A_fbr[1, 3]
4.4

julia> A_fbr(3)(1)
4.4
```

When we print the tree in text, positions are numbered from top to bottom.

Because our array is sparse, (mostly zero, or another fill value), it would be
more efficient to store only the nonzero values. In Finch, each level is
represented with a different format. A sparse level only stores non-fill values.
This time, we'll use a tensor constructor with `sl` (for "`SparseList` of
nonzeros") instead of `d` (for "`Dense`"):

```jldoctest example1
julia> A_fbr = Tensor(Dense(SparseList(Element(0.0))), A)
4×3 Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}:
 0.0  0.0  4.4
 1.1  0.0  0.0
 2.2  0.0  5.5
 3.3  0.0  0.0

julia> tensor_tree(A_fbr)
4×3-Tensor
└─ Dense [:,1:3]
   ├─ [:, 1]: SparseList (0.0) [1:4]
   │  ├─ [2]: 1.1
   │  ├─ [3]: 2.2
   │  └─ [4]: 3.3
   ├─ [:, 2]: SparseList (0.0) [1:4]
   └─ [:, 3]: SparseList (0.0) [1:4]
      ├─ [1]: 4.4
      └─ [3]: 5.5
```

Our `Dense(SparseList(Element(0.0)))` format is also known as
["CSC"](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_.28CSC_or_CCS.29)
and is equivalent to
[`SparseMatrixCSC`](https://sparsearrays.juliasparse.org/dev/#man-csc). The
[`Tensor`](@ref) function will perform a zero-cost copy between Finch fibers and
sparse matrices, when available.  CSC is an excellent general-purpose
representation when we expect most of the columns to have a few nonzeros.
However, when most of the columns are entirely fill (a situation known as
hypersparsity), it is better to compress the root level as well:

```jldoctest example1
julia> A_fbr = Tensor(SparseList(SparseList(Element(0.0))), A)
4×3 Tensor{SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}:
 0.0  0.0  4.4
 1.1  0.0  0.0
 2.2  0.0  5.5
 3.3  0.0  0.0

julia> tensor_tree(A_fbr)
4×3-Tensor
└─ SparseList (0.0) [:,1:3]
   ├─ [:, 1]: SparseList (0.0) [1:4]
   │  ├─ [2]: 1.1
   │  ├─ [3]: 2.2
   │  └─ [4]: 3.3
   └─ [:, 3]: SparseList (0.0) [1:4]
      ├─ [1]: 4.4
      └─ [3]: 5.5
```

Here we see that the entirely zero column has also been compressed. The
`SparseList(SparseList(Element(0.0)))` format is also known as
["DCSC"](https://ieeexplore.ieee.org/document/4536313).

The
["COO"](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html)
(or "Coordinate") format is often used in practice for ease of interchange
between libraries. In an `N`-dimensional array `A`, COO stores `N` lists of
indices `I_1, ..., I_N` where `A[I_1[p], ..., I_N[p]]` is the `p`^th stored
value in column-major numbering. In Finch, `COO` is represented as a multi-index
level, which can handle more than one index at once. We use curly brackets to
declare the number of indices handled by the level:

```jldoctest example1
julia> A_fbr = Tensor(SparseCOO{2}(Element(0.0)), A)
4×3 Tensor{SparseCOOLevel{2, Tuple{Int64, Int64}, Vector{Int64}, Tuple{Vector{Int64}, Vector{Int64}}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}:
 0.0  0.0  4.4
 1.1  0.0  0.0
 2.2  0.0  5.5
 3.3  0.0  0.0

julia> tensor_tree(A_fbr)
4×3-Tensor
└─ SparseCOO{2} (0.0) [:,1:3]
   ├─ [2, 1]: 1.1
   ├─ [3, 1]: 2.2
   ├─ ⋮
   ├─ [1, 3]: 4.4
   └─ [3, 3]: 5.5
```

The COO format is compact and straightforward, but doesn't support random
access. For random access, one should use the `SparseDict` or `SparseBytemap` format. A full listing
of supported formats is described after a rough description of shared common internals of level,
relating to types and storage.

## Types and Storage of Level

All levels have a `postype`, typically denoted as `Tp` in the constructors, used for internal pointer types but accessible by the
function:

```@docs
postype
```

Additionally, many levels have a `Vp` or `Vi` in their constructors; these stand for vector of element type `Tp` or `Ti`.
More generally, levels are paramterized by the types that they use for storage. By default, all levels use `Vector`, but a user
could could change any or all of the storage types of a tensor so that the tensor would be stored on a GPU or CPU or some combination thereof,
or even just via a vector with a different allocation mechanism.  The storage type should behave like `AbstractArray`
and needs to implement the usual abstract array functions and `Base.resize!`. See the tests for an example.

When levels are constructed in short form as in the examples above, the index, position, and storage types are inferred
from the level below. All the levels at the bottom of a Tensor (`Element, Pattern, Repeater`) specify an index type, position type,
and storage type even if they don't need them. These are used by levels that take these as parameters.

## Level Methods

Tensor levels are implemented using the following methods:

```@docs
declare_level!
assemble_level!
reassemble_level!
freeze_level!
level_ndims
level_size
level_axes
level_eltype
level_fill_value
```

# Combinator Interface

Tensor Combinators allow us to modify the behavior of tensors. The `AbstractCombinator` interface (defined in [`src/tensors/abstract_combinator.jl`](https://github.com/finch-tensor/Finch.jl/blob/main/src/tensors/abstract_combinator.jl)) is the interface through which Finch understands tensor combinators. The interface requires the combinator to overload all of the tensor methods, as well as the methods used by Looplets when lowering ranges, etc. For a minimal example, read the definitions in [`/src/tensors/combinators/offset.jl`](https://github.com/finch-tensor/Finch.jl/blob/main/src/tensors/combinators/offset.jl).
