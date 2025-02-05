```@meta
CurrentModule = Finch
```

# Constructing Tensors

You can build a finch tensor with the `Tensor` constructor. In general, the
`Tensor` constructor mirrors Julia's [`Array`](https://docs.julialang.org/en/v1/base/arrays/#Core.Array) constructor, but with an additional
prefixed argument which specifies the formatted storage for the tensor.

```@docs
Tensor
Tensor(lvl::AbstractLevel)
Tensor(lvl::AbstractLevel, dims::Number...)
Tensor(lvl::AbstractLevel, init::UndefInitializer)
Tensor(lvl::AbstractLevel, arr)
Tensor(arr)
```

A few predefined formats are available for use in the first argument to the `Tensor` constructor:

```@docs
DenseFormat
CSCFormat
CSFFormat
DCSCFormat
DCSFFormat
COOFormat
HashFormat
ByteMapFormat
```

For example, to construct an empty sparse matrix:

```jldoctest example1; setup=:(using Finch)
julia> A_fbr = Tensor(Dense(SparseList(Element(0.0))), 4, 3)
4×3 Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}:
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

julia> tensor_tree(A_fbr)
4×3-Tensor
└─ Dense [:,1:3]
   ├─ [:, 1]: SparseList (0.0) [1:4]
   ├─ [:, 2]: SparseList (0.0) [1:4]
   └─ [:, 3]: SparseList (0.0) [1:4]
```

To initialize a sparse matrix with some values:

```jldoctest example1
julia> A = [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0]
4×3 Matrix{Float64}:
 0.0  0.0  4.4
 1.1  0.0  0.0
 2.2  0.0  5.5
 3.3  0.0  0.0

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

# Custom Storage Tree Level Formats

This section describes the formatted storage for Finch tensors, the first
argument to the [`Tensor`](@ref) constructor. Level storage types holds all of
the tensor data, and can be nested hierarchichally.

Finch represents tensors hierarchically in a tree, where each node in the tree
is a vector of subtensors and the leaves are the elements.  Thus, a matrix is
analogous to a vector of vectors, and a 3-tensor is analogous to a vector of
vectors of vectors.  The vectors at each level of the tensor all have the same
structure, which can be selected by the user. You can visualize the tree using the [`tensor_tree`](@ref) function.

```@docs
tensor_tree(::AbstractTensor)
tensor_tree(::IO, ::AbstractTensor)
```

In a Finch tensor tree, the child of each node is selected by an array index.
All of the children at the same level will use the same format and share the
same storage. Finch is column major, so in an expression `A[i_1, ..., i_N]`, the
rightmost dimension `i_N` corresponds to the root level of the tree, and the
leftmost dimension `i_1` corresponds to the leaf level.

Our example could be visualized as follows:

![CSC Format Index Tree](../assets/LevelsVsFibers-matrix.png)

# Types of Level Storage

Finch supports a variety of storage formats for each level of the tensor tree,
each with advantages and disadvantages. Some storage formats support in-order
access, while others support random access. Some storage formats must be written
to in column-major order, while others support out-of-order writes. The
capabilities of each level are summarized in the following tables along with
some general descriptions.

| Level Format Name | Group    | Data Characteristic | Column-Major Reads | Random Reads | Column-Major Bulk Update | Random Bulk Update | Random Updates | Status |
|:----------------- |:-------- |:------------------- |:------------------:|:------------:|:------------------------:|:------------------:|:--------------:|:------:|
| Dense             | Core     | Dense               | ✅                  | ✅            | ✅                        | ✅                  | ✅              | ✅      |
| SparseTree        | Core     | Sparse              | ✅                  | ✅            | ✅                        | ✅                  | ✅              | ⚙️     |
| SparseRunListTree | Core     | Sparse Runs         | ✅                  | ✅            | ✅                        | ✅                  | ✅              | ⚙️     |
| Element           | Core     | Leaf                | ✅                  | ✅            | ✅                        | ✅                  | ✅              | ✅      |
| Pattern           | Core     | Leaf                | ✅                  | ✅            | ✅                        | ✅                  | ✅              | ✅      |
| SparseList        | Advanced | Sparse              | ✅                  | ❌            | ✅                        | ❌                  | ❌              | ✅      |
| SparseRunList     | Advanced | Sparse Runs         | ✅                  | ❌            | ✅                        | ❌                  | ❌              | ✅      |
| SparseBlockList   | Advanced | Sparse Blocks       | ✅                  | ❌            | ✅                        | ❌                  | ❌              | ✅      |
| SparsePoint       | Advanced | Single Sparse       | ✅                  | ✅            | ✅                        | ❌                  | ❌              | ✅      |
| SparseInterval    | Advanced | Single Sparse Run   | ✅                  | ✅            | ✅                        | ❌                  | ❌              | ✅      |
| SparseBand        | Advanced | Single Sparse Block | ✅                  | ✅            | ✅                        | ❌                  | ❌              | ⚙️     |
| RunList           | Advanced | Dense Runs          | ✅                  | ❌            | ✅                        | ❌                  | ❌              | ⚙️     |
| SparseBytemap     | Advanced | Sparse              | ✅                  | ✅            | ✅                        | ✅                  | ❌              | ✅      |
| SparseDict        | Advanced | Sparse              | ✅                  | ✅            | ✅                        | ✅                  | ❌              | ✅️     |
| MutexLevel        | Modifier | No Data             | ✅                  | ✅            | ✅                        | ✅                  | ✅              | ⚙️     |
| SeperationLevel   | Modifier | No Data             | ✅                  | ✅            | ✅                        | ✅                  | ✅              | ⚙️     |
| SparseCOO         | Legacy   | Sparse              | ✅                  | ✅            | ✅                        | ❌                  | ✅              | ✅️     |

The "Level Format Name" is the name of the level datatype. Other columns have descriptions below.

### Status

| Symbol | Meaning                                                                    |
|:------ |:-------------------------------------------------------------------------- |
| ✅      | Indicates the level is ready for serious use.                              |
| ⚙️     | Indicates the level is experimental and under development.                 |
| 🕸️     | Indicates the level is deprecated, and may be removed in a future release. |

### Groups

#### Core Group

Contains the basic, minimal set of levels one should use to build and
manipulate tensors.  These levels can be efficiently read and written to in any
order.

#### Advanced Group

Contains levels which are more specialized, and geared
towards bulk updates. These levels may be more efficient in certain cases, but are
also more restrictive about access orders and intended for more advanced usage.

#### Modifier Group

Contains levels which are also more specialized, but not towards a sparsity pattern.
These levels modify other levels in a variety of ways, but don't store novel sparsity patterns.
Typically, they modify how levels are stored or attach data to levels to support the utilization
of various hardware features.

#### Legacy Group

Contains levels which are not recommended for new code, but
are included for compatibility with older code.

### Data Characteristics

| Level Type        | Description                                                                                       |
|:----------------- |:------------------------------------------------------------------------------------------------- |
| **Dense**         | Levels which store every subtensor.                                                               |
| **Leaf**          | Levels which store only scalars, used for the leaf level of the tree.                             |
| **Sparse**        | Levels which store only non-fill values, used for levels with few nonzeros.                       |
| **Sparse Runs**   | Levels which store runs of repeated non-fill values.                                              |
| **Sparse Blocks** | Levels which store Blocks of repeated non-fill values.                                            |
| **Dense Runs**    | Levels which store runs of repeated values, and no compile-time zero annihilation.                |
| **No Data**       | Levels which don't store data but which alter the storage pattern or attach additional meta-data. |

Note that the `Single` sparse levels store a single instance of each nonzero, run, or block. These are useful with a parent level to represent IDs.

### Access Characteristics

| Operation Type                 | Description                                                                                                          |
|:------------------------------ |:-------------------------------------------------------------------------------------------------------------------- |
| **Column-Major Reads**         | Indicates efficient reading of data in column-major order.                                                           |
| **Random Reads**               | Indicates efficient reading of data in random-access order.                                                          |
| **Column-Major Bulk Update**   | Indicates efficient writing of data in column-major order, the total time roughly linear to the size of the tensor.  |
| **Column-Major Random Update** | Indicates efficient writing of data in random-access order, the total time roughly linear to the size of the tensor. |
| **Random Update**              | Indicates efficient writing of data in random-access order, the total time roughly linear to the number of updates.  |

### Diagrams

The following diagrams illustrate the structure of the levels individually.

![Diagram of Core Level Structures](../assets/Structures-levels.png)

The following diagrams illustrate the way that levels can be combined to form a tensor tree.

![Diagram of Core Level Structures](../assets/Structures-structures.png)

# Examples of Popular Formats in Finch

Finch levels can be used to construct a variety of popular sparse formats. A few examples follow:

| Format Type               | Syntax                                                         |
|:------------------------- |:-------------------------------------------------------------- |
| Sparse Vector             | `Tensor(SparseList(Element(0.0)), args...)`                    |
| CSC Matrix                | `Tensor(Dense(SparseList(Element(0.0))), args...)`             |
| CSF 3-Tensor              | `Tensor(Dense(SparseList(SparseList(Element(0.0)))), args...)` |
| DCSC (Hypersparse) Matrix | `Tensor(SparseList(SparseList(Element(0.0))), args...)`        |
| COO Matrix                | `Tensor(SparseCOO{2}(Element(0.0)), args...)`                  |
| COO 3-Tensor              | `Tensor(SparseCOO{3}(Element(0.0)), args...)`                  |
| Run-Length-Encoded Image  | `Tensor(Dense(RunList(Element(0.0))), args...)`                |

# Level Constructors

## Core Levels

```@docs
DenseLevel
ElementLevel
PatternLevel
```

## Advanced Levels

```@docs
SparseListLevel
RunListLevel
SparseRunListLevel
SparseBlockListLevel
SparseBandLevel
SparsePointLevel
SparseIntervalLevel
SparseByteMapLevel
SparseDictLevel
```

## Legacy Levels

```@docs
SparseCOOLevel
```
