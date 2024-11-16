# Finch.jl

[docs]:https://finch-tensor.github.io/Finch.jl/stable
[ddocs]:https://finch-tensor.github.io/Finch.jl/dev
[ci]:https://github.com/finch-tensor/Finch.jl/actions/workflows/CI.yml?query=branch%3Amain
[cov]:https://codecov.io/gh/finch-tensor/Finch.jl
[example]:https://github.com/finch-tensor/Finch.jl/tree/main/docs/examples

[docs_ico]:https://img.shields.io/badge/docs-stable-blue.svg
[ddocs_ico]:https://img.shields.io/badge/docs-dev-blue.svg
[ci_ico]:https://github.com/finch-tensor/Finch.jl/actions/workflows/CI.yml/badge.svg?branch=main
[cov_ico]:https://codecov.io/gh/finch-tensor/Finch.jl/branch/main/graph/badge.svg
[example_ico]:https://img.shields.io/badge/examples-docs%2Fexamples-blue.svg

| **Documentation**                             | **Build Status**                      | **Examples**    |
|:---------------------------------------------:|:-------------------------------------:|:---------------------:|
| [![][docs_ico]][docs] [![][ddocs_ico]][ddocs] | [![][ci_ico]][ci] [![][cov_ico]][cov] | [![][example_ico]][example] |

Finch is a Julia-to-Julia compiler for sparse or structured multidimensional arrays. Finch empowers users to write high-level array programs which are transformed behind-the-scenes into fast sparse code.

## Why Finch.jl?

Finch was built to make sparse and structured array programming easier and more efficient.  Finch.jl leverages compiler technology to automatically generate customized, fused sparse kernels for each specific
use case. This allows users to write readable, high-level sparse array programs without worrying about the performance of the generated code. Finch can automatically generate efficient implementations even for unique problems that lack existing library solutions.

### How it Works
Finch uses state-of-the-art schedulers to compile high-level programs such as matrix addition or multiplication into a custom intermediate representation (IR) called Finch IR. Finch IR is then lowered into efficient, sparse code. Finch can specialize each program to each combination of sparse formats and algebraic properties, such as `x * 0 => 0`, eliminating unnecessary computations in sparse code automatically. 

### Sparse and Structured Tensors

Finch supports most major sparse formats (CSR, CSC, DCSR, DCSC, CSF, COO, Hash, Bytemap). Finch also allows users to define their own sparse formats with a parameterized format language.

Finch also supports a wide variety of array structure beyond sparsity. Whether you're dealing with [custom background (zero) values](https://en.wikipedia.org/wiki/GraphBLAS), [run-length encoding](https://en.wikipedia.org/wiki/Run-length_encoding), or matrices with [special structures](https://en.wikipedia.org/wiki/Sparse_matrix#Special_structure) like banded or triangular matrices, Finch’s compiler can understand and optimize various data patterns and computational rules to adapt to the structure of data.

### Examples:

Finch supports many high-level array operations out of the box, such as `+`, `*`, `maximum`, `sum`, `map`, `broadcast`, and `reduce`.

```julia
julia> using Finch

julia> A = Tensor(Dense(SparseList(Element(0.0))), [0 1.1 0; 2.2 0 3.3; 4.4 0 0; 0 0 5.5])
4×3 Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}:
 0.0  1.1  0.0
 2.2  0.0  3.3
 4.4  0.0  0.0
 0.0  0.0  5.5

julia> B = Tensor(Dense(SparseList(Element(0.0))), [0 1 1; 1 0 0; 0 0 1; 0 0 1])
4×3 Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}:
 0.0  1.0  1.0
 1.0  0.0  0.0
 0.0  0.0  1.0
 0.0  0.0  1.0

julia> C = A .* B
4×3 Tensor{DenseLevel{Int64, SparseDictLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, Dict{Tuple{Int64, Int64}, Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}:
 0.0  1.1  0.0
 2.2  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  5.5

julia> D = sum(C, dims=2)
4 Tensor{SparseDictLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, Dict{Tuple{Int64, Int64}, Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}:
 1.1
 2.2
 0.0
 5.5
```

For situations where more complex operations are needed, Finch supports an `@einsum` syntax on sparse and structured tensors.
```julia
julia> @einsum E[i] += A[i, j] * B[i, j]
4 Tensor{SparseDictLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, Dict{Tuple{Int64, Int64}, Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}:
 1.1
 2.2
 0.0
 5.5

julia> @einsum F[i] <<max>>= A[i, j] + B[i, j]
4 Tensor{DenseLevel{Int64, ElementLevel{-Inf, Float64, Int64, Vector{Float64}}}}:
 2.1
 3.3
 4.4
 6.5
```

Finch even allows users to fuse multiple operations into a single kernel with `lazy` and `compute`.

```julia
julia> C = lazy(A) .+ lazy(B)
?×?-LazyTensor{Float64}

julia> D = sum(C, dims=2)
?-LazyTensor{Float64}

julia> compute(D)
4 Tensor{SparseDictLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, Dict{Tuple{Int64, Int64}, Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}:
 3.1
 6.5
 5.4
 6.5
```

# Installation

At the [Julia](https://julialang.org/downloads/) REPL, install the latest stable version by running:

````julia
julia> using Pkg; Pkg.add("Finch")
````

## Learn More

The following manuscripts provide a good description of the research behind Finch:

[Finch: Sparse and Structured Array Programming with Control Flow](https://arxiv.org/abs/2404.16730).
Willow Ahrens, Teodoro Fields Collin, Radha Patel, Kyle Deeds, Changwan Hong, Saman Amarasinghe.

[Looplets: A Language for Structured Coiteration](https://doi.org/10.1145/3579990.3580020). CGO 2023. 
Willow Ahrens, Daniel Donenfeld, Fredrik Kjolstad, Saman Amarasinghe.

## Beyond Finch

The following research efforts use Finch:

[SySTeC: A Symmetric Sparse Tensor Compiler](https://arxiv.org/abs/2406.09266).
Radha Patel, Willow Ahrens, Saman Amarasinghe.

[The Continuous Tensor Abstraction: Where Indices are Real](https://arxiv.org/abs/2407.01742).
Jaeyeon Won, Willow Ahrens, Joel S. Emer, Saman Amarasinghe.

[Galley: Modern Query Optimization for Sparse Tensor Programs](https://arxiv.org/abs/2408.14706). [Galley.jl](https://github.com/kylebd99/Galley.jl).
Kyle Deeds, Willow Ahrens, Magda Balazinska, Dan Suciu.
