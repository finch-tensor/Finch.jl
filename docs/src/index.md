# Finch.jl

Finch is a Julia-to-Julia compiler for sparse or structured multidimensional arrays. Finch empowers users to write high-level array programs which are transformed behind-the-scenes into fast sparse code.

## Why Finch.jl?

Finch was built to make sparse and structured array programming easier and more efficient.  Finch.jl leverages compiler technology to automatically generate customized, fused sparse kernels for each specific
use case. This allows users to write readable, high-level sparse array programs without worrying about the performance of the generated code. Finch can automatically generate efficient implementations even for unique problems that lack existing library solutions.

# Installation

At the [Julia](https://julialang.org/downloads/) REPL, install the latest stable version by running:

```julia
julia> using Pkg; Pkg.add("Finch")
```

## Quickstart

```julia
julia> using Finch

# Create a sparse tensor
julia> A = Tensor(CSCFormat(), [1 0 0; 0 2 0; 0 0 3])
3×3 Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}:
 1.0  0.0  0.0
 0.0  2.0  0.0
 0.0  0.0  3.0

# Perform a simple operation
julia> B = A + A
3×3 Tensor{DenseLevel{Int64, SparseDictLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, Dict{Tuple{Int64, Int64}, Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}:
 2.0  0.0  0.0
 0.0  4.0  0.0
 0.0  0.0  6.0
```

### Sparse and Structured Tensors

Finch supports most major sparse formats (CSR, CSC, DCSR, DCSC, CSF, COO, Hash, Bytemap). Finch also allows users to define their own sparse formats with a parameterized format language.

```
CSC_matrix = Tensor(CSCFormat())
CSR_matrix = swizzle(Tensor(CSCFormat()), 2, 1)
CSF_tensor = Tensor(CSFFormat(3))
```

Finch also supports a wide variety of array structure beyond sparsity. Whether you're dealing with [custom background (zero) values](https://en.wikipedia.org/wiki/GraphBLAS), [run-length encoding](https://en.wikipedia.org/wiki/Run-length_encoding), or matrices with [special structures](https://en.wikipedia.org/wiki/Sparse_matrix#Special_structure) like banded or triangular matrices, Finch’s compiler can understand and optimize various data patterns and computational rules to adapt to the structure of data.

### Examples:

Finch supports many high-level array operations out of the box, such as `+`, `*`, `maximum`, `sum`, `map`, `broadcast`, and `reduce`.

```julia
julia> using Finch

# Define sparse tensor A
julia> A = Tensor(Dense(SparseList(Element(0.0))), [0 1.1 0; 2.2 0 3.3; 4.4 0 0; 0 0 5.5])

# Define sparse tensor B
julia> B = Tensor(Dense(SparseList(Element(0.0))), [0 1 1; 1 0 0; 0 0 1; 0 0 1])

# Element-wise multiplication
julia> C = A .* B

# Element-wise max
julia> C = max.(A, B)

# Sum over rows
julia> D = sum(C, dims=2)
```

For situations where more complex operations are needed, Finch supports an `@einsum` syntax on sparse and structured tensors.
```julia
julia> @einsum E[i] += A[i, j] * B[i, j]

julia> @einsum F[i, k] <<max>>= A[i, j] + B[j, k]

```

Finch even allows users to fuse multiple operations into a single kernel with `lazy` and `compute`.  The `lazy` function creates a lazy tensor, which is a symbolic representation of the computation. The `compute` function evaluates the computation.
Different optimizers can be used with `compute`, such as the state-of-the-art `Galley` optimizer, which can adapt to the
sparsity patterns of the inputs.

```julia
julia> using Finch, BenchmarkTools

julia> A = fsprand(1000, 1000, 0.1); B = fsprand(1000, 1000, 0.1); C = fsprand(1000, 1000, 0.0001);

julia> A = lazy(A); B = lazy(B); C = lazy(C);

julia> sum(A * B * C)

julia> @btime compute(sum(A * B * C));
  263.612 ms (1012 allocations: 185.08 MiB)

julia> @btime compute(sum(A * B * C), ctx=galley_scheduler());
  153.708 μs (667 allocations: 29.02 KiB)
```

### How it Works
Finch first translates high-level array code into **FinchLogic**, a custom intermediate representation that captures operator fusion and enables loop ordering optimizations. Using advanced schedulers, Finch optimizes FinchLogic and lowers it to **FinchNotation**, a more refined representation that precisely defines control flow. This optimized FinchNotation is then compiled into highly efficient, sparsity-aware code. Finch can specialize to each combination of sparse formats and algebraic properties, such as `x * 0 => 0`, eliminating unnecessary computations in sparse code automatically. 

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

[Galley: Modern Query Optimization for Sparse Tensor Programs](https://arxiv.org/abs/2408.14706).
Kyle Deeds, Willow Ahrens, Magda Balazinska, Dan Suciu.

## Contributing

Contributions are welcome! Please see our [contribution guidelines](CONTRIBUTING.md) for more information.
