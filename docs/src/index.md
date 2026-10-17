# Finch.jl

Finch is a Julia-to-Julia compiler for sparse or structured multidimensional arrays. Finch empowers users to write loop-based array programs which are transformed behind-the-scenes into fast sparse code.

## Why Finch.jl?

Finch was built to make sparse and structured array programming easier and more efficient.  Finch.jl leverages compiler technology to automatically generate customized, fused sparse kernels for each specific
use case. This allows users to write readable sparse array programs without worrying about the performance of the generated code. Finch can automatically generate efficient implementations even for unique problems that lack existing library solutions.

# Installation

At the [Julia](https://julialang.org/downloads/) REPL, install the latest stable version by running:

```julia
julia> using Pkg;
       Pkg.add("Finch");

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

# Create dense input and output vectors

julia> x = Tensor(Dense(Element(0.0)), [1.0, 2.0, 3.0]);

julia> y = Tensor(Dense(Element(0.0)), 3);

# Compute a sparse matrix-vector product

julia> @finch begin
           y .= 0
           for j = _, i = _
               y[i] += A[i, j] * x[j]
           end
       end;

julia> y
3 Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}:
 1.0
 4.0
 9.0
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

Finch programs are loop nests written with the `@finch` macro. Finch compiles each program for the formats of the tensors it touches, so a single kernel can fuse several operations and skip the zeros of every sparse input.

```julia
julia> using Finch

# Define sparse tensor A

julia> A = Tensor(Dense(SparseList(Element(0.0))), [0 1.1 0; 2.2 0 3.3; 4.4 0 0; 0 0 5.5]);

# Define sparse tensor B

julia> B = Tensor(Dense(SparseList(Element(0.0))), [0 1 1; 1 0 0; 0 0 1; 0 0 1]);

# Define outputs: a sparse matrix and a dense vector with a fill value of -Inf

julia> C = Tensor(Dense(SparseList(Element(0.0))));

julia> D = Tensor(Dense(Element(-Inf)));

# Element-wise multiplication and a max-plus row reduction, fused into one kernel

julia> @finch begin
           C .= 0
           D .= -Inf
           for j = _, i = _
               C[i, j] = A[i, j] * B[i, j]
               D[i] <<max>>= A[i, j] + B[i, j]
           end
       end;
```

To generate a reusable function ahead of time, use the `@finch_kernel` macro. See the
[documentation](https://finch-tensor.github.io/Finch.jl/stable/docs/language/calling_finch/)
for details.

### How it Works

Finch programs are expressed in **FinchNotation**, a loop-based representation that precisely defines control flow. Finch compiles FinchNotation into highly efficient, sparsity-aware code by specializing each loop to the structure of the tensors it iterates over. Finch can specialize to each combination of sparse formats and algebraic properties, such as `x * 0 => 0`, eliminating unnecessary computations in sparse code automatically.

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
