```@meta
CurrentModule = Finch
```

# High-Level Array API

Finch tensors also support many of the basic array operations one might expect,
including indexing, slicing, and elementwise maps, broadcast, and reduce.
For example:

```jldoctest example1; setup = :(using Finch)
julia> A = fsparse([1, 1, 2, 3], [2, 4, 5, 6], [1.0, 2.0, 3.0])
3×6 Tensor{SparseCOOLevel{2, Tuple{Int64, Int64}, Vector{Int64}, Tuple{Vector{Int64}, Vector{Int64}}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}:
 0.0  1.0  0.0  2.0  0.0  0.0
 0.0  0.0  0.0  0.0  3.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0

julia> A + 0
3×6 Tensor{DenseLevel{Int64, DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}:
 0.0  1.0  0.0  2.0  0.0  0.0
 0.0  0.0  0.0  0.0  3.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0

julia> A + 1
3×6 Tensor{DenseLevel{Int64, DenseLevel{Int64, ElementLevel{1.0, Float64, Int64, Vector{Float64}}}}}:
 1.0  2.0  1.0  3.0  1.0  1.0
 1.0  1.0  1.0  1.0  4.0  1.0
 1.0  1.0  1.0  1.0  1.0  1.0

julia> B = A .* 2
3×6 Tensor{SparseDictLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, Dict{Tuple{Int64, Int64}, Int64}, Vector{Int64}, SparseDictLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, Dict{Tuple{Int64, Int64}, Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}:
 0.0  2.0  0.0  4.0  0.0  0.0
 0.0  0.0  0.0  0.0  6.0  0.0
 0.0  0.0  0.0  0.0  0.0  0.0

julia> B[1:2, 1:2]
2×2 Tensor{SparseDictLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, Dict{Tuple{Int64, Int64}, Int64}, Vector{Int64}, SparseDictLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, Dict{Tuple{Int64, Int64}, Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}:
 0.0  2.0
 0.0  0.0

julia> map(x -> x^2, B)
3×6 Tensor{SparseDictLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, Dict{Tuple{Int64, Int64}, Int64}, Vector{Int64}, SparseDictLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, Dict{Tuple{Int64, Int64}, Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}:
 0.0  4.0  0.0  16.0   0.0  0.0
 0.0  0.0  0.0   0.0  36.0  0.0
 0.0  0.0  0.0   0.0   0.0  0.0
```

# Einsum

Finch also supports a highly general `@einsum` macro which supports any reduction over any simple pointwise array expression.

```@docs
@einsum
```

# Array Fusion

Finch supports array fusion, which allows you to compose multiple array operations
into a single kernel. This can be a significant performance optimization, as it
allows the compiler to optimize the entire operation at once. The two functions
the user needs to know about are `lazy` and `compute`. You can use `lazy` to
mark an array as an input to a fused operation, and call `compute` to execute
the entire operation at once. For example:

```jldoctest example1
julia> C = lazy(A);

julia> D = lazy(B);

julia> E = (C .+ D)/2;

julia> compute(E)
3×6 Tensor{SparseDictLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, Dict{Tuple{Int64, Int64}, Int64}, Vector{Int64}, SparseDictLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, Dict{Tuple{Int64, Int64}, Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}:
 0.0  1.5  0.0  3.0  0.0  0.0
 0.0  0.0  0.0  0.0  4.5  0.0
 0.0  0.0  0.0  0.0  0.0  0.0

```

In the above example, `E` is a fused operation that adds `C` and `D` together
and then divides the result by 2. The `compute` function examines the entire
operation and decides how to execute it in the most efficient way possible.
In this case, it would likely generate a single kernel that adds the elements of `A` and `B`
together and divides each result by 2, without materializing an intermediate.

```@docs
lazy
compute
fused
```

The `lazy` and `compute` functions allow the compiler to fuse operations together, resulting in asymptotically more efficient code.

```julia
julia> using BenchmarkTools

julia> A = fsprand(1000, 1000, 100); B = Tensor(rand(1000, 1000)); C = Tensor(rand(1000, 1000));

julia> @btime A .* (B * C);
  145.940 ms (859 allocations: 7.69 MiB)

julia> @btime compute(lazy(A) .* (lazy(B) * lazy(C)));
  694.666 μs (712 allocations: 60.86 KiB)

```

## Optimizers

Different optimizers can be used with `compute`, such as the state-of-the-art
Galley optimizer, which can adapt to the sparsity patterns of the inputs. The
optimizer can be set as an argument `ctx` to the `compute` function, or using
`set_scheduler` or `with_scheduler`.

```@docs
set_scheduler!
with_scheduler
default_scheduler
```

### The Galley Optimizer

Galley is a cost-based optimizer for Finch's lazy evaluation interface based on techniques from database 
query optimization. To use Galley, you just add the parameter `ctx=galley_optimizer()` to the `compute` 
function. While the default optimizer (`ctx=default_scheduler()`) makes decisions entirely based on
the types of the inputs, Galley gathers statistics on their sparsity to make cost-based based optimization
decisions.

```@docs
galley_scheduler
```

```julia
julia> A = fsprand(1000, 1000, 0.1); B = fsprand(1000, 1000, 0.1); C = fsprand(1000, 1000, 0.0001);

julia> A = lazy(A); B = lazy(B); C = lazy(C);

julia> @btime compute(sum(A * B * C));
  282.503 ms (1018 allocations: 184.43 MiB)

julia> @btime compute(sum(A * B * C), ctx=galley_scheduler());
  152.792 μs (672 allocations: 28.81 KiB)

```

By taking advantage of the fact that C is highly sparse, Galley can better structure the computation. In the matrix chain multiplication,
it always starts with the C,B matmul before multiplying with A. In the summation, it takes advantage of distributivity to pushing the reduction
down to the inputs. It first sums over A and C, then multiplies those vectors with B.

Because Galley adapts to the sparsity patterns of the first input tensor, it can
be useful to distinguish between different uses of the same function using the
`tag` keyword argument to `compute` or `fuse`.  For example, we may wish to
distinguish one spmv from another, as follows:

```jldoctest example2; setup=:(using Finch)
julia> A = rand(1000, 1000); B = rand(1000, 1000); C = fsprand(1000, 1000, 0.0001);

julia> fused((A, B, C) -> C .* (A * B), A, B, C, tag=:very_sparse_sddmm);

julia> C = fsprand(1000, 1000, 0.9);

julia> fused((A, B, C) -> C .* (A * B), A, B, C, tag=:very_dense_sddmm);

```

By distinguishing between the two uses of the same function, Galley can make
better decisions about how to optimize each computation separately.