```@meta
CurrentModule = Finch
```

# Benchmarking Tips

Julia code is [notoriously
fussy](https://github.com/JuliaCI/BenchmarkTools.jl#why-does-this-package-exist)
to benchmark.
We'll use [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl)
to automatically follow best practices for getting reliable julia benchmarks. We'll also
follow the [Julia Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/).

Finch is even trickier to benchmark, for a few reasons:
1. The first time an @finch function is called, it is compiled, which takes an
   extra long time. @finch can also incur dynamic dispatch costs if the array
   types are not [type
   stable](https://docs.julialang.org/en/v1/manual/faq/#man-type-stability). We
   can remedy this by using [`@finch_kernel`](@ref), which simplifies
   benchmarking by compiling the function ahead of time, so it behaves like a
   normal Julia function. If you must use `@finch`, try to ensure that the code
   is type-stable.
2. Finch tensors reuse memory from previous calls, so the first time a tensor is
   used in a Finch function, it will allocate memory, but subsequent times not so
   much. If we want to benchmark the memory allocation, we can reconstruct the
   tensor each time. Otherwise, we can let the repeated executions of the kernel
   measure the non-allocating runtime.
3. Runtime for sparse code often depends on the sparsity pattern, so it's
   important to benchmark with representative data. Using standard matrices or tensors from
   [MatrixDepot.jl](https://github.com/JuliaLinearAlgebra/MatrixDepot.jl) or
   [TensorDepot.jl](https://github.com/finch-tensor/TensorDepot.jl) is a good
   way to do this.

````julia
using Finch
using BenchmarkTools
using SparseArrays
using MatrixDepot
````

Load a sparse matrix from MatrixDepot.jl and convert it to a Finch tensor

````julia
A = Tensor(Dense(SparseList(Element(0.0))), matrixdepot("HB/west0067"))
(m, n) = size(A)

x = Tensor(Dense(Element(0.0)), rand(n))
y = Tensor(Dense(Element(0.0)))
````

````
Dense [1:0]
````

Construct a Finch kernel for sparse matrix-vector multiply

````julia
eval(@finch_kernel function spmv(y, A, x)
    y .= 0
    for j = _, i = _
        y[i] += A[i, j] * x[j]
    end
end)
````

````
spmv (generic function with 1 method)
````

Benchmark the kernel, ignoring allocation costs for y

````julia
@benchmark spmv($y, $A, $x)
````

````
BenchmarkTools.Trial: 10000 samples with 211 evaluations.
 Range (min … max):  355.450 ns … 517.379 ns  ┊ GC (min … max): 0.00% … 0.00%
 Time  (median):     358.014 ns               ┊ GC (median):    0.00%
 Time  (mean ± σ):   359.839 ns ±   9.413 ns  ┊ GC (mean ± σ):  0.00% ± 0.00%

  ▇▃▇█▃               ▁▂▃                                       ▂
  █████▁▁▁▁▁▁▁▃▃▁▁▄▃▃████▄▆▄▅▆▄▅▅▄▅▆▅▆▆▇▇▇▆▆▅▆▆▄▆▅▆▅▅▄▆▅▅▆▃▆▄▃▆ █
  355 ns        Histogram: log(frequency) by time        407 ns <

 Memory estimate: 0 bytes, allocs estimate: 0.
````

The `@benchmark` macro will benchmark a function in local scope, and it will run
the function a few times to estimate the runtime. It will also try to avoid
first-time compilation costs by running the function once before benchmarking
it. Note the use of `$` to interpolate the arrays into the benchmark, bringing
them into the local scope.

We can also benchmark the memory allocation of the kernel by constructing `y` in the
benchmark kernel

````julia
@benchmark begin
    y = Tensor(Dense(Element(0.0)))
    y = spmv(y, $A, $x).y
end
````

````
BenchmarkTools.Trial: 10000 samples with 202 evaluations.
 Range (min … max):  387.168 ns …   3.504 μs  ┊ GC (min … max): 0.00% … 86.82%
 Time  (median):     391.297 ns               ┊ GC (median):    0.00%
 Time  (mean ± σ):   403.645 ns ± 136.241 ns  ┊ GC (mean ± σ):  1.68% ±  4.34%

   ▂▇█▇▅▂           ▂▃▂▂▂▂▁▁▁                                   ▂
  ▆███████▅▄▄▃▅▄▄▃▄▇████████████▇▆▆▆▇█▇███▇▇▇▆▇█▇█▇▇▆▅▅▄▄▆▅▅▅▄▅ █
  387 ns        Histogram: log(frequency) by time        452 ns <

 Memory estimate: 608 bytes, allocs estimate: 2.
````

