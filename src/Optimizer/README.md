# Overview

Galley is a system for declarative sparse tensor algebra. It combines techniques from database query optimization with the flexible formats and execution of sparse tensor compilers, specifically the Finch compiler included in Galley.jl. Details about the theory and system design can be found in our paper [here](https://arxiv.org/abs/2408.14706). There are two main ways to interact with Galley:

  - Galley can be used as an optimizer for the array interface of Galley.jl.
  - Users can write programs directly in Galley's notation.

The optimizer lives in `Galley.Optimizer`. Import `Galley` for tensor storage and compilation, and import `Galley.Optimizer` for the direct query API.

## Galley: An Optimizer for Galley.jl

Using Galley with Galley's array API just requires setting the `ctx` parameter to `galley_scheduler()` in the `compute` function!

```
using Galley

A = lazy(Tensor(Dense(SparseList(Element(0.0))), fsprand(100, 100, .1)))
B = lazy(Tensor(Dense(SparseList(Element(0.0))), fsprand(100, 100, .1)))
C = compute(A * B, ctx = galley_scheduler())
```

Alternatively, you can specify that Galley should be used at the top of the file:

```
using Galley

Galley.set_scheduler!(galley_scheduler())
A = lazy(Tensor(Dense(SparseList(Element(0.0))), fsprand(100, 100, .1)))
B = lazy(Tensor(Dense(SparseList(Element(0.0))), fsprand(100, 100, .1)))
C = compute(A * B)
```

This is the preferred usage pattern for Galley.

## Galley: Direct Usage

Instead of going through the array API, you can specify your computation directly with Galley. This uses the grammar defined in `PlanAST/plan.jl`. For example, we can specify the same matrix multiplication as above using:

```
using Galley
using Galley.Optimizer

A = Tensor(Dense(SparseList(Element(0.0))), fsprand(100, 100, .1))
B = Tensor(Dense(SparseList(Element(0.0))), fsprand(100, 100, .1))
C = galley(Mat(:i, :k, Agg(+, :j, MapJoin(*, Input(A, :i, :j), Input(B, :j, :k))))).value[1]
C == compute(lazy(A) * B, ctx=galley_scheduler()) # True
```
