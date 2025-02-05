# Iteration Protocols

Finch is a flexible tensor compiler with many ways to iterate over the same
data. For example, consider the case where we are intersecting two sparse
vectors `x[i]` and `y[i]`. By default, we would iterate over all of the nonzeros
of each vector. However, if we want to skip over the nonzeros in `y` based on
the nonzeros in `x`, we could declare the tensor `x` as the leader tensor with an
`x[gallop(i)]` protocol. When `x` leads the iteration, the generated code uses
the nonzeros of `x` as an outer loop and the nonzeros of `y` as an inner loop.
If we know that the nonzero datastructure of `y` supports efficient random access,
we might ask to iterate over `y` with a `y[follow(i)]` protocol, where we look up
each value of `y[i]` only when `x[i]` is nonzero.

Finch supports several iteration protocols, documented below. Note that not all
formats support all protocols, consult the documentation for each format to figure out
which protocols are supported.

```@docs
follow
walk
gallop
extrude
laminate
```
