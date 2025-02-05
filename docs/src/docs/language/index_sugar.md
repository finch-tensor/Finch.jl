# Index Sugar and Tensor Modifiers

In Finch, expressions like `x[i + 1]` are compiled using tensor modifiers, like
`offset(x, 1)[i]`. The user can construct tensor modifiers directly, e.g.
`offset(x, 1)`, or implicitly using the syntax `x[i + 1]`. Recognizable index
expressions are converted to tensor modifiers before dimensionalization, so that
the modified tensor will participate in dimensionalization.

While tensor modifiers may change the behavior of a tensor, they reference their
parent tensor as the `root` tensor. Modified tensors are not understoond as
distinct from their roots. For example, all accesses to the root tensor must
obey lifecycle and dimensionalization rules. Additionally, root tensors which
are themselves modifiers are unwrapped at the beginning of the program, so that
modifiers are not obscured and the new root tensor is not a modifier.

The following table lists the recognized index expressions and their equivalent
tensor expressions, where `i` is an index, `a, b` are constants, `p` is an
iteration protocol, and `x` is an expression:

| Original Expression | Transformed Expression |
|:------------------- |:---------------------- |
| `A[i + a]`          | `offset(A, 1)[i]`      |
| `A[i + x]`          | `toeplitz(A, 1)[i, x]` |
| `A[(a:b)(i)]`       | `window(A, a:b)[i]`    |
| `A[a * i]`          | `scale(A, (3,))[i]`    |
| `A[i * x]`          | `products(A, 1)[i, j]` |
| `A[~i]`             | `permissive(A)[i]`     |
| `A[p(i)]`           | `protocolize(A, p)[i]` |

Each of these tensor modifiers is described below:

```@docs
offset
toeplitz
window
scale
products
permissive
protocolize
```
