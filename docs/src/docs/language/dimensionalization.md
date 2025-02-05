In Finch, all tensors accessed by a particular index must have the same dimension
along the corresponding mode. Finch determines the dimension of a loop index
`i` from all of the tensors using `i` in an access, as well as the bounds in the
loop itself.

For example, consider the following code

```julia
A = fsprand(3, 4, 0.5)
B = fsprand(4, 5, 0.5)
C = Tensor(Dense(SparseList(Element(0.0))))
@finch begin
    C .= 0
    for i in 1:3
        for j in _
            for k in _
                C[i, j] += A[i, k] * B[k, j]
            end
        end
    end
end
```

In the above code, the second dimension of `A` must match the first dimension of
`B`.  Also, the first dimension of `A` must match the `i` loop dimension, `1:3`.
Finch will also resize declared tensors to match indices used in writes, so `C`
is resized to `(1:3, 1:5)`. If no dimensions are specified elsewhere, then Finch
will use the dimension of the declared tensor.

Dimensionalization occurs after wrapper arrays are de-sugared. You can therefore
exempt a tensor from dimensionalization by wrapping the corresponding index in
`~`. For example,

```julia
@finch begin
    y .= 0
    for i in 1:3
        y[~i] += x[i]
    end
end
```

does not set the dimension of `y`, and `y` does not participate in
dimensionalization.

In summary, the rules of index dimensionalization are as follows:

  - Indices have dimensions
  - Using an index in an access “hints” that the index should have the corresponding dimension
  - Loop dimensions are equal to the “meet” of all hints in the loop body and the loop bounds
  - The meet usually asserts that dimensions match, but may also e.g. propagate info about parallelization

The rules of declaration dimensionalization are as follows:

  - Declarations have dimensions
  - Left hand side (updating) tensor access “hint” the size of that tensor
  - The dimensions of a declaration are the “meet” of all hints from the declaration to the first read
  - The new dimensions of the declared tensor are used when the tensor is on the right hand side (reading) access.

```@docs
Finch.FinchNotation.Auto
```
