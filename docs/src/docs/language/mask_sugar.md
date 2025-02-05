# Mask Sugar

In Finch, expressions like `i == j` are treated as a sugar for mask tensors, which can
be used to encode fancy iteration patterns. For example, the expression `i == j`
is converted to a diagonal boolean mask tensor `DiagMask()[i, j]`, which allows
an expression like

```julia
@finch begin
    for i in _, j in _
        if i == j
            s[] += A[i, j]
        end
    end
end
```

to compile to something like

```julia
for i in 1:n
    s[] += A[i, i]
end
```

There are several mask tensors and syntaxes available, summarized in the
following table where `i, j` are indices:

| Expression | Transformed Expression  |
|:---------- |:----------------------- |
| `i < j`    | `UpTriMask()[i, j - 1]` |
| `i <= j`   | `UpTriMask()[i, j]`     |
| `i > j`    | `LoTriMask()[i, j + 1]` |
| `i >= j`   | `LoTriMask()[i, j]`     |
| `i == j`   | `DiagMask()[i, j]`      |
| `i != j`   | `!(DiagMask()[i, j])`   |

Note that either `i` or `j` may be expressions, so long as the expression is
constant with respect to the loop over the index.

The mask tensors are described below:

```@docs
uptrimask
lotrimask
diagmask
bandmask
chunkmask
```
