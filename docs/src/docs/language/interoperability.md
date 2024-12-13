# Using Finch with Other Languages

## Python

Finch has a dedicated Python frontend which can be installed with `pip install finch-tensor`. The frontend is [Array-API](https://data-apis.org/array-api/latest/) compliant. The code for the python wrapper is available at the [finch-tensor-python](https://github.com/finch-tensor/finch-tensor-python) repo. Finch is also available as a backend to [pydata/sparse](https://sparse.pydata.org/en/stable/).

You can also use [juliacall](https://github.com/JuliaPy/PythonCall.jl) to access more advanced Finch features from Python.

## Other languages

You can use Finch in other languages using Julia interfaces such as
[julia.h](https://docs.julialang.org/en/v1/manual/embedding/), making
considerations for converting between 0-indexed and 1-indexed arrays.

## 0-Index Compatibility

Julia, Matlab, Fortran, etc. index arrays [starting at
1](https://docs.julialang.org/en/v1/devdocs/offset-arrays/). C, python, etc.
index starting at 0. In a dense array, we can simply subtract one from the
index, and in fact, this is what Julia will does under the hood when you pass a
vector [between C to
Julia](https://docs.julialang.org/en/v1/manual/embedding/#Working-with-Arrays).

However, for sparse array formats, it's not just a matter of subtracting one
from the index, as the internal lists of indices, positions, etc all start from
zero as well. To remedy the situation, Finch leverages `PlusOneVector` and `CIndex`.

### `PlusOneVector`

`PlusOneVector` is a view that adds `1` on access to an underlying 0-Index vector.
This allows to use Python/NumPy vector, without copying, as a mutable index array.

```jldoctest example2; setup = :(using Finch)
julia> v = Vector([1, 0, 2, 3])
4-element Vector{Int64}:
 1
 0
 2
 3

julia> obov = PlusOneVector(v)
4-element PlusOneVector{Int64, Vector{Int64}}:
 2
 1
 3
 4

julia> obov[1] += 8
10

julia> obov
4-element PlusOneVector{Int64, Vector{Int64}}:
 10
  1
  3
  4

julia> obov.data
4-element Vector{Int64}:
 9
 0
 2
 3

```

### `CIndex`

!!! warning
    `CIndex` is no longer recommended - use `PlusOneVector` instead.

Finch also interoperates with the [CIndices](https://github.com/JuliaSparse/CIndices.jl)
package, which exports a type called `CIndex`. The internal representation of `CIndex`
is one less than the value it represents, and we can use `CIndex` as the index or
position type of a Finch array to represent arrays in other languages.

For example, if `idx_c`, `ptr_c`, and `val_c` are the internal arrays of a CSC
matrix in a zero-indexed language, we can represent that matrix as a one-indexed
Finch array without copying by calling
```@meta
DocTestSetup = quote
    using Finch
    using CIndices
end
```
```jldoctest example2
julia> m = 4; n = 3; ptr_c = [0, 3, 3, 5]; idx_c = [1, 2, 3, 0, 2]; val_c = [1.1, 2.2, 3.3, 4.4, 5.5];

julia> ptr_jl = reinterpret(CIndex{Int}, ptr_c)
4-element reinterpret(CIndex{Int64}, ::Vector{Int64}):
 1
 4
 4
 6

julia> idx_jl = reinterpret(CIndex{Int}, idx_c)
5-element reinterpret(CIndex{Int64}, ::Vector{Int64}):
 2
 3
 4
 1
 3

julia> A = Tensor(Dense(SparseList{CIndex{Int}}(Element{0.0, Float64, CIndex{Int}}(val_c), m, ptr_jl, idx_jl), n))
CIndex{Int64}(4)×3 Tensor{DenseLevel{Int64, SparseListLevel{CIndex{Int64}, Base.ReinterpretArray{CIndex{Int64}, 1, Int64, Vector{Int64}, false}, Base.ReinterpretArray{CIndex{Int64}, 1, Int64, Vector{Int64}, false}, ElementLevel{0.0, Float64, CIndex{Int64}, Vector{Float64}}}}}:
 0.0  0.0  4.4
 1.1  0.0  0.0
 2.2  0.0  5.5
 3.3  0.0  0.0

julia> tensor_tree(A)
CIndex{Int64}(4)×3-Tensor
└─ Dense [:,1:3]
   ├─ [:, 1]: SparseList (0.0) [1:CIndex{Int64}(4)]
   │  ├─ [CIndex{Int64}(2)]: 1.1
   │  ├─ [CIndex{Int64}(3)]: 2.2
   │  └─ [CIndex{Int64}(4)]: 3.3
   ├─ [:, 2]: SparseList (0.0) [1:CIndex{Int64}(4)]
   └─ [:, 3]: SparseList (0.0) [1:CIndex{Int64}(4)]
      ├─ [CIndex{Int64}(1)]: 4.4
      └─ [CIndex{Int64}(3)]: 5.5
```

We can also convert between representations by copying to or from `CIndex` fibers.
