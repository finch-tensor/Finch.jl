```@meta
CurrentModule = Galley
```

# Galley Tensor File Input/Output

All of the file formats supported by Galley are listed below. Each format has a
corresponding `read` and `write` function, and can be selected automatically
based on the file extension with the following functions:

```@docs
fread
fwrite
```

## Binsparse Format (`.bsp`)

Galley supports the most recent revision of the
[Binsparse](https://github.com/GraphBLAS/binsparse-specification) binary sparse
tensor format, including the v2.0 tensor extension. This is a good option for
those who want an efficient way to transfer sparse tensors between supporting
libraries and languages. The Binsparse format represents the tensor format as a
JSON string in the underlying data container, which can be either HDF5 or a
combination of NPY or JSON files.  Binsparse arrays are stored 0-indexed.

The repository also ships small CLI wrappers under `bin/` for interoperability
testing, including the canonical dense-HDF5 checker used by
`binsparse-tests`.

```@docs
bspwrite
bspread
```

## TensorMarket (`.mtx`, `.ttx`)

Galley supports the [MatrixMarket](https://math.nist.gov/MatrixMarket/formats.html#MMformat) and [TensorMarket](https://github.com/finch-tensor/TensorMarket.jl) formats, which prioritize readability and archiveability, storing matrices and tensors in plaintext.

```@docs
fttwrite
fttread
```

## FROSTT (`.tns`)

Galley supports the [FROSTT](http://frostt.io/tensors/) format for legacy codes that still use it.

```@docs
ftnswrite
ftnsread
```
