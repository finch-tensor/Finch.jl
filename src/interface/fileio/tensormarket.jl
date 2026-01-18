"""
    fttwrite(filename, tns)

Write a sparse Finch tensor to a TensorMarket file.

[TensorMarket](https://github.com/willow-ahrens/TensorMarket.jl) must be loaded for this function to be available.

See also: [ttwrite](http://willowahrens.io/TensorMarket.jl/stable/#TensorMarket.ttwrite)
"""
function fttwrite(args...)
    throw(
        FinchExtensionError(
            "TensorMarket.jl must be loaded to use write .ttx files (hint: `using TensorMarket`)"
        ),
    )
end

"""
    fttread(filename, infoonly=false, retcoord=false)

Read the TensorMarket file into a Finch tensor. The tensor will be dense or
COO depending on the format of the file.

[TensorMarket](https://github.com/willow-ahrens/TensorMarket.jl) must be loaded for this function to be available.

See also: [ttread](http://willowahrens.io/TensorMarket.jl/stable/#TensorMarket.ttread)
"""
function fttread(args...)
    throw(
        FinchExtensionError(
            "TensorMarket.jl must be loaded to use read .ttx files (hint: `using TensorMarket`)"
        ),
    )
end

"""
    ftnswrite(filename, tns)

Write a sparse Finch tensor to a FROSTT `.tns` file.

[TensorMarket](https://github.com/willow-ahrens/TensorMarket.jl) must be loaded for this function to be available.

!!! danger
    This file format does not record the size or eltype of the tensor, and is provided for
    archival purposes only.

See also: [tnswrite](http://willowahrens.io/TensorMarket.jl/stable/#TensorMarket.tnswrite)
"""
function ftnswrite(args...)
    throw(
        FinchExtensionError(
            "TensorMarket.jl must be loaded to write .tns files (hint: `using TensorMarket`)"
        ),
    )
end

"""
    ftnsread(filename)

Read the contents of the FROSTT `.tns` file 'filename' into a Finch COO Tensor.

[TensorMarket](https://github.com/willow-ahrens/TensorMarket.jl) must be loaded for this function to be available.

!!! danger
    This file format does not record the size or eltype of the tensor, and is provided for
    archival purposes only.

See also: [tnsread](http://willowahrens.io/TensorMarket.jl/stable/#TensorMarket.tnsread)
"""
function ftnsread(args...)
    throw(
        FinchExtensionError(
            "TensorMarket.jl must be loaded to read .tns files (hint: `using TensorMarket`)"
        ),
    )
end
