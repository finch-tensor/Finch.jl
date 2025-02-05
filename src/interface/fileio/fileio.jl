"""
    fwrite(filename::AbstractString, tns::Finch.Tensor)

Write the Finch tensor to a file using a file format determined by the file extension.
The following file extensions are supported:

- `.bsp.h5`: [Binsparse](https://github.com/GraphBLAS/binsparse-specification) HDF5 file format
- `.bspnpy`: [Binsparse](https://github.com/GraphBLAS/binsparse-specification) NumPy and JSON subdirectory format
- `.mtx`: [MatrixMarket](https://math.nist.gov/MatrixMarket/formats.html#MMformat) `.mtx` text file format
- `.ttx`: [TensorMarket](https://github.com/willow-ahrens/TensorMarket.jl) `.ttx` text file format
- `.tns`: [FROSTT](http://frostt.io/tensors/) `.tns` text file format
"""
function fwrite(filename::AbstractString, tns)
    if endswith(filename, ".tns")
        ftnswrite(filename, tns)
    elseif endswith(filename, ".ttx") || endswith(filename, ".mtx")
        fttwrite(filename, tns)
    elseif endswith(filename, ".bsp.h5") || endswith(filename, ".bsp.hdf5") ||
        endswith(filename, ".bspnpy")
        bspwrite(filename, tns)
    else
        error("Unknown file extension for file $filename")
    end
end

"""
    fread(filename::AbstractString)

Read the Finch tensor from a file using a file format determined by the file extension.
The following file extensions are supported:

- `.bsp.h5`: [Binsparse](https://github.com/GraphBLAS/binsparse-specification) HDF5 file format
- `.bspnpy`: [Binsparse](https://github.com/GraphBLAS/binsparse-specification) NumPy and JSON subdirectory format
- `.mtx`: [MatrixMarket](https://math.nist.gov/MatrixMarket/formats.html#MMformat) `.mtx` text file format
- `.ttx`: [TensorMarket](https://github.com/willow-ahrens/TensorMarket.jl) `.ttx` text file format
- `.tns`: [FROSTT](http://frostt.io/tensors/) `.tns` text file format
"""
function fread(filename::AbstractString)
    if endswith(filename, ".tns")
        ftnsread(filename)
    elseif endswith(filename, ".ttx") || endswith(filename, ".mtx")
        fttread(filename)
    elseif endswith(filename, ".bsp.h5") || endswith(filename, ".bsp.hdf5") ||
        endswith(filename, ".bspnpy")
        bspread(filename)
    else
        error("Unknown file extension for file $filename")
    end
end

include("binsparse.jl")
include("tensormarket.jl")
