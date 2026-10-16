module HDF5Ext

using Galley
using Galley.JSON
using Galley.DataStructures

isdefined(Base, :get_extension) ? (using HDF5) : (using ..HDF5)

const HDF5Node = Union{HDF5.File,HDF5.Group}

function Galley.bspread_h5(fname::AbstractString)
    h5open(fname, "r") do io
        Galley.bspread(io)
    end
end

function Galley.bspwrite_h5(fname::AbstractString, arr, attrs=OrderedDict(); kwargs...)
    h5open(fname, "w") do io
        Galley.bspwrite(io, arr, attrs; kwargs...)
    end
    fname
end

Galley.bspread_header(f::HDF5Node) = JSON.parse(read(attributes(f)["binsparse"]))
Galley.bspwrite_header(f::HDF5Node, str::String) = (attributes(f)["binsparse"] = str)
Galley.bspread_vector(g::HDF5Node, key) = read(g[key])
Galley.bspwrite_vector(g::HDF5Node, vec, key) = (g[key] = vec)

end
