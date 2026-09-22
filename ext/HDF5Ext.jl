module HDF5Ext

using Finch
using Finch.JSON
using Finch.DataStructures

isdefined(Base, :get_extension) ? (using HDF5) : (using ..HDF5)

const HDF5Node = Union{HDF5.File,HDF5.Group}

function Finch.bspread_h5(fname::AbstractString)
    h5open(fname, "r") do io
        Finch.bspread(io)
    end
end

function Finch.bspwrite_h5(fname::AbstractString, arr, attrs=OrderedDict())
    h5open(fname, "w") do io
        Finch.bspwrite(io, arr, attrs)
    end
    fname
end

Finch.bspread_header(f::HDF5Node) = JSON.parse(read(attributes(f)["binsparse"]))
Finch.bspwrite_header(f::HDF5Node, str::String) = (attributes(f)["binsparse"] = str)
Finch.bspread_vector(g::HDF5Node, key) = read(g[key])
Finch.bspwrite_vector(g::HDF5Node, vec, key) = (g[key] = vec)

end
