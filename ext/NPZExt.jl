module NPZExt

using Finch
using Finch: NPYPath
using Finch.JSON
using Finch.DataStructures

isdefined(Base, :get_extension) ? (using NPZ) : (using ..NPZ)

function Base.getindex(g::NPYPath, key::AbstractString)
    path = joinpath(g.dirname, key)
    if isfile("$path.npy")
        npzread("$path.npy")
    else
        NPYPath(path)
    end
end
function Base.setindex!(g::NPYPath, val::AbstractArray, key::AbstractString)
    npzwrite(joinpath(mkpath(g.dirname), "$(key).npy"), val)
end
function Base.setindex!(g::NPYPath, val::AbstractDict, key::AbstractString)
    for (key_2, val) in val
        setindex!(g[key], val, key_2)
    end
end
function Base.haskey(g::NPYPath, key::AbstractString)
    isfile(joinpath(mkpath(g.dirname), "$(key).npy")) ||
        isdir(joinpath(mkpath(g.dirname), key))
end

Finch.bspread_header(g::NPYPath) = JSON.parsefile(joinpath(g.dirname, "binsparse.json"))
function Finch.bspwrite_header(g::NPYPath, str::String)
    write(joinpath(mkpath(g.dirname), "binsparse.json"), str)
end
Finch.bspread_vector(g::NPYPath, key) = g[key]
Finch.bspwrite_vector(g::NPYPath, vec, key) = (g[key] = vec)

function Finch.bspread_bspnpy(fname::AbstractString)
    bspread(NPYPath(fname))
end

function Finch.bspwrite_bspnpy(fname::AbstractString, arr, attrs=OrderedDict())
    bspwrite(NPYPath(fname), arr, attrs)
    fname
end

end
