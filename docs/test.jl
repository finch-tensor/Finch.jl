#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(PackageSpec(; path=joinpath(@__DIR__, "..")))
    Pkg.instantiate()
end

using Test
using Documenter
using Galley

DocMeta.setdocmeta!(
    Galley, :DocTestSetup, :(using Galley; using SparseArrays); recursive=true
)

doctest(Galley)
