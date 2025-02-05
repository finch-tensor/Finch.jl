#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(PackageSpec(; path=joinpath(@__DIR__, "..")))
    Pkg.instantiate()
end

using Test
using Documenter
using Finch

root = joinpath(@__DIR__, "..")

DocMeta.setdocmeta!(
    Finch, :DocTestSetup, :(using Finch; using SparseArrays); recursive=true
)

doctest(Finch; fix=true)
