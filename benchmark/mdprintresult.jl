#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end

# This file was copied from Transducers.jl
# which is available under an MIT license (see LICENSE).
using PkgBenchmark
include("pprinthelper.jl")
result = PkgBenchmark.readresults(joinpath(@__DIR__, "result.json"))

md = sprint(export_markdown, result)
println(md)
