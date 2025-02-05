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
group_target = PkgBenchmark.readresults(joinpath(@__DIR__, "result-target.json"))
group_baseline = PkgBenchmark.readresults(joinpath(@__DIR__, "result-baseline.json"))
judgement = judge(group_target, group_baseline)

md = sprint(export_markdown, judgement)
md = replace(md, ":x:" => "❌")
md = replace(md, ":white_check_mark:" => "✅")
println(md)
