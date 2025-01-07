#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..")))
    Pkg.instantiate()
end

# This file was copied from Transducers.jl
# which is available under an MIT license (see LICENSE).
using PkgBenchmark

function mkconfig(; kwargs...)
    return BenchmarkConfig(env = Dict("JULIA_NUM_THREADS" => "1", "FINCH_BENCHMARK_ARGS" => get(ENV, "FINCH_BENCHMARK_ARGS", join(ARGS, " "))); kwargs...)
end

script = tempname(joinpath(@__DIR__))

cp(joinpath(@__DIR__, "benchmarks.jl"), script)

group_target = benchmarkpkg(
    dirname(@__DIR__),
    mkconfig(),
    resultfile = joinpath(@__DIR__, "result-target.json"),
    script = script,
)

group_baseline = benchmarkpkg(
    dirname(@__DIR__),
    mkconfig(id = "main"),
    resultfile = joinpath(@__DIR__, "result-baseline.json"),
    script = script,
)

judgement = judge(group_target, group_baseline)

include("pprintjudge.jl")