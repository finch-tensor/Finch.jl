#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..")))
    Pkg.instantiate()
end

using Test
using ArgParse

s = ArgParseSettings("Run Finch.jl tests. By default, all tests are run unless --include or --exclude options are provided. 
Finch compares to reference output which depends on the system word size (currently $(Sys.WORD_SIZE)-bit).To overwrite $(Sys.WORD_SIZE==32 ? 64 : 32)-bit output, run this with a $(Sys.WORD_SIZE==32 ? 64 : 32)-bit julia executable.
If the environment variable FINCH_TEST_ARGS is set, it will override the given arguments.")

@add_arg_table! s begin
    "--overwrite", "-w"
    action = :store_true
    help = "overwrite reference output for $(Sys.WORD_SIZE)-bit systems"
    "--include", "-i"
    nargs = '*'
    default = []
    help = "list of test suites to include, e.g., --include constructors merges"

    "--exclude", "-e"
    nargs = '*'
    default = []
    help = "list of test suites to exclude, e.g., --exclude parallel algebra"
end

if "FINCH_TEST_ARGS" in keys(ENV)
    ARGS = split(ENV["FINCH_TEST_ARGS"], " ")
end

parsed_args = parse_args(ARGS, s)

"""
    check_output(fname, arg)

Compare the output of `println(arg)` with standard reference output, stored
in a file named `fname`. Call `julia runtests.jl --help` for more information on
how to overwrite the reference output.
"""
function check_output(fname, arg)
    global parsed_args
    ref_dir = joinpath(@__DIR__, "reference$(Sys.WORD_SIZE)")
    ref_file = joinpath(ref_dir, fname)
    if parsed_args["overwrite"]
        mkpath(dirname(ref_file))
        open(ref_file, "w") do f
            println(f, arg)
        end
        true
    else
        reference = replace(read(ref_file, String), "\r"=>"")
        result = replace(sprint(println, arg), "\r"=>"")
        if reference == result
            return true
        else
            println("disagreement with reference output")
            println("reference")
            println(reference)
            println("result")
            println(result)
            return false
        end
    end
end

function should_run(name)
    global parsed_args
    inc = parsed_args["include"]
    exc = parsed_args["exclude"]
    return (isempty(inc) || name in inc) && !(name in exc)
end

macro repl(io, ex, quiet = false)
    quote
        println($(esc(io)), "julia> ", Finch.striplines($(QuoteNode(ex))))
        if $(esc(quiet))
            $(esc(ex))
        else
            show($(esc(io)), MIME("text/plain"), $(esc(ex)))
        end
        println($(esc(io)))
    end
end

using Finch

include("data_matrices.jl")
include("continuous_data.jl")

include("utils.jl")

@testset "Finch.jl" begin
    if should_run("print") include("test_print.jl") end
    if should_run("constructors") include("test_constructors.jl") end
    if should_run("representation") include("test_representation.jl") end
    if should_run("merges") include("test_merges.jl") end
    if should_run("index") include("test_index.jl") end
    if should_run("typical") include("test_typical.jl") end
    if should_run("kernels") include("test_kernels.jl") end
    if should_run("issues") include("test_issues.jl") end
    if should_run("interface") include("test_interface.jl") end
    if should_run("galley") include("test_galley.jl") end
    #if should_run("continuous") include("test_continuous.jl") end # TODO fails often because of https://github.com/willow-ahrens/Finch.jl/issues/378
    if should_run("continuousexamples") include("test_continuousexamples.jl") end
    if should_run("simple") include("test_simple.jl") end
    if should_run("examples") include("test_examples.jl") end
    if should_run("fileio") include("test_fileio.jl") end
    if should_run("docs") && Sys.WORD_SIZE == 64
        @testset "Documentation" begin
            if parsed_args["overwrite"]
                include("../docs/fix.jl")
            else
                include("../docs/test.jl")
            end
        end
    end
    if should_run("parallel") include("test_parallel.jl") end
    if should_run("scheduler") include("test_scheduler.jl") end
    #algebra goes at the end since it calls refresh()
    if should_run("algebra") include("test_algebra.jl") end
end
