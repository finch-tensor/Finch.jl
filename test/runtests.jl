#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end

using Test
using ArgParse

s = ArgParseSettings("Run Finch.jl tests. All tests are run by default. Specific
test suites may be specified as positional arguments. Finch compares to
reference output which depends on the system word size (currently
$(Sys.WORD_SIZE)-bit). To overwrite $(Sys.WORD_SIZE==32 ? 64 : 32)-bit output,
run this with a $(Sys.WORD_SIZE==32 ? 64 : 32)-bit julia executable.")

@add_arg_table! s begin
    "--overwrite", "-w"
        action = :store_true
        help = "overwrite reference output for $(Sys.WORD_SIZE)-bit systems"
    "suites"
        nargs = '*'
        default = ["all"]
        help = "names of test suites to run"
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
        mkpath(ref_dir)
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
            @debug "disagreement with reference output" reference result
            return false
        end
    end
end

function should_run(name)
    global parsed_args
    return ("all" in parsed_args["suites"] || name in parsed_args["suites"])
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

include("utils.jl")

@testset "Finch.jl" begin
    if should_run("print") include("test_print.jl") end
    if should_run("representation") include("test_representation.jl") end
    if should_run("constructors") include("test_constructors.jl") end
    if should_run("conversions") include("test_conversions.jl") end
    if should_run("merges") include("test_merges.jl") end
    if should_run("index") include("test_index.jl") end
    if should_run("typical") include("test_typical.jl") end
    if should_run("kernels") include("test_kernels.jl") end
    if should_run("issues") include("test_issues.jl") end
    if should_run("base") include("test_base.jl") end
    if should_run("apps") include("test_apps.jl") end
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
    if should_run("embed") include("embed/test_embed.jl") end
    if should_run("debug") include("test_debug.jl") end
    if should_run("parallel") include("test_parallel.jl") end
    if should_run("continuous") include("test_continuous.jl") end
    #algebra goes at the end since it calls refresh()
    if should_run("algebra") include("test_algebra.jl") end
end
