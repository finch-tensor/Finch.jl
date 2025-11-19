#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(PackageSpec(; path=joinpath(@__DIR__, "..")))
    Pkg.instantiate()
end

using Test
using ArgParse

s = ArgParseSettings(
    "Run Finch.jl tests. By default, all tests are run unless --include or --exclude options are provided. 
Finch compares to reference output which depends on the system word size (currently $(Sys.WORD_SIZE)-bit).To overwrite $(Sys.WORD_SIZE==32 ? 64 : 32)-bit output, run this with a $(Sys.WORD_SIZE==32 ? 64 : 32)-bit julia executable.
If the environment variable FINCH_TEST_ARGS is set, it will override the given arguments.",
)

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
    "--nprocs", "-p"
    default = 0
    arg_type = Int
    help = "number of processors to use for parallelization (0 to disable)"
    "--nthreads", "-t"
    default = 2
    arg_type = Int
    help = "number of threads to use on each processor (if nprocs >= 1)"
end

if "FINCH_TEST_ARGS" in keys(ENV)
    ARGS = split(ENV["FINCH_TEST_ARGS"], " ")
end

parsed_args = parse_args(ARGS, s)

function test_filter(name)
    global parsed_args
    inc = parsed_args["include"]
    exc = parsed_args["exclude"]
    return (isempty(inc) || name in inc) && !(name in exc)
end

using Finch

if parsed_args["nprocs"] == 0
    using Test

    macro testitem(nm, exs...)
        default_imports = true
        retries = 0
        timeout = nothing
        tags = Symbol[]
        setup = Any[]
        skip = false
        failfast = nothing
        _id = nothing
        _run = true  # useful for testing `@testitem` itself
        _source = QuoteNode(__source__)
        if length(exs) > 1
            kw_seen = Set{Symbol}()
            for ex in exs[1:(end - 1)]
                ex.head == :(=) ||
                    error("`@testitem` options must be passed as keyword arguments")
                kw = ex.args[1]
                kw in kw_seen && error("`@testitem` has duplicate keyword `$kw`")
                push!(kw_seen, kw)
                if kw == :tags
                    tags = ex.args[2]
                    @assert(
                        tags isa Expr &&
                            all(t -> t isa QuoteNode && t.value isa Symbol, tags.args),
                        "`tags` keyword must be passed a collection of `Symbol`s"
                    )
                elseif kw == :default_imports
                    default_imports = ex.args[2]
                    @assert default_imports isa Bool "`default_imports` keyword must be passed a `Bool`"
                elseif kw == :setup
                    setup = ex.args[2]
                    @assert setup isa Expr "`setup` keyword must be passed a collection of `@testsetup` names"
                    setup = map(Symbol, setup.args)
                elseif kw == :retries
                    retries = ex.args[2]
                    @assert retries isa Integer "`retries` keyword must be passed an `Integer`"
                elseif kw == :timeout
                    t = ex.args[2]
                    @assert t isa Real "`timeout` keyword must be passed a `Real`"
                    @assert t > 0 "`timeout` keyword must be passed a positive number. Got `timeout=$t`"
                    timeout = ceil(Int, t)
                elseif kw == :skip
                    skip = ex.args[2]
                    # If the `Expr` doesn't evaluate to a Bool, throws at runtime.
                    @assert skip isa Union{Bool,Expr} "`skip` keyword must be passed a `Bool`"
                elseif kw == :failfast
                    failfast = ex.args[2]
                    @assert failfast isa Bool "`failfast` keyword must be passed a `Bool`. Got `failfast=$failfast`"
                elseif kw == :_id
                    _id = ex.args[2]
                    # This will always be written to the JUnit XML as a String, require the user
                    # gives us a String, so that we write exactly what the user expects.
                    # If given an `Expr` that doesn't evaluate to a String, throws at runtime.
                    @assert _id isa Union{AbstractString,Expr} "`id` keyword must be passed a string"
                elseif kw == :_run
                    _run = ex.args[2]
                    @assert _run isa Bool "`_run` keyword must be passed a `Bool`"
                elseif kw == :_source
                    _source = ex.args[2]
                    @assert isa(_source, Union{QuoteNode,Expr})
                else
                    error("unknown `@testitem` keyword arg `$(ex.args[1])`")
                end
            end
        end
        @assert !_run || nm isa String "`@testitem` expects a `String` literal name as the first argument"
        if isempty(exs) || !(exs[end] isa Expr && exs[end].head == :block)
            error("expected `@testitem` to have a body")
        end
        esc(
            quote
                if test_filter($nm)
                    if $skip
                        @testset $nm begin
                            @test true skip = true
                        end
                    else
                        @testset $nm failfast = $failfast begin
                            @info "Running test item: $($nm)"
                            $(exs[end])
                        end
                    end
                end
            end,
        )
    end

    macro testsetup(mod)
        (mod isa Expr && mod.head == :module) ||
            error("`@testsetup` expects a `module ... end` argument")
        _, name, code = mod.args
        name isa Symbol || error("`@testsetup module` expects a valid module name")
        esc(
            quote
                @eval $mod
                using .$(name)
            end,
        )
    end

    @testset "Finch" begin
        include("modules/checkoutput_testsetup.jl")
        include("suites/algebra_tests.jl")
        include("suites/constructors_tests.jl")
        include("suites/continuous_tests.jl")
        include("suites/continuousexamples_tests.jl")
        include("suites/docs_tests.jl")
        include("suites/examples_tests.jl")
        include("suites/fileio_tests.jl")
        include("suites/galley_tests.jl")
        include("suites/index_tests.jl")
        include("suites/interface_tests.jl")
        include("suites/issue_tests.jl")
        include("suites/kernel_tests.jl")
        include("suites/merge_tests.jl")
        include("suites/parallel_tests.jl")
        include("suites/print_tests.jl")
        include("suites/representation_tests.jl")
        include("suites/scheduler_tests.jl")
        include("suites/simple_tests.jl")
        include("suites/style_tests.jl")
        include("suites/typical_tests.jl")
    end

else
    using ReTestItems

    runtests(
        (ti) -> test_filter(ti.name),
        Finch;
        nworkers=parsed_args["nprocs"],
        nworker_threads=parsed_args["nthreads"],
        worker_init_expr=quote
            using Finch
            using SparseArrays
            parsed_args = $parsed_args
        end)
end
