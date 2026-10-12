@testsetup module CheckOutput
export @repl, check_output

using ..Main: parsed_args

"""
striplines(ex)

Remove line numbers. `ex` is the target Julia expression
"""
function striplines(ex::Expr)
    if ex.head == :block
        Expr(:block, filter(x -> !(x isa LineNumberNode), map(striplines, ex.args))...)
    elseif ex.head == :macrocall
        Expr(:macrocall, ex.args[1], nothing, map(striplines, ex.args[3:end])...)
    else
        Expr(ex.head, map(striplines, ex.args)...)
    end
end
striplines(ex) = ex

macro repl(io, ex, quiet=false)
    quote
        println($(esc(io)), "julia> ", striplines($(QuoteNode(ex))))
        if $(esc(quiet))
            $(esc(ex))
        else
            show($(esc(io)), MIME("text/plain"), $(esc(ex)))
        end
        println($(esc(io)))
    end
end

function normalize_sparse_dicts(output)
    # Dict iteration order can change between Julia versions. Sort only the
    # integer coordinate-to-position entries printed by SparseDict levels.
    entry = raw"\(-?\d+, -?\d+\) => -?\d+"
    dict = Regex(
        raw"\bDict(?:\{Tuple\{Int\d+, Int\d+\}, Int\d+\})?\(" *
        entry * "(?:, " * entry * raw")*\)",
    )
    return replace(
        output, dict => function (value)
            prefix = value[1:findfirst('(', value)]
            entries = [m.match for m in eachmatch(Regex(entry), value)]
            return prefix * join(sort!(entries), ", ") * ")"
        end
    )
end

"""
check_output(fname, arg)

Compare the output of `println(arg)` with standard reference output, stored
in a file named `fname`. Call `julia runtests.jl --help` for more information on
how to overwrite the reference output.
"""
function check_output(fname, arg)
    global parsed_args
    ref_dir = joinpath(@__DIR__, "../reference$(Sys.WORD_SIZE)")
    ref_file = joinpath(ref_dir, fname)
    if parsed_args["overwrite"]
        mkpath(dirname(ref_file))
        open(ref_file, "w") do f
            println(f, arg)
        end
        true
    else
        reference = replace(read(ref_file, String), "\r" => "")
        result = replace(sprint(println, arg), "\r" => "")
        if normalize_sparse_dicts(reference) == normalize_sparse_dicts(result)
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
end
