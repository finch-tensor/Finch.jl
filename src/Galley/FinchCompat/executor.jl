mutable struct GalleyOptimizer
    estimator
    verbose
end

Base.:(==)(a::GalleyOptimizer, b::GalleyOptimizer) = a.verbose == b.verbose && a.estimator == b.estimator
Base.hash(a::GalleyOptimizer, h::UInt) = hash(GalleyOptimizer, hash(a.verbose, hash(a.estimator, h)))

GalleyOptimizer(; verbose = false, estimator=DCStats) = GalleyOptimizer(estimator, verbose)

function (ctx::GalleyOptimizer)(prgm)
    finch_mode = ctx.verbose ? :safe : :fast
    verbosity = ctx.verbose ? 3 : 0
    produce_node = prgm.bodies[end]
    output_vars = [Alias(a.name) for a in produce_node.args]
    galley_prgm = Plan(finch_hl_to_galley(normalize_hl(prgm))...)
    tns_inits, instance_prgm = galley(galley_prgm, ST=ctx.estimator, output_aliases=output_vars, verbose=verbosity, output_program_instance=true)
    timer_idx = 1
    julia_prgm = :()
    if operation(instance_prgm) == Finch.block
        for body in instance_prgm.bodies
            if ctx.verbose 
                timer_symbol = Symbol("t_$timer_idx")
                julia_prgm = :($julia_prgm;
                                $timer_symbol = time(); 
                                @finch mode=$(QuoteNode(finch_mode)) begin $(finch_unparse_program(nothing, body)) end;
                                println("Kernel ", $timer_idx, " Runtime: $(time() - $timer_symbol)"))
                timer_idx += 1
            else
                julia_prgm = :($julia_prgm; @finch mode=$(QuoteNode(finch_mode)) begin $(finch_unparse_program(nothing, body)) end)
            
            end
        end
    else
        julia_prgm = :(@finch mode=$(QuoteNode(finch_mode)) begin $(finch_unparse_program(nothing, instance_prgm)) end)
    end
    for init in tns_inits
        julia_prgm = :($init; $julia_prgm)
    end
    julia_prgm = :($julia_prgm; return Tuple([$([v.name for v in output_vars]...)]))
    julia_prgm
end

function Finch.set_options(ctx::GalleyOptimizer; estimator=DCStats)
    ctx.estimator=estimator
    return ctx
end

"""
    galley_scheduler(verbose = false, estimator=DCStats)

The galley scheduler uses the sparsity patterns of the inputs to optimize the computation.
The first set of inputs given to galley is used to optimize, and the `estimator` is used to
estimate the sparsity of intermediate computations during optimization.
"""
galley_scheduler(; verbose = false, estimator=DCStats) = Finch.LogicExecutor(GalleyOptimizer(verbose=verbose, estimator=estimator); verbose=verbose)