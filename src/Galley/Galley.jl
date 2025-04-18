# This file defines a prototype front-end which allows users to define tensor expressions and get their results.
module Galley

using AutoHashEquals
using Base: hash, copy, convert, getproperty, getfield, setfield!, getindex, ==, show
using DataStructures
using IterTools: subsets
using RewriteTools
using RewriteTools.Rewriters
using SyntaxInterface
using AbstractTrees
using Statistics
using Finch
using Finch: Element, SparseListLevel, SparseDict, Dense, SparseCOO, fsparse_impl,
    compute_parse,
    isimmediate, set_options, flatten_plans, initmax, initmin
using Finch: StableSet
using Finch.FinchNotation: index_instance, variable_instance, tag_instance,
    literal_instance,
    access_instance, reader_instance, updater_instance, assign_instance,
    loop_instance, declare_instance, block_instance, define_instance,
    call_instance, freeze_instance, thaw_instance, finch_unparse_program,
    Auto
using Finch.FinchLogic

export galley
export PlanNode,
    Value, Index, Alias, Input, MapJoin, Aggregate, Materialize, Query, Plan, IndexExpr
export Scalar, Σ, Mat, Agg
export DCStats, NaiveStats, TensorDef, DC, insert_statistics
export naive, greedy, pruned, exact
export GalleyOptimizer, AdaptiveExecutor, AdaptiveExecutorCode, galley_scheduler

IndexExpr = Symbol
TensorId = Symbol
# This defines the list of access protocols allowed by the Finch API
@enum AccessProtocol t_walk = 1 t_lead = 2 t_follow = 3 t_gallop = 4 t_default = 5

# A subset of the allowed level formats provided by the Finch API
@enum LevelFormat t_sparse_list = 1 t_coo = 2 t_dense = 3 t_hash = 4 t_bytemap = 5 t_undef =
    6

# The set of optimizers implemented by Galley
@enum FAQ_OPTIMIZERS greedy naive pruned exact

name_counter::UInt = 0

function galley_gensym(s::String)
    global name_counter += 1
    return Symbol(s * "_$name_counter")
end
galley_gensym(s::Symbol) = galley_gensym(string(s))

include("utility-funcs.jl")
include("PlanAST/PlanAST.jl")
include("TensorStats/TensorStats.jl")
include("LogicalOptimizer/LogicalOptimizer.jl")
include("PhysicalOptimizer/PhysicalOptimizer.jl")
include("CSEElimination/CSEElimination.jl")
include("ExecutionEngine/ExecutionEngine.jl")
include("FinchCompat/FinchCompat.jl")

# Galley takes in a series of high level queries which define required outputs.
# Each query has the form:
#       Query(name, Materialize(formats..., indices..., expr))
# The inner expr can be any combination of MapJoin(op, args...) and Aggregate(op, init, idxs..., arg)
# with the leaves being Input(tns, idxs...), Alias(name, init, idxs...), or Value(v) where name refers
# to the results of a previous query.
function galley(input_plan::PlanNode;
    faq_optimizer::FAQ_OPTIMIZERS=greedy,
    ST=DCStats,
    update_cards=true,
    simple_cse=true,
    max_kernel_size=8,
    output_logical_plan=false,
    output_physical_plan=false,
    output_aliases=nothing,
    output_program_instance=false,
    verbose=0)
    if input_plan.kind == Materialize
        input_plan = Query(:out, input_plan)
    end
    if input_plan.kind == Query
        input_plan = Plan(input_plan)
    end
    counter_start = Galley.name_counter

    overall_start = time()
    # To avoid input corruption, we start by copying the input queries (except for the data)
    input_plan = plan_copy(input_plan)
    if verbose >= 1
        println("Input Queries : ")
        for input_query in input_plan.queries
            println(input_query)
        end
    end

    # First, we perform high level optimization where each query is translated to one or
    # more queries with a simpler structure: Query(name, Aggregate(op, init, idxs, point_expr))
    # where point_expr is made up of just MapJoin, Input, and Alias nodes.
    alias_stats, alias_hash = OrderedDict{IndexExpr,TensorStats}(),
    OrderedDict{IndexExpr,UInt}()
    output_aliases = if isnothing(output_aliases)
        [input_query.name for input_query in input_plan.queries]
    else
        output_aliases
    end
    output_orders = OrderedDict(
        input_query.name => input_query.expr.idx_order for input_query in input_plan.queries
    )
    opt_start = time()

    # Optimize Aggregation & Materialization
    faq_opt_start = time()
    logical_plan = high_level_optimize(
        faq_optimizer, input_plan, ST, alias_stats, alias_hash, verbose
    )
    faq_opt_time = time() - faq_opt_start

    if verbose >= 1
        println("FAQ Opt Time: $faq_opt_time")
        println("--------------- Logical Plan ---------------")
        for query in logical_plan.queries
            println(query)
        end
        println("--------------------------------------------")
    end

    # Split-Up Large Queries
    split_start = time()
    split_plan = split_plan_to_kernel_limit(
        logical_plan, ST, max_kernel_size, alias_stats, verbose
    )
    total_split_time = split_start

    # Loop Order Selection
    phys_opt_start = time()
    alias_to_loop_order = OrderedDict{IndexExpr,Vector{IndexExpr}}()
    physical_plan = split_plan_to_physical_plan(
        split_plan, ST, alias_to_loop_order, alias_stats
    )

    # Tensor Format Selection
    physical_plan = modify_plan_formats!(physical_plan, alias_to_loop_order, alias_stats)

    # Access Protocol Selection
    physical_plan = modify_plan_protocols!(physical_plan, ST, alias_stats)
    total_phys_opt_time = time() - phys_opt_start

    # Duplicate Query Elmination
    cse_plan = naive_cse!(physical_plan)
    total_opt_time = time() - opt_start

    if verbose >= 1
        println("Physical Opt Time: $faq_opt_time")
        println("--------------- Physical Plan ---------------")
        for query in cse_plan.queries
            println(query)
        end
        println("--------------------------------------------")
    end

    if output_program_instance
        return get_execute_code(cse_plan, verbose)
    end

    # Execute Queries
    exec_start = time()
    alias_result = execute_plan(cse_plan::PlanNode, verbose)
    total_exec_time = time() - exec_start

    total_overall_time = time() - overall_start
    verbose >= 2 && println("Time to FAQ Opt: ", faq_opt_time)
    verbose >= 2 && println("Time to Split Opt: ", total_split_time)
    verbose >= 2 && println("Time to Phys Opt: ", total_phys_opt_time)
    verbose >= 1 && println(
        "Time to Optimize: ", (faq_opt_time + total_split_time + total_phys_opt_time)
    )
    verbose >= 1 && println("Time to Execute: ", total_exec_time)
    verbose >= 1 && println("Time to count: ", total_count_time)
    verbose >= 1 && println("Overall Time: ", total_overall_time)
    global name_counter = counter_start
    return (value=[alias_result[alias.name] for alias in output_aliases],
        opt_time=total_opt_time,
        execute_time=total_exec_time,
        overall_time=total_overall_time)
end

end
