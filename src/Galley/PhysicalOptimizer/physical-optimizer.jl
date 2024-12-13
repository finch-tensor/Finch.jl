function get_input_stats(expr::PlanNode; include_aliases=true)
    input_stats = Dict{Int, TensorStats}()
    for n in PostOrderDFS(expr)
        if n.kind == Input || (include_aliases && n.kind == Alias)
            input_stats[n.node_id] = n.stats
        end
    end
    return input_stats
end

function reorder_input(input, expr, loop_order::Vector{IndexExpr})
    input_order = get_index_order(input.stats)
    fixed_order = relative_sort(input_order, loop_order, rev=true)
    agg_expr = Aggregate(initwrite(get_default_value(input.stats)), get_default_value(input.stats), input)
    agg_expr.stats = input.stats
    formats = select_output_format(agg_expr.stats, reverse(input_order), fixed_order)
    reorder_query = Query(Alias(galley_gensym("A")), Materialize(formats..., fixed_order..., agg_expr), reverse(input_order)...)
    reorder_stats = copy_stats(input.stats)
    reorder_def = get_def(reorder_stats)
    reorder_def.index_order = fixed_order
    reorder_def.level_formats = formats
    reorder_query.expr.stats = reorder_stats
    final_alias_expr = Alias(reorder_query.name.name)
    final_alias_expr.stats = copy_stats(reorder_stats)
    expr = Rewrite(Postwalk(@rule ~n => final_alias_expr where n.node_id == input.node_id))(expr)
    return [reorder_query], expr
end

function split_plan_to_physical_plan(split_plan::PlanNode, ST, alias_to_loop_order, alias_stats::Dict{IndexExpr, TensorStats}; only_add_loop_order=true, transpose_aliases=false, verbose = 0)
    physical_queries = []
    for s_query in split_plan.queries
        p_queries = logical_query_to_physical_queries(s_query, ST, alias_stats; only_add_loop_order=false, transpose_aliases=false)
        for p_query in p_queries
            alias_stats[p_query.name.name] = p_query.expr.stats
            for n in PostOrderDFS(p_query.expr)
                if n.kind == Alias
                    alias_to_loop_order[n.name] = IndexExpr[idx.name for idx in p_query.loop_order]
                end
            end
        end
        append!(physical_queries, p_queries)
    end
    return Plan(physical_queries...)
end

# This function takes in a query and outputs one or more queries
# The input query should take the form:
#       Query(name, Materialize(formats..., index_order..., expr))
#       or
#       Query(name, Aggregate(op, init, idxs..., map_expr))
#       or
#       Query(name, map_expr)
# The output queries should be of the form:
#       Query(name, Materialize(formats..., output_order..., expr), loop_order...)
#       or
#       Query(name, Aggregate(op, init, idxs..., map_expr), loop_order...)
# The former are either transposition queries or outputs which have a user-specified format. 
# `alias_stats` is a dictionary which holds stats objects for the results of any previous
# queries. This is needed to get the stats for `Alias` inputs.
function logical_query_to_physical_queries(query::PlanNode, ST, alias_stats::Dict{IndexExpr, TensorStats}; only_add_loop_order=true, transpose_aliases=false, verbose = 0)
    if !(@capture query Query(~name, Materialize(~args...))) &&
            !(@capture query Query(~name, Aggregate(~args...))) &&
            !(@capture query Query(~name, MapJoin(~args...)))
        throw(ErrorException("Physical optimizer only takes in properly formatted single queries."))
    end
    insert_statistics!(ST, query, bindings=alias_stats)
    insert_node_ids!(query)
    id_to_node = Dict()
    for node in PreOrderDFS(query)
        id_to_node[node.node_id] = node
    end
    expr = query.expr
    output_formats = nothing
    output_order = nothing
    if expr.kind == Materialize
        if !any([f.val == t_undef for f in expr.formats])
            output_formats = [f.val for f in expr.formats]
        end
        output_order = IndexExpr[idx.name for idx in expr.idx_order]
        expr = expr.expr
    end

    agg_op = nothing
    agg_init = nothing
    reduce_idxs = Set{IndexExpr}()
    if expr.kind == Aggregate
        agg_op = expr.op
        agg_init = expr.init
        reduce_idxs = Set{IndexExpr}([i.name for i in expr.idxs])
        expr = expr.arg
    end

    # Determine the optimal loop order for the query
    transposable_stats = get_input_stats(expr; include_aliases=transpose_aliases)
    for (id, stats) in transposable_stats
        @assert !isnothing(get_index_order(stats)) "query: $query stats: $stats"
    end
    disjuncts_and_conjuncts = get_conjunctive_and_disjunctive_inputs(expr)
    disjunct_and_conjunct_stats = (conjuncts=[s.stats for s in disjuncts_and_conjuncts.conjuncts],
                                    disjuncts=[s.stats for s in disjuncts_and_conjuncts.disjuncts])
    agg_op = isnothing(agg_op) ? initwrite(get_default_value(expr.stats)) : agg_op

    output_stats = reduce_tensor_stats(agg_op, agg_init, reduce_idxs, expr.stats)
    if !isnothing(output_order)
        for idx in output_order
            if idx ∉ get_index_set(output_stats)
                dummy_stat = ST(get_default_value(output_stats))
                add_dummy_idx!(dummy_stat, idx, idx_pos=1)
                push!(disjunct_and_conjunct_stats.conjuncts, dummy_stat)
            end
        end
    end

    loop_order = get_join_loop_order(disjunct_and_conjunct_stats, collect(values(transposable_stats)), output_stats, output_order)
    queries = []
    for (id, stats) in transposable_stats
        if !only_add_loop_order && !is_sorted_wrt_index_order(get_index_order(stats), loop_order; loop_order=true)
            reorder_queries, expr = reorder_input(id_to_node[id], expr, loop_order)
            append!(queries, reorder_queries)
        end
    end

    expr = Aggregate(agg_op, agg_init, reduce_idxs..., expr)
    expr.stats = output_stats

    if !isnothing(output_order)
        for i in eachindex(output_order)
            if output_order[i] ∉ get_index_set(output_stats)
                add_dummy_idx!(output_stats, output_order[i])
            end
        end
        first_formats =  select_output_format(output_stats, loop_order, output_order)
        if !isnothing(output_formats) && first_formats != output_formats
            intermediate_query = Query(Alias(galley_gensym("A")), Materialize(first_formats..., output_order..., expr), loop_order...)
            reorder_stats = copy_stats(expr.stats)
            reorder_def = get_def(reorder_stats)
            reorder_def.index_order = output_order
            reorder_def.level_formats = first_formats
            alias_stats[intermediate_query.name.name] = reorder_stats
            intermediate_query.expr.stats = reorder_stats
            push!(queries, intermediate_query)
            best_formats = !isnothing(output_formats) ? output_formats : select_output_format(output_stats, reverse(output_order), output_order)
            alias_expr = Alias(intermediate_query.name.name)
            alias_expr.stats = reorder_stats
            result_expr = Aggregate(initwrite(get_default_value(alias_expr.stats)), get_default_value(alias_expr.stats), alias_expr)
            result_expr.stats = reorder_stats
            result_query = Query(query.name, Materialize(best_formats... , output_order..., result_expr), reverse(output_order)...)
            result_stats = copy_stats(reorder_stats)
            result_def = get_def(result_stats)
            result_def.index_order = output_order
            result_def.level_formats = best_formats
            result_query.expr.stats = result_stats
            push!(queries, result_query)
        else
            output_formats = isnothing(output_formats) ? select_output_format(output_stats, loop_order, output_order) : output_formats
            result_query = Query(query.name, Materialize(output_formats..., output_order..., expr), loop_order...)
            reorder_stats = copy_stats(expr.stats)
            reorder_def = get_def(reorder_stats)
            reorder_def.index_order = output_order
            reorder_def.level_formats = output_formats
            result_query.expr.stats = reorder_stats
            push!(queries, result_query)
        end
    else
        result_query = Query(query.name, expr, loop_order...)
        push!(queries, result_query)
    end

    return queries
end
