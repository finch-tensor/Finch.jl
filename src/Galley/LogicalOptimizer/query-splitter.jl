function count_index_occurences(nodes)
    vars = Set()
    occurences = 0
    for n in nodes
        for c in PostOrderDFS(n)
            if c.kind == Input
                occurences += length(c.idxs)
                union!(vars, [idx.val for idx in c.idxs])
            elseif c.kind == Alias
                occurences += length(get_index_set(c.stats))
                union!(vars, get_index_set(c.stats))
            end
        end
    end
    return occurences - length(vars)
end

function get_connected_subsets(nodes, max_kernel_size)
    ss = []
    for k in 2:4
        for s in subsets(nodes, k)
            if count_index_occurences(s) > max_kernel_size
                continue
            end
            is_cross_prod = false
            for n in s
                if isempty(
                    ∩(
                        get_index_set(n.stats),
                        union(
                            [
                                get_index_set(n2.stats) for
                                n2 in s if n2.node_id != n.node_id
                            ]...,
                        ),
                    ),
                )
                    is_cross_prod = true
                    break
                end
            end
            if !is_cross_prod
                push!(ss, s)
            end
        end
    end
    return ss
end

# This function takes in queries of the form:
#   Query(name, Aggregate(agg_op, idxs..., map_expr))
#   Query(name, Materialize(formats.., idxs..., Aggregate(agg_op, idxs..., map_expr)))
# It outputs a set of queries where the final one binds `name` and each
# query has less than `MAX_INDEX_OCCURENCES` index occurences.
function split_plan_to_kernel_limit(
    logical_plan::PlanNode, ST, max_kernel_size, alias_stats, verbose
)
    split_queries = PlanNode[]
    for l_query in logical_plan.queries
        s_queries = split_query(l_query, ST, max_kernel_size, alias_stats, verbose)
        append!(split_queries, s_queries)
    end
    return Plan(split_queries...)
end

# This function takes in queries of the form:
#   Query(name, Aggregate(agg_op, idxs..., map_expr))
#   Query(name, Materialize(formats.., idxs..., Aggregate(agg_op, idxs..., map_expr)))
# It outputs a set of queries where the final one binds `name` and each
# query has less than `MAX_INDEX_OCCURENCES` index occurences.
function split_query(q::PlanNode, ST, max_kernel_size, alias_stats, verbose)
    insert_node_ids!(q)
    aq = AnnotatedQuery(q, ST)
    pe = aq.point_expr
    insert_statistics!(ST, pe; bindings=alias_stats)
    has_agg = length(aq.reduce_idxs) > 0
    agg_op = has_agg ? aq.idx_op[first(aq.reduce_idxs)] : nothing
    agg_init = has_agg ? aq.idx_init[first(aq.reduce_idxs)] : nothing
    node_id_counter = maximum([n.node_id for n in PostOrderDFS(pe)]) + 1
    queries = []
    cost_cache = Dict()
    cur_occurences = count_index_occurences([pe])
    if verbose > 2 && cur_occurences > max_kernel_size
        println(
            "Splitting the following query (cur_occurences = $cur_occurences) (max_kernel_size = $max_kernel_size): "
        )
        println(q)
    end
    while cur_occurences > max_kernel_size
        nodes_to_remove = nothing
        new_query = nothing
        new_agg_idxs = Set()
        min_cost = Inf
        for node in PostOrderDFS(pe)
            if node.kind in (Value, Input, Alias, Index) ||
                count_index_occurences([node]) == 0
                continue
            end
            cache_key = [node.node_id]
            if !haskey(cost_cache, cache_key)
                n_reduce_idxs = get_reducible_idxs(aq, node)
                should_reduce = has_agg && (length(n_reduce_idxs) > 0)
                n_mat_stats = if should_reduce
                    reduce_tensor_stats(agg_op, agg_init, n_reduce_idxs, node.stats)
                else
                    node.stats
                end
                cost_cache[cache_key] = (n_reduce_idxs,
                    n_mat_stats,
                    estimate_nnz(n_mat_stats))
            end
            n_reduce_idxs, n_mat_stats, n_cost = cost_cache[cache_key]
            if n_cost < min_cost && count_index_occurences([node]) < max_kernel_size
                nodes_to_remove = [node.node_id]
                new_expr = if (has_agg && !isempty(n_reduce_idxs))
                    Aggregate(agg_op, n_reduce_idxs..., node)
                else
                    node
                end
                new_expr.stats = n_mat_stats
                new_query = Query(Alias(galley_gensym("A")), new_expr)
                new_agg_idxs = n_reduce_idxs
                min_cost = n_cost + get_forced_transpose_cost(node)
            end
            if node.kind == MapJoin && isassociative(node.op.val)
                for s in get_connected_subsets(node.args, max_kernel_size)
                    cache_key = sort([n.node_id for n in s])
                    if !haskey(cost_cache, cache_key)
                        s_stat = merge_tensor_stats(node.op.val, [n.stats for n in s]...)
                        s_reduce_idxs = Set{IndexExpr}()
                        for idx in n_reduce_idxs
                            if !any([
                                idx ∈ get_index_set(n.stats) for n in setdiff(node.args, s)
                            ])
                                push!(s_reduce_idxs, idx)
                            end
                        end
                        should_reduce = has_agg && (length(s_reduce_idxs) > 0)
                        s_mat_stats = if should_reduce
                            reduce_tensor_stats(agg_op, agg_init, s_reduce_idxs, s_stat)
                        else
                            s_stat
                        end
                        s_cost =
                            estimate_nnz(s_mat_stats) * AllocateCost +
                            get_forced_transpose_cost(MapJoin(node.op.val, s...))
                        cost_cache[cache_key] = (s_reduce_idxs, s_mat_stats, s_cost)
                    end
                    s_reduce_idxs, s_mat_stats, s_cost = cost_cache[cache_key]
                    if s_cost < min_cost
                        nodes_to_remove = [n.node_id for n in s]
                        new_expr = if (has_agg && !isempty(s_reduce_idxs))
                            Aggregate(agg_op, s_reduce_idxs..., MapJoin(node.op, s...))
                        else
                            MapJoin(node.op, s...)
                        end
                        new_expr.stats = s_mat_stats
                        new_query = Query(Alias(galley_gensym("A")), new_expr)
                        new_agg_idxs = s_reduce_idxs
                        # We want to prefer larger kernels here.
                        min_cost = s_cost - length(s) * 1000
                    end
                end
            end
        end
        # If there is no valid way to reduce index occurrences, we just break and handle
        # the remainder.
        if isnothing(new_query)
            break
        end
        push!(queries, new_query)
        setdiff!(aq.reduce_idxs, new_agg_idxs)
        condense_stats!(new_query.expr.stats; cheap=false)
        alias = Alias(new_query.name.name)
        alias.stats = copy_stats(new_query.expr.stats)
        alias.node_id = node_id_counter
        node_id_counter += 1
        replace_and_remove_nodes!(pe, nodes_to_remove[1], alias, nodes_to_remove[2:end])
        cur_occurences = count_index_occurences([pe])
    end
    remainder_expr = has_agg ? Aggregate(agg_op, agg_init, aq.reduce_idxs..., pe) : pe
    if !isnothing(aq.output_order)
        remainder_expr = Materialize(
            aq.output_format..., aq.output_order..., remainder_expr
        )
    end
    final_query = Query(q.name, remainder_expr)
    push!(queries, final_query)
    for query in queries
        insert_statistics!(ST, query)
        insert_node_ids!(query)
    end
    return queries
end

function split_queries(ST, max_kernel_size, p::PlanNode; alias_stats=Dict(), verbose)
    new_queries = []
    for query in p.queries
        append!(new_queries, split_query(query, ST, max_kernel_size, alias_stats, verbose))
    end
    new_plan = Plan(new_queries..., p.outputs)
    insert_node_ids!(new_plan)
    return new_plan
end
