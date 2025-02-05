function branch_and_bound(
    input_aq::AnnotatedQuery,
    component,
    k,
    max_subquery_costs,
    alias_hash,
    cost_cache=Dict{UInt,Float64}(),
)
    input_aq = copy_aq(input_aq)
    PLAN_AND_COST = Tuple{Vector{IndexExpr},Vector{PlanNode},AnnotatedQuery,Float64}
    optimal_orders = Dict{Set{IndexExpr},PLAN_AND_COST}(
        Set{IndexExpr}() => (PlanNode[], PlanNode[], input_aq, 0)
    )
    prev_new_optimal_orders = optimal_orders
    # To speed up inference, we cache cost calculations for each set of already reduced idxs
    # and proposed reduction index.
    for _ in 1:length(component)
        best_idx_ext = Dict{
            Set{IndexExpr},
            Tuple{AnnotatedQuery,IndexExpr,Vector{IndexExpr},Vector{PlanNode},Float64},
        }()
        for (vars, pc) in prev_new_optimal_orders
            aq = pc[3]
            prev_cost = pc[4]
            for idx in get_reducible_idxs(aq) ∩ component
                cost, reduced_vars = cost_of_reduce(idx, aq, cost_cache, alias_hash)
                cost += prev_cost
                new_vars = union(vars, IndexExpr[i for i in reduced_vars])
                bound = Inf
                for vars2 in keys(max_subquery_costs)
                    if vars2 ⊇ new_vars
                        bound = min(bound, max_subquery_costs[vars2])
                    end
                end
                cheapest_cost = min(
                    get(best_idx_ext, new_vars, (nothing, nothing, nothing, nothing, Inf))[5],
                    get(optimal_orders, new_vars, (nothing, nothing, nothing, Inf))[4],
                    bound)
                if cost <= cheapest_cost + 10 # We add 1 to avoid FP issues
                    best_idx_ext[new_vars] = (aq, idx, pc[1], pc[2], cost)
                end
            end
        end
        if length(best_idx_ext) == 0
            break
        end
        num_to_keep = Int(min(k, length(best_idx_ext)))
        # At each step, we only keep 'k' options for the next index.
        top_k_idx_ext = Dict{
            Set{IndexExpr},
            Tuple{AnnotatedQuery,IndexExpr,Vector{IndexExpr},Vector{PlanNode},Float64},
        }(
            sort(collect(best_idx_ext); by=(v_p) -> v_p[2][5])[1:num_to_keep]
        )

        new_optimal_orders = Dict{Set{IndexExpr},PLAN_AND_COST}()
        for (new_vars, idx_ext_info) in top_k_idx_ext
            aq, idx, old_order, old_queries, cost = idx_ext_info
            new_aq = copy_aq(aq)
            reduce_query = reduce_idx!(idx, new_aq)
            alias_hash[reduce_query.name.name] = cannonical_hash(
                reduce_query.expr, alias_hash
            )
            new_queries = PlanNode[old_queries..., reduce_query]
            new_order = IndexExpr[old_order..., idx]
            new_optimal_orders[new_vars] = (new_order, new_queries, new_aq, cost)
        end
        merge!(optimal_orders, new_optimal_orders)
        prev_new_optimal_orders = new_optimal_orders
    end

    # During the greedy pass, we compute upper bounds on the cost of each subquery which
    # will be used in the pruned pass.
    optimal_subquery_costs = Dict{Set{IndexExpr},Float64}()
    if k == 1
        for vars in keys(optimal_orders)
            optimal_subquery_costs[vars] = optimal_orders[vars][4]
        end
    end
    if haskey(optimal_orders, Set{IndexExpr}([i for i in component]))
        return optimal_orders[Set{IndexExpr}([i for i in component])],
        optimal_subquery_costs,
        cost_cache
    else
        println("Component: $component")
        println("Optimal Orders: $(keys(optimal_orders))")
        println("Reducible Idxs: $(get_reducible_idxs(input_aq))")
        return nothing
    end
end

function pruned_query_to_plan(
    input_aq::AnnotatedQuery,
    cost_cache::Dict{UInt,Float64},
    alias_hash::Dict{IndexExpr,UInt};
    use_greedy=false,
)
    total_cost = 0
    elimination_order = IndexExpr[]
    queries = PlanNode[]
    cur_aq = copy_aq(input_aq)
    while !isempty(get_reducible_idxs(cur_aq))
        component = cur_aq.connected_components[1]
        if isempty(component ∩ get_reducible_idxs(cur_aq))
            continue
        end

        (greedy_order, greedy_queries, greedy_aq, greedy_cost), greedy_subquery_costs, cost_cache = branch_and_bound(
            cur_aq, component, 1, Dict(), alias_hash, cost_cache
        )
        if length(component) >= 10 || use_greedy
            append!(elimination_order, greedy_order)
            for idx in greedy_order
                reduce_query = reduce_idx!(idx, cur_aq)
                alias_hash[reduce_query.name.name] = cannonical_hash(
                    reduce_query.expr, alias_hash
                )
                push!(queries, reduce_query)
            end
            total_cost += greedy_cost
            continue
        end
        exact_opt_result = branch_and_bound(
            cur_aq, component, Inf, greedy_subquery_costs, alias_hash, cost_cache
        )
        if !isnothing(exact_opt_result)
            (exact_order, exact_queries, exact_aq, exact_cost),
            exact_subquery_costs,
            cost_cache = exact_opt_result
            append!(elimination_order, exact_order)
            for idx in exact_order
                reduce_query = reduce_idx!(idx, cur_aq)
                alias_hash[reduce_query.name.name] = cannonical_hash(
                    reduce_query.expr, alias_hash
                )
                push!(queries, reduce_query)
            end
            total_cost += exact_cost
        else
            println("WARNING: Pruned Optimizer Failed. Falling Back to Greedy Plan.")
            append!(elimination_order, greedy_order)
            for idx in greedy_order
                reduce_query = reduce_idx!(idx, cur_aq)
                alias_hash[reduce_query.name.name] = cannonical_hash(
                    reduce_query.expr, alias_hash
                )
                push!(queries, reduce_query)
            end
            total_cost += greedy_cost
        end
    end

    remaining_q = get_remaining_query(cur_aq)
    if !isnothing(remaining_q)
        push!(queries, remaining_q)
    end
    last_query = queries[end]
    # The final query should produce the result, so we ensure that it has the result's name
    last_query.name = cur_aq.output_name
    last_query.expr = Materialize(
        cur_aq.output_format..., cur_aq.output_order..., last_query.expr
    )
    last_query.expr.stats = copy_stats(last_query.expr.expr.stats)
    get_def(last_query.expr.stats).index_order = [idx for idx in cur_aq.output_order]
    get_def(last_query.expr.stats).level_formats = [f for f in cur_aq.output_format]
    alias_hash[last_query.name.name] = cannonical_hash(last_query.expr, alias_hash)
    return queries, total_cost, cost_cache
end

function exact_query_to_plan(input_aq::AnnotatedQuery, cost_cache, alias_hash)
    total_cost = 0
    elimination_order = []
    for component in input_aq.connected_components
        (exact_order, exact_queries, exact_aq, exact_cost), exact_subquery_costs, cost_cache = branch_and_bound(
            input_aq, component, Inf, Dict(), alias_hash, cost_cache
        )
        append!(elimination_order, exact_order)
        total_cost += exact_cost
    end
    queries = order_to_queries(input_aq, elimination_order, alias_hash)
    return queries, total_cost, cost_cache
end
