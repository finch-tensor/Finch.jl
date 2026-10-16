function greedy_query_to_plan(input_aq::AnnotatedQuery, cost_cache, alias_hash)
    aq = copy_aq(input_aq)
    queries = []
    total_cost = 0
    reducible_idxs = get_reducible_idxs(aq)
    while !isempty(reducible_idxs)
        cheapest_idx = nothing
        cheapest_cost = Inf
        for idx in reducible_idxs
            cost, _ = cost_of_reduce(idx, aq, cost_cache, alias_hash)
            if cost < cheapest_cost
                cheapest_idx = idx
                cheapest_cost = cost
            end
        end
        if cheapest_cost == Inf
            println("REDUCIBLE IDXs: $reducible_idxs")
            println("$(aq.point_expr)")
        end
        query = reduce_idx!(cheapest_idx, aq)

        alias_hash[query.name.name] = cannonical_hash(query.expr, alias_hash)
        push!(queries, query)
        reducible_idxs = get_reducible_idxs(aq)
        total_cost += cheapest_cost
    end
    remaining_q = get_remaining_query(aq)
    if !isnothing(remaining_q)
        push!(queries, remaining_q)
    end
    last_query = queries[end]
    # The final query should produce the result, so we ensure that it has the result's name
    last_query.name = aq.output_name
    last_query.expr = Materialize(aq.output_format..., aq.output_order..., last_query.expr)
    last_query.expr.stats = last_query.expr.expr.stats
    get_def(last_query.expr.stats).index_order = [idx for idx in aq.output_order]
    get_def(last_query.expr.stats).level_formats = [f for f in aq.output_format]
    alias_hash[last_query.name.name] = cannonical_hash(last_query.expr, alias_hash)
    return queries, total_cost, cost_cache
end
