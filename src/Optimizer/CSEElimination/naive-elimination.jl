function naive_cse!(plan::PlanNode)
    hash_to_alias = OrderedDict()
    alias_to_hash = OrderedDict()
    cannonical_alias = OrderedDict()
    cse_queries = PlanNode[]
    for query in plan.queries
        for n in PostOrderDFS(query.expr)
            if n.kind == Alias
                n.val = cannonical_alias[n.val]
            end
        end
        q_hash = cannonical_hash(query.expr, alias_to_hash)
        if haskey(hash_to_alias, q_hash)
            cannonical_alias[query.name.name] = hash_to_alias[q_hash]
        else
            cannonical_alias[query.name.name] = query.name.name
            hash_to_alias[q_hash] = query.name.name
            alias_to_hash[query.name.name] = q_hash
            push!(cse_queries, query)
        end
    end
    return Plan(cse_queries...)
end
