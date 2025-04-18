include("annotated-query.jl")
include("greedy-optimizer.jl")
include("pruned-optimizer.jl")
include("query-splitter.jl")

function one_step_distribute(q::PlanNode)
    distribution_triangles = []
    for root in PreOrderDFS(q)
        if root.kind == MapJoin
            for child2 in root.args
                if child2.kind == MapJoin && isdistributive(root.op.val, child2.op.val)
                    for child1 in root.args
                        if child1.node_id != child2.node_id
                            push!(
                                distribution_triangles,
                                (root.node_id, child1.node_id, child2.node_id),
                            )
                        end
                    end
                end
            end
        end
    end
    plans = []
    for (r, c1, c2) in distribution_triangles
        new_plan = plan_copy(q)
        root = nothing
        for node in PreOrderDFS(new_plan)
            if node.node_id == r
                root = node
            end
        end

        child1 = nothing
        for node in root.args
            if node.node_id == c1
                child1 = node
            end
        end

        child2 = nothing
        for node in root.args
            if node.node_id == c2
                child2 = node
            end
        end
        root.args = [n for n in root.args if n.node_id != c1]
        child2.args = [
            n.kind == Value ? n : MapJoin(root.op, child1, n) for n in child2.args
        ]
        new_plan = canonicalize(new_plan, false)
        push!(plans, new_plan)
    end
    return plans
end

function enumerate_distributed_plans(q::PlanNode, visited_plans, alias_hash, max_depth=1)
    plans = []
    plan_frontier = [plan_copy(q)]
    for _ in 1:max_depth
        new_plans = []
        for plan in plan_frontier
            for new_plan in one_step_distribute(plan)
                phash = cannonical_hash(new_plan, alias_hash)
                if phash ∉ visited_plans
                    push!(new_plans, new_plan)
                    push!(visited_plans, phash)
                end
            end
        end
        if length(new_plans) == 0
            break
        end
        append!(plans, new_plans)
        plan_frontier = new_plans
    end
    return plans
end
function high_level_optimize(
    faq_optimizer::FAQ_OPTIMIZERS,
    input_plan::PlanNode,
    ST,
    alias_stats::OrderedDict{IndexExpr,TensorStats},
    alias_hash::OrderedDict{IndexExpr,UInt},
    verbose,
)
    logical_plan = PlanNode[]
    for input_query in input_plan.queries
        logical_queries = high_level_optimize_query(
            faq_optimizer, input_query, ST, alias_stats, alias_hash, verbose
        )
        for query in logical_queries
            alias_hash[query.name.name] = cannonical_hash(query.expr, alias_hash)
            alias_stats[query.name.name] = query.expr.stats
        end
        append!(logical_plan, logical_queries)
    end
    return Plan(logical_plan...)
end

function high_level_optimize_query(
    faq_optimizer::FAQ_OPTIMIZERS,
    q::PlanNode,
    ST,
    alias_stats::OrderedDict{IndexExpr,TensorStats},
    alias_hash::OrderedDict{IndexExpr,UInt},
    verbose,
)
    insert_statistics!(ST, q; bindings=alias_stats)
    if faq_optimizer === naive
        insert_node_ids!(q)
        return PlanNode[q]
    end

    # If there's the possibility of distributivity, we attempt that pushdown and see
    # whether it benefits the computation.
    check_dnf = !allequal([n.op.val for n in PostOrderDFS(q) if n.kind === MapJoin])
    q_non_dnf = canonicalize(plan_copy(q), false)
    input_aq = AnnotatedQuery(q_non_dnf, ST)
    logical_queries, cnf_cost, cost_cache = high_level_optimize_annotated_query(
        faq_optimizer, input_aq, alias_hash, OrderedDict{UInt,Float64}(), verbose
    )
    if check_dnf
        min_cost = cnf_cost
        min_query = canonicalize(plan_copy(q), false)
        visited_queries = OrderedSet()
        finished = false
        while !finished
            finished = true
            for query in
                enumerate_distributed_plans(min_query, visited_queries, alias_hash, 1)
                input_aq = AnnotatedQuery(query, ST)
                queries, cost, cost_cache = high_level_optimize_annotated_query(
                    faq_optimizer, input_aq, alias_hash, cost_cache, verbose
                )
                if cost < min_cost
                    logical_queries = queries
                    min_cost = cost
                    min_query = plan_copy(query)
                    finished = false
                end
            end
        end

        # We check the fully distributed option too just to see
        q_dnf = canonicalize(q, true)
        if cannonical_hash(q_dnf, alias_hash) ∉ visited_queries
            dnf_aq = AnnotatedQuery(q_dnf, ST)
            dnf_queries, dnf_cost, cost_cache = high_level_optimize_annotated_query(
                faq_optimizer, dnf_aq, alias_hash, cost_cache, verbose
            )
            if dnf_cost < min_cost
                verbose >= 1 && println("USED FULL DNF")
                logical_queries = dnf_queries
                min_cost = dnf_cost
                min_query = q_dnf
            end
        end
        verbose >= 1 && println("Used DNF: $(min_cost < cnf_cost) \n QUERY: $min_query")
    end
    return logical_queries
end

function high_level_optimize_annotated_query(
    faq_optimizer::FAQ_OPTIMIZERS,
    aq::AnnotatedQuery,
    alias_hash::OrderedDict{IndexExpr,UInt},
    cost_cache::OrderedDict{UInt,Float64},
    verbose,
)
    if faq_optimizer == greedy
        return pruned_query_to_plan(aq, cost_cache, alias_hash; use_greedy=true)
    elseif faq_optimizer == exact
        return exact_query_to_plan(aq, cost_cache, alias_hash)
    elseif faq_optimizer == pruned
        return pruned_query_to_plan(aq, cost_cache, alias_hash)
    end
end
