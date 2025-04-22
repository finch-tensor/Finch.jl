
function select_leader_protocol(format::LevelFormat)
    if format == t_sparse_list
        return t_walk
    elseif format == t_coo
        return t_default
    elseif format == t_dense
        return t_default
    elseif format == t_bytemap
        return t_walk
    elseif format == t_hash
        return t_walk
    end
end

function select_follower_protocol(format::LevelFormat)
    if format == t_sparse_list
        return t_follow
    elseif format == t_coo
        return t_default
    elseif format == t_dense
        return t_follow
    elseif format == t_bytemap
        return t_follow
    elseif format == t_hash
        return t_follow
    end
end

function modify_plan_protocols!(plan::PlanNode, ST, alias_stats)
    for query in plan.queries
        # Propagate the index order and format info from the last step.
        insert_node_ids!(query)
        insert_statistics!(ST, query; bindings=alias_stats)
        # Choose access protocols
        modify_protocols!(query.expr)
        alias_stats[query.name.name] = query.expr.stats
        @assert !isnothing(get_index_order(alias_stats[query.name.name])) "$(query.name.name)"
    end
    return plan
end

function modify_protocols!(expr)
    inputs = get_conjunctive_and_disjunctive_inputs(expr)
    conjuncts = [input.stats for input in inputs.conjuncts]
    disjuncts = [input.stats for input in inputs.disjuncts]

    if length(conjuncts) == 0 && length(disjuncts) == 0
        return nothing
    end

    # Start by initializing the protocol lists for each input
    for input in conjuncts
        get_def(input).index_protocols = [t_default for _ in get_index_order(input)]
    end
    for input in disjuncts
        get_def(input).index_protocols = [t_default for _ in get_index_order(input)]
    end

    vars = union(
        [get_index_set(i) for i in conjuncts]..., [get_index_set(i) for i in disjuncts]...
    )
    for var in vars
        relevant_conjuncts = [i for i in conjuncts if var ∈ get_index_set(i)]
        relevant_disjuncts = [i for i in disjuncts if var ∈ get_index_set(i)]
        if length(relevant_conjuncts) == 0
            # If there are no covering conjuncts, then we need to walk all disjuncts
            for input in relevant_disjuncts
                input_def = get_def(input)
                var_index = findall(x -> x == var, get_index_order(input))
                input_def.index_protocols[var_index] .= select_leader_protocol(
                    get_index_format(input, var)
                )
            end
        else
            # If there is at least one covering conjunct, all disjuncts are followers
            for input in relevant_disjuncts
                input_def = get_def(input)
                var_index = findall(x -> x == var, get_index_order(input))
                input_def.index_protocols[var_index] .= select_follower_protocol(
                    get_index_format(input, var)
                )
            end
            costs = []
            for input in relevant_conjuncts
                if get_index_format(input, var) == t_dense
                    push!(costs, get_dim_size(input, var))
                    continue
                end
                size_before_var = 1
                indices_before_var = []
                for index in reverse(get_index_order(input))
                    index == var && break
                    push!(indices_before_var, index)
                end
                # The choice of `|` below is arbitrary because the actual agg_op doesn't affect
                # the nnz (barring things like prod reductions which might be a TODO).
                # TODO: Replace this with conditional estimates
                if length(indices_before_var) > 0
                    size_before_var = estimate_nnz(
                        reduce_tensor_stats(
                            |,
                            false,
                            setdiff(get_index_set(input), indices_before_var),
                            set_fill_value!(input, false),
                        ),
                    )
                end
                size_after_var = estimate_nnz(
                    reduce_tensor_stats(
                        |,
                        false,
                        setdiff(get_index_set(input), [indices_before_var..., var]),
                        set_fill_value!(input, false),
                    ),
                )
                push!(costs, max(1, size_after_var / size_before_var))
            end
            min_cost = minimum(costs)

            # For the conjuncts, we identify a single leader then have the rest follow.
            needs_leader = true
            for i in eachindex(relevant_conjuncts)
                input = relevant_conjuncts[i]
                input_def = get_def(input)
                format = get_index_format(input, var)
                var_index = findall(x -> x == var, get_index_order(input))
                is_leader = costs[i] == min_cost
                if is_leader && needs_leader
                    input_def.index_protocols[var_index] .= select_leader_protocol(format)
                    needs_leader = false
                else
                    input_def.index_protocols[var_index] .= select_follower_protocol(format)
                end
            end
        end
    end
end
