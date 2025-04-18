
function modify_plan_formats!(plan::PlanNode, alias_to_loop_order, alias_stats)
    for query in plan.queries
        if query.expr.kind === Aggregate
            loop_order_when_used = alias_to_loop_order[query.name.name]
            output_stats = query.expr.stats
            output_order = relative_sort(
                get_index_set(output_stats), loop_order_when_used; rev=true
            )
            loop_order_when_built = IndexExpr[idx.name for idx in query.loop_order]
            # Determine the optimal output format & add a further query to reformat if necessary.
            output_formats = select_output_format(
                output_stats, loop_order_when_built, output_order
            )
            query.expr = Materialize(output_formats..., output_order..., query.expr)
            reorder_stats = copy_stats(output_stats)
            reorder_def = get_def(reorder_stats)
            reorder_def.index_order = output_order
            reorder_def.level_formats = output_formats
            query.expr.stats = reorder_stats
            alias_stats[query.name.name] = query.expr.stats
            @assert !isnothing(get_index_order(alias_stats[query.name.name])) "$(query.name.name)"
        end
    end
    return plan
end

function select_output_format(output_stats::TensorStats,
    loop_order::Vector{IndexExpr},
    output_indices::Vector{IndexExpr})
    if length(output_indices) == 0
        return LevelFormat[]
    end

    formats = []
    for i in eachindex(output_indices)
        prefix = output_indices[(length(output_indices) - i + 1):end]
        needs_rw = !fully_compat_with_loop_prefix(prefix, loop_order)
        # We use division here rather than `conditional_indices` because lower=more conservative
        # here.
        prev_nnz = estimate_nnz(output_stats; indices=prefix[2:end])
        new_nnz = estimate_nnz(output_stats; indices=prefix)
        approx_nnz_per = new_nnz / prev_nnz
        # We prefer to be conservative on formats so we scale down the sparsity a bit
        approx_sparsity = approx_nnz_per / get_dim_size(output_stats, prefix[1])
        dense_memory_footprint = prev_nnz * get_dim_size(output_stats, prefix[1])
        if approx_sparsity > 0.5 && dense_memory_footprint < 3 * 10^10
            #            if get_dim_space_size(output_stats, OrderedSet(prefix)) > 10^10
            #                throw(OutOfMemoryError())
            #            end
            push!(formats, t_dense)
        elseif approx_sparsity > 0.05 && dense_memory_footprint < 3 * 10^10 &&
            (length(formats) == 0 ? true : formats[end] != t_bytemap) # TODO: Check out finch double bytemap bug
            #            if get_dim_space_size(output_stats, OrderedSet(prefix)) > 10^10
            #                throw(OutOfMemoryError())
            #            end
            push!(formats, t_bytemap)
        elseif needs_rw
            push!(formats, t_hash)
        else
            push!(formats, t_sparse_list)
        end
    end
    return reverse(formats)
end
