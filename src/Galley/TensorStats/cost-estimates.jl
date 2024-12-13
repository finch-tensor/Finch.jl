
# We start by defining some basic cost parameters. These will need to be adjusted somewhat
# through testing.
const SeqReadCost = 1
const SeqWriteCost = 5
const RandomReadCost = 5
const RandomWriteCost = 10

const ComputeCost = 1
const AllocateCost = 10
const DenseAllocateCost = .5
const SparseAllocateCost = 60

# We estimate the prefix cost based on the number of iterations in that prefix.
function get_loop_lookups(vars::Set{IndexExpr}, rel_conjuncts, rel_disjuncts)
    # This tensor stats doesn't actually correspond to a particular place in the expr tree,
    # so we unfortunately have to mangle the statistics interface a bit.
    join_stats = if length(rel_disjuncts) == 0 || vars ⊆ union(get_index_set(stat) for stat in rel_conjuncts)
        join_def = merge_tensor_def(+, [get_def(stat) for stat in rel_conjuncts]...)
        merge_tensor_stats_join(+, join_def, rel_conjuncts...)
    elseif length(rel_conjuncts) == 0
        new_def = merge_tensor_def(+, [get_def(stat) for stat in rel_disjuncts]...)
        merge_tensor_stats_union(+, new_def, rel_disjuncts...)
    else
        disjunct_def = merge_tensor_def(+, [get_def(stat) for stat in rel_disjuncts]...)
        disjunct_stats = merge_tensor_stats_union(+, disjunct_def, rel_disjuncts...)

        join_def = merge_tensor_def(+, [get_def(stat) for stat in rel_conjuncts]..., disjunct_def)
        merge_tensor_stats_join(+, join_def, rel_conjuncts..., disjunct_stats)
    end

    lookups = estimate_nnz(join_stats, indices=vars)
    return lookups
end

# The prefix cost is equal to the number of valid iterations times the number of tensors
# which we need to access to handle that final iteration.
function get_prefix_cost(new_prefix::Vector{IndexExpr},  output_vars, conjunct_stats, disjunct_stats)
    new_var = new_prefix[end]
    prefix_set = Set(new_prefix)
    rel_conjuncts = [stat for stat in conjunct_stats if !isempty(get_index_set(stat) ∩ prefix_set)]
    rel_disjuncts = [stat for stat in disjunct_stats if !isempty(get_index_set(stat) ∩ prefix_set)]
    lookups = get_loop_lookups(prefix_set, rel_conjuncts, rel_disjuncts)
    lookup_factor = 0
    for stat in union(rel_conjuncts, rel_disjuncts)
        if new_var ∉ get_index_set(stat)
            continue
        end
        if isnothing(get_index_order(stat)) || needs_reformat(stat, new_prefix)
            rel_vars = get_index_set(stat) ∩ prefix_set
            approx_sparsity = estimate_nnz(stat; indices=rel_vars, conditional_indices=setdiff(rel_vars, [new_var])) / get_dim_size(stat, new_var)
            is_dense = approx_sparsity > .05
            lookup_factor += is_dense ? SeqReadCost / 5 : SeqReadCost
            continue
        end

        format = get_index_format(stat, new_var)
        if format == t_dense
            lookup_factor += SeqReadCost / 5
        elseif format == t_bytemap
            lookup_factor += SeqReadCost / 2.5
        else
            lookup_factor += SeqReadCost
        end
    end

    if output_vars isa Vector && new_var ∈ output_vars
        new_var_idx = only(indexin([new_var], output_vars))
        min_var_idx = minimum([x for x in indexin(new_prefix, output_vars) if !isnothing(x)])
        is_rand_write = new_var_idx != min_var_idx
        if is_rand_write
            lookup_factor += RandomWriteCost
        else
            lookup_factor += SeqWriteCost
        end
    else
        lookup_factor += SeqWriteCost
    end

    return lookups * lookup_factor
end
