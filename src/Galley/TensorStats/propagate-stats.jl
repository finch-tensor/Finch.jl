# This file defines how stats are produced for a logical expression based on its children.

# We begin by defining the necessary interface for a statistics object.
function merge_tensor_stats_join(op, all_stats::Vararg{TensorStats})
    throw(error("merge_tensor_stats_join not implemented for: ", typeof(all_stats[1])))
end

function merge_tensor_stats_union(op, all_stats::Vararg{TensorStats})
    throw(error("merge_tensor_stats_union not implemented for: ", typeof(all_stats[1])))
end

function reduce_tensor_stats(
    op, init, reduce_indices::StableSet{IndexExpr}, stats::TensorStats
)
    throw(error("reduce_tensor_stats not implemented for: ", typeof(stats)))
end

function transpose_tensor_stats(index_order::Vector{IndexExpr}, stats::TensorStats)
    throw(error("transpose_tensor_stats not implemented for: ", typeof(stats)))
end

# We now define a set of functions for manipulating the TensorDefs that will be shared
# across all statistics types
function merge_tensor_def(op, all_defs::Vararg{TensorDef})
    new_fill_value = op([def.fill_val for def in all_defs]...)
    new_index_set = union([def.index_set for def in all_defs]...)
    new_dim_sizes = OrderedDict{IndexExpr,Float64}()
    for index in new_index_set
        for def in all_defs
            if index in def.index_set
                new_dim_sizes[index] = def.dim_sizes[index]
            end
        end
    end
    #    @assert new_index_set ⊆ keys(new_dim_sizes)
    return TensorDef(
        new_index_set, new_dim_sizes, new_fill_value, nothing, nothing, nothing
    )
end

function reduce_tensor_def(op, init, reduce_indices::StableSet{IndexExpr}, def::TensorDef)
    op = op isa PlanNode ? op.val : op
    init = init isa PlanNode ? init.val : init
    if isnothing(init)
        if isnothing(op) && isnothing(init)
            init = def.fill_val
        elseif isidentity(op, def.fill_val) || isidempotent(op)
            init = op(def.fill_val, def.fill_val)
        elseif op == +
            init = def.fill_val * prod([def.dim_sizes[x] for x in reduce_indices])
        elseif op == *
            init = def.fill_val^prod([def.dim_sizes[x] for x in reduce_indices])
        else
            # This is going to be VERY SLOW. Should raise a warning about reductions over non-identity fill values.
            # Depending on the semantics of reductions, we might be able to do this faster.
            println(
                "Warning: A reduction can take place over a tensor whose fill value is not the reduction operator's identity. \\
                        This can result in a large slowdown as the new fill is calculated.",
            )
            init = op(
                [
                    def.fill_val for
                    _ in prod([def.dim_sizes[x] for x in reduce_indices])
                ]...,
            )
        end
    end
    @assert !isnothing(init)
    new_index_set = setdiff(def.index_set, reduce_indices)
    new_dim_sizes = OrderedDict{IndexExpr,Float64}()
    for index in new_index_set
        new_dim_sizes[index] = def.dim_sizes[index]
    end
    return TensorDef(new_index_set, new_dim_sizes, init, nothing, nothing, nothing)
end

# This function determines whether a binary operation is union-like or join-like and creates
# new statistics objects accordingly.
function merge_tensor_stats(op, all_stats::Vararg{ST}) where {ST<:TensorStats}
    new_def::TensorDef = merge_tensor_def(op, [get_def(stats) for stats in all_stats]...)
    join_like_args = ST[]
    union_like_args = ST[]
    for stats in all_stats
        if length(get_index_set(stats)) == 0
            continue
        end
        if isannihilator(op, get_fill_value(stats))
            push!(join_like_args, stats)
        else
            push!(union_like_args, stats)
        end
    end
    if length(union_like_args) == 0 && length(join_like_args) == 0
        return ST(get_fill_value(new_def))
    elseif length(union_like_args) == 0
        return merge_tensor_stats_join(op, new_def, join_like_args...)
    elseif length(join_like_args) == 0
        return merge_tensor_stats_union(op, new_def, union_like_args...)
    elseif union([get_index_set(stats) for stats in join_like_args]...) ==
        get_index_set(new_def)
        # Currently we glean no information from non-join-like args
        return merge_tensor_stats_join(op, new_def, join_like_args...)
    else
        return merge_tensor_stats_union(op, new_def, join_like_args..., union_like_args...)
    end
end

function merge_tensor_stats(op::PlanNode, all_stats::Vararg{ST}) where {ST<:TensorStats}
    return merge_tensor_stats(op.val, all_stats...)
end

function reduce_tensor_stats(
    op, init, reduce_indices::Union{Vector{PlanNode},StableSet{PlanNode}}, stats::ST
) where {ST<:TensorStats}
    return reduce_tensor_stats(
        op, init, StableSet{IndexExpr}([idx.name for idx in reduce_indices]), stats
    )
end

function transpose_tensor_def(index_order::Vector{IndexExpr}, def::TensorDef)
    return reindex_def(index_order, def)
end

################# NaiveStats Propagation ##################################################
# We do everything in log for numerical stability
function merge_tensor_stats_join(op, new_def::TensorDef, all_stats::Vararg{NaiveStats})
    new_dim_space_size = sum([
        log2(get_dim_size(new_def, idx)) for idx in new_def.index_set
    ])
    prob_non_fill = sum([
        log2(stats.cardinality) -
        sum([log2(get_dim_size(stats, idx)) for idx in get_index_set(stats)]) for
        stats in all_stats
    ])
    new_cardinality = 2^(prob_non_fill + new_dim_space_size)
    return NaiveStats(new_def, new_cardinality)
end

function merge_tensor_stats_union(op, new_def::TensorDef, all_stats::Vararg{NaiveStats})
    new_dim_space_size = sum([
        log2(get_dim_size(new_def, idx)) for idx in new_def.index_set
    ])
    prob_fill = sum([
        log2(
            1 -
            2^(
                log2(stats.cardinality) -
                sum([log2(get_dim_size(stats, idx)) for idx in get_index_set(stats)])
            ),
        ) for stats in all_stats
    ])
    new_cardinality = 2^(log2(1 - 2^prob_fill) + new_dim_space_size)
    return NaiveStats(new_def, new_cardinality)
end

function reduce_tensor_stats(
    op, init, reduce_indices::StableSet{IndexExpr}, stats::NaiveStats
)
    if length(reduce_indices) == 0
        return copy_stats(stats)
    end
    new_def = reduce_tensor_def(op, init, reduce_indices, get_def(stats))
    new_dim_space_size = sum([
        log2(get_dim_size(new_def, idx)) for idx in new_def.index_set
    ])
    old_dim_space_size = sum([
        log2(get_dim_size(stats, idx)) for idx in get_index_set(stats)
    ])
    prob_fill_value = 1 - 2^(log2(stats.cardinality) - old_dim_space_size)
    prob_non_fill_subspace =
        1 - 2^(log2(prob_fill_value) * 2^(old_dim_space_size - new_dim_space_size))
    new_cardinality = 2^(new_dim_space_size + log2(prob_non_fill_subspace))
    return NaiveStats(new_def, new_cardinality)
end

function transpose_tensor_stats(index_order::Vector{IndexExpr}, stats::NaiveStats)
    stats = copy_stats(stats)
    stats.def = transpose_tensor_def(index_order, get_def(stats))
    return stats
end

################# DCStats Propagation ##################################################

function unify_dc_ints(all_stats, new_def)
    final_idx_2_int = OrderedDict{IndexExpr,Int}()
    final_int_2_idx = OrderedDict{Int,IndexExpr}()
    max_int = 1
    for (i, idx) in enumerate(union([keys(stat.idx_2_int) for stat in all_stats]...))
        final_idx_2_int[idx] = max_int
        final_int_2_idx[max_int] = idx
        max_int += 1
    end
    for idx in get_index_set(new_def)
        if !haskey(final_idx_2_int, idx)
            final_idx_2_int[idx] = max_int
            final_int_2_idx[max_int] = idx
            max_int += 1
        end
    end
    final_idx_2_int, final_int_2_idx
end

convert_bitset(int_to_int, b) = BitSet([int_to_int[x] for x in b])

function merge_tensor_stats_join(op, new_def::TensorDef, all_stats::Vararg{DCStats})
    if length(all_stats) == 1
        return DCStats(
            new_def,
            copy(all_stats[1].idx_2_int),
            copy(all_stats[1].int_2_idx),
            copy(all_stats[1].dcs),
        )
    end
    final_idx_2_int, final_int_2_idx = unify_dc_ints(all_stats, new_def)
    new_dc_dict = OrderedDict{DCKey,Float64}()
    for stats in all_stats
        for dc in stats.dcs
            dc_key = (X=BitSet(Int[final_idx_2_int[stats.int_2_idx[x]] for x in dc.X]),
                Y=BitSet(Int[final_idx_2_int[stats.int_2_idx[y]] for y in dc.Y]))
            current_dc = get(new_dc_dict, dc_key, Inf)
            if dc.d < current_dc
                new_dc_dict[dc_key] = dc.d
            end
        end
    end
    new_stats = DCStats(
        new_def,
        final_idx_2_int,
        final_int_2_idx,
        StableSet{DC}(DC(key.X, key.Y, d) for (key, d) in new_dc_dict),
    )
    return new_stats
end

function merge_tensor_stats_union(op, new_def::TensorDef, all_stats::Vararg{DCStats})
    if length(all_stats) == 1
        return DCStats(
            new_def,
            copy(all_stats[1].idx_2_int),
            copy(all_stats[1].int_2_idx),
            copy(all_stats[1].dcs),
        )
    end
    final_idx_2_int, final_int_2_idx = unify_dc_ints(all_stats, new_def)
    dc_keys = counter(DCKey)
    stats_dcs = []
    # We start by extending all arguments' dcs to the new dimensions and infer dcs as needed
    for stats in all_stats
        dcs = OrderedDict{DCKey,Float64}()
        Z = setdiff(get_index_set(new_def), get_index_set(stats))
        Z_dimension_space_size = get_dim_space_size(new_def, Z)
        for dc in stats.dcs
            new_key::DCKey = (
                X=BitSet(Int[final_idx_2_int[stats.int_2_idx[x]] for x in dc.X]),
                Y=BitSet(Int[final_idx_2_int[stats.int_2_idx[y]] for y in dc.Y]))
            dcs[new_key] = dc.d
            inc!(dc_keys, new_key)
            ext_dc_key = (X=new_key.X, Y=(∪(new_key.Y, idxs_to_bitset(final_idx_2_int, Z))))
            if !haskey(dcs, ext_dc_key)
                inc!(dc_keys, ext_dc_key)
            end
            dcs[ext_dc_key] = min(get(dcs, ext_dc_key, Inf), dc.d * Z_dimension_space_size)
        end
        push!(stats_dcs, dcs)
    end

    # We only keep DCs which can be inferred from all inputs. Otherwise, we might miss
    # important information which simply wasn't inferred
    new_dcs = OrderedDict{DCKey,Float64}()
    for (key, count) in dc_keys
        if count == length(all_stats)
            new_dcs[key] = min(
                typemax(UInt), sum([get(dcs, key, Float64(0)) for dcs in stats_dcs])
            )
            if key.Y ⊆ idxs_to_bitset(final_idx_2_int, get_index_set(new_def))
                new_dcs[key] = min(
                    new_dcs[key],
                    get_dim_space_size(new_def, bitset_to_idxs(final_int_2_idx, key.Y)),
                )
            end
        end
    end

    #=
        for Y in subsets(collect(get_index_set(new_def)))
            proj_dc_key = (X=BitSet(), Y=idxs_to_bitset(final_idx_2_int, Y))
            new_dcs[proj_dc_key] = min(get(new_dcs, proj_dc_key, typemax(UInt)/2), get_dim_space_size(new_def, StableSet(Y)))
        end
     =#
    return DCStats(
        new_def,
        final_idx_2_int,
        final_int_2_idx,
        StableSet{DC}(DC(key.X, key.Y, d) for (key, d) in new_dcs),
    )
end

function reduce_tensor_stats(
    op, init, reduce_indices::StableSet{IndexExpr}, stats::DCStats
)
    if length(reduce_indices) == 0
        return copy_stats(stats)
    end
    new_def = reduce_tensor_def(op, init, reduce_indices, get_def(stats))
    new_dcs = copy(stats.dcs)
    new_idx_2_int = copy(stats.idx_2_int)
    new_int_2_idx = copy(stats.int_2_idx)
    new_stats = DCStats(new_def, new_idx_2_int, new_int_2_idx, new_dcs)
    return new_stats
end

function transpose_tensor_stats(index_order::Vector{IndexExpr}, stats::DCStats)
    stats = copy_stats(stats)
    stats.def = transpose_tensor_def(index_order, get_def(stats))
    return stats
end
