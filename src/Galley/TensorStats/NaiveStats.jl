#################  NaiveStats Definition ###################################################

@auto_hash_equals mutable struct NaiveStats <: TensorStats
    def::TensorDef
    cardinality::Float64

    function NaiveStats(def::TensorDef, cardinality)
        return new(def, cardinality)
    end

    function NaiveStats(tensor, indices)
        if !(tensor isa Tensor)
            tensor = Tensor(tensor)
        end
        def = TensorDef(tensor, indices)
        cardinality = countstored(tensor)
        return new(def, cardinality)
    end
end

get_def(stat::NaiveStats) = stat.def
function get_cannonical_stats(stat::NaiveStats, rel_granularity=4)
    NaiveStats(copy_def(stat.def), geometric_round(rel_granularity, stat.cardinality))
end

# This function assumes that stat1 and stat2 have ONLY differ in the value of their DCs.
function issimilar(stat1::NaiveStats, stat2::NaiveStats, rel_granularity)
    return abs(
        log(rel_granularity, stat1.cardinality) - log(rel_granularity, stat2.cardinality)
    ) <= 1
end

function estimate_nnz(
    stat::NaiveStats; indices=get_index_set(stat),
    conditional_indices=StableSet{IndexExpr}(),
)
    return stat.cardinality / get_dim_space_size(stat, conditional_indices)
end

condense_stats!(::NaiveStats; timeout=100000, cheap=true) = nothing
function fix_cardinality!(stat::NaiveStats, card)
    stat.cardinality = card
end

copy_stats(stat::NaiveStats) = NaiveStats(copy_def(stat.def), stat.cardinality)

function NaiveStats(index_set, dim_sizes, cardinality, fill_val)
    NaiveStats(TensorDef(index_set, dim_sizes, fill_val, nothing), cardinality)
end

function NaiveStats(x)
    def = TensorDef(
        StableSet{IndexExpr}(), OrderedDict{IndexExpr,Int}(), x, nothing, nothing, nothing
    )
    return NaiveStats(def, 1)
end
function reindex_stats(stat::NaiveStats, indices)
    return NaiveStats(reindex_def(indices, stat.def), stat.cardinality)
end

function set_fill_value!(stat::NaiveStats, fill_val)
    return NaiveStats(set_fill_value!(stat.def, fill_val), stat.cardinality)
end

function relabel_index!(stats::NaiveStats, i::IndexExpr, j::IndexExpr)
    relabel_index!(stats.def, i, j)
end

function add_dummy_idx!(stats::NaiveStats, i::IndexExpr; idx_pos=-1)
    add_dummy_idx!(stats.def, i; idx_pos=idx_pos)
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
