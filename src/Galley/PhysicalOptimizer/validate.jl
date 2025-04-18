
# This function does a variety of sanity checks on the kernel before we attempt to execute it.
# Such as:
#  1. Check that the loop order is a permutation of the input indices
#  2. Check that the output indices are the inputs minus any that are aggregate_indices
#  3. Check that the inputs are all sorted w.r.t. the loop order

function get_input_indices(n::PlanNode)
    return if n.kind == Input
        get_index_set(n.stats)
    elseif n.kind == Alias
        get_index_set(n.stats)
    elseif n.kind == Value
        StableSet{IndexExpr}()
    elseif n.kind == Aggregate
        get_input_indices(n.arg)
    elseif n.kind == MapJoin
        union([get_input_indices(input) for input in n.args]...)
    elseif n.kind == Materialize
        get_input_indices(n.expr)
    end
end

function get_output_indices(n::PlanNode)
    return if n.kind == Input
        get_index_set(n.stats)
    elseif n.kind == Alias
        get_index_set(n.stats)
    elseif n.kind == Aggregate
        setdiff(get_input_indices(n.arg), n.idxs)
    elseif n.kind == MapJoin
        union([get_input_indices(input) for input in n.args]...)
    elseif n.kind == Materialize
        get_input_indices(n.expr)
    end
end

function get_input_indices_and_dims(n::PlanNode)
    return if n.kind == Input
        [(idx, get_dim_size(n.stats, idx)) for idx in get_index_set(n.stats)]
    elseif n.kind == Alias
        [(idx, get_dim_size(n.stats, idx)) for idx in get_index_set(n.stats)]
    elseif n.kind == Value
        []
    elseif n.kind == Aggregate
        union([(idx, get_dim_size(n.stats, idx)) for idx in get_index_set(n.stats)],
            get_input_indices_and_dims(n.arg))
    elseif n.kind == MapJoin
        union([(idx, get_dim_size(n.stats, idx)) for idx in get_index_set(n.stats)],
            [get_input_indices_and_dims(input) for input in n.args]...)
    elseif n.kind == Materialize
        union([(idx, get_dim_size(n.stats, idx)) for idx in get_index_set(n.stats)],
            get_input_indices_and_dims(n.expr))
    end
end

function check_sorted_inputs(n::PlanNode, loop_order)
    return if n.kind == Input
        @assert is_sorted_wrt_index_order(
            [idx.name for idx in n.idxs], loop_order; loop_order=true
        )
    elseif n.kind == Alias
        @assert is_sorted_wrt_index_order(
            get_index_order(n.stats), loop_order; loop_order=true
        )
    elseif n.kind == Aggregate
        check_sorted_inputs(n.arg, loop_order)
    elseif n.kind == MapJoin
        for arg in n.args
            check_sorted_inputs(arg, loop_order)
        end
    elseif n.kind == Materialize
        check_sorted_inputs(n.expr, loop_order)
    end
end

function check_protocols(n::PlanNode)
    return if n.kind == Input
        @assert !isnothing(get_index_protocols(n.stats))
        @assert length(get_index_set(n.stats)) == length(get_index_protocols(n.stats))
    elseif n.kind == Alias
        @assert !isnothing(get_index_protocols(n.stats))
        @assert length(get_index_set(n.stats)) == length(get_index_protocols(n.stats))
    elseif n.kind == Aggregate
        check_protocols(n.arg)
    elseif n.kind == MapJoin
        for arg in n.args
            check_protocols(arg)
        end
    elseif n.kind == Materialize
        check_protocols(n.expr)
    end
end

function check_formats(n::PlanNode)
    return if n.kind == Input
        @assert !isnothing(get_index_formats(n.stats))
        @assert length(get_index_set(n.stats)) == length(get_index_formats(n.stats))
    elseif n.kind == Alias
        @assert !isnothing(get_index_formats(n.stats))
        @assert length(get_index_set(n.stats)) == length(get_index_formats(n.stats))
    elseif n.kind == Aggregate
        check_formats(n.arg)
    elseif n.kind == MapJoin
        for arg in n.args
            check_formats(arg)
        end
    elseif n.kind == Materialize
        check_formats(n.expr)
    end
end

function validate_physical_query(q::PlanNode)
    q = plan_copy(q)
    input_indices = get_input_indices(q.expr)
    indices_and_dims = get_input_indices_and_dims(q.expr)
    idx_dim = OrderedDict()
    for (idx, dim) in indices_and_dims
        if !haskey(idx_dim, idx)
            idx_dim[idx] = dim
        end
        @assert idx_dim[idx] == dim "idx:$idx dim:$dim query:$q "
    end
    output_indices = StableSet([idx.name for idx in q.expr.idx_order])
    @assert input_indices ∪ output_indices == StableSet([idx.name for idx in q.loop_order])
    @assert StableSet(output_indices) == StableSet([idx.name for idx in q.expr.idx_order])
    check_sorted_inputs(q.expr, [idx.name for idx in q.loop_order])
    check_protocols(q.expr)
    check_formats(q.expr)
end
