
function merge_mapjoins(plan::PlanNode)
    Rewrite(
        Postwalk(
            Chain([
                (@rule MapJoin(~f, ~a..., MapJoin(~f, ~b...), ~c...) =>
                    MapJoin(f, a..., b..., c...) where {isassociative(f.val)}),
                (@rule MapJoin(~f, ~a..., MapJoin(~f, ~b...)) =>
                    MapJoin(f, a..., b...) where {isassociative(f.val)}),
                (@rule MapJoin(~f, MapJoin(~f, ~a...), ~b...) =>
                    MapJoin(f, a..., b...) where {isassociative(f.val)}),
                (@rule Aggregate(
                    ~f, ~init, ~idxs1..., Aggregate(~f, ~init, ~idxs2..., ~a)
                ) => Aggregate(f, init, idxs1..., idxs2..., a)),
            ]),
        ),
    )(
        plan
    )
end

function relabel_index(n::PlanNode, i::IndexExpr, j::IndexExpr)
    for node in PostOrderDFS(n)
        if node.kind == Index && node.name == i
            node.name = j
        end
        if !isnothing(node.stats)
            relabel_index!(node.stats, i, j)
        end
    end
end

# In cannonical form, two aggregates in the plan shouldn't reduce out the same variable.
# E.g. MapJoin(*, Aggregate(+, Input(tns, i, j)))
function unique_indices(scope_dict::Dict{IndexExpr,IndexExpr}, n::PlanNode, counter)
    if n.kind === Plan
        return Plan(
            [unique_indices(scope_dict, query, counter) for query in n.queries]...,
            n.outputs,
        )
    elseif n.kind === Query
        return Query(n.name, unique_indices(scope_dict, n.expr, counter))
    elseif n.kind === Materialize
        return Materialize(
            n.formats..., n.idx_order..., unique_indices(scope_dict, n.expr, counter)
        )
    elseif n.kind === MapJoin
        return MapJoin(
            n.op, [unique_indices(scope_dict, arg, counter) for arg in n.args]...
        )
    elseif n.kind === Input
        return relabel_input(
            n, [unique_indices(scope_dict, idx, counter).name for idx in n.idxs]...
        )
    elseif n.kind === Aggregate
        new_idxs = []
        for idx in n.idxs
            old_idx = idx.val
            new_idx =
                haskey(scope_dict, old_idx) ? Symbol("$(idx.val)_$(counter[1])") : idx.val
            if new_idx != idx.val
                counter[1] += 1
            end
            push!(new_idxs, new_idx)
            scope_dict[old_idx] = new_idx
        end
        return Aggregate(
            n.op, n.init, new_idxs..., unique_indices(scope_dict, n.arg, counter)
        )
    elseif n.kind === Index
        return Index(get(scope_dict, n.name, n.name))
    else
        return n
    end
end

function _insert_statistics!(
    ST, expr::PlanNode; bindings=Dict{IndexExpr,TensorStats}(), replace=false
)
    if expr.kind === MapJoin
        expr.stats = merge_tensor_stats(expr.op.val, ST[arg.stats for arg in expr.args]...)
    elseif expr.kind === Aggregate
        expr.stats = reduce_tensor_stats(
            expr.op.val,
            expr.init.val,
            Set{IndexExpr}([idx.name for idx in expr.idxs]),
            expr.arg.stats,
        )
    elseif expr.kind === Materialize
        expr.stats = copy_stats(expr.expr.stats)
        def = get_def(expr.stats)
        def.level_formats = [f.val for f in expr.formats]
        def.index_order = [idx.name for idx in expr.idx_order]
        for idx in def.index_order
            if idx ∉ get_index_set(expr.expr.stats)
                add_dummy_idx!(expr.stats, idx)
            end
        end
    elseif expr.kind === Alias
        if haskey(bindings, expr.name)
            expr.stats = bindings[expr.name]
        end
        if !isnothing(expr.stats)
            idxs = [idx.name for idx in expr.idxs]
            stats_order = get_index_order(expr.stats)
            @assert length(idxs) == 0 || !isnothing(stats_order)
            if !isempty(idxs) && stats_order != idxs
                expr.stats = reindex_stats(expr.stats, idxs)
            end
        end
    elseif expr.kind === Input
        if isnothing(expr.stats) || replace
            expr.stats = ST(expr.tns.val, IndexExpr[idx.val for idx in expr.idxs])
        end
    elseif expr.kind === Value
        expr.stats = ST(expr.val)
    end
end

# Often, we will only have changed a small part of the expression, e.g. by performing a
# reduction, so we only update the stats objects which were involved with those indices.
function insert_statistics!(
    ST,
    plan::PlanNode;
    bindings=Dict{IndexExpr,TensorStats}(),
    replace=false,
    reduce_idx=nothing,
)
    check_reduce_idxs = !isnothing(reduce_idx)
    for expr in PostOrderDFS(plan)
        if check_reduce_idxs && !isnothing(expr.stats) &&
            reduce_idx ∉ get_index_set(expr.stats)
            continue
        end
        _insert_statistics!(ST, expr; bindings=bindings, replace=replace)
    end
end

# This function labels every node with an id. These ids respect a topological ordering where
# children have id's that are larger than parents.
function insert_node_ids!(plan::PlanNode)
    cur_id = 1
    for expr in PreOrderDFS(plan)
        expr.node_id = cur_id
        cur_id += 1
    end
end

function distribute_mapjoins(plan::PlanNode, use_dnf)
    if use_dnf
        Rewrite(
            Fixpoint(
                Postwalk(
                    Chain([
                        (@rule MapJoin(
                            ~f, ~a..., Aggregate(~g, ~init, ~idxs..., ~arg), ~c...
                        ) => Aggregate(
                            g, init, idxs..., MapJoin(f, a..., arg, c...)
                        ) where {isdistributive(f.val, g.val)}),
                        (@rule MapJoin(~f, ~a..., Aggregate(~g, ~init, ~idxs..., ~arg)) =>
                            Aggregate(
                                g, init, idxs..., MapJoin(f, a..., arg)
                            ) where {isdistributive(f.val, g.val)}),
                        (@rule MapJoin(~f, Aggregate(~g, ~init, ~idxs..., ~arg), ~b...) =>
                            Aggregate(
                                g, init, idxs..., MapJoin(f, arg, b...)
                            ) where {isdistributive(f.val, g.val)}),
                        (@rule MapJoin(~f, ~x..., MapJoin(~g, ~args...)) => MapJoin(
                            g, [MapJoin(f, x..., arg) for arg in args]...
                        ) where {isdistributive(f.val, g.val)}),
                        (@rule MapJoin(~f, MapJoin(~g, ~args...), ~x...) => MapJoin(
                            g, [MapJoin(f, arg, x...) for arg in args]...
                        ) where {isdistributive(f.val, g.val)}),
                        (@rule MapJoin(~f, ~x..., MapJoin(~g, ~args...), ~y...) =>
                            MapJoin(
                                g, [MapJoin(f, x..., arg, y...) for arg in args]...
                            ) where {isdistributive(f.val, g.val)})]),
                ),
            ),
        )(
            plan
        )
    else
        Rewrite(
            Fixpoint(
                Postwalk(
                    Chain([
                        (@rule MapJoin(
                            ~f, ~a..., Aggregate(~g, ~init, ~idxs..., ~arg), ~c...
                        ) => Aggregate(
                            g, init, idxs..., MapJoin(f, a..., arg, c...)
                        ) where {isdistributive(f.val, g.val)}),
                        (@rule MapJoin(~f, ~a..., Aggregate(~g, ~init, ~idxs..., ~arg)) =>
                            Aggregate(
                                g, init, idxs..., MapJoin(f, a..., arg)
                            ) where {isdistributive(f.val, g.val)}),
                        (@rule MapJoin(~f, Aggregate(~g, ~init, ~idxs..., ~arg), ~b...) =>
                            Aggregate(
                                g, init, idxs..., MapJoin(f, arg, b...)
                            ) where {isdistributive(f.val, g.val)})]),
                ),
            ),
        )(
            plan
        )
    end
end

function remove_extraneous_mapjoins(plan::PlanNode)
    Rewrite(
        Fixpoint(
            Postwalk(
                Chain([
                    (@rule MapJoin(~f, ~x..., ~v) =>
                        v where {(v.kind == Value && isannihilator(f.val, v.val))}),
                    (@rule MapJoin(~f, ~v, ~x...) =>
                        v where {(v.kind == Value && isannihilator(f.val, v.val))}),
                    (@rule MapJoin(~f, ~x..., ~v, ~y...) =>
                        v where {(v.kind == Value && isannihilator(f.val, v.val))}),
                    (@rule MapJoin(~f, ~x) => x where {(isunarynull(f.val))})]),
            ),
        ),
    )(
        plan
    )
end

function canonicalize(plan::PlanNode, use_dnf)
    counter = [1]
    plan = unique_indices(Dict{IndexExpr,IndexExpr}(), plan, counter)
    plan = merge_mapjoins(plan)
    plan = distribute_mapjoins(plan, use_dnf)
    plan = remove_extraneous_mapjoins(plan)
    plan = merge_mapjoins(plan)
    plan = distribute_mapjoins(plan, use_dnf)
    plan = merge_mapjoins(plan)
    # Each aggregate should correspond to a unique variable, which we ensure here.
    plan = unique_indices(Dict{IndexExpr,IndexExpr}(), plan, counter)
    # Sometimes rewrites will cause an implicit DAG, so we recopy the plan to avoid overwriting
    # later on.
    plan = plan_copy(plan)
    insert_node_ids!(plan)
    return plan
end

gen_alias_name(hash) = Symbol("A_$hash")
gen_idx_name(count::Int) = Symbol("i_$count")

function cannonical_hash(plan::PlanNode, alias_hash)
    plan = plan_copy(plan; copy_statistics=false)
    idx_translate_dict = Dict{IndexExpr,IndexExpr}()
    for n in PostOrderDFS(plan)
        if n.kind === Index
            if !haskey(idx_translate_dict, n.name)
                idx_translate_dict[n.name] = gen_idx_name(length(idx_translate_dict))
            end
            n.name = idx_translate_dict[n.name]
        elseif n.kind === Alias
            n.name = gen_alias_name(alias_hash[n.name])
        end
    end
    return hash(plan)
end
