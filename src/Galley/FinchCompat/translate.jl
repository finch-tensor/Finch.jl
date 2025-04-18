function push_relabels(prgm)
    prgm = Rewrite(
        Fixpoint(
            Prewalk(
                Chain([
                    (@rule relabel(mapjoin(~op, ~args...), ~idxs...) => begin
                        idxs_2 = getfields(mapjoin(op, args...))
                        mapjoin(
                            op,
                            map(arg -> relabel(reorder(arg, idxs_2...), idxs...), args)...,
                        )
                    end),
                    (@rule relabel(aggregate(~op, ~init, ~arg, ~idxs1...), ~idxs2...) =>
                        begin
                            arg_idxs = Finch.getfields(arg)
                            arg_relabel_idxs = filter((i) -> i ∉ idxs1, arg_idxs)
                            relabel_dict = merge(
                                OrderedDict(i => i for i in idxs1),
                                OrderedDict(
                                    arg_relabel_idxs[i] => idxs2[i] for
                                    i in eachindex(idxs2)
                                ),
                            )
                            aggregate(
                                op,
                                init,
                                relabel(arg, [relabel_dict[i] for i in arg_idxs]...),
                                idxs1...,
                            )
                        end),
                    (@rule relabel(relabel(~arg, ~idxs...), ~idxs_2...) =>
                        relabel(~arg, ~idxs_2...)),
                    (@rule relabel(reorder(~arg, ~idxs_1...), ~idxs_2...) => begin
                        idxs_3 = getfields(arg)
                        reidx = OrderedDict(map(Pair, idxs_1, idxs_2)...)
                        idxs_4 = map(idx -> get(reidx, idx, idx), idxs_3)
                        reorder(relabel(arg, idxs_4...), idxs_2...)
                    end),
                    (@rule relabel(table(~arg, ~idxs_1...), ~idxs_2...) => begin
                        table(arg, idxs_2...)
                    end),
                    (@rule relabel(~arg::isimmediate) => arg),
                ]),
            ),
        ),
    )(
        prgm
    )
end

function aggs_to_mapjoins(prgm)
    Rewrite(
        Fixpoint(
            Prewalk(
                Chain([
                    (@rule aggregate(~op, ~init, ~arg) => mapjoin(op, init, arg)),
                    (@rule mapjoin(overwrite, nothing, ~arg) => arg),
                ]),
            ),
        ),
    )(
        prgm
    )
end

function compatible_order(order1, order2)
    resulting_order = []
    cur_lpos = 1
    cur_rpos = 1
    while cur_lpos <= length(order1) && cur_rpos <= length(order2)
        if order1[cur_lpos].val == order2[cur_rpos].val
            push!(resulting_order, order1[cur_lpos])
            cur_lpos += 1
            cur_rpos += 1
        elseif order1[cur_lpos] ∉ order2
            push!(resulting_order, order1[cur_lpos])
            cur_lpos += 1
        else
            push!(resulting_order, order2[cur_rpos])
            cur_rpos += 1
        end
    end
    while cur_lpos <= length(order1)
        push!(resulting_order, order1[cur_lpos])
        cur_lpos += 1
    end
    while cur_rpos <= length(order2)
        push!(resulting_order, order2[cur_rpos])
        cur_rpos += 1
    end
    return resulting_order
end

function remove_reorders(prgm::LogicNode)
    queries = collect(prgm.bodies[1:(end - 1)])
    new_queries = []
    for q in queries
        expr = q.rhs
        format = nothing
        if operation(expr) == Finch.reformat
            format = expr.tns
            expr = expr.arg
        end
        output_idx_order = Finch.getfields(expr)
        expr = Rewrite(
            Fixpoint(
                Postwalk(
                    @rule reorder(~arg, ~idxs2...) =>
                        if length(idxs2) < length(getfields(arg))
                            reorder(
                                aggregate(
                                    overwrite, nothing, arg,
                                    setdiff(getfields(arg), idxs2)...,
                                ),
                                idxs2...,
                            )
                        end
                ),
            ),
        )(
            expr
        )
        bc_idxs = SortedSet()
        for n in PostOrderDFS(expr)
            if n.kind == reorder
                bc_idxs = bc_idxs ∪ setdiff(n.idxs, getfields(n.arg))
            end
        end
        table_idxs = SortedSet()
        for n in PostOrderDFS(expr)
            if n.kind == table
                table_idxs = table_idxs ∪ n.idxs
            end
        end
        bc_idxs = setdiff(bc_idxs, table_idxs)
        expr = Rewrite(
            Fixpoint(
                Postwalk(
                    Chain([
                        (@rule reorder(~arg, ~idxs1...) => arg),
                        (@rule aggregate(~op, ~init, ~arg, ~agg_idxs...) =>
                            aggregate(op, init, arg, setdiff(agg_idxs, bc_idxs)...))]),
                ),
            ),
        )(
            expr
        )
        expr = reorder(expr, output_idx_order...)
        if !isnothing(format)
            expr = reformat(format, expr)
        end
        push!(new_queries, query(q.lhs, expr))
    end
    prgm = plan(new_queries..., prgm.bodies[end])
    prgm
end

function unwrap_subqueries(prgm::LogicNode)
    Rewrite(Postwalk(Chain([@rule subquery(~lhs, ~rhs) => rhs])))(prgm)
end

function normalize_hl(prgm::LogicNode)
    # Currently, we just remove sub-query wrapping. This may introduce
    # common sub expressions.
    # TODO: If a sub-query is also produced, we shouldn't unwrap it. We should just refer to it.
    # However, the way query results are currently gensymed will mean that this never happens.
    prgm = unwrap_subqueries(prgm)
    prgm = flatten_plans(prgm)
    # We push all relabels down to the inputs
    prgm = push_relabels(prgm)
    # We remove any interior reorder statements and wrap the expression in one exterior reorder.
    # In particular, whenever a broadcast removes a size-1 dim, we introduce an aggregate. Whenever a
    # broadcast introduces a size-1 dim then aggregates it out later, we drop that dim from both.
    prgm = remove_reorders(prgm)
    # If an aggregate doesn't contain any idxs, we turn it into a mapjoin with this init value.
    prgm = aggs_to_mapjoins(prgm)
    return prgm
end

# Galley takes in a Finch Logic Program with the following form:
# pgrm := plan
# plan := query..., produces
# produces := symbols...
# query := reorder | reformat
# reformat := format..., reorder
# reorder := field..., expr
# expr := aggregate | mapjoin | alias | table
# table := symbol, field...
# aggregate := op, init, expr, field...
function finch_hl_to_galley(prgm::LogicNode)
    if prgm.kind == plan
        query_nodes = [q for q in prgm.bodies if q.kind == query]
        return PlanNode[finch_hl_to_galley(q) for q in query_nodes]
    elseif prgm.kind == produces
        return nothing # TODO: Change Galley to accept a produces list
    elseif prgm.kind == query
        rhs = finch_hl_to_galley(prgm.rhs)
        if rhs.kind != Materialize
            rhs = Mat([IndexExpr(i.val) for i in FinchLogic.getfields(prgm.rhs)]..., rhs)
        end
        lhs = finch_hl_to_galley(prgm.lhs)
        return Query(lhs, rhs)
    elseif prgm.kind == reformat
        formats = get_tensor_formats(prgm.tns.val)
        @assert prgm.arg.kind == reorder
        reorder_arg = prgm.arg
        idxs = [IndexExpr(i.val) for i in reorder_arg.idxs]
        return Mat(formats..., idxs..., finch_hl_to_galley(reorder_arg.arg))
    elseif prgm.kind == reorder
        idxs = [IndexExpr(i.val) for i in prgm.idxs]
        return Mat(idxs..., finch_hl_to_galley(prgm.arg))
    elseif prgm.kind == aggregate
        return Aggregate(finch_hl_to_galley(prgm.op),
            finch_hl_to_galley(prgm.init),
            [finch_hl_to_galley(i) for i in prgm.idxs]...,
            finch_hl_to_galley(prgm.arg))
    elseif prgm.kind == mapjoin
        return MapJoin(finch_hl_to_galley(prgm.op),
            [finch_hl_to_galley(arg) for arg in prgm.args]...)
    elseif prgm.kind == table
        if prgm.tns.kind == deferred
            if prgm.tns.imm isa Tensor
                return Input(
                    prgm.tns.imm,
                    [finch_hl_to_galley(i) for i in prgm.idxs]...,
                    string(prgm.tns.ex),
                )
            else
                return Input(
                    Tensor(prgm.tns.imm),
                    [finch_hl_to_galley(i) for i in prgm.idxs]...,
                    string(prgm.tns.ex),
                )
            end
        else
            if prgm.tns.val isa Tensor
                return Input(prgm.tns.val, [finch_hl_to_galley(i) for i in prgm.idxs]...)
            else
                return Input(
                    Tensor(prgm.tns.val), [finch_hl_to_galley(i) for i in prgm.idxs]...
                )
            end
        end
    elseif prgm.kind == relabel
        @assert prgm.arg.kind == alias
        return Alias(prgm.arg.val, [finch_hl_to_galley(i) for i in prgm.idxs]...)
    elseif prgm.kind == alias
        return Alias(prgm.val)
    elseif prgm.kind == field
        return IndexExpr(prgm.val)
    elseif prgm.kind == immediate
        return Value(prgm.val)
    else
        throw(
            error("Finch Logic statements of kind $(prgm.kind) is not supported by Galley.")
        )
    end
end
