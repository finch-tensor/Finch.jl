# To reduce compilation overhead, we try and roughly cannonicalize the inputs
# to operator_expr by sorting them.
function sort_mapjoin_args(args)
    immediate_args = [arg for arg in args if arg.kind == Input || arg.kind == Alias]
    remainder = [arg for arg in args if !(arg.kind == Input || arg.kind == Alias)]
    perm = sortperm([
        (
            length(get_index_order(arg.stats)),
            get_index_formats(arg.stats)...,
            get_index_protocols(arg.stats)...,
            get_index_order(arg.stats),
        ) for arg in immediate_args
    ])
    return [immediate_args[perm]..., remainder...]
end

function initialize_access(
    tensor_id::Symbol,
    tensor,
    index_ids,
    protocols,
    index_sym_dict;
    read=true,
    cannonicalize=true,
    virtual_tns=false,
)
    if !isnothing(tensor) && isbits(tensor)
        return literal_instance(tensor)
    end

    mode = read ? reader_instance() : updater_instance(auto)
    index_expressions = []
    for i in range(1, length(index_ids))
        index = if cannonicalize
            if !haskey(index_sym_dict, index_ids[i])
                idx_num = length(index_sym_dict)
                index_sym_dict[index_ids[i]] = get_index_symbol(idx_num)
            end
            index_instance(index_sym_dict[index_ids[i]])
        else
            index_instance(index_ids[i])
        end
        if read == true
            if protocols[i] == t_walk
                index = call_instance(literal_instance(walk), index)
            elseif protocols[i] == t_gallop
                index = call_instance(literal_instance(gallop), index)
            elseif protocols[i] == t_lead
                index = call_instance(literal_instance(lead), index)
            elseif protocols[i] == t_follow
                index = call_instance(literal_instance(follow), index)
            end
        end
        push!(index_expressions, index)
    end
    tensor_var = variable_instance(tensor_id)
    if virtual_tns
        return access_instance(tensor_var, mode, index_expressions...)
    else
        tensor_tag = tag_instance(tensor_var, tensor)
        tensor_access = access_instance(tensor_tag, mode, index_expressions...)
        return tensor_access
    end
end

function translate_rhs(
    alias_dict,
    tensor_counter,
    index_sym_dict,
    rhs::PlanNode,
    cannonicalize::Bool,
    virtual_aliases,
)
    if rhs.kind == Alias
        idxs = get_index_order(rhs.stats)
        protocols = [get_index_protocol(rhs.stats, idx) for idx in idxs]
        t_name = if cannonicalize
            tensor_counter[1] += 1
            get_tensor_symbol(tensor_counter[1])
        else
            rhs.name
        end
        if virtual_aliases
            return initialize_access(
                t_name,
                nothing,
                idxs,
                protocols,
                index_sym_dict;
                read=true,
                cannonicalize=cannonicalize,
                virtual_tns=virtual_aliases,
            )
        else
            tns = alias_dict[rhs.name]
            @assert all([
                get_dim_size(rhs.stats, idxs[i]) == size(tns)[i] for i in eachindex(idxs)
            ]) "$(size(tns)) $(idxs) $([(X, Int64(x)) for (X,x) in rhs.stats.def.dim_sizes])"
            return initialize_access(
                t_name,
                tns,
                idxs,
                protocols,
                index_sym_dict;
                read=true,
                cannonicalize=cannonicalize,
            )
        end

    elseif rhs.kind === Input
        idxs = get_index_order(rhs.stats)
        protocols = [get_index_protocol(rhs.stats, idx) for idx in idxs]
        t_name = if cannonicalize
            tensor_counter[1] += 1
            get_tensor_symbol(tensor_counter[1])
        else
            rhs.id
        end
        return initialize_access(
            t_name,
            rhs.tns.val,
            idxs,
            protocols,
            index_sym_dict;
            read=true,
            cannonicalize=cannonicalize,
        )
    elseif rhs.kind == Value
        return literal_instance(rhs.val)
    elseif rhs.kind === MapJoin
        if iscommutative(rhs.op.val)
            rhs.args = sort_mapjoin_args(rhs.args)
        end
        if is_binary(rhs.op.val)
            instance = translate_rhs(
                alias_dict,
                tensor_counter,
                index_sym_dict,
                rhs.args[1],
                cannonicalize,
                virtual_aliases,
            )
            for arg in rhs.args[2:end]
                instance = call_instance(
                    literal_instance(rhs.op.val),
                    translate_rhs(
                        alias_dict,
                        tensor_counter,
                        index_sym_dict,
                        arg,
                        cannonicalize,
                        virtual_aliases,
                    ),
                    instance,
                )
            end
            return instance
        else
            return call_instance(literal_instance(rhs.op.val),
                [
                    translate_rhs(
                        alias_dict,
                        tensor_counter,
                        index_sym_dict,
                        arg,
                        cannonicalize,
                        virtual_aliases,
                    ) for arg in rhs.args
                ]...)
        end
    else
        throw(
            ErrorException(
                "RHS expression cannot contain anything except Alias, Input, and MapJoin: $rhs"
            ),
        )
    end
end

# To be executed, a query must be in the following format:
# Query(name, Materialize(formats..., index_order..., Aggregate(op, idxs..., map_expr)))
# TODO: use loop_order to label indexes
function execute_query(alias_dict, q::PlanNode, verbose, cannonicalize, return_prgm)
    tensor_counter = [0]
    index_sym_dict = Dict{IndexExpr,IndexExpr}()
    name = q.name.name
    mat_expr = q.expr
    loop_order = [idx.name for idx in q.loop_order]
    output_formats = [f.val for f in mat_expr.formats]
    output_idx_order = [idx.name for idx in mat_expr.idx_order]
    agg_expr = mat_expr.expr
    output_default = get_default_value(agg_expr.stats)
    output_dimensions = [get_dim_size(mat_expr.stats, idx) for idx in output_idx_order]
    agg_op = agg_expr.op.val
    rhs_expr = agg_expr.arg
    rhs_instance = translate_rhs(
        alias_dict, tensor_counter, index_sym_dict, rhs_expr, cannonicalize, return_prgm
    )

    output_tensor = initialize_tensor(output_formats,
        output_dimensions,
        output_default)
    output_name = cannonicalize ? :output_tensor : name
    output_access = initialize_access(output_name,
        output_tensor,
        output_idx_order,
        [t_default for _ in output_idx_order],
        index_sym_dict;
        read=false,
        cannonicalize=cannonicalize)
    dec_instance = declare_instance(variable_instance(output_name),
        literal_instance(output_default), literal_instance(auto))

    prgm_instance = assign_instance(output_access, literal_instance(agg_op), rhs_instance)
    loop_order = [
        cannonicalize ? index_instance(index_sym_dict[i]) : index_instance(i) for
        i in loop_order
    ]
    for index in reverse(loop_order)
        prgm_instance = loop_instance(index, Auto(), prgm_instance)
    end
    prgm_instance = block_instance(dec_instance, prgm_instance)

    verbose >= 4 && display(prgm_instance)
    verbose >= 5 && println(
        Finch.unquote_literals(
            Finch.dataflow(
                Finch.unresolve(
                    Finch.pretty(
                        Finch.execute_code(
                            :ex, typeof(prgm_instance); mode=:fast
                        ),
                    ),
                ),
            ),
        ),
    )
    verbose >= 2 && println("Expected Output Size: $(estimate_nnz(agg_expr.stats))")

    if return_prgm
        output_tensor_init = tensor_initializer(
            output_formats, output_dimensions, output_default
        )
        return :($output_name = $output_tensor_init), prgm_instance
    end
    start_time = time()
    Finch.execute(prgm_instance; mode=:fast)
    verbose >= 2 && println("Kernel Execution Took: ", time() - start_time)
    verbose >= 2 && println("Stored Entries: ", count_stored(output_tensor))
    verbose >= 2 && println("Non Default Entries: ", count_non_default(output_tensor))
    alias_dict[name] = output_tensor
end

function execute_plan(cse_plan::PlanNode, verbose)
    alias_result = Dict{IndexExpr,Any}()
    for query in cse_plan.queries
        verbose > 2 && println("--------------- Computing: $(query.name) ---------------")
        verbose > 2 && println(query)
        verbose > 3 && validate_physical_query(query)
        execute_query(alias_result, query, verbose, true, false)
    end
    return alias_result
end

function get_execute_code(cse_plan::PlanNode, verbose)
    tensor_inits = []
    bodies = []
    for query in cse_plan.queries
        tensor_init, body = execute_query(Dict(), query, verbose, false, true)
        push!(tensor_inits, tensor_init)
        push!(bodies, body)
    end
    return tensor_inits, block_instance(bodies...)
end
