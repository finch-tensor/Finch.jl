flatten_plans = Rewrite(
    Postwalk(
        Fixpoint(
            Chain([
                (@rule plan(~s1..., plan(~s2...), ~s3...) => plan(s1, s2, s3)),
                (@rule plan(~s1..., produces(~p...), ~s2...) => plan(s1, produces(p))),
            ]),
        ),
    ),
)

isolate_aggregates = Rewrite(
    Postwalk(
        @rule aggregate(~op, ~init, ~arg, ~idxs...) => begin
            name = alias(gensym(:A))
            subquery(name, aggregate(~op, ~init, ~arg, ~idxs...))
        end
    ),
)

isolate_reformats = Rewrite(
    Postwalk(
        @rule reformat(~tns, ~arg) => begin
            name = alias(gensym(:A))
            subquery(name, reformat(tns, arg))
        end
    ),
)

isolate_tables = Rewrite(
    Postwalk(
        @rule table(~tns, ~idxs...) => begin
            name = alias(gensym(:A))
            subquery(name, table(tns, idxs...))
        end
    ),
)

function lift_subqueries_expr(node::LogicNode, bindings)
    if node.kind === subquery
        if !haskey(bindings, node.lhs)
            arg_2 = lift_subqueries_expr(node.arg, bindings)
            bindings[node.lhs] = arg_2
        end
        node.lhs
    elseif istree(node)
        similarterm(
            node,
            operation(node),
            map(n -> lift_subqueries_expr(n, bindings), arguments(node)),
        )
    else
        node
    end
end

"""
    lift_subqueries

Creates a plan that lifts all subqueries to the top level of the program, with
unique queries for each distinct subquery alias. This function processes the rhs
of each subquery once, to carefully extract SSA form from any nested pointer
structure. After calling lift_subqueries, it is safe to map over the program
(recursive pointers to subquery structures will not incur exponential overhead).
"""
function lift_subqueries(node::LogicNode)
    if node.kind === plan
        plan(map(lift_subqueries, node.bodies))
    elseif node.kind === query
        bindings = OrderedDict()
        rhs_2 = lift_subqueries_expr(node.rhs, bindings)
        plan(
            map(((lhs, rhs),) -> query(lhs, rhs), collect(bindings)), query(node.lhs, rhs_2)
        )
    elseif node.kind === produces
        node
    else
        error()
    end
end

function pretty_labels(root)
    fields = Dict()
    aliases = Dict()
    Rewrite(
        Postwalk(
            Chain([
                (@rule ~i::isfield => get!(fields, i, field(Symbol(:i, length(fields))))),
                (@rule ~a::isalias => get!(aliases, a, alias(Symbol(:A, length(aliases))))),
            ]),
        ),
    )(
        root
    )
end

"""
push_fields(node)

This program modifies all `EXPR` statements in the program, as
defined in the following grammar:
```
    LEAF := relabel(ALIAS, FIELD...) |
            table(IMMEDIATE, FIELD...) |
            IMMEDIATE
    EXPR := LEAF |
            reorder(EXPR, FIELD...) |
            relabel(EXPR, FIELD...) |
            mapjoin(IMMEDIATE, EXPR...) |
            aggregate(IMMEDIATE, IMMEDIATE, EXPR, FIELD...)
```
Pushes all reorder and relabel statements down to LEAF nodes of each EXPR.
Output LEAF nodes will match the form `reorder(relabel(LEAF, FIELD...),
FIELD...)`, omitting reorder or relabel if not present as an ancestor of the
LEAF in the original EXPR. Tables and immediates will absorb relabels.
Copies of reorders are left in place, but relabels are removed.
"""
function push_fields(root)
    root = Rewrite(
        Prewalk(
            Fixpoint(
                Chain([
                    (@rule relabel(mapjoin(~op, ~args...), ~idxs...) => begin
                        reidx = Dict(map(Pair, getfields(mapjoin(op, args...)), idxs)...)
                        mapjoin(
                            op,
                            map(
                                arg -> relabel(
                                    arg, map(idx -> reidx[idx], getfields(arg))...
                                ),
                                args,
                            )...,
                        )
                    end),
                    (@rule relabel(aggregate(~op, ~init, ~arg, ~idxs...), ~idxs_2...) =>
                        begin
                            idxs_3 = setdiff(getfields(arg), idxs)
                            reidx = Dict(map(Pair, idxs_3, idxs_2)...)
                            idxs_4 = map(idx -> get(reidx, idx, idx), getfields(arg))
                            aggregate(op, init, relabel(arg, idxs_4...), idxs...)
                        end),
                    (@rule relabel(relabel(~arg, ~idxs...), ~idxs_2...) =>
                        relabel(~arg, ~idxs_2...)),
                    (@rule relabel(reorder(~arg, ~idxs_1...), ~idxs_2...) => begin
                        idxs_3 = getfields(arg)
                        reidx = Dict(map(Pair, idxs_1, idxs_2)...)
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
        root
    )

    root = Rewrite(
        Prewalk(
            Chain([
                Fixpoint(
                    @rule reorder(reorder(~arg, ~idxs...), ~idxs_2...) =>
                        reorder(~arg, ~idxs_2...)
                ),
                (@rule reorder(mapjoin(~op, ~args...), ~idxs...) =>
                    reorder(
                        mapjoin(
                            op,
                            map(
                                arg -> reorder(arg, intersect(idxs, getfields(arg))...),
                                args,
                            )...,
                        ),
                        idxs...,
                    )),
                (@rule reorder(aggregate(~op, ~init, ~arg, ~idxs...), ~idxs_2...) =>
                    if !issubsequence(intersect(getfields(arg), idxs_2), idxs_2)
                        reorder(
                            aggregate(
                                op,
                                init,
                                reorder(arg, withsubsequence(idxs_2, getfields(arg))...),
                                idxs...,
                            ), idxs_2...)
                    end),
            ]),
        ),
    )(
        root
    )
    root
end

"""
propagate_transpose_queries(root)

Removes non-materializing permutation queries by propagating them to the
expressions they contain. Pushes fields and also removes copies. Removes queries of the form:
```
    query(ALIAS, reorder(relabel(ALIAS, FIELD...), FIELD...))
```

Does not remove queries which define production aliases.

Accepts programs of the form:
```
       TABLE  := table(IMMEDIATE, FIELD...)
       ACCESS := reorder(relabel(ALIAS, FIELD...), FIELD...)
    POINTWISE := ACCESS | mapjoin(IMMEDIATE, POINTWISE...) | reorder(IMMEDIATE, FIELD...) | IMMEDIATE
    MAPREDUCE := POINTWISE | aggregate(IMMEDIATE, IMMEDIATE, POINTWISE, FIELD...)
  INPUT_QUERY := query(ALIAS, TABLE)
COMPUTE_QUERY := query(ALIAS, reformat(IMMEDIATE, MAPREDUCE)) | query(ALIAS, MAPREDUCE))
         PLAN := plan(STEP...)
         STEP := COMPUTE_QUERY | INPUT_QUERY | PLAN | produces((ALIAS | ACCESS)...)
         ROOT := STEP
```
"""
function propagate_transpose_queries(node, bindings=Dict{LogicNode,LogicNode}())
    if @capture node plan(~stmts...)
        stmts = map(stmts) do stmt
            propagate_transpose_queries(stmt, bindings)
        end
        plan(stmts...)
    elseif @capture node query(~lhs, ~rhs)
        rhs = push_fields(Rewrite(Postwalk((node) -> get(bindings, node, node)))(rhs))
        if @capture rhs reorder(relabel(~tns::isalias, ~idxs_1...), ~idxs_2...)
            bindings[lhs] = rhs
            plan()
        elseif @capture rhs relabel(~tns::isalias, ~idxs_1...)
            bindings[lhs] = rhs
            plan()
        elseif isalias(rhs)
            bindings[lhs] = rhs
            plan()
        else
            query(lhs, rhs)
        end
    elseif @capture node produces(~args...)
        push_fields(Rewrite(Postwalk((node) -> get(bindings, node, node)))(node))
    else
        throw(ArgumentError("Unrecognized program in propagate_transpose_queries"))
    end
end

"""
materialize_squeeze_expand_productions(root)

Makes separate kernels for squeeze and expand operations in produces statements,
since swizzle does not support this.
"""
function materialize_squeeze_expand_productions(root)
    Rewrite(
        Postwalk(
            @rule produces(~args...) => begin
                preamble = []
                args_2 = map(args) do arg
                    if (@capture arg reorder(relabel(~tns::isalias, ~idxs_1...), ~idxs_2...)) && Set(idxs_1) != Set(idxs_2)
                        tns_2 = alias(gensym(:A))
                        idxs_3 = withsubsequence(intersect(idxs_1, idxs_2), idxs_2)
                        push!(
                            preamble,
                            query(tns_2, reorder(relabel(tns, idxs_1), idxs_3)),
                        )
                        if idxs_3 == idxs_2
                            tns_2
                        else
                            reorder(relabel(tns_2, idxs_3), idxs_2)
                        end
                    elseif @capture(arg, reorder(relabel(~tns, ~i...), ~i...))
                        tns
                    else
                        arg
                    end
                end
                plan(preamble, produces(args_2))
            end
        ),
    )(
        root
    )
end

function propagate_copy_queries(root)
    copies = Dict()
    for node in PostOrderDFS(root)
        if @capture node query(~a, ~b::isalias)
            copies[a] = get(copies, b, b)
        end
    end
    Rewrite(
        Postwalk(Chain([
            (a -> get(copies, a, nothing)),
            (@rule query(~a, ~a) => plan()),
        ])),
    )(
        root
    )
end

"""
This one is a placeholder that places reorder statements inside aggregate and mapjoin query nodes.
only works on the output of propagate_fields(push_fields(prgm))
"""
function lift_fields(prgm)
    Rewrite(
        Postwalk(
            Chain([
                (@rule aggregate(~op, ~init, ~arg, ~idxs_1...) => begin
                    idxs_2 = getfields(arg)
                    aggregate(op, init, reorder(arg, idxs_2...), idxs_1...)
                end),
                (@rule query(~lhs, ~rhs) => if rhs.kind === mapjoin
                    idxs = getfields(rhs)
                    query(lhs, reorder(rhs, idxs...))
                end),
                (@rule query(~lhs, reformat(~tns, ~arg)) => if arg.kind === mapjoin
                    idxs = getfields(arg)
                    query(lhs, reformat(tns, reorder(arg, idxs...)))
                end),
            ]),
        ),
    )(
        prgm
    )
end

pad_labels = Rewrite(
    Postwalk(
        @rule relabel(~arg, ~idxs...) => reorder(relabel(~arg, ~idxs...), idxs...)
    ),
)

function propagate_into_reformats(root)
    Rewrite(
        Postwalk(
            Chain([
                (@rule plan(
                    ~a1..., query(~b, ~c), ~a2..., query(~d, reformat(~tns, ~b)), ~a3...
                ) => begin
                    if !(b in PostOrderDFS(plan(a2..., a3...))) && (
                        c.kind === mapjoin || c.kind === aggregate || c.kind === reorder
                    )
                        plan(a1..., query(d, reformat(tns, c)), a2..., a3...)
                    end
                end),
            ]),
        ),
    )(
        root
    )
end

function issubsequence(a, b)
    a = collect(a)
    b = collect(b)
    return issubset(a, b) && intersect(b, a) == a
end

function withsubsequence(a, b)
    a = collect(a)
    b = collect(b)
    view(b, findall(idx -> idx in a, b)) .= a
    b
end

"""
    concordize(root)

Accepts a program of the following form:

```
        TABLE := table(IMMEDIATE, FIELD...)
       ACCESS := reorder(relabel(ALIAS, FIELD...), FIELD...)
      COMPUTE := ACCESS |
                 mapjoin(IMMEDIATE, COMPUTE...) |
                 aggregate(IMMEDIATE, IMMEDIATE, COMPUTE, FIELD...) |
                 reformat(IMMEDIATE, COMPUTE) |
                 IMMEDIATE
COMPUTE_QUERY := query(ALIAS, COMPUTE)
  INPUT_QUERY := query(ALIAS, TABLE)
         STEP := COMPUTE_QUERY | INPUT_QUERY | produces((ALIAS | ACCESS)...)
         ROOT := PLAN(STEP...)
```

Inserts permutation statements of the form `query(ALIAS, reorder(ALIAS,
FIELD...))` and updates `relabel`s so that
they match their containing `reorder`s. Modified `ACCESS` statements match the form:

```
ACCESS := reorder(relabel(ALIAS, idxs_1::FIELD...), idxs_2::FIELD...) where issubsequence(idxs_1, idxs_2)
```
"""
function concordize(root)
    needed_swizzles = Dict()
    spc = Namespace()
    map(
        node -> freshen(spc, node.name),
        unique(filter(Or(isfield, isalias), collect(PostOrderDFS(root)))),
    )
    #Exempt productions from swizzling
    root = flatten_plans(root)
    @assert @capture root plan(~s..., produces(~p...))
    root = plan(s)
    #Collect the needed swizzles
    root = Rewrite(
        Postwalk(
            @rule reorder(relabel(~a::isalias, ~idxs_1...), ~idxs_2...) => begin
                idxs_3 = intersect(idxs_1, idxs_2)
                if !issubsequence(idxs_3, idxs_2)
                    idxs_4 = withsubsequence(intersect(idxs_2, idxs_1), idxs_1)
                    perm = map(idx -> findfirst(isequal(idx), idxs_1), idxs_4)
                    reorder(
                        relabel(
                            get!(
                                get!(needed_swizzles, a, OrderedDict()),
                                perm,
                                alias(freshen(spc, a.name)),
                            ),
                            idxs_4,
                        ),
                        idxs_2...,
                    )
                end
            end
        ),
    )(
        root
    )
    #Insert the swizzles
    root = Rewrite(
        Postwalk(
            Chain([
                (@rule query(~a, ~b) => begin
                    if haskey(needed_swizzles, a)
                        idxs = getfields(b)
                        swizzle_queries =
                            map(collect(needed_swizzles[a])) do (perm, c)
                                query(c, reorder(relabel(a, idxs...), idxs[perm]...))
                            end
                        plan(query(a, b), swizzle_queries...)
                    end
                end),
            ]),
        ),
    )(
        root
    )
    root = flatten_plans(plan(root, produces(p)))
end

drop_noisy_reorders = Rewrite(
    Postwalk(
        @rule reorder(relabel(~arg, ~idxs...), ~idxs...) => relabel(arg, idxs...)
    ),
)

function format_queries(node::LogicNode, defer=false, bindings=Dict())
    if @capture node plan(~stmts...)
        stmts = map(stmts) do stmt
            format_queries(stmt, defer, bindings)
        end
        plan(stmts...)
    elseif (@capture node query(~lhs, ~rhs)) && rhs.kind !== reformat && rhs.kind !== table
        rep = SuitableRep(bindings)(rhs)
        bindings[lhs] = rep
        if defer
            tns = deferred(fiber_ctr(rep), typeof(rep_construct(rep)))
        else
            tns = immediate(rep_construct(rep))
        end
        query(lhs, reformat(tns, rhs))
    elseif @capture node query(~lhs, ~rhs)
        bindings[lhs] = SuitableRep(bindings)(rhs)
        node
    else
        node
    end
end

function dropdims_rep(rep, dims)
    rep = aggregate_rep(
        initwrite(fill_value(rep)),
        fill_value(rep),
        rep,
        dims,
    )
end
struct SuitableRep
    bindings::Dict
end
function (ctx::SuitableRep)(ex)
    res = if ex.kind === alias
        ctx.bindings[ex]
    elseif @capture ex table(~tns::isimmediate, ~idxs...)
        data_rep(ex.tns.val)
    elseif @capture ex table(~tns::isdeferred, ~idxs...)
        data_rep(ex.tns.type)
    elseif @capture ex reorder(mapjoin(~op, ~args...), ~idxs...)
        idxs_2 = toposort(vcat(map(collect, map(getfields, args)), [collect(idxs)]))
        @assert idxs_2 !== nothing
        reps = map(args) do arg
            rep = ctx(arg)
            dims = findall(idx -> !(idx in getfields(arg)), idxs)
            expanddims_rep(rep, dims)
        end
        rep = map_rep(op.val, reps...)
        dropdims_rep(rep, findall(idx -> !(idx in idxs), idxs_2))
    elseif ex.kind === aggregate
        idxs = getfields(ex.arg)
        aggregate_rep(
            ex.op.val,
            ex.init.val,
            ctx(ex.arg),
            map(idx -> findfirst(isequal(idx), idxs), ex.idxs),
        )
    elseif ex.kind === reorder
        rep = ctx(ex.arg)
        idxs = getfields(ex.arg)
        #first drop dimensions
        rep = dropdims_rep(rep, findall(idx -> idx in setdiff(idxs, ex.idxs), idxs))
        #then permute remaining dimensions to match
        perm = sortperm(
            intersect(idxs, ex.idxs); by=idx -> findfirst(isequal(idx), ex.idxs)
        )
        rep = permutedims_rep(rep, perm)
        #then add new dimensions
        expanddims_rep(rep, findall(idx -> !(idx in idxs), ex.idxs))
    elseif ex.kind === relabel
        ctx(ex.arg)
    elseif ex.kind === reformat
        data_rep(ex.tns.val)
    elseif ex.kind === immediate
        ElementData(ex.val, typeof(ex.val))
    else
        error("Unrecognized expression: $(ex.kind)")
    end
    try
        if ndims(res) != length(getfields(ex))
            display(ex)
            println(res)
            println()
        end
    catch
    end
    res
end

function propagate_map_queries(root)
    root = Rewrite(Postwalk(@rule aggregate(~op, ~init, ~arg) => mapjoin(op, init, arg)))(
        root
    )
    rets = getproductions(root)
    props = Dict()
    for node in PostOrderDFS(root)
        if @capture node query(~a, mapjoin(~op, ~args...))
            if !(a in rets)
                props[a] = mapjoin(op, args...)
            end
        end
    end
    root = Rewrite(
        Prewalk(
            Chain([
                (a -> if haskey(props, a)
                    props[a]
                end),
                (@rule query(~a, ~b) => if haskey(props, a)
                    plan()
                end),
            ]),
        ),
    )(
        root
    )
    Rewrite(Postwalk(@rule plan(~a1..., plan(), ~a2...) => plan(a1..., a2...)))(root)
end

function propagate_map_queries_backward(root)
    root = Rewrite(Postwalk(@rule aggregate(~op, ~init, ~arg) => mapjoin(op, init, arg)))(
        root
    )
    uses = Dict{LogicNode,Int}()
    defs = Dict{LogicNode,LogicNode}()
    rets = getproductions(root)
    for node in PostOrderDFS(root)
        if node.kind === alias
            uses[node] = get(uses, node, 0) + 1
        elseif @capture node query(~a, ~b)
            uses[a] = get(uses, a, 0) - 1
            defs[a] = b
        end
    end
    root = Rewrite(
        Prewalk(
            Chain([
                (@rule query(~a, ~b) => if uses[a] == 1 && !(a in rets)
                    plan()
                end),
                (@rule ~a => if get(uses, a, 0) == 1 && !(a in rets)
                    get(defs, a, a)
                end),
            ]),
        ),
    )(
        root
    )
    root = push_fields(root)
    root = Rewrite(
        Fixpoint(
            Prewalk(
                Chain([
                    (@rule mapjoin(
                        ~f::isimmediate,
                        ~a1...,
                        aggregate(~g::isimmediate, ~init::isimmediate, ~arg, ~idxs...),
                        ~a2...,
                    ) => begin
                        if isdistributive(DefaultAlgebra(), literal(f.val), literal(g.val)) &&
                            isannihilator(
                                DefaultAlgebra(), literal(f.val), literal(init.val)
                            ) &&
                            length(getfields(aggregate(g, init, arg, idxs...))) ==
                            length(getfields(mapjoin(f, a1..., a2...)))
                            aggregate(g, init, mapjoin(f, a1..., arg, a2...), idxs...)
                        end
                    end),
                    (@rule aggregate(~op::isimmediate, ~init::isimmediate,
                        aggregate(~op, ~init_2, ~arg, ~idxs...),
                        ~idxs_2...,
                    ) => begin
                        if isidentity(
                            DefaultAlgebra(), literal(op.val), literal(init_2.val)
                        )
                            aggregate(op, init, arg, idxs..., idxs_2...)
                        end
                    end),
                    (@rule reorder(aggregate(~op, ~init, ~arg, ~idxs...), ~idxs_2...) =>
                        aggregate(
                            op,
                            init,
                            reorder(arg, vcat(idxs_2, idxs)...),
                            idxs...,
                        )
                ),
                ]),
            ),
        ),
    )(
        root
    )
    root
end

function normalize_names(ex)
    spc = Namespace()
    scope = Dict()
    normname(sym) =
        get!(scope, sym) do
            if isgensym(sym)
                sym = gensymname(sym)
            end
            freshen(spc, sym)
        end
    ex = Rewrite(Postwalk(@rule ~a::isalias => alias(normname(a.name))))(ex)
    Rewrite(Postwalk(@rule ~a::isfield => field(normname(a.name))))(ex)
end

function toposort(chains::Vector{Vector{T}}) where {T}
    chains = filter(!isempty, deepcopy(chains))
    parents = Dict{T,Int}(map(chain -> first(chain) => 0, chains))
    for chain in chains, u in chain[2:end]
        parents[u] = get(parents, u, 0) + 1
    end
    roots = filter(u -> parents[u] == 0, keys(parents))
    perm = []
    while !isempty(parents)
        isempty(roots) && return nothing
        push!(perm, pop!(roots))
        for chain in chains
            if !isempty(chain) && first(chain) == last(perm)
                popfirst!(chain)
                if !isempty(chain)
                    parents[first(chain)] -= 1
                    if parents[first(chain)] == 0
                        push!(roots, first(chain))
                    end
                end
            end
        end
        pop!(parents, last(perm))
    end
    return perm
end

function heuristic_loop_order(root, reps)
    chains = Vector{LogicNode}[]
    for node in PostOrderDFS(root)
        if @capture node reorder(relabel(~arg, ~idxs...), ~idxs_2...)
            push!(chains, intersect(idxs, idxs_2, getfields(root)))
        end
    end
    for idx in getfields(root)
        push!(chains, [idx])
    end
    res = something(toposort(chains), getfields(root))
    if mapreduce(length, max, chains; init=0) < length(unique(reduce(vcat, chains)))
        counts = Dict()
        for chain in chains
            for idx in chain
                counts[idx] = get(counts, idx, 0) + 1
            end
        end
        sort!(res; by=idx -> counts[idx] == 1, alg=Base.MergeSort)
    end
    return res
end

"""
set_loop_order(root)

Heuristically chooses a total order for all loops in the program by inserting
`reorder` statments inside reformat, query, and aggregate nodes.

Accepts programs of the form:
```
      REORDER := reorder(relabel(ALIAS, FIELD...), FIELD...)
       ACCESS := reorder(relabel(ALIAS, idxs_1::FIELD...), idxs_2::FIELD...) where issubsequence(idxs_1, idxs_2)
    POINTWISE := ACCESS | mapjoin(IMMEDIATE, POINTWISE...) | reorder(IMMEDIATE, FIELD...) | IMMEDIATE
    MAPREDUCE := POINTWISE | aggregate(IMMEDIATE, IMMEDIATE, POINTWISE, FIELD...)
       TABLE  := table(IMMEDIATE, FIELD...)
COMPUTE_QUERY := query(ALIAS, reformat(IMMEDIATE, arg::(REORDER | MAPREDUCE)))
  INPUT_QUERY := query(ALIAS, TABLE)
         STEP := COMPUTE_QUERY | INPUT_QUERY
         ROOT := PLAN(STEP..., produces(ALIAS...))
```
"""
function set_loop_order(node, perms=Dict(), reps=Dict())
    if @capture node plan(~stmts...)
        stmts = map(stmts) do stmt
            set_loop_order(stmt, perms, reps)
        end
        plan(stmts...)
    elseif @capture node query(~lhs, reformat(~tns, ~rhs::isalias))
        rhs_2 = perms[rhs]
        perms[lhs] = lhs
        reps[lhs] = SuitableRep(reps)(rhs_2)
        query(lhs, reformat(tns, rhs_2))
    elseif @capture node query(~lhs, reformat(~tns, ~rhs))
        arg = alias(gensym(:A))
        set_loop_order(plan(
                query(A, rhs),
                query(lhs, reformat(tns, A)),
            ), perms, reps)
    elseif @capture node query(~lhs, table(~tns, ~idxs...))
        reps[lhs] = SuitableRep(reps)(node.rhs)
        perms[lhs] = lhs
        node
    elseif @capture node query(~lhs, aggregate(~op, ~init, ~arg, ~idxs...))
        arg = push_fields(Rewrite(Postwalk(tns -> get(perms, tns, tns)))(arg))
        idxs_2 = heuristic_loop_order(arg, reps)
        rhs_2 = aggregate(op, init, reorder(arg, idxs_2...), idxs...)
        reps[lhs] = SuitableRep(reps)(rhs_2)
        perms[lhs] = reorder(relabel(lhs, getfields(rhs_2)), getfields(node.rhs))
        query(lhs, rhs_2)
    elseif @capture node query(
        ~lhs, reorder(relabel(~tns::isalias, ~idxs_1...), ~idxs_2...)
    )
        tns = get(perms, tns, tns)
        reps[lhs] = SuitableRep(reps)(node.rhs)
        perms[lhs] = lhs
        node
    elseif @capture node query(~lhs, ~rhs)
        #assuming rhs is a bunch of mapjoins
        rhs = push_fields(Rewrite(Postwalk(tns -> get(perms, tns, tns)))(rhs))
        idxs = heuristic_loop_order(rhs, reps)
        rhs_2 = reorder(rhs, idxs...)
        reps[lhs] = SuitableRep(reps)(rhs_2)
        perms[lhs] = reorder(relabel(lhs, idxs), getfields(rhs))
        query(lhs, rhs_2)
    elseif @capture node produces(~args...)
        Rewrite(Postwalk(tns -> get(perms, tns, tns)))(node)
    else
        throw(ArgumentError("Unrecognized program in set_loop_order"))
    end
end

function optimize(prgm)
    #deduplicate and lift inline subqueries to regular queries
    prgm = lift_subqueries(prgm)

    #At this point in the program, all statements should be unique, so
    #it is okay to name different occurences of things.

    prgm = propagate_map_queries_backward(prgm)

    #these steps lift reformat, aggregate, and table nodes into separate
    #queries, using subqueries as temporaries.
    prgm = isolate_reformats(prgm)
    prgm = isolate_aggregates(prgm)
    prgm = isolate_tables(prgm)
    prgm = lift_subqueries(prgm)

    #I shouldn't use gensym but I do, so this cleans up the names
    prgm = pretty_labels(prgm)

    #These steps fuse copy, permutation, and mapjoin statements
    #into later expressions.
    #Only reformat statements preserve intermediate breaks in computation
    prgm = propagate_fields(prgm)
    prgm = propagate_copy_queries(prgm)
    prgm = propagate_transpose_queries(prgm)
    prgm = propagate_map_queries(prgm)

    #These steps assign a global loop order to each statement.
    prgm = propagate_fields(prgm)
    prgm = push_fields(prgm)
    prgm = lift_fields(prgm)
    prgm = push_fields(prgm)

    prgm = propagate_transpose_queries(prgm)
    prgm = set_loop_order(prgm)
    prgm = push_fields(prgm)

    #After we have a global loop order, we concordize the program
    prgm = concordize(prgm)

    prgm = materialize_squeeze_expand_productions(prgm)
    prgm = propagate_copy_queries(prgm)

    #Add reformat statements where there aren't any
    prgm = propagate_into_reformats(prgm)
    prgm = propagate_copy_queries(prgm)

    #Normalize names for caching
    prgm = normalize_names(prgm)
    prgm
end

"""
    DefaultLogicOptimizer(ctx)

The default optimizer for finch logic programs. Optimizes to a structure
suitable for the LogicCompiler or LogicInterpreter, then calls `ctx` on the
resulting program.
"""
struct DefaultLogicOptimizer
    ctx
end

Base.:(==)(a::DefaultLogicOptimizer, b::DefaultLogicOptimizer) = a.ctx == b.ctx
Base.hash(a::DefaultLogicOptimizer, h::UInt) = hash(DefaultLogicOptimizer, hash(a.ctx, h))

function (ctx::DefaultLogicOptimizer)(prgm)
    prgm = optimize(prgm)
    ctx.ctx(prgm)
end

function set_options(ctx::DefaultLogicOptimizer; kwargs...)
    DefaultLogicOptimizer(set_options(ctx.ctx; kwargs...))
end
