
function push_labels(root)
    root = Rewrite(
        Postwalk(
            Chain([
                (@rule relabel(~arg, ~idxs...) =>
                    reorder(relabel(~arg, ~idxs...), idxs...)),
                #TODO: This statement sets the loop order, which should probably be done with more forethought as a separate step before concordization.
                (@rule mapjoin(~args...) =>
                    reorder(mapjoin(args...), getfields(mapjoin(args...))...)),
            ]),
        ),
    )(
        root
    )

    root = Rewrite(
        Fixpoint(
            Prewalk(
                Chain([
                    (@rule reorder(mapjoin(~op, ~args...), ~idxs...) =>
                        mapjoin(op, map(arg -> reorder(arg, ~idxs...), args)...)),
                    (@rule relabel(mapjoin(~op, ~args...), ~idxs...) => begin
                        idxs_2 = getfields(mapjoin(op, args...))
                        mapjoin(
                            op,
                            map(arg -> relabel(reorder(arg, idxs_2...), idxs...), args)...,
                        )
                    end),
                    (@rule reorder(reorder(~arg, ~idxs...), ~idxs_2...) =>
                        reorder(~arg, ~idxs_2...)),
                    (@rule relabel(relabel(~arg, ~idxs...), ~idxs_2...) =>
                        relabel(~arg, ~idxs_2...)),
                    (@rule relabel(
                        reorder(relabel(~arg, ~idxs...), ~idxs_2...), ~idxs_3...
                    ) => begin
                        reidx = Dict(map(Pair, idxs_2, idxs_3)...)
                        idxs_4 = map(idx -> get(reidx, idx, idx), idxs)
                        reorder(relabel(arg, idxs_4...), idxs_3...)
                    end),
                ]),
            ),
        ),
    )(
        root
    )
end
#=
    root = Rewrite(Fixpoint(Postwalk(Chain([
        (@rule plan(~a1..., query(~b, relabel(~c, ~i...)), ~a2...) => begin
            d = alias(gensym(:A))
            bindings[d] = c
            rw = Rewrite(Postwalk(@rule b => relabel(d, i...)))
            plan(a1..., query(d, c), map(rw, a2)...)
        end),
        (@rule plan(~a1..., query(~b, reorder(~c, ~i...)), ~a2...) => begin
            d = alias(gensym(:A))
            bindings[d] = c
            rw = Rewrite(Postwalk(@rule b => reorder(d, i...)))
            plan(a1..., query(d, c), map(rw, a2)...)
        end),
        #(@rule reformat(~tns, relabel(~arg, ~idxs...)) => relabel(reformat(tns, arg), idxs...)),
    ]))))(root)
=#
#=
function pull_reorders(root, bindings)
    root = Rewrite(Fixpoint(Prewalk(Chain([
        (@rule mapjoin(~args...) => reorder(mapjoin(args...), getfields(mapjoin(args...), bindings)...))
    ]))))(root)
    root = Rewrite(Fixpoint(Prewalk(Chain([
        (@rule mapjoin(~a1..., reorder(~b, ~i...), ~a2...) => mapjoin(a1..., b, a2...)),
    ]))))(root)
end

function push_reorders(root, bindings)
    Rewrite(Fixpoint(Postwalk(Chain([
        (@rule plan(~a1..., query(~b, reorder(~c, ~i...)), ~a2...) => begin
            d = alias(gensym(:A))
            bindings[d] = c
            rw = Rewrite(Postwalk(@rule b => reorder(d, i...)))
            plan(a1..., query(d, c), map(rw, a2)...)
        end),
    ]))))(root)
end
=#

#=
function toposort(perms::Vector{Vector{T}}) where T
    graph = Dict{T, Set{T}}()
    for perm in perms
        i = nothing
        for j in perm
            if i != nothing
                push!(get!(graph, i, Set{T}()), j)
            end
            i = j
        end
    end
    #https://rosettacode.org/wiki/Topological_sort#Julia
    for (k, v) in graph
        delete!(v, k)
    end
    extraitems = setdiff(reduce(union, values(graph)), keys(graph))
    for item in extraitems
        graph[item] = Set{T}()
    end
    rst = Vector{T}()
    while true
        ordered = Set(item for (item, dep) in graph if isempty(dep))
        if isempty(ordered) break end
        append!(rst, ordered)
        graph = Dict{T,Set{T}}(item => setdiff(dep, ordered) for (item, dep) in graph if item ∉ ordered)
    end
    if isempty(graph)
        return rst
    else
        # a cyclic dependency exists amongst $(keys(graph))
        return nothing
    end
end

function heuristic_loop_order(node)
    perms = Vector{LogicNode}[]
    for node in PostOrderDFS(node)
        if @capture node relabel(~arg, ~idxs...)
            push!(perms, idxs)
        end
    end
    sort!(perms, by=length)
    return something(toposort(perms), getfields(node))
end

heuristic_pointwise_loop_order(node) = getfields(node)
heuristic_aggregate_loop_order(node) = getfields(node)

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
function set_loop_order(prgm, fields = Dict())
    if @capture node plan(~stmts..., produces(~args...))
        stmts = map(stmts) do stmt
            set_loop_order(stmt, fields)
        end
        plan(stmts..., produces(args...))
    elseif @capture node query(~lhs, table(~tns, ~idxs...))
        node
    elseif @capture node query(~lhs, aggregate(~op, ~init, ~arg, ~idxs...))
        idxs_2 = heuristic_aggregate_loop_order(arg, idxs)
        rhs = aggregate(op, init, reorder(arg, idxs_2...), idxs...)
        fields[lhs] = getfields(rhs)
        query(lhs, rhs)
    elseif @capture node query(~lhs, reformat(~tns, aggregate(~op, ~init, ~arg, ~idxs...)))
        idxs_2 = heuristic_aggregate_loop_order(arg, idxs)
        rhs = reformat(tns, aggregate(op, init, reorder(arg, idxs_2...), idxs...))
        fields[lhs] = getfields(rhs)
        query(lhs, rhs)
    elseif @capture node query(~lhs, reformat(~tns, ~arg))
        idxs = heuristic_pointwise_loop_order(arg)
        rhs = reformat(tns, reorder(arg, idxs...))
        fields[lhs] = getfields(rhs)
        query(lhs, rhs)
    elseif @capture node query(~lhs, ~rhs)
        idxs = heuristic_pointwise_loop_order(rhs)
        rhs = reorder(rhs, idxs...)
        fields[lhs] = getfields(rhs)
        query(lhs, reorder(rhs, idxs...))
    else
        throw(ArgumentError("Unrecognized program in set_loop_order"))
    end
end
=#
#=
"""
propagate_loop_order(root)

Heuristically chooses a total order for all loops in the program by inserting
`reorder` statments inside reformat, query, and aggregate nodes. This function
attempts to preserve the ordering of inputs wherever possible.
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
function propagate_loop_order(node::LogicNode, bindings = Dict{LogicNode, Any}())
    if @capture node plan(~stmts..., produces(~args...))
        stmts = map(stmts) do stmt
            propagate_loop_order(stmt, fields)
        end
        plan(stmts..., produces(args...))
    elseif @capture node query(~lhs, table(~tns, ~idxs...))
        bindings[lhs] = idxs
        node
    elseif @capture node query(~lhs, reformat(~tns, ~rhs))
        bindings[lhs] = fields(rhs)

        node
    elseif @capture node query(~lhs, ~rhs)
        rhs = propagate_fields(rhs, fields)
        fields[lhs] = getfields(rhs, Dict())
        query(lhs, rhs)
    elseif @capture node relabel(alias, ~idxs...)
        node
    elseif isalias(node)
        relabel(node, fields[node]...)
    elseif istree(node)
        similarterm(node, operation(node), map(x -> propagate_fields(x, fields), arguments(node)))
    else
        node
    end
end

function propagate_loop_order_helper(node)
    orders = Dict()
    for node in PostOrderDFS(node)
        if @capture node reorder(~arg, ~idxs...)
            push!(orders, intersect(idxs, getfields(arg))) #only the nontrivial indices need to be ordered
            setdiff(idxs, getfields(arg))
        end
    end
    #attempt to construct a total ordering of the arguments
    #this is a topo sort
    #if we can't, nbd
    reorder(node, total_idxs...)
end
=#

"""
compute(args..., ctx = ctx) =

FinchInterpreter(DefaultOptimizer())(prgm)
FinchInterpreter(DefaultNormalizer(DefaultHeuristic()))(prgm)
FinchInterpreter(DefaultNormalizer())(prgm)
FinchCompiler(DefaultNormalizer())(prgm)
FinchExecutor(FinchCompiler(DefaultNormalizer()))(prgm)
FinchExecutor(ChipCacher(FinchCompiler(DefaultNormalizer())))(prgm)
FinchExecutor(ChipCacher(FinchCompiler(DefaultNormalizer())))(prgm)
"""

#=
function hash_structure(node::LogicNode, s, cache=IDDict(), names=Dict())
    get!(cache, node) do
        if istree(node)
            s = hash(node.kind, s)
            for arg in arguments(node)
                s = hash_structure(arg, s)
            end
            s
        elseif node.kind === table
            s = hash(table, s)
            s = hash(typeof(node.tbl.val), s)
            for idx in node.idxs
                s = hash_structure(idx, s)
            end
            s
        elseif node.kind === alias || node.kind === index
            node
        end
    end
end
sha256(reinterpret(UInt8, [4]))
bytes(x) = reinterpret(NTuple{sizeof(x), UInt8}, x)
import SHA
import Serialization
bytes(x) = reinterpret(NTuple{sizeof(x), UInt8}, x)
function update_structure_hash!(ctx, a::LogicNode, names=Dict{LogicNode,Int}())
    #delimit each hash blob with the kind of the node
    SHA.update!(ctx, bytes(Int(a.kind)))
    if a.kind === field || a.kind === alias
        SHA.update!(ctx, bytes(get!(names, a, length(names))))
    elseif a.kind === immediate
        buffer = IOBuffer()
        Serialization.serialize(buffer, a.val)
        SHA.update!(ctx, take!(buffer))
    elseif a.kind === subquery
        if haskey(names, a.lhs)
            SHA.update!(ctx, a.lhs)
        else
            names[a.lhs] = length(names)
            update_structure_hash!(ctx, a.lhs)
            update_structure_hash!(ctx, a.arg)
        end
    elseif a.kind === table
        update_structure_hash!(ctx, immediate(typeof(a.tns.val)))
        for idx in a.idxs
            update_structure_hash!(ctx, idx)
        end
    elseif istree(a)
        for child in a.children
            update_structure_hash!(ctx, child)
        end
    else
        error("unimplemented")
    end
    #terminate each hash with an unused node kind
    SHA.update!(ctx, bytes(typemax(Int)))
end

structure_hash(node::LogicNode) = begin
    ctx = SHA.SHA1_CTX()
    update_structure_hash!(ctx, node)
    SHA.digest!(ctx)
end

import SHA
import Serialization
bytes(x) = reinterpret(NTuple{sizeof(x), UInt8}, x)

function isequal_structure(a::LogicStructure, b::LogicNode, names::Dict{LogicNode,Int} = Dict{LogicNode,Int}())
    if a.kind !== b.kind
        return false
    end
    if a.kind === field || a.kind === alias
        return get!(names, a, length(names)) === get!(names, b, length(names))
    end
    elseif a.kind === field || a.kind === alias
        return get!(names, a, length(names)) === get!(names, b, length(names))
    end
    if a.kind === field || a.kind === alias
        s = hash(get!(names, a, length(names)), s)
    elseif a.kind === immediate
        s = hash(a.val, s)
    elseif a.kind === subquery
        if haskey(names, a.lhs)
            s = hash_structure(a.lhs, s)
        else
            names[a.lhs] = length(names)
            s = hash_structure(a.lhs, s)
            s = hash_structure(a.arg, s)
        end
    elseif a.kind === table
        s = hash(immediate(typeof(a.tns.val)), s)
        for idx in a.idxs
            s = hash_structure(idx, s)
        end
    elseif istree(a)
        for child in a.children
            s = hash_structure(child, s)
        end
    else
        error("unimplemented")
    end
    #terminate each hash with an unused node kind
    return hash(typemax(Int), s)
end

struct LogicStructure
    node::LogicNode
    reps::Dict{LogicNode,LogicNode}
end

function LogicStructure(node::LogicNode)
    names = Dict{LogicNode,Int}()
    reps = IDDict{LogicNode,LogicNode}()
    for node in PostOrderDFS(node)
        if node.kind === alias
            reps[node] = get!(names, node, alias(Symbol(:A_, length(names))))
        elseif node.kind === field
            reps[node] = get!(names, node, field(Symbol(:A_, length(names))))
        elseif node.kind === table
            reps[node] = get!(names, node, immediate(Symbol(:A_, length(names))))
        end
    end
end

Base.hash(s::LogicStructure, h::UInt) = hash_structure(s.node, h)

function hash_structure(a::LogicNode, s::UInt, names::Dict{LogicNode,Int} = Dict{LogicNode,Int}())
    #delimit each hash blob with the kind of the node
    s = hash(Int(a.kind), s)
    if a.kind === field || a.kind === alias
        s = hash(get!(names, a, length(names)), s)
    elseif a.kind === immediate
        s = hash(a.val, s)
    elseif a.kind === subquery
        if haskey(names, a.lhs)
            s = hash_structure(a.lhs, s)
        else
            names[a.lhs] = length(names)
            s = hash_structure(a.lhs, s)
            s = hash_structure(a.arg, s)
        end
    elseif a.kind === table
        s = hash(immediate(typeof(a.tns.val)), s)
        for idx in a.idxs
            s = hash_structure(idx, s)
        end
    elseif istree(a)
        for child in a.children
            s = hash_structure(child, s)
        end
    else
        error("unimplemented")
    end
    #terminate each hash with an unused node kind
    return hash(typemax(Int), s)
end
=#
