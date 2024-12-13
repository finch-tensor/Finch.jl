const IS_TREE = 1
const IS_STATEFUL = 2
const ID = 4

@enum LogicNodeKind begin
    immediate =  0ID
    deferred  =  1ID
    field     =  2ID
    alias     =  3ID
    table     =  4ID | IS_TREE
    mapjoin   =  5ID | IS_TREE
    aggregate =  6ID | IS_TREE
    reorder   =  7ID | IS_TREE
    relabel   =  8ID | IS_TREE
    reformat  =  9ID | IS_TREE
    subquery  = 10ID | IS_TREE
    query     = 11ID | IS_TREE | IS_STATEFUL
    produces  = 12ID | IS_TREE | IS_STATEFUL
    plan      = 13ID | IS_TREE | IS_STATEFUL
end

"""
    immediate(val)

Logical AST expression for the literal value `val`.
"""
immediate

"""
    deferred(ex, [type])

Logical AST expression for an expression `ex` of type `type`, yet to be evaluated.
"""
deferred

"""
    field(name)

Logical AST expression for an field named `name`.
"""
field

"""
    alias(name)

Logical AST expression for an alias named `name`.
"""
alias

"""
    table(tns, idxs...)

Logical AST expression for a tensor object `val`, indexed by fields `idxs...`.
"""
table

"""
    mapjoin(op, args...)

Logical AST expression for mapping the function `op` across `args...`.
The order of fields in the mapjoin is `unique(vcat(map(getfields, args)...))`
"""
mapjoin

"""
    aggregate(op, init, arg, idxs...)

Logical AST statement that reduces `arg` using `op`, starting with `init`.
`idxs` are the dimensions to reduce. May happen in any order.
"""
aggregate

"""
    reorder(arg, idxs...)

Logical AST statement that reorders the dimensions of `arg` to be `idxs...`.
Dimensions known to be length 1 may be dropped. Dimensions that do not exist in
`arg` may be added.
"""
reorder

"""
    relabel(arg, idxs...)

Logical AST statement that relabels the dimensions of `arg` to be `idxs...`
"""
relabel

"""
    reformat(tns, arg)

Logical AST statement that reformats `arg` into the tensor `tns`.
"""
reformat

"""
    subquery(lhs, arg)

Logical AST statement that evaluates `arg`, binding the result to `lhs`, and returns `arg`.
"""
subquery

"""
    query(lhs, rhs)

Logical AST statement that evaluates `rhs`, binding the result to `lhs`.
"""
query

"""
    produces(args...)

Logical AST statement that returns `args...` from the current plan. Halts
execution of the program.
"""
produces

"""
    plan(bodies...)

Logical AST statement that executes a sequence of statements `bodies...`.
"""
plan

"""
    LogicNode

A Finch Logic IR node. Finch uses a variant of Concrete Field Notation as an
intermediate representation.

The LogicNode struct represents many different Finch IR nodes. The nodes are
differentiated by a `FinchLogic.LogicNodeKind` enum.
"""
mutable struct LogicNode
    kind::LogicNodeKind
    val::Any
    type::Type
    children::Vector{LogicNode}
end

"""
    isimmediate(node)

Returns true if the node is a finch immediate
"""
isimmediate(ex::LogicNode) = ex.kind === immediate

"""
    isdeferred(node)

Returns true if the node is a finch immediate
"""
isdeferred(ex::LogicNode) = ex.kind === deferred

"""
    isalias(node)

Returns true if the node is a finch alias
"""
isalias(ex::LogicNode) = ex.kind === alias

"""
    isfield(node)

Returns true if the node is a finch field
"""
isfield(ex::LogicNode) = ex.kind === field

isstateful(node::LogicNode) = Int(node.kind) & IS_STATEFUL != 0
SyntaxInterface.istree(node::LogicNode) = Int(node.kind) & IS_TREE != 0
AbstractTrees.children(node::LogicNode) = node.children
SyntaxInterface.arguments(node::LogicNode) = node.children
SyntaxInterface.operation(node::LogicNode) = node.kind

function SyntaxInterface.similarterm(::Type{LogicNode}, op::LogicNodeKind, args)
    @assert Int(op) & IS_TREE != 0
    LogicNode(op, nothing, Any, args)
end

function LogicNode_concatenate_args(args)
    n_args = 0
    for arg in args
        if arg isa AbstractArray
            n_args += length(arg)
        else
            n_args += 1
        end
    end
    args_2 = Vector{LogicNode}(undef, n_args)
    i = 0
    for arg in args
        if arg isa AbstractArray
            for arg_2 in arg
                args_2[i += 1] = arg_2
            end
        else
            args_2[i += 1] = arg
        end
    end
    args_2
end

function LogicNode(kind::LogicNodeKind, args::Vector)
    if (kind === immediate || kind === field || kind === alias || kind === deferred) && length(args) == 1
        return LogicNode(kind, args[1], Any, LogicNode[])
    elseif kind === deferred && length(args) == 2
        return LogicNode(kind, args[1], args[2], LogicNode[])
    elseif kind === deferred && length(args) == 3
        return LogicNode(kind, (args[1], args[3]), args[2], LogicNode[])
    else
        args = LogicNode_concatenate_args(args)
        if (kind === table && length(args) >= 1) ||
            (kind === mapjoin && length(args) >= 1) ||
            (kind === aggregate && length(args) >= 3) ||
            (kind === reorder && length(args) >= 1) ||
            (kind === relabel && length(args) >= 1) ||
            (kind === reformat && length(args) == 2) ||
            (kind === subquery && length(args) == 2) ||
            (kind === query && length(args) == 2) ||
            (kind === produces) ||
            (kind === plan) 
            return LogicNode(kind, nothing, Any, args)
        else
            error("wrong number of arguments to $kind(...)")
        end
    end
end

function (kind::LogicNodeKind)(args...)
    LogicNode(kind, Any[args...,])
end

function Base.getproperty(node::LogicNode, sym::Symbol)
    if sym === :kind || sym === :val || sym === :type || sym === :children
        return Base.getfield(node, sym)
    elseif node.kind === deferred && sym === :ex node.val isa Tuple ? node.val[1] : node.val
    elseif node.kind === deferred && sym === :imm node.val[2]
    elseif node.kind === field && sym === :name node.val::Symbol
    elseif node.kind === alias && sym === :name node.val::Symbol
    elseif node.kind === table && sym === :tns node.children[1]
    elseif node.kind === table && sym === :idxs @view node.children[2:end]
    elseif node.kind === mapjoin && sym === :op node.children[1]
    elseif node.kind === mapjoin && sym === :args @view node.children[2:end]
    elseif node.kind === aggregate && sym === :op node.children[1]
    elseif node.kind === aggregate && sym === :init node.children[2]
    elseif node.kind === aggregate && sym === :arg node.children[3]
    elseif node.kind === aggregate && sym === :idxs @view node.children[4:end]
    elseif node.kind === reorder && sym === :arg node.children[1]
    elseif node.kind === reorder && sym === :idxs @view node.children[2:end]
    elseif node.kind === relabel && sym === :arg node.children[1]
    elseif node.kind === relabel && sym === :idxs @view node.children[2:end]
    elseif node.kind === reformat && sym === :tns node.children[1]
    elseif node.kind === reformat && sym === :arg node.children[2]
    elseif node.kind === subquery && sym === :lhs node.children[1]
    elseif node.kind === subquery && sym === :arg node.children[2]
    elseif node.kind === query && sym === :lhs node.children[1]
    elseif node.kind === query && sym === :rhs node.children[2]
    elseif node.kind === produces && sym === :args node.children
    elseif node.kind === plan && sym === :bodies node.children
    else
        error("type LogicNode($(node.kind), ...) has no property $sym")
    end
end

function Base.show(io::IO, node::LogicNode)
    if node.kind === immediate || node.kind === field || node.kind === alias
        print(io, node.kind, "(", node.val, ")")
    elseif node.kind === deferred
        print(io, node.kind, "(", node.val, ", ", node.type, ")")
    else
        print(io, node.kind, "("); join(io, node.children, ", "); print(io, ")")
    end
end

function Base.show(io::IO, mime::MIME"text/plain", node::LogicNode)
    print(io, "Finch Logic: ")
    try
        if isstateful(node)
            display_statement(io, mime, node, 0)
        else
            display_expression(io, mime, node)
        end
    catch
        println(io, "error showing: ", node)
        rethrow()
    end
end

function display_statement(io, mime, node, indent)
    if operation(node) == query
        display_expression(io, mime, node.lhs)
        print(io, " = ")
        display_expression(io, mime, node.rhs)
    elseif operation(node) == plan
        println(io, "plan")
        for body in node.bodies
            print(io, " " ^ (indent + 2))
            display_statement(io, mime, body, indent + 2)
            println(io)
        end
        print(io, " " ^ indent, "end")
    elseif operation(node) == produces
        print(io, "return (")
        for arg in node.args[1:end - 1]
            display_expression(io, mime, arg)
            print(io, ", ")
        end
        if length(node.args) > 0
            display_expression(io, mime, node.args[end])
        end
        print(io, ")")
    else
        throw(ArgumentError("Expected statement but got $(operation(node))"))
    end
end

function display_expression(io, mime, node)
    if operation(node) === immediate
        print(io, node.val)
    elseif operation(node) === deferred
        print(io, node.ex)
        print(io, "::")
        print(io, node.type)
    elseif operation(node) === field
        print(io, node.name)
    elseif operation(node) === alias
        print(io, node.name)
    elseif operation(node) == subquery
        print(io, "(")
        display_expression(io, mime, node.lhs)
        print(io, " = ")
        display_expression(io, mime, node.arg)
        print(io, ")")
    elseif istree(node)
        print(io, operation(node), "(")
        for child in node.children[1:end-1]
            display_expression(io, mime, child)
            print(io, ", ")
        end
        if length(node.children) > 0
            display_expression(io, mime, node.children[end])
        end
        print(io, ")")
    else
        throw(ArgumentError("Expected expression but got $(operation(node))"))
    end
end

function Base.:(==)(a::LogicNode, b::LogicNode)
    if a.kind === value
        return b.kind === value && a.val == b.val && a.type === b.type
    elseif a.kind === immediate
        return b.kind === immediate && a.val === b.val
    elseif a.kind === deferred
        return b.kind === deferred && a.val === b.val && a.type === b.type
    elseif a.kind === field
        return b.kind === field && a.name == b.name
    elseif a.kind === alias
        return b.kind === alias && a.name == b.name
    elseif istree(a)
        return a.kind === b.kind && a.children == b.children
    else
        error("unimplemented")
    end
end

function Base.hash(a::LogicNode, h::UInt)
    if a.kind === immediate || a.kind === field || a.kind === alias
        return hash(a.kind, hash(a.val, h))
    elseif istree(a)
        return hash(a.kind, hash(a.children, h))
    elseif a.kind === deferred
        return hash(a.kind, hash(a.val, hash(a.type, h)))
    else
        error("unimplemented")
    end
end

"""
    logic_leaf(x)

Return a terminal finch node wrapper around `x`. A convenience function to
determine whether `x` should be understood by default as a immediate or value.
"""
logic_leaf(arg) = immediate(arg)
logic_leaf(arg::Type) = immediate(arg)
logic_leaf(arg::Function) = immediate(arg)
logic_leaf(arg::LogicNode) = arg

Base.convert(::Type{LogicNode}, x) = logic_leaf(x)
Base.convert(::Type{LogicNode}, x::LogicNode) = x

#overload RewriteTools pattern constructor so we don't need
#to wrap leaf nodes.
finch_pattern(arg) = logic_leaf(arg)
finch_pattern(arg::RewriteTools.Slot) = arg
finch_pattern(arg::RewriteTools.Segment) = arg
finch_pattern(arg::RewriteTools.Term) = arg
function RewriteTools.term(f::LogicNodeKind, args...; type = nothing)
    RewriteTools.Term(f, [finch_pattern.(args)...])
end

function getfields(node::LogicNode, bindings=Dict())
    if node.kind == field
        throw(ArgumentError("getfields($(node.kind)) is undefined"))
    elseif node.kind == immediate
        return []
    elseif node.kind == alias
        throw(ArgumentError("getfields(alias) is undefined, try calling `propagate_fields` on the whole plan to resolve alias fields."))
    elseif node.kind == table
        return node.idxs
    elseif node.kind == subquery
        getfields(node.arg, bindings)
    elseif node.kind == mapjoin
        #TODO this is wrong here: the overall order should at least be concordant with the args if the args are concordant
        return unique(vcat(map(arg -> getfields(arg, bindings), node.args)...))
    elseif node.kind == aggregate
        return setdiff(getfields(node.arg, bindings), node.idxs)
    elseif node.kind == reorder
        return node.idxs
    elseif node.kind == relabel
        return node.idxs
    elseif node.kind == reformat
        return getfields(node.arg, bindings)
    else
        throw(ArgumentError("getfields($(node.kind)) is undefined"))
    end
end

function propagate_fields(node::LogicNode, fields = Dict{LogicNode, Any}())
    if @capture node plan(~stmts...)
        stmts = map(stmts) do stmt
            propagate_fields(stmt, fields)
        end
        plan(stmts...)
    elseif @capture node query(~lhs, ~rhs)
        rhs = propagate_fields(rhs, fields)
        fields[lhs] = getfields(rhs, Dict())
        query(lhs, rhs)
    elseif @capture node relabel(alias, ~idxs...)
        node
    elseif isalias(node)
        relabel(node, fields[node]...)
    elseif node.kind === produces
        node
    elseif istree(node)
        similarterm(node, operation(node), map(x -> propagate_fields(x, fields), arguments(node)))
    else
        node
    end
end

function getproductions(node::LogicNode)
    for node in PostOrderDFS(node)
        if node.kind == produces
            res = []
            for arg in PostOrderDFS(node)
                if isalias(arg)
                    push!(res, arg)
                end
            end
            return res
        end
    end
    return []
end
