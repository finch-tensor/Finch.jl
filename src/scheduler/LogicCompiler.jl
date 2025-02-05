function simple_map(::Type{T}, f::F, args) where {T,F}
    res = Vector{T}(undef, length(args))
    for i in 1:length(args)
        res[i] = f(args[i])
    end
    res
end

"""
    get_structure(root::LogicNode)

Quickly produce a normalized structure for a logic program. Note: the result
will not be a runnable logic program, but can be hashed and compared for
equality. Two programs will have equal structure if their tensors have the same
type and their program structure is equivalent up to renaming.
"""
function get_structure(
    node::LogicNode,
    fields::Dict{Symbol,LogicNode}=Dict{Symbol,LogicNode}(),
    aliases::Dict{Symbol,LogicNode}=Dict{Symbol,LogicNode}())
    if node.kind === field
        get!(fields, node.name, immediate(length(fields) + length(aliases)))
    elseif node.kind === alias
        get!(aliases, node.name, immediate(length(fields) + length(aliases)))
    elseif node.kind === subquery
        if haskey(aliases, node.lhs.name)
            aliases[node.lhs.name]
        else
            #this will define the alias in aliases dict
            subquery(
                get_structure(node.lhs, fields, aliases),
                get_structure(node.arg, fields, aliases),
            )
        end
    elseif node.kind === table
        table(
            immediate(typeof(node.tns.val)),
            simple_map(LogicNode, idx -> get_structure(idx, fields, aliases), node.idxs),
        )
    elseif istree(node)
        similarterm(
            node,
            operation(node),
            simple_map(
                LogicNode, arg -> get_structure(arg, fields, aliases), arguments(node)
            ),
        )
    else
        node
    end
end

@kwdef struct LogicLowerer
    mode = :fast
end

@kwdef struct PointwiseLowerer
    bound_idxs = []
end

function compile_pointwise_logic(ex)
    ctx = PointwiseLowerer()
    code = ctx(ex)
    bound_idxs = ctx.bound_idxs
    (code, bound_idxs)
end

function (ctx::PointwiseLowerer)(ex)
    if @capture ex mapjoin(~op, ~args...)
        :($(op.val)($(map(ctx, args)...)))
    elseif (@capture ex reorder(relabel(~arg::isalias, ~idxs_1...), ~idxs_2...))
        append!(ctx.bound_idxs, idxs_1)
        :($(arg.name)[$(map(idx -> idx in idxs_2 ? idx.name : 1, idxs_1)...)]) #TODO need a trait for the first index
    elseif (@capture ex reorder(~arg::isimmediate, ~idxs...))
        arg.val
    elseif ex.kind === immediate
        ex.val
    else
        error("Unrecognized logic: $(ex)")
    end
end

function compile_logic_constant(node)
    if node.kind === immediate
        val = node.val
        if Base.isoperator(Symbol(val))
            Symbol(val)
        else
            val
        end
    elseif node.kind === deferred
        :($(node.ex)::$(node.type))
    else
        error()
    end
end

function logic_constant_type(node)
    if node.kind === immediate
        typeof(node.val)
    elseif node.kind === deferred
        node.type
    else
        error()
    end
end

function (ctx::LogicLowerer)(ex)
    if @capture ex query(~lhs::isalias, table(~tns, ~idxs...))
        :($(lhs.name) = $(compile_logic_constant(tns)))
    elseif @capture ex query(
        ~lhs::isalias,
        reformat(~tns, reorder(relabel(~arg::isalias, ~idxs_1...), ~idxs_2...)),
    )
        loop_idxs = map(idx -> idx.name, withsubsequence(intersect(idxs_1, idxs_2), idxs_2))
        lhs_idxs = map(idx -> idx.name, idxs_2)
        (rhs, rhs_idxs) = compile_pointwise_logic(
            reorder(relabel(arg, idxs_1...), idxs_2...)
        )
        body = :($(lhs.name)[$(lhs_idxs...)] = $rhs)
        for idx in loop_idxs
            if field(idx) in rhs_idxs
                body = :(
                    for $idx in _
                        $body
                    end
                )
            elseif idx in lhs_idxs
                body = :(
                    for $idx in 1:1
                        $body
                    end
                )
            end
        end
        quote
            $(lhs.name) = $(compile_logic_constant(tns))
            @finch mode = $(QuoteNode(ctx.mode)) begin
                $(lhs.name) .= $(fill_value(logic_constant_type(tns)))
                $body
                return $(lhs.name)
            end
        end
    elseif @capture ex query(~lhs::isalias, reformat(~tns, mapjoin(~args...)))
        z = fill_value(logic_constant_type(tns))
        ctx(
            query(
                lhs, reformat(tns, aggregate(initwrite(z), immediate(z), mapjoin(args...)))
            ),
        )
    elseif @capture ex query(~lhs, reformat(~tns, aggregate(~op, ~init, ~arg, ~idxs_1...)))
        idxs_2 = map(idx -> idx.name, getfields(arg))
        lhs_idxs = map(idx -> idx.name, setdiff(getfields(arg), idxs_1))
        (rhs, rhs_idxs) = compile_pointwise_logic(arg)
        body = :($(lhs.name)[$(lhs_idxs...)] << $(compile_logic_constant(op)) >>= $rhs)
        for idx in idxs_2
            if field(idx) in rhs_idxs
                body = :(
                    for $idx in _
                        $body
                    end
                )
            elseif idx in lhs_idxs
                body = :(
                    for $idx in 1:1
                        $body
                    end
                )
            end
        end
        quote
            $(lhs.name) = $(compile_logic_constant(tns))
            @finch mode = $(QuoteNode(ctx.mode)) begin
                $(lhs.name) .= $(fill_value(logic_constant_type(tns)))
                $body
                return $(lhs.name)
            end
        end
    elseif @capture ex produces(~args...)
        return :(
            return (
                $(
                    map(args) do arg
                        if @capture(
                            arg, reorder(relabel(~tns::isalias, ~idxs_1...), ~idxs_2...)
                        ) && Set(idxs_1) == Set(idxs_2)
                            :(swizzle(
                                $(tns.name),
                                $([findfirst(isequal(idx), idxs_1) for idx in idxs_2]...),
                            ))
                        elseif @capture(arg, reorder(~tns::isalias, ~idxs...))
                            tns.name
                        elseif isalias(arg)
                            arg.name
                        else
                            error("Unrecognized logic: $(arg)")
                        end
                    end...
                ),
            )
        )
    elseif @capture ex plan(~bodies...)
        Expr(:block, map(ctx, bodies)...)
    else
        error("Unrecognized logic: $(ex)")
    end
end

"""
    LogicCompiler

The LogicCompiler is a simple compiler for finch logic programs. The interpreter is
only capable of executing programs of the form:
      REORDER := reorder(relabel(ALIAS, FIELD...), FIELD...)
       ACCESS := reorder(relabel(ALIAS, idxs_1::FIELD...), idxs_2::FIELD...) where issubsequence(idxs_1, idxs_2)
    POINTWISE := ACCESS | mapjoin(IMMEDIATE, POINTWISE...) | reorder(IMMEDIATE, FIELD...) | IMMEDIATE
    MAPREDUCE := POINTWISE | aggregate(IMMEDIATE, IMMEDIATE, POINTWISE, FIELD...)
       TABLE  := table(IMMEDIATE | DEFERRED, FIELD...)
COMPUTE_QUERY := query(ALIAS, reformat(IMMEDIATE, arg::(REORDER | MAPREDUCE)))
  INPUT_QUERY := query(ALIAS, TABLE)
         STEP := COMPUTE_QUERY | INPUT_QUERY | produces(ALIAS...)
         ROOT := PLAN(STEP...)
"""
@kwdef struct LogicCompiler
    mode = :fast
end

Base.:(==)(a::LogicCompiler, b::LogicCompiler) = a.mode == b.mode
Base.hash(a::LogicCompiler, h::UInt) = hash(LogicCompiler, hash(a.mode, h))

function set_options(ctx::LogicCompiler; mode=ctx.mode, kwargs...)
    LogicCompiler(; mode=mode)
end

function (ctx::LogicCompiler)(prgm::LogicNode)
    prgm = format_queries(prgm, true)
    LogicLowerer(; mode=ctx.mode)(prgm)
end

codes = Dict()
function compute_impl(prgm, ::LogicLowerer)
    f = get!(codes, get_structure(prgm)) do
        eval(compile(prgm))
    end
    return Base.invokelatest(f, prgm)
end
