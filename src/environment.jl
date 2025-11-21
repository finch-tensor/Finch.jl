abstract type AbstractCompiler end

"""
    Namespace

A namespace for managing variable names and aesthetic fresh variable generation.
"""
struct Namespace
    counts
end
Namespace() = Namespace(Dict())

"""
    freshen(ctx, tags...)

Return a fresh variable in the current context named after `Symbol(tags...)`
"""
function freshen(spc::Namespace, tags...)
    name = Symbol(tags...)
    m = match(r"^(.*)_(\d*)$", string(name))
    if m === nothing
        tag = name
        n = 1
    else
        tag = Symbol(m.captures[1])
        n = parse(BigInt, m.captures[2])
    end
    n = max(get(spc.counts, tag, 0) + 1, n)
    spc.counts[tag] = n
    if n == 1
        return Symbol(tag)
    else
        return Symbol(tag, :_, n)
    end
end

"""
    JuliaContext

A context for compiling Julia code, managing side effects, parallelism, and
variable names in the generated code of the executing environment.
"""
@kwdef mutable struct JuliaContext <: AbstractCompiler
    namespace::Namespace = Namespace()
    preamble::Vector{Any} = []
    epilogue::Vector{Any} = []
    task = VirtualSerialTask()
end

"""
    push_preamble!(ctx, thunk)

Push the thunk onto the preamble in the currently executing context. The
preamble will be evaluated before the code returned by the given function in the
context.
"""
push_preamble!(ctx::JuliaContext, thunk) = push!(ctx.preamble, thunk)

"""
    push_epilogue!(ctx, thunk)

Push the thunk onto the epilogue in the currently executing context. The
epilogue will be evaluated after the code returned by the given function in the
context.
"""
push_epilogue!(ctx::JuliaContext, thunk) = push!(ctx.epilogue, thunk)

"""
    get_task(ctx)

Get the task which will execute code in this context
"""
get_task(ctx) = ctx.task

"""
    virtualize(ctx, ex, T, [tag])

Return the virtual program corresponding to the Julia expression `ex` of type
`T` in the `JuliaContext` `ctx`. Implementaters may support the optional `tag`
argument is used to name the resulting virtual variable.
"""
virtualize(ctx, ex, T, tag) = virtualize(ctx, ex, T)
function virtualize(ctx, ex, T::Type{NamedTuple{names,args}}) where {names,args}
    Dict(map(zip(names, args.parameters)) do (name, arg)
        name => virtualize(ctx, :($ex.$(QuoteNode(name))), arg, name)
    end...)
end

freshen(ctx::JuliaContext, tags...) = freshen(ctx.namespace, tags...)

contain_epilogue_helper(node, epilogue) = node
function contain_epilogue_helper(node::Expr, epilogue)
    if @capture node :for(~ext, ~body)
        return node
    elseif @capture node :while(~ext, ~body)
        return node
    elseif @capture node :break()
        return Expr(:block, epilogue, node)
    else
        return Expr(node.head, map(x -> contain_epilogue_helper(x, epilogue), node.args)...)
    end
end

"""
    contain(f, ctx)

Call f on a subcontext of `ctx` and return the result. Variable bindings,
preambles, and epilogues defined in the subcontext will not escape the call to
contain.
"""
function contain(f, ctx::JuliaContext; task=nothing)
    task_2 = something(task, ctx.task)
    preamble = Expr(:block)
    epilogue = Expr(:block)
    ctx_2 = JuliaContext(ctx.namespace, preamble.args, epilogue.args, task_2)
    body = f(ctx_2)
    if epilogue == Expr(:block)
        return quote
            $preamble
            $body
        end
    else
        res = freshen(ctx_2, :res)
        return quote
            $preamble
            $res = $(contain_epilogue_helper(body, epilogue))
            $epilogue
            $res
        end
    end
end
