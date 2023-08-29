@kwdef mutable struct LowerJulia <: AbstractCompiler
    code = JuliaContext()
    algebra = DefaultAlgebra()
    bindings::Dict{Any, Any} = Dict()
    mode = fastfinch
    modes::Dict{Any, Any} = Dict()
    scope = Set()
    shash = StaticHash()
    program_rules = get_program_rules(algebra, shash)
    bounds_rules = get_bounds_rules(algebra, shash)
end

function contain(f, ctx::LowerJulia)
    contain(ctx.code) do code_2
        f(LowerJulia(code_2, ctx.algebra, ctx.bindings, ctx.mode, ctx.modes, ctx.scope, ctx.shash, ctx.program_rules, ctx.bounds_rules))
    end
end

struct StaticHash
    counts::Dict{Any, Int}
end
StaticHash() = StaticHash(Dict{Any, Int}())

function (h::StaticHash)(x)
    if haskey(h.counts, x)
        return h.counts[x]
    else
        return (h.counts[x] = UInt(length(h.counts)))
    end
end

(ctx::AbstractCompiler)(root) = ctx(root, Stylize(root, ctx)(root))
#(ctx::AbstractCompiler)(root, style) = (display(root); display(style); lower(root, ctx, style))
(ctx::AbstractCompiler)(root, style) = lower(root, ctx, style)

function open_scope(prgm, ctx::AbstractCompiler)
    ctx_2 = shallowcopy(ctx)
    ctx_2.scope = Set()
    res = ctx_2(prgm)
    for tns in ctx_2.scope
        pop!(ctx_2.modes, tns, nothing)
    end
    ctx.bindings = copy(ctx.bindings)
    res
end

function cache!(ctx::AbstractCompiler, var, val)
    val = finch_leaf(val)
    isconstant(val) && return val
    var = freshen(ctx.code,var)
    val = simplify(val, ctx)
    push!(ctx.code.preamble, quote
        $var = $(contain(ctx_2 -> ctx_2(val), ctx))
    end)
    return cached(value(var, Any), literal(val))
end

resolve(node, ctx) = node
function resolve(node::FinchNode, ctx::AbstractCompiler)
    if node.kind === virtual
        return node.val
    elseif node.kind === variable
        return resolve(ctx.bindings[node], ctx)
    elseif node.kind === index
        return resolve(ctx.bindings[node], ctx)
    else
        error("unimplemented $node")
    end
end

(ctx::AbstractCompiler)(root::Union{Symbol, Expr}, ::DefaultStyle) = root

function lower(root, ctx::AbstractCompiler, ::DefaultStyle)
    node = finch_leaf(root)
    if node.kind === virtual
        error("don't know how to lower $root")
    end
    ctx(node)
end

function lower(root::FinchNode, ctx::AbstractCompiler, ::DefaultStyle)
    if root.kind === value
        return root.val
    elseif root.kind === index
        @assert haskey(ctx.bindings, root) "index $(root) unbound"
        return ctx(ctx.bindings[root]) #This unwraps indices that are virtuals. Arguably these virtuals should be precomputed, but whatevs.
    elseif root.kind === literal
        if typeof(root.val) === Symbol ||
          typeof(root.val) === Expr ||
          typeof(root.val) === Missing
            return QuoteNode(root.val)
        else
            return root.val
        end
    elseif root.kind === block
        if isempty(root.bodies)
            return quote end
        else
            head = root.bodies[1]
            body = block(root.bodies[2:end]...)
            preamble = quote end

            if head.kind === define
                @assert head.lhs.kind === variable
                ctx.bindings[head.lhs] = cache!(ctx, head.lhs.name, head.rhs)
                push!(ctx.scope, head.lhs)
            elseif head.kind === declare
                @assert head.tns.kind === variable
                @assert get(ctx.modes, head.tns, reader()).kind === reader
                ctx.bindings[head.tns] = declare!(ctx.bindings[head.tns], ctx, head.init) #TODO should ctx.bindings be scoped?
                push!(ctx.scope, head.tns)
                ctx.modes[head.tns] = updater()
            elseif head.kind === freeze
                @assert ctx.modes[head.tns].kind === updater
                ctx.bindings[head.tns] = freeze!(ctx.bindings[head.tns], ctx)
                ctx.modes[head.tns] = reader()
            elseif head.kind === thaw
                @assert get(ctx.modes, head.tns, reader()).kind === reader
                ctx.bindings[head.tns] = thaw!(ctx.bindings[head.tns], ctx)
                ctx.modes[head.tns] = updater()
            else
                preamble = contain(ctx) do ctx_2
                    ctx_2(instantiate!(head, ctx_2))
                end
            end

            quote
                $preamble
                $(contain(ctx) do ctx_2
                    (ctx_2)(body)
                end)
            end
        end
    elseif root.kind === access
        return lower_access(ctx, root, resolve(root.tns, ctx))
    elseif root.kind === call
        if root.op == literal(and)
            if isempty(root.args)
                return true
            else
                reduce((x, y) -> :($x && $y), map(ctx, root.args)) #TODO This could be better. should be able to handle empty case
            end
        elseif root.op == literal(or)
            if isempty(root.args)
                return false
            else
                reduce((x, y) -> :($x || $y), map(ctx, root.args))
            end
        else
            :($(ctx(root.op))($(map(ctx, root.args)...)))
        end
    elseif root.kind === cached
        return ctx(root.arg)
    elseif root.kind === loop
        @assert root.idx.kind === index
        @assert root.ext.kind === virtual
        lower_loop(ctx, root, root.ext.val)
    elseif root.kind === sieve
        cond = freshen(ctx.code,:cond)
        push!(ctx.code.preamble, :($cond = $(ctx(root.cond))))
    
        return quote
            if $cond
                $(contain(ctx) do ctx_2
                    open_scope(root.body, ctx_2)
                end)
            end
        end
    elseif root.kind === virtual
        ctx(root.val)
    elseif root.kind === assign
        if root.lhs.kind === access
            @assert root.lhs.mode.kind === updater
            rhs = ctx(simplify(call(root.op, root.lhs, root.rhs), ctx))
        else
            rhs = ctx(root.rhs)
        end
        lhs = ctx(root.lhs)
        return :($lhs = $rhs)
    elseif root.kind === variable
        return ctx(ctx.bindings[root])
    else
        error("unimplemented ($root)")
    end
end

function lower_access(ctx, node, tns)
    tns = ctx(tns)
    idxs = map(ctx, node.idxs)
    :($(ctx(tns))[$(idxs...)])
end

function lower_access(ctx, node, tns::Number)
    @assert node.mode.kind === reader
    tns
end

function lower_loop(ctx, root, ext)
    root_2 = Rewrite(Postwalk(@rule access(~tns, ~mode, ~idxs...) => begin
        if !isempty(idxs) && root.idx == idxs[end]
            protos = [(mode.kind === reader ? defaultread : defaultupdate) for _ in idxs]
            tns_2 = unfurl(tns, ctx, root.ext.val, mode, protos...)
            access(tns_2, mode, idxs...)
        end
    end))(root)
    return ctx(root_2, result_style(LookupStyle(), Stylize(root_2, ctx)(root_2)))
end

function lower_loop(ctx, root, ext::ParallelDimension)
    root = ensure_concurrent(root, ctx)
    
    tid = index(freshen(ctx.code, :tid))
    i = freshen(ctx.code, :i)
    root_2 = loop(tid, Extent(value(i, Int), value(i, Int)),
        loop(root.idx, ext.ext,
            sieve(access(VirtualSplitMask(value(:(Threads.nthreads()))), reader(), root.idx, tid),
                root.body
            )
        )
    )
    return quote
        Threads.@threads for $i = 1:Threads.nthreads()
            $(contain(ctx) do ctx_2
                ctx_2(instantiate!(root_2, ctx_2))
            end)
        end
    end
end
