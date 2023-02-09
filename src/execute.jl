execute(ex) = execute(ex, DefaultAlgebra())
function register(algebra)
    Base.eval(Finch, quote
        @generated function execute(ex, a::$algebra)
            execute_code(:ex, ex, a())
        end
    end)
end

#struct Lifetime <: IndexNode
#    body
#end
#
#lifetime(arg) = Lifetime(arg)
#
#SyntaxInterface.istree(::Lifetime) = true
#SyntaxInterface.arguments(ex::Lifetime) = [ex.body]
#SyntaxInterface.operation(::Lifetime) = lifetime
#SyntaxInterface.similarterm(::Type{<:IndexNode}, ::typeof(lifetime), args) = Lifetime(args...)
#
#IndexNotation.isliteral(::Lifetime) =  false
#
#struct LifetimeStyle end
#
#Base.show(io, ex::Lifetime) = Base.show(io, MIME"text/plain", ex)
#function Base.show(io::IO, mime::MIME"text/plain", ex::Lifetime)
#    print(io, "Lifetime(")
#    print(io, ex.body)
#    print(io, ")")
#end
#
#(ctx::Stylize{LowerJulia})(node::Lifetime) = result_style(LifetimeStyle(), ctx(node.body))
#combine_style(a::DefaultStyle, b::LifetimeStyle) = LifetimeStyle()
#combine_style(a::ThunkStyle, b::LifetimeStyle) = ThunkStyle()
#combine_style(a::LifetimeStyle, b::LifetimeStyle) = LifetimeStyle()
#
#function (ctx::LowerJulia)(prgm::Lifetime, ::LifetimeStyle)
#    prgm = prgm.body
#    quote
#        $(contain(ctx) do ctx_3
#            prgm = OpenScope(ctx = ctx_3)(prgm)
#            ctx_3(prgm)
#        end)
#        $(contain(ctx) do ctx_3
#            prgm = freeze(ctx = ctx_3)(prgm)
#            :(($(map(getresults(prgm)) do tns
#                :($(getname(tns)) = $(ctx_3(tns)))
#            end...), ))
#        end)
#    end
#end

function execute_code(ex, T, algebra = DefaultAlgebra())
    prgm = nothing
    code = contain(LowerJulia(algebra = algebra)) do ctx
        quote
            $(begin
                prgm = virtualize(ex, T, ctx)
                prgm = TransformSSA(Freshen())(prgm)
                prgm = ThunkVisitor(ctx)(prgm) #TODO this is a bit of a hack.
                (prgm, dims) = dimensionalize!(prgm, ctx)
                #The following call separates tensor and index names from environment symbols.
                #TODO we might want to keep the namespace around, and/or further stratify index
                #names from tensor names
                contain(ctx) do ctx_2
                    prgm2 = OpenScope(ctx = ctx_2)(prgm)
                    prgm2 = ThunkVisitor(ctx_2)(prgm2) #TODO this is a bit of a hack.
                    prgm2 = simplify(prgm2, ctx_2)
                    ctx_2(prgm2)
                end
            end)
            $(contain(ctx) do ctx_2
                prgm = CloseScope(ctx = ctx_2)(prgm)
                :(($(map(getresults(prgm)) do acc
                    @assert acc.tns.kind === variable
                    name = acc.tns.name
                    tns = trim!(ctx.bindings[name], ctx_2)
                    :($name = $(ctx_2(tns)))
                end...), ))
            end)
        end
    end
    #=
    code = quote
        @inbounds begin
            $code
        end
    end
    =#
    code = code |>
        lower_caches |>
        lower_cleanup
    #quote
    #    println($(QuoteNode(code |>         striplines |>
    #    unblock |>
    #    unquote_literals)))
    #    $code
    #end
end

macro finch(args_ex...)
    @assert length(args_ex) >= 1
    (args, ex) = (args_ex[1:end-1], args_ex[end])
    results = Set()
    prgm = IndexNotation.finch_parse_instance(ex, results)
    thunk = quote
        res = $execute($prgm, $(map(esc, args)...))
    end
    for tns in results
        push!(thunk.args, quote
            $(esc(tns)) = res.$tns
        end)
    end
    push!(thunk.args, quote
        res
    end)
    thunk
end

macro finch_code(args_ex...)
    @assert length(args_ex) >= 1
    (args, ex) = (args_ex[1:end-1], args_ex[end])
    prgm = IndexNotation.finch_parse_instance(ex)
    return quote
        $execute_code(:ex, typeof($prgm), $(map(esc, args)...)) |>
        striplines |>
        unblock |>
        unquote_literals
    end
end

"""
    OpenScope(ctx)

A transformation to initialize tensors that have just entered into scope.

See also: [`initialize!`](@ref)
"""
@kwdef struct OpenScope{Ctx}
    ctx::Ctx
    target=nothing
    escape=[]
end
initialize!(tns, ctx) = tns
get_reader(tns, ctx, protos...) = tns
get_updater(tns, ctx, protos...) = tns
function (ctx::OpenScope)(node)
    if istree(node)
        return similarterm(node, operation(node), map(ctx, arguments(node)))
    else
        return node
    end
end

function gettns(acc)
    @assert acc.kind === access
    return acc.tns
end

function (ctx::OpenScope)(node::IndexNode)
    if node.kind === access
        tns = node.tns
        if tns.kind === variable
            tns = ctx.ctx.bindings[tns.name]
        elseif tns.kind === virtual
            tns = tns.val
        end
        if node.tns.kind === variable
            if (ctx.target === nothing || (node.tns in ctx.target)) && !(node.tns in ctx.escape)
                protos = map(idx -> idx.kind === protocol ? idx.mode.val : nothing, node.idxs)
                idxs = map(idx -> idx.kind === protocol ? ctx(idx.idx) : ctx(idx), node.idxs)
                if node.mode.kind === reader
                    return access(get_reader(tns, ctx.ctx, protos...), node.mode, idxs...)
                else
                    tns = initialize!(tns, ctx.ctx)
                    return access(get_updater(tns, ctx.ctx, protos...), node.mode, idxs...)
                end
            else
                return access(ctx(node.tns), node.mode, map(ctx, node.idxs)...)
            end
        end
        return access(ctx(node.tns), node.mode, map(ctx, node.idxs)...)
    elseif node.kind === with
        ctx_2 = OpenScope(ctx.ctx, ctx.target, union(ctx.escape, map(gettns, getresults(node.prod))))
        with(ctx_2(node.cons), ctx_2(node.prod))
    elseif istree(node)
        return similarterm(node, operation(node), map(ctx, arguments(node)))
    else
        return node
    end
end

"""
    CloseScope(ctx)

A transformation to freeze output tensors before they leave scope and are
returned to the caller.

See also: [`freeze!`](@ref)
"""
@kwdef struct CloseScope{Ctx}
    ctx::Ctx
    target=nothing
    escape=[]
end
freeze!(tns, ctx, mode, idxs...) = tns
function (ctx::CloseScope)(node)
    if istree(node)
        return similarterm(node, operation(node), map(ctx, arguments(node)))
    else
        return node
    end
end

function (ctx::CloseScope)(node::IndexNode)
    if node.kind === access
        if node.tns.kind === variable
            if (ctx.target === nothing || (node.tns in ctx.target)) && !(node.tns in ctx.escape)
                if node.mode.kind !== reader
                    freeze!(ctx.ctx.bindings[node.tns.name], ctx.ctx, node.mode, node.idxs...)
                end
            end
        end
        return similarterm(node, operation(node), map(ctx, arguments(node)))
    elseif node.kind === with
        ctx_2 = CloseScope(ctx.ctx, ctx.target, union(ctx.escape, map(gettns, getresults(node.prod))))
        with(ctx_2(node.cons), ctx_2(node.prod))
    elseif istree(node)
        return similarterm(node, operation(node), map(ctx, arguments(node)))
    else
        return node
    end
end