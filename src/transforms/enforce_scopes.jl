@kwdef struct EnforceScopesVisitor
    namespace = Namespace()
    vars = Dict(index(:(:)) => index(:(:)))
    scope = Set()
    global_scope = scope
end

freshen(ctx::EnforceScopesVisitor, tags...) = freshen(ctx.namespace, tags...)

"""
    enforce_scopes(prgm)

A transformation which gives all loops unique index names and enforces that
tensor roots are declared in a containing scope and enforces that variables are
declared once within their scope. Note that `loop` and `sieve` both introduce new scopes.
"""
enforce_scopes(prgm) = EnforceScopesVisitor()(prgm)

struct ScopeError
    msg
end

function open_scope(ctx::EnforceScopesVisitor, prgm)
    prgm = EnforceScopesVisitor(; kwfields(ctx)..., vars=copy(ctx.vars), scope=Set())(prgm)
end

function (ctx::EnforceScopesVisitor)(node::FinchNode)
    if @capture node loop(~idx, ~ext, ~body)
        ctx.vars[idx] = index(freshen(ctx, idx.name))
        loop(ctx(idx), ctx(ext), open_scope(ctx, body))
    elseif @capture node sieve(~cond, ~body)
        sieve(ctx(cond), open_scope(ctx, body))
    elseif @capture node declare(~tns, ~init, ~op)
        push!(ctx.scope, tns)
        declare(ctx(tns), init, op)
    elseif @capture node freeze(~tns, ~op)
        node.tns in ctx.scope || ctx.scope === ctx.global_scope ||
            throw(ScopeError("cannot freeze $tns not defined in this scope"))
        freeze(ctx(tns), op)
    elseif @capture node thaw(~tns, ~op)
        node.tns in ctx.scope || ctx.scope === ctx.global_scope ||
            throw(ScopeError("cannot thaw $tns not defined in this scope"))
        thaw(ctx(tns), op)
    elseif node.kind === variable
        if !(node in ctx.scope)
            push!(ctx.global_scope, node)
        end
        node
    elseif node.kind === index
        haskey(ctx.vars, node) || throw(ScopeError("unbound index $node"))
        ctx.vars[node]
    elseif node.kind == define
        if node.lhs.kind != variable
            throw(ScopeError("cannot define a non-variable $node.lhs"))
        end
        #TODO why not just freshen variables?
        rhs = ctx(node.rhs)
        var = node.lhs
        haskey(ctx.vars, var) &&
            throw(ScopeError("In node $(node) variable $(var) is already bound."))
        ctx.vars[var] = node.rhs
        define(node.lhs, rhs, open_scope(ctx, node.body))
    elseif istree(node)
        return similarterm(node, operation(node), map(ctx, arguments(node)))
    else
        return node
    end
end
