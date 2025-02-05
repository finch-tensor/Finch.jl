@kwdef struct EnforceLifecyclesVisitor
    uses = OrderedDict()
    scoped_uses = Dict()
    global_uses = uses
    modes = Dict()
end

struct EnforceLifecyclesError
    msg
end

function open_scope(ctx::EnforceLifecyclesVisitor, prgm)
    ctx_2 = EnforceLifecyclesVisitor(; kwfields(ctx)..., uses=Dict())
    close_scope(prgm, ctx_2)
end

function getmodified(node::FinchNode)
    if node.kind === block
        return unique(mapreduce(getmodified, vcat, node.bodies; init=[]))
    elseif node.kind === declare || node.kind === thaw
        return [node.tns]
    else
        return []
    end
end

function close_scope(prgm, ctx::EnforceLifecyclesVisitor)
    prgm = ctx(prgm)
    for tns in getmodified(prgm)
        if ctx.modes[tns].kind !== reader
            prgm = block(prgm, freeze(tns, ctx.modes[tns].op))
        end
    end
    prgm
end

"""
    enforce_lifecycles(prgm)

A transformation which adds `freeze` and `thaw` statements automatically to
tensor roots, depending on whether they appear on the left or right hand side.
"""
function enforce_lifecycles(prgm)
    prgm = infer_declare_ops(prgm, Dict())
    close_scope(prgm, EnforceLifecyclesVisitor())
end

#assumes arguments to prgm have been visited already and their uses collected
function open_stmt(prgm, ctx::EnforceLifecyclesVisitor)
    for (tns, mode) in ctx.uses
        cur_mode = get(ctx.modes, tns, reader())
        if mode.kind === reader && cur_mode.kind === updater
            prgm = block(freeze(tns, cur_mode.op), prgm)
        elseif mode.kind === updater && cur_mode.kind === reader
            prgm = block(thaw(tns, mode.op), prgm)
        end
        ctx.modes[tns] = mode
    end
    empty!(ctx.uses)
    prgm
end

function (ctx::EnforceLifecyclesVisitor)(node::FinchNode)
    if node.kind === loop
        open_stmt(loop(node.idx, ctx(node.ext), open_scope(ctx, node.body)), ctx)
    elseif node.kind === sieve
        open_stmt(sieve(ctx(node.cond), open_scope(ctx, node.body)), ctx)
    elseif node.kind === define
        open_stmt(define(node.lhs, ctx(node.rhs), open_scope(ctx, node.body)), ctx)
    elseif node.kind === declare
        ctx.scoped_uses[node.tns] = ctx.uses
        mode = get(ctx.modes, node.tns, reader())
        if mode.kind === updater
            node = block(freeze(node.tns, mode.op), node)
        end
        ctx.modes[node.tns] = updater(node.op)
        node
    elseif node.kind === freeze
        haskey(ctx.modes, node.tns) ||
            throw(EnforceLifecyclesError("cannot freeze undefined $(node.tns)"))
        ctx.modes[node.tns].kind === reader && return block()
        ctx.modes[node.tns] = reader()
        node
    elseif node.kind === thaw
        mode = get(ctx.modes, node.tns, reader())
        ctx.modes[node.tns] = updater(node.op)
        mode == updater(node.op) && return block()
        mode == reader() && return node
        #mode.kind === updater
        return block(freeze(node.tns, mode.op), node)
    elseif node.kind === assign
        return open_stmt(assign(ctx(node.lhs), ctx(node.op), ctx(node.rhs)), ctx)
    elseif node.kind === access
        idxs = map(ctx, node.idxs)
        uses = get(ctx.scoped_uses, getroot(node.tns), ctx.global_uses)
        mode = get(uses, getroot(node.tns), node.mode)
        mode.kind != node.mode.kind &&
            throw(
                EnforceLifecyclesError(
                    "cannot mix reads and writes to $(node.tns) outside of defining scope (hint: perhaps add a declaration like `var .= 0` or use an updating operator like `var += 1`)"
                ),
            )
        mode.kind === updater && mode.op != node.mode.op &&
            throw(
                EnforceLifecyclesError(
                    "cannot mix reduction operators to $(node.tns) outside of defining scope (hint: perhaps add a declaration like `var .= 0` or use an updating operator like `var += 1`)"
                ),
            )
        uses[getroot(node.tns)] = node.mode
        access(node.tns, node.mode, idxs...)
    elseif node.kind === yieldbind
        args_2 = map(node.args) do arg
            uses = get(ctx.scoped_uses, getroot(arg), ctx.global_uses)
            get(uses, getroot(arg), reader()).kind !== reader &&
                throw(
                    EnforceLifecyclesError(
                        "cannot return $(arg) outside of defining scope"
                    ),
                )
            uses[getroot(arg)] = reader()
            ctx(arg)
        end
        open_stmt(yieldbind(args_2...), ctx)
    elseif istree(node)
        return similarterm(
            node, operation(node), simple_map(FinchNode, ctx, arguments(node))
        )
    else
        return node
    end
end

function infer_declare_ops(node, ops=Dict())
    if node.kind === declare
        declare(node.tns, node.init, get(ops, getroot(node.tns), overwrite))
    else
        if node.kind === access && node.mode === updater
            ops[getroot(node.tns)] = node.mode.op
        end
        if istree(node)
            similarterm(
                node,
                operation(node),
                reverse(
                    simple_map(
                        FinchNode, n -> infer_declare_ops(n, ops), reverse(arguments(node))
                    ),
                ),
            )
        else
            node
        end
    end
end
