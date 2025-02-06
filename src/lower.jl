"""
    FinchCompiler

The core compiler for Finch, lowering canonicalized Finch IR to Julia code.
"""
@kwdef mutable struct FinchCompiler <: AbstractCompiler
    code = JuliaContext()
    algebra = DefaultAlgebra()
    mode = :fast
    result = freshen(code, :result)
    symbolic = SymbolicContext(; algebra=algebra)
    scope = ScopeContext()
end

"""
    get_result(ctx)

Return a variable which evaluates to the result of the program which should be
returned to the user.
"""
get_result(ctx::FinchCompiler) = ctx.result
"""
    get_mode_flag(ctx)

Return the mode flag given in `@finch mode = ?`.
"""
get_mode_flag(ctx::FinchCompiler) = ctx.mode

get_binding(ctx::FinchCompiler, var) = get_binding(ctx.scope, var)
has_binding(ctx::FinchCompiler, var) = has_binding(ctx.scope, var)
set_binding!(ctx::FinchCompiler, var, val) = set_binding!(ctx.scope, var, val)
set_declared!(ctx::FinchCompiler, var, val, op) = set_declared!(ctx.scope, var, val, op)
set_frozen!(ctx::FinchCompiler, var, val) = set_frozen!(ctx.scope, var, val)
set_thawed!(ctx::FinchCompiler, var, val, op) = set_thawed!(ctx.scope, var, val, op)
get_tensor_mode(ctx::FinchCompiler, var) = get_tensor_mode(ctx.scope, var)
function open_scope(f::F, ctx::FinchCompiler) where {F}
    open_scope(ctx.scope) do scope_2
        f(FinchCompiler(ctx.code, ctx.algebra, ctx.mode, ctx.result, ctx.symbolic, scope_2))
    end
end

push_preamble!(ctx::FinchCompiler, thunk) = push_preamble!(ctx.code, thunk)
push_epilogue!(ctx::FinchCompiler, thunk) = push_epilogue!(ctx.code, thunk)
get_task(ctx::FinchCompiler) = get_task(ctx.code)
freshen(ctx::FinchCompiler, tags...) = freshen(ctx.code, tags...)

get_algebra(ctx::FinchCompiler) = ctx.algebra
get_static_hash(ctx::FinchCompiler) = get_static_hash(ctx.symbolic)
prove(ctx::FinchCompiler, root) = prove(ctx.symbolic, root)
simplify(ctx::FinchCompiler, root) = simplify(ctx.symbolic, root)

function contain(f, ctx::FinchCompiler; kwargs...)
    contain(ctx.code; kwargs...) do code_2
        f(FinchCompiler(code_2, ctx.algebra, ctx.mode, ctx.result, ctx.symbolic, ctx.scope))
    end
end

(ctx::AbstractCompiler)(root) = ctx(root, get_style(ctx, root))
(ctx::AbstractCompiler)(root, style) = lower(ctx, root, style)
#(ctx::AbstractCompiler)(root, style) = (println(); println(); display(root); display(style); lower(ctx, root, style))
function cache!(ctx::AbstractCompiler, var, val)
    val = finch_leaf(val)
    isconstant(val) && return val
    var = freshen(ctx, var)
    val = simplify(ctx, val)
    push_preamble!(
        ctx,
        quote
            $var = $(contain(ctx_2 -> ctx_2(val), ctx))
        end,
    )
    return cached(value(var, Any), literal(val))
end

resolve(ctx, node) = node
function resolve(ctx::AbstractCompiler, node::FinchNode)
    if node.kind === virtual
        return node.val
    elseif node.kind === variable
        return resolve(ctx, get_binding(ctx, node))
    elseif node.kind === index
        return resolve(ctx, get_binding(ctx, node))
    else
        error("unimplemented $node")
    end
end

(ctx::AbstractCompiler)(root::Union{Symbol,Expr}, ::DefaultStyle) = root

"""
    get_style(ctx, root)

return the style to use for lowering `root` in `ctx`. This method is used to
determine which pass should be used to lower a given node. The default
implementation returns `DefaultStyle()`. Overload the three argument form
of this method, `get_style(ctx, node, root)` and specialize on `node`.
"""
get_style(ctx, root) = get_style(ctx, root, root)

get_style(ctx, node, root) = DefaultStyle()

function get_style(ctx, node::FinchNode, root)
    if node.kind === virtual
        return get_style(ctx, node.val, root)
    elseif istree(node)
        return mapreduce(
            arg -> get_style(ctx, arg, root),
            result_style,
            arguments(node);
            init=DefaultStyle(),
        )
    else
        return DefaultStyle()
    end
end

function lower(ctx::AbstractCompiler, root, ::DefaultStyle)
    node = finch_leaf(root)
    if node.kind === virtual
        error("don't know how to lower $root")
    end
    ctx(node)
end

function lower(ctx::AbstractCompiler, root::FinchNode, ::DefaultStyle)
    if root.kind === value
        return root.val
    elseif root.kind === index
        return ctx(get_binding(ctx, root)) #This unwraps indices that are virtuals. Arguably these virtuals should be precomputed, but whatevs.
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

            #The idea here is that we expect parent blocks to eagerly process
            #child blocks, so the effects of the statements like freeze or thaw
            #should always be visible to any subsequent statement, even if its
            #in a different block.
            if head.kind === block
                ctx(block(head.bodies..., body))
            elseif head.kind === declare
                val_2 = declare!(ctx, get_binding(ctx, head.tns), head.init)
                set_declared!(ctx, head.tns, val_2, head.op)
            elseif head.kind === freeze
                val_2 = freeze!(ctx, get_binding(ctx, head.tns))
                set_frozen!(ctx, head.tns, val_2)
            elseif head.kind === thaw
                val_2 = thaw!(ctx, get_binding(ctx, head.tns))
                set_thawed!(ctx, head.tns, val_2, head.op)
            else
                preamble = contain(ctx) do ctx_2
                    ctx_2(instantiate!(ctx_2, head))
                end
            end

            quote
                $preamble
                $(
                    contain(ctx) do ctx_2
                        (ctx_2)(body)
                    end
                )
            end
        end
    elseif root.kind === define
        @assert root.lhs.kind === variable
        set_binding!(ctx, root.lhs, cache!(ctx, root.lhs.name, root.rhs))
        contain(ctx) do ctx_2
            open_scope(ctx_2) do ctx_3
                ctx_3(root.body)
            end
        end
    elseif (root.kind === declare || root.kind === freeze || root.kind === thaw)
        #these statements only apply to subsequent statements in a block
        #if we try to lower them directly they are a no-op
        #arguably, the declare, freeze, or thaw nodes should never reach this case but we'll leave that alone for now
        quote end
    elseif root.kind === access
        tns = resolve(ctx, root.tns)
        if length(root.idxs) > 0
            throw(FinchCompileError("Finch failed to completely lower an access to $tns"))
        end
        @assert (root.mode.kind === reader) || (root.mode.kind === updater)
        return lower_access(ctx, tns, root.mode)
    elseif root.kind === call
        root = simplify(ctx, root)
        if root.kind === call
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
        else
            return ctx(root)
        end
    elseif root.kind === cached
        return ctx(root.arg)
    elseif root.kind === loop
        @assert root.idx.kind === index
        @assert root.ext.kind === virtual
        lower_loop(ctx, root, root.ext.val)
    elseif root.kind === sieve
        cond = freshen(ctx, :cond)
        push_preamble!(ctx, :($cond = $(ctx(root.cond))))

        return quote
            if $cond
                $(
                    contain(ctx) do ctx_2
                        open_scope(ctx_2) do ctx_3
                            ctx_3(root.body)
                        end
                    end
                )
            end
        end
    elseif root.kind === virtual
        ctx(root.val)
    elseif root.kind === assign
        @assert root.lhs.kind === access
        @assert root.lhs.mode.kind === updater
        if length(root.lhs.idxs) > 0
            throw(FinchCompileError("Finch failed to completely lower an access to $tns"))
        end
        rhs = simplify(ctx, root.rhs)
        tns = resolve(ctx, root.lhs.tns)
        return lower_assign(ctx, tns, root.lhs.mode, root.op, rhs)
    elseif root.kind === variable
        return ctx(get_binding(ctx, root))
    elseif root.kind === yieldbind
        contain(ctx) do ctx_2
            quote
                $(get_result(ctx)) = (; $(map(root.args) do tns
                    name = getroot(tns).name
                    Expr(:kw, name, ctx_2(tns))
                end...),)
            end
        end
    else
        error("unimplemented ($root)")
    end
end

function lower_access(ctx, tns, mode)
    tns = ctx(tns)
    :($(ctx(tns))[])
end

function lower_assign(ctx, tns, mode, op, rhs)
    tns = ctx(tns)
    op = ctx(op)
    rhs = ctx(rhs)
    :($tns[] = $op($tns[], $rhs))
end

function lower_access(ctx, tns::Number, mode)
    @assert node.mode.kind === reader
    tns
end

"""
    unfurl(ctx, tns, ext, proto)
Return an array object (usually a looplet nest) for lowering the outermost
dimension of virtual tensor `tns`. `ext` is the extent of the looplet. `proto`
is the protocol that should be used for this index, but one doesn't need to
unfurl all the indices at once.
"""
unfurl(ctx, tns, ext, mode, proto) =
    throw(FinchProtocolError("$tns does not support $mode with protocol $proto"))

function lower_loop(ctx, root, ext)
    contain(ctx) do ctx_2
        root_2 = Rewrite(
            Postwalk(
                @rule access(~tns, ~mode, ~idxs...) => begin
                    if !isempty(idxs) && root.idx == idxs[end]
                        tns_2 = unfurl(
                            ctx_2,
                            tns,
                            root.ext.val,
                            mode,
                            (mode.kind === reader ? defaultread : defaultupdate),
                        )
                        access(Unfurled(resolve(ctx_2, tns), tns_2), mode, idxs...)
                    end
                end
            ),
        )(
            root
        )
        return ctx_2(root_2, result_style(LookupStyle(), get_style(ctx_2, root_2)))
    end
end

function lower_loop(ctx, root, ext::ParallelDimension)
    lower_parallel_loop(ctx, root, ext, ext.device)
end
function lower_parallel_loop(ctx, root, ext::ParallelDimension, device::VirtualCPU)
    root = ensure_concurrent(root, ctx)

    decl_in_scope = unique(
        filter(
            !isnothing,
            map(node -> begin
                    if @capture(node, declare(~tns, ~init, ~op))
                        tns
                    end
                end, PostOrderDFS(root.body)),
        ),
    )

    used_in_scope = unique(
        filter(
            !isnothing,
            map(node -> begin
                    if @capture(node, access(~tns, ~mode, ~idxs...))
                        getroot(tns)
                    end
                end, PostOrderDFS(root.body)),
        ),
    )

    for tns in setdiff(used_in_scope, decl_in_scope)
        set_binding!(ctx, tns, virtual_transfer(ctx, resolve(ctx, tns), device, bcast_send))
    end
    for tns in intersect(used_in_scope, decl_in_scope)
        set_binding!(
            ctx, tns, virtual_transfer(ctx, resolve(ctx, tns), device, scatter_send)
        )
    end

    virtual_parallel_region(ctx, device) do ctx_2
        subtask = get_task(ctx_2)
        tid = get_task_num(subtask)
        for tns in setdiff(used_in_scope, decl_in_scope)
            set_binding!(
                ctx_2,
                tns,
                virtual_transfer(ctx_2, resolve(ctx_2, tns), subtask, bcast_recv),
            )
        end
        for tns in intersect(used_in_scope, decl_in_scope)
            set_binding!(
                ctx_2,
                tns,
                virtual_transfer(ctx_2, resolve(ctx_2, tns), subtask, scatter_recv),
            )
        end
        contain(ctx_2) do ctx_3
            open_scope(ctx_3) do ctx_4
                i = index(freshen(ctx, :i))
                root_2 = loop(i, Extent(tid, tid),
                    loop(root.idx, ext.ext,
                        sieve(access(VirtualSplitMask(device.n), reader(), root.idx, i),
                            root.body,
                        ),
                    ),
                )
                ctx_4(instantiate!(ctx_4, root_2))
            end
        end
        for tns in intersect(used_in_scope, decl_in_scope)
            set_binding!(
                ctx_2,
                tns,
                virtual_transfer(ctx_2, resolve(ctx_2, tns), subtask, gather_send),
            )
        end
    end
    for tns in intersect(used_in_scope, decl_in_scope)
        set_binding(
            ctx_2, tns, virtual_transfer(ctx_2, resolve(ctx_2, tns), subtask, gather_recv)
        )
    end
end
