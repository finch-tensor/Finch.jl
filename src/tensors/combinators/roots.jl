
virtual_size(ctx, tns::FinchNode) = virtual_size(ctx, resolve(ctx, tns))
virtual_resize!(ctx, tns::FinchNode, dims...) = virtual_resize!(ctx, resolve(ctx, tns), dims...)
virtual_fill_value(ctx, tns::FinchNode) = virtual_fill_value(ctx, resolve(ctx, tns))

function unfurl_posthook(ctx::AbstractCompiler, tns::FinchNode, mode)
    if tns.kind === virtual
        return unfurl_posthook(ctx, tns.val, mode)
    elseif tns.kind === variable
        return Unfurled(tns, unfurl_posthook(ctx, resolve(ctx, tns), mode))
    else
        return tns
    end
end

declare!(ctx::AbstractCompiler, tns::FinchNode, init) = declare!(ctx, resolve(ctx, tns), init)
thaw!(ctx::AbstractCompiler, tns::FinchNode) = thaw!(ctx, resolve(ctx, tns))
freeze!(ctx::AbstractCompiler, tns::FinchNode) = freeze!(ctx, resolve(ctx, tns))

unfurl(ctx, tns::FinchNode, ext, mode, proto) =
    unfurl(ctx, resolve(ctx, tns), ext, mode, proto)

lower_access(ctx::AbstractCompiler, node, tns::FinchNode) =
    lower_access(ctx, node, resolve(ctx, tns))

is_injective(ctx, lvl::FinchNode) = is_injective(ctx, resolve(ctx, lvl))
is_atomic(ctx, lvl::FinchNode) = is_atomic(ctx, resolve(ctx, lvl))
is_concurrent(ctx, lvl::FinchNode) = is_concurrent(ctx, resolve(ctx, lvl))

function getroot(node::FinchNode)
    if node.kind === virtual
        return getroot(node.val)
    elseif node.kind === variable
        return node
    else
        error("could not get root of $(node)")
    end
end
