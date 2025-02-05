
virtual_size(ctx, tns::FinchNode) = virtual_size(ctx, resolve(ctx, tns))
function virtual_resize!(ctx, tns::FinchNode, dims...)
    virtual_resize!(ctx, resolve(ctx, tns), dims...)
end
virtual_fill_value(ctx, tns::FinchNode) = virtual_fill_value(ctx, resolve(ctx, tns))

function instantiate(ctx::AbstractCompiler, tns::FinchNode, mode)
    if tns.kind === virtual
        return instantiate(ctx, tns.val, mode)
    elseif tns.kind === variable
        return Unfurled(tns, instantiate(ctx, resolve(ctx, tns), mode))
    else
        return tns
    end
end

function declare!(ctx::AbstractCompiler, tns::FinchNode, init)
    declare!(ctx, resolve(ctx, tns), init)
end
thaw!(ctx::AbstractCompiler, tns::FinchNode) = thaw!(ctx, resolve(ctx, tns))
freeze!(ctx::AbstractCompiler, tns::FinchNode) = freeze!(ctx, resolve(ctx, tns))

function unfurl(ctx, tns::FinchNode, ext, mode, proto)
    unfurl(ctx, resolve(ctx, tns), ext, mode, proto)
end

function lower_access(ctx::AbstractCompiler, tns::FinchNode, mode)
    lower_access(ctx, resolve(ctx, tns), mode)
end

function lower_assign(ctx::AbstractCompiler, tns::FinchNode, mode, op, rhs)
    lower_assign(ctx, resolve(ctx, tns), mode, op, rhs)
end

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
