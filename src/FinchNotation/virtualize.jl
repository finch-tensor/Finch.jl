#TODO delete this line and figure this out with finch_leaf_instance
Finch.virtualize(ctx, ex, (@nospecialize T)) = value(ex, T)

function Finch.virtualize(ctx, ex, ::Type{FinchNotation.LiteralInstance{val}}) where {val}
    literal(val)
end
function Finch.virtualize(ctx, ex, ::Type{FinchNotation.IndexInstance{name}}) where {name}
    freshen(ctx, name)
    index(name)
end
function Finch.virtualize(
    ctx, ex, ::Type{FinchNotation.DefineInstance{Lhs,Rhs,Body}}
) where {Lhs,Rhs,Body}
    define(
        virtualize(ctx, :($ex.lhs), Lhs),
        virtualize(ctx, :($ex.rhs), Rhs),
        virtualize(ctx, :($ex.body), Body),
    )
end
function Finch.virtualize(
    ctx, ex, ::Type{FinchNotation.DeclareInstance{Tns,Init,Op}}
) where {Tns,Init,Op}
    declare(
        virtualize(ctx, :($ex.tns), Tns),
        virtualize(ctx, :($ex.init), Init),
        virtualize(ctx, :($ex.op), Op),
    )
end
function Finch.virtualize(
    ctx, ex, ::Type{FinchNotation.FreezeInstance{Tns,Op}}
) where {Tns,Op}
    freeze(virtualize(ctx, :($ex.tns), Tns), virtualize(ctx, :($ex.op), Op))
end
function Finch.virtualize(
    ctx, ex, ::Type{FinchNotation.ThawInstance{Tns,Op}}
) where {Tns,Op}
    thaw(virtualize(ctx, :($ex.tns), Tns), virtualize(ctx, :($ex.op), Op))
end
function Finch.virtualize(
    ctx, ex, ::Type{FinchNotation.BlockInstance{Bodies}}
) where {Bodies}
    bodies = map(enumerate(Bodies.parameters)) do (n, Body)
        virtualize(ctx, :($ex.bodies[$n]), Body)
    end
    block(bodies...)
end
function Finch.virtualize(
    ctx, ex, ::Type{FinchNotation.SieveInstance{Cond,Body}}
) where {Cond,Body}
    cond = virtualize(ctx, :($ex.cond), Cond)
    body = virtualize(ctx, :($ex.body), Body)
    sieve(cond, body)
end
function Finch.virtualize(
    ctx, ex, ::Type{FinchNotation.LoopInstance{Idx,Ext,Body}}
) where {Idx,Ext,Body}
    idx = virtualize(ctx, :($ex.idx), Idx)
    ext = virtualize(ctx, :($ex.ext), Ext)
    body = virtualize(ctx, :($ex.body), Body)
    loop(idx, ext, body)
end
function Finch.virtualize(
    ctx, ex, ::Type{FinchNotation.AssignInstance{Lhs,Op,Rhs}}
) where {Lhs,Op,Rhs}
    assign(
        virtualize(ctx, :($ex.lhs), Lhs),
        virtualize(ctx, :($ex.op), Op),
        virtualize(ctx, :($ex.rhs), Rhs),
    )
end
function Finch.virtualize(
    ctx, ex, ::Type{FinchNotation.CallInstance{Op,Args}}
) where {Op,Args}
    op = virtualize(ctx, :($ex.op), Op)
    args = map(enumerate(Args.parameters)) do (n, Arg)
        virtualize(ctx, :($ex.args[$n]), Arg)
    end
    call(op, args...)
end
function Finch.virtualize(
    ctx, ex, ::Type{FinchNotation.AccessInstance{Tns,Mode,Idxs}}
) where {Tns,Mode,Idxs}
    tns = virtualize(ctx, :($ex.tns), Tns)
    idxs = map(enumerate(Idxs.parameters)) do (n, Idx)
        virtualize(ctx, :($ex.idxs[$n]), Idx)
    end
    access(tns, virtualize(ctx, :($ex.mode), Mode), idxs...)
end
Finch.virtualize(ctx, ex, ::Type{FinchNotation.ReaderInstance}) = reader()
function Finch.virtualize(ctx, ex, ::Type{FinchNotation.UpdaterInstance{Op}}) where {Op}
    op = virtualize(ctx, :($ex.op), Op)
    updater(op)
end
function Finch.virtualize(ctx, ex, ::Type{FinchNotation.VariableInstance{tag}}) where {tag}
    variable(tag)
end
function Finch.virtualize(
    ctx, ex, ::Type{FinchNotation.TagInstance{Var,Bind}}
) where {Var,Bind}
    var = virtualize(ctx, :($ex.var), Var)
    bind = virtualize(ctx, :($ex.bind), Bind, var.name)
    tag(var, bind)
end
function Finch.virtualize(
    ctx, ex, ::Type{FinchNotation.YieldBindInstance{Args}}
) where {Args}
    args = map(enumerate(Args.parameters)) do (n, Arg)
        virtualize(ctx, :($ex.args[$n]), Arg)
    end
    yieldbind(args...)
end
