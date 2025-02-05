using Finch.FinchNotation: block_instance, declare_instance, call_instance, loop_instance,
    index_instance, variable_instance, tag_instance, access_instance, reader_instance,
    updater_instance, assign_instance, literal_instance, yieldbind_instance

@kwdef struct PointwiseMachineLowerer
    ctx
    bound_idxs = []
end

function lower_pointwise_logic(ctx, ex)
    ctx = PointwiseMachineLowerer(; ctx=ctx)
    code = ctx(ex)
    return (code, ctx.bound_idxs)
end

function (ctx::PointwiseMachineLowerer)(ex)
    if @capture ex mapjoin(~op, ~args...)
        call_instance(literal_instance(op.val), map(ctx, args)...)
    elseif (@capture ex reorder(relabel(~arg::isalias, ~idxs_1...), ~idxs_2...))
        append!(ctx.bound_idxs, idxs_1)
        idxs_3 = map(enumerate(idxs_1)) do (n, idx)
            if idx in idxs_2
                index_instance(idx.name)
            else
                first(axes(ctx.ctx.scope[arg])[n])
            end
        end
        access_instance(
            tag_instance(variable_instance(arg.name), ctx.ctx.scope[arg]),
            reader_instance(),
            idxs_3...,
        )
    elseif (@capture ex reorder(~arg::isimmediate, ~idxs...))
        literal_instance(arg.val)
    elseif ex.kind === immediate
        literal_instance(ex.val)
    else
        error("Unrecognized logic: $(ex)")
    end
end

@kwdef struct LogicMachine
    scope = Dict{Any,Any}()
    verbose = false
    mode = :fast
end

function (ctx::LogicMachine)(ex)
    if ex.kind === alias
        ex.scope[ex]
    elseif @capture ex query(~lhs, ~rhs)
        ctx.scope[lhs] = ctx(rhs)
        (ctx.scope[lhs],)
    elseif @capture ex table(~tns, ~idxs...)
        return tns.val
    elseif @capture ex reformat(
        ~tns, reorder(relabel(~arg::isalias, ~idxs_1...), ~idxs_2...)
    )
        loop_idxs = withsubsequence(intersect(idxs_1, idxs_2), idxs_2)
        lhs_idxs = idxs_2
        res = tag_instance(variable_instance(:res), tns.val)
        lhs = access_instance(
            res, updater_instance(auto), map(idx -> index_instance(idx.name), lhs_idxs)...
        )
        (rhs, rhs_idxs) = lower_pointwise_logic(
            ctx, reorder(relabel(arg, idxs_1...), idxs_2...)
        )
        body = assign_instance(lhs, literal_instance(initwrite(fill_value(tns.val))), rhs)
        for idx in loop_idxs
            if idx in rhs_idxs
                body = loop_instance(index_instance(idx.name), auto, body)
            elseif idx in lhs_idxs
                body = loop_instance(
                    index_instance(idx.name),
                    call_instance(
                        literal_instance(extent), literal_instance(1), literal_instance(1)
                    ),
                    body,
                )
            end
        end
        body = block_instance(
            declare_instance(
                res, literal_instance(fill_value(tns.val)), literal_instance(auto)
            ),
            body,
            yieldbind_instance(res),
        )
        if ctx.verbose
            print("Running: ")
            display(body)
        end
        execute(body; mode=ctx.mode).res
    elseif @capture ex reformat(~tns, mapjoin(~args...))
        z = fill_value(tns.val)
        ctx(reformat(tns, aggregate(initwrite(z), immediate(z), mapjoin(args...))))
    elseif @capture ex reformat(~tns, aggregate(~op, ~init, ~arg, ~idxs_1...))
        loop_idxs = getfields(arg)
        lhs_idxs = setdiff(getfields(arg), idxs_1)
        res = tag_instance(variable_instance(:res), tns.val)
        lhs = access_instance(
            res, updater_instance(auto), map(idx -> index_instance(idx.name), lhs_idxs)...
        )
        (rhs, rhs_idxs) = lower_pointwise_logic(ctx, arg)
        body = assign_instance(lhs, literal_instance(op.val), rhs)
        for idx in loop_idxs
            if idx in rhs_idxs
                body = loop_instance(index_instance(idx.name), auto, body)
            elseif idx in lhs_idxs
                body = loop_instance(
                    index_instance(idx.name),
                    call_instance(
                        literal_instance(extent), literal_instance(1), literal_instance(1)
                    ),
                    body,
                )
            end
        end
        body = block_instance(
            declare_instance(
                res, literal_instance(fill_value(tns.val)), literal_instance(auto)
            ),
            body,
            yieldbind_instance(res),
        )
        if ctx.verbose
            print("Running: ")
            display(body)
        end
        execute(body; mode=ctx.mode).res
    elseif @capture ex produces(~args...)
        return map(args) do arg
            if @capture(arg, reorder(relabel(~tns::isalias, ~idxs_1...), ~idxs_2...)) &&
                Set(idxs_1) == Set(idxs_2)
                return swizzle(
                    ctx.scope[tns], [findfirst(isequal(idx), idxs_1) for idx in idxs_2]...
                )
            elseif @capture(arg, reorder(~tns::isalias, ~idxs...))
                ctx.scope[tns]
            elseif isalias(arg)
                ctx.scope[arg]
            else
                error("Unrecognized logic: $(arg)")
            end
        end
    elseif @capture ex plan(~head)
        ctx(head)
    elseif @capture ex plan(~head, ~tail...)
        ctx(head)
        return ctx(plan(tail...))
    else
        error("Unrecognized logic: $(ex)")
    end
end

"""
    LogicInterpreter(scope = Dict(), verbose = false, mode = :fast)

The LogicInterpreter is a simple interpreter for finch logic programs. The interpreter is
only capable of executing programs of the form:
      REORDER := reorder(relabel(ALIAS, FIELD...), FIELD...)
       ACCESS := reorder(relabel(ALIAS, idxs_1::FIELD...), idxs_2::FIELD...) where issubsequence(idxs_1, idxs_2)
    POINTWISE := ACCESS | mapjoin(IMMEDIATE, POINTWISE...) | reorder(IMMEDIATE, FIELD...) | IMMEDIATE
    MAPREDUCE := POINTWISE | aggregate(IMMEDIATE, IMMEDIATE, POINTWISE, FIELD...)
       TABLE  := table(IMMEDIATE, FIELD...)
COMPUTE_QUERY := query(ALIAS, reformat(IMMEDIATE, arg::(REORDER | MAPREDUCE)))
  INPUT_QUERY := query(ALIAS, TABLE)
         STEP := COMPUTE_QUERY | INPUT_QUERY | produces(ALIAS...)
         ROOT := PLAN(STEP...)
"""
@kwdef struct LogicInterpreter
    verbose = false
    mode = :fast
end

function Base.:(==)(a::LogicInterpreter, b::LogicInterpreter)
    a.verbose == b.verbose && a.mode == b.mode
end
function Base.hash(a::LogicInterpreter, h::UInt)
    hash(LogicInterpreter, hash(a.verbose, hash(a.mode, h)))
end

function set_options(ctx::LogicInterpreter; verbose=ctx.verbose, mode=ctx.mode, kwargs...)
    LogicInterpreter(; verbose=verbose, mode=mode)
end

function (ctx::LogicInterpreter)(prgm)
    prgm = format_queries(prgm)
    LogicMachine(; verbose=ctx.verbose, mode=ctx.mode)(prgm)
end
