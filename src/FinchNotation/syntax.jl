const incs = Dict(:+= => :+, :*= => :*, :&= => :&, :|= => :|, :(:=) => :overwrite)
const evaluable_exprs = [
    :Inf,
    :Inf16,
    :Inf32,
    :Inf64,
    :(-Inf),
    :(-Inf16),
    :(-Inf32),
    :(-Inf64),
    :NaN,
    :NaN16,
    :NaN32,
    :NaN64,
    :nothing,
    :missing,
    :Eps,
]

const program_nodes = (
    index=index,
    loop=loop,
    sieve=sieve,
    block=block,
    define=define,
    declare=declare,
    freeze=freeze,
    thaw=thaw,
    assign=assign,
    call=call,
    access=access,
    yieldbind=yieldbind,
    reader=reader,
    updater=updater,
    variable=variable,
    tag=(ex) -> :(finch_leaf($(esc(ex)))),
    literal=literal,
    leaf=(ex) -> :(finch_leaf($(esc(ex)))),
    auto=:(finch_leaf(auto)),
)

const instance_nodes = (
    index=index_instance,
    loop=loop_instance,
    sieve=sieve_instance,
    block=block_instance,
    define=define_instance,
    declare=declare_instance,
    freeze=freeze_instance,
    thaw=thaw_instance,
    assign=assign_instance,
    call=call_instance,
    access=access_instance,
    yieldbind=yieldbind_instance,
    reader=reader_instance,
    updater=updater_instance,
    variable=variable_instance,
    tag=(ex) ->
        :($tag_instance($(variable_instance(ex)), $finch_leaf_instance($(esc(ex))))),
    literal=literal_instance,
    leaf=(ex) -> :($finch_leaf_instance($(esc(ex)))),
    auto=:($finch_leaf_instance(auto)),
)

d() = 1
d(args...) = 0
and() = true
and(x) = x
and(x, y, tail...) = x && and(y, tail...)
or() = false
or(x) = x
or(x, y, tail...) = x || or(y, tail...)

struct InitWriter{Vf} end

(f::InitWriter{Vf})(x) where {Vf} = x
function (f::InitWriter{Vf})(x, y) where {Vf}
    @debug begin
        @assert isequal(x, Vf)
    end
    y
end

"""
    initwrite(z)(a, b)

`initwrite(z)` is a function which may assert that `a`
[`isequal`](https://docs.julialang.org/en/v1/base/base/#Base.isequal) to `z`,
and `returns `b`.  By default, `lhs[] = rhs` is equivalent to `lhs[]
<<initwrite(fill_value(lhs))>>= rhs`.
"""
initwrite(z) = InitWriter{z}()

"""
    overwrite(z)(a, b)

`overwrite(z)` is a function which returns `b` always. `lhs[] := rhs` is equivalent to
`lhs[] <<overwrite>>= rhs`.

```jldoctest setup=:(using Finch)
julia> a = Tensor(SparseList(Element(0.0)), [0, 1.1, 0, 4.4, 0])
5 Tensor{SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}:
 0.0
 1.1
 0.0
 4.4
 0.0

julia> x = Scalar(0.0); @finch for i=_; x[] <<overwrite>>= a[i] end;

julia> x[]
0.0
```
"""
overwrite(l, r) = r

"""
    Auto()

A singleton type representing the lack of a dimension.  This is used in place of
a dimension when we want to avoid dimensionality checks. In the `@finch` macro,
you can write `Auto()` with an underscore as `for i = _`, allowing
finch to pick up the loop bounds from the tensors automatically.
"""
struct Auto end
const auto = Auto()
function extent end
function realextent end

struct FinchParserVisitor
    nodes
end

function (ctx::FinchParserVisitor)(ex::Symbol)
    if ex == :_ || ex == :(:)
        return :($auto)
    elseif ex in evaluable_exprs
        return ctx.nodes.literal(@eval($ex))
    else
        ctx.nodes.tag(ex)
    end
end
(ctx::FinchParserVisitor)(ex::QuoteNode) = ctx.nodes.literal(ex.value)
(ctx::FinchParserVisitor)(ex) = ctx.nodes.literal(ex) #TODO error on any unrecognized syntax like this.

struct FinchSyntaxError
    msg
end

function (ctx::FinchParserVisitor)(ex::Expr)
    islinenum(ex) = ex isa LineNumberNode
    if @capture ex :if(~cond, ~body)
        return :($(ctx.nodes.sieve)($(ctx(cond)), $(ctx(body))))
    elseif @capture ex :if(~cond, ~body, ~tail)
        throw(
            FinchSyntaxError(
                "Finch does not support else, elseif, or the ternary operator. Consider using multiple if blocks, or the ifelse() function instead."
            ),
        )
    elseif @capture ex :elseif(~args...)
        throw(FinchSyntaxError("Finch does not support elseif."))
    elseif @capture ex :(.=)(~tns, ~init)
        return :($(ctx.nodes.declare)(
            $(ctx(tns)), $(ctx(init)), $(ctx.nodes.literal)(auto)
        ))
    elseif @capture ex :macrocall($(Symbol("@declare")), ~ln::islinenum, ~tns, ~init, ~op)
        return :($(ctx.nodes.declare)($(ctx(tns)), $(ctx(init)), $(ctx(op))))
    elseif @capture ex :macrocall($(Symbol("@freeze")), ~ln::islinenum, ~tns, ~op)
        return :($(ctx.nodes.freeze)($(ctx(tns)), $(ctx(op))))
    elseif @capture ex :macrocall($(Symbol("@thaw")), ~ln::islinenum, ~tns, ~op)
        return :($(ctx.nodes.thaw)($(ctx(tns)), $(ctx(op))))
    elseif @capture ex :for(:block(), ~body)
        return ctx(body)
    elseif @capture ex :for(:block(:(=)(~idx, ~ext), ~tail...), ~body)
        if isempty(tail)
            return ctx(:(
                for $idx in $ext
                    $body
                end
            ))
        else
            return ctx(:(
                for $idx in $ext
                    $(Expr(:for, Expr(:block, tail...), body))
                end
            ))
        end
    elseif @capture ex :for(:(=)(~idx, ~ext), ~body)
        ext = ctx(ext)
        body = ctx(body)
        if idx isa Symbol
            return quote
                let $(esc(idx)) = $(ctx.nodes.index(idx))
                    $(ctx.nodes.loop)($(esc(idx)), $ext, $body)
                end
            end
        else
            return quote
                $(ctx.nodes.loop)($(ctx(idx)), $ext, $body)
            end
        end
    elseif @capture ex :let(:block(), ~body)
        return ctx(body)
    elseif @capture ex :let(:block(:(=)(~lhs, ~rhs), ~tail...), ~body)
        if isempty(tail)
            return ctx(:(
                let $lhs = $rhs
                    $body
                end
            ))
        else
            return ctx(:(
                let $lhs = $rhs
                    $(Expr(:let, Expr(:block, tail...), body))
                end
            ))
        end
    elseif @capture ex :let(:(=)(~lhs, ~rhs), ~body)
        rhs = ctx(rhs)
        body = ctx(body)
        if lhs isa Symbol
            return quote
                let $(esc(lhs)) = $(ctx.nodes.variable(lhs))
                    $(ctx.nodes.define)($(esc(lhs)), $rhs, $body)
                end
            end
        else
            return quote
                $(ctx.nodes.define)($(ctx(lhs)), $rhs, $body)
            end
        end
    elseif @capture ex :block(~bodies...)
        bodies = filter(!islinenum, bodies)
        if length(bodies) == 1
            return ctx(:($(bodies[1])))
        else
            return :($(ctx.nodes.block)($(map(ctx, bodies)...)))
        end
    elseif @capture ex :return(:tuple(~args...))
        return :($(ctx.nodes.yieldbind)($(map(ctx, args)...)))
    elseif @capture ex :return(~arg)
        return :($(ctx.nodes.yieldbind)($(ctx(arg))))
    elseif @capture ex :ref(~tns, ~idxs...)
        mode = ctx.nodes.reader
        return :($(ctx.nodes.access)($(ctx(tns)), $mode(), $(map(ctx, idxs)...)))
    elseif (@capture ex (~op)(~lhs, ~rhs)) && haskey(incs, op)
        return ctx(:($lhs << $(incs[op]) >>= $rhs))
    elseif @capture ex :(=)(:ref(~tns, ~idxs...), ~rhs)
        mode = ctx.nodes.updater
        op = :($(ctx.nodes.literal)($initwrite))
        lhs = :($(ctx.nodes.access)($(ctx(tns)), $mode($op), $(map(ctx, idxs)...)))
        return :($(ctx.nodes.assign)($lhs, $op, $(ctx(rhs))))
    elseif @capture ex :>>=(:call(:<<, :ref(~tns, ~idxs...), ~op), ~rhs)
        mode = ctx.nodes.updater
        lhs = :($(ctx.nodes.access)($(ctx(tns)), $mode($(ctx(op))), $(map(ctx, idxs)...)))
        return :($(ctx.nodes.assign)($lhs, $(ctx(op)), $(ctx(rhs))))
    elseif @capture ex :>>=(:call(:<<, ~lhs, ~op), ~rhs)
        error("Finch doesn't support incrementing definitions of variables")
    elseif @capture ex :(=)(~lhs, ~rhs)
        error("Finch doesn't support variable bindings outside of let statements")
    elseif @capture ex :tuple(~args...)
        return ctx(:(tuple($(args...))))
    elseif @capture ex :comparison(~a, ~cmp, ~b)
        return ctx(:($cmp($a, $b)))
    elseif @capture ex :comparison(~a, ~cmp, ~b, ~tail...)
        return ctx(:($cmp($a, $b) && $(Expr(:comparison, b, tail...))))
    elseif @capture ex :&&(~a, ~b)
        return ctx(:($and($a, $b)))
    elseif @capture ex :||(~a, ~b)
        return ctx(:($or($a, $b)))
    elseif @capture ex :call(~op, ~args...)
        if op == :(:)
            return :($(ctx.nodes.call)($(ctx(:extent)), $(map(ctx, args)...)))
        else
            return :($(ctx.nodes.call)($(ctx(op)), $(map(ctx, args)...)))
        end
    elseif @capture ex :(...)(~arg) #TODO error on any unrecognized syntax like this.
        return esc(ex)
    elseif @capture ex :$(~arg)
        return esc(arg)
    elseif ex in evaluable_exprs
        return ctx.nodes.literal(@eval(ex))
    else
        return ctx.nodes.leaf(ex)
    end
end

finch_parse_program(ex) = FinchParserVisitor(program_nodes)(ex)
finch_parse_instance(ex) = FinchParserVisitor(instance_nodes)(ex)
function finch_parse_yieldbind(ex)
    if @capture ex :$(~arg)
        return nothing
    elseif @capture ex :macrocall(~args)
        return nothing
    elseif @capture ex :return(~arg)
        if arg isa Symbol
            return [arg]
        elseif @capture arg :tuple(~args...)
            return filter(arg_2 -> arg_2 isa Symbol, collect(args))
        end
    elseif ex isa Expr
        return mapreduce(
            finch_parse_yieldbind,
            (x, y) -> something(x, y, Some(nothing)),
            ex.args;
            init=nothing,
        )
    end
end

function finch_parse_default_yieldbind(ex)
    if @capture ex :block(~args...)
        return mapreduce(finch_parse_default_yieldbind, vcat, args)
    elseif (@capture ex :(.=)(~tns, ~init)) && tns isa Symbol
        return [tns]
    else
        return []
    end
end

macro finch_program(ex)
    return finch_parse_program(ex)
end

macro finch_program_instance(ex)
    return :(
        let
            $(finch_parse_instance(ex))
        end
    )
end

display_expression(io, mime, ex) = show(IOContext(io, :compact => true), mime, ex) # TODO virtual or value is currently determined in virtualize.
function display_expression(io, mime, node::Union{FinchNode,FinchNodeInstance})
    if operation(node) === value
        print(io, node.val)
        if node.type !== Any
            print(io, "::")
            print(io, node.type)
        end
    elseif operation(node) === literal
        print(io, node.val)
    elseif operation(node) === index
        print(io, node.name)
    elseif operation(node) === variable
        print(io, node.name)
    elseif operation(node) === cached
        print(io, "cached(")
        display_expression(io, mime, node.arg)
        print(io, ", ")
        display_expression(io, mime, node.ref.val)
        print(io, ")")
    elseif operation(node) === tag
        print(io, "tag(")
        display_expression(io, mime, node.var)
        print(io, ", ")
        display_expression(io, mime, node.bind)
        print(io, ")")
    elseif operation(node) === virtual
        print(io, "virtual(")
        #print(io, node.val)
        summary(io, node.val)
        print(io, ")")
    elseif operation(node) === access
        display_expression(io, mime, node.tns)
        print(io, "[")
        if length(node.idxs) >= 1
            for idx in node.idxs[1:(end - 1)]
                display_expression(io, mime, idx)
                print(io, ", ")
            end
            display_expression(io, mime, node.idxs[end])
        end
        print(io, "]")
    elseif operation(node) === call
        display_expression(io, mime, node.op)
        print(io, "(")
        for arg in node.args[1:(end - 1)]
            display_expression(io, mime, arg)
            print(io, ", ")
        end
        if !isempty(node.args)
            display_expression(io, mime, node.args[end])
        end
        print(io, ")")
    elseif istree(node)
        print(io, operation(node))
        print(io, "(")
        for arg in arguments(node)[1:(end - 1)]
            print(io, arg)
            print(io, ",")
        end
        if !isempty(arguments(node))
            print(arguments(node)[end])
        end
    else
        error("unimplemented")
    end
end

function display_statement(io, mime, node::Union{FinchNode,FinchNodeInstance}, indent)
    if operation(node) === loop
        print(io, " "^indent * "for ")
        display_expression(io, mime, node.idx)
        print(io, " = ")
        display_expression(io, mime, node.ext)
        body = node.body
        while operation(body) === loop
            print(io, ", ")
            display_expression(io, mime, body.idx)
            print(io, " = ")
            display_expression(io, mime, body.ext)
            body = body.body
        end
        println(io)
        display_statement(io, mime, body, indent + 2)
        println(io)
        print(io, " "^indent * "end")
    elseif operation(node) === define
        print(io, " "^indent * "let ")
        display_expression(io, mime, node.lhs)
        print(io, " = ")
        display_expression(io, mime, node.rhs)
        body = node.body
        while operation(body) === define
            print(io, ", ")
            display_expression(io, mime, body.lhs)
            print(io, " = ")
            display_expression(io, mime, body.rhs)
            body = body.body
        end
        println(io)
        display_statement(io, mime, body, indent + 2)
        println(io)
        print(io, " "^indent * "end")
    elseif operation(node) === sieve
        print(io, " "^indent * "if ")
        while operation(node.body) === sieve
            display_expression(io, mime, node.cond)
            print(io, " && ")
            node = node.body
        end
        display_expression(io, mime, node.cond)
        println(io)
        node = node.body
        display_statement(io, mime, node, indent + 2)
        println(io)
        print(io, " "^indent * "end")
    elseif operation(node) === assign
        print(io, " "^indent)
        display_expression(io, mime, node.lhs)
        print(io, " <<")
        display_expression(io, mime, node.op)
        print(io, ">>= ")
        display_expression(io, mime, node.rhs)
    elseif operation(node) === declare
        if operation(node.op) === literal && node.op.val === auto
            print(io, " "^indent)
            display_expression(io, mime, node.tns)
            print(io, " .= ")
            display_expression(io, mime, node.init)
        else
            print(io, " "^indent * "@declare(")
            display_expression(io, mime, node.tns)
            print(io, ", ")
            display_expression(io, mime, node.init)
            print(io, ", ")
            display_expression(io, mime, node.op)
            print(io, ")")
        end
    elseif operation(node) === freeze
        print(io, " "^indent * "@freeze(")
        display_expression(io, mime, node.tns)
        print(io, ", ")
        display_expression(io, mime, node.op)
        print(io, ")")
    elseif operation(node) === thaw
        print(io, " "^indent * "@thaw(")
        display_expression(io, mime, node.tns)
        print(io, ", ")
        display_expression(io, mime, node.op)
        print(io, ")")
    elseif operation(node) === yieldbind
        print(io, " "^indent * "return (")
        for arg in node.args[1:(end - 1)]
            display_expression(io, mime, arg)
            print(io, ", ")
        end
        if !isempty(node.args)
            display_expression(io, mime, node.args[end])
        end
        print(io, ")")
    elseif operation(node) === block
        print(io, " "^indent * "begin\n")
        for body in node.bodies
            display_statement(io, mime, body, indent + 2)
            println(io)
        end
        print(io, " "^indent * "end")
    else
        println(node)
        error("unimplemented")
    end
end

finch_unparse_program(ctx, node) = finch_unparse_program(ctx, finch_leaf(node))
function finch_unparse_program(ctx, node::Union{FinchNode,FinchNodeInstance})
    if operation(node) === value
        node.val
    elseif operation(node) === literal
        node.val
    elseif operation(node) === index
        node.name
    elseif operation(node) === variable
        node.name
    elseif operation(node) === cached
        finch_unparse_program(ctx, node.arg)
    elseif operation(node) === tag
        @assert operation(node.var) === variable
        node.var.name
    elseif operation(node) === virtual
        if node.val == auto
            :_
        else
            ctx(node)
        end
    elseif operation(node) === access
        tns = finch_unparse_program(ctx, node.tns)
        idxs = map(x -> finch_unparse_program(ctx, x), node.idxs)
        :($tns[$(idxs...)])
    elseif operation(node) === call
        op = finch_unparse_program(ctx, node.op)
        args = map(x -> finch_unparse_program(ctx, x), node.args)
        :($op($(args...)))
    elseif operation(node) === loop
        idx = finch_unparse_program(ctx, node.idx)
        ext = finch_unparse_program(ctx, node.ext)
        body = finch_unparse_program(ctx, node.body)
        :(
            for $idx in $ext
                $body
            end
        )
    elseif operation(node) === define
        lhs = finch_unparse_program(ctx, node.lhs)
        rhs = finch_unparse_program(ctx, node.rhs)
        body = finch_unparse_program(ctx, node.body)
        :(
            let $lhs = $rhs
                $body
            end
        )
    elseif operation(node) === sieve
        cond = finch_unparse_program(ctx, node.cond)
        body = finch_unparse_program(ctx, node.body)
        :(
            if $cond
                $body
            end
        )
    elseif operation(node) === assign
        lhs = finch_unparse_program(ctx, node.lhs)
        op = finch_unparse_program(ctx, node.op)
        rhs = finch_unparse_program(ctx, node.rhs)
        if haskey(incs, op)
            Expr(incs[op], lhs, rhs)
        else
            :($lhs << $op >>= $rhs)
        end
    elseif operation(node) === declare
        tns = finch_unparse_program(ctx, node.tns)
        init = finch_unparse_program(ctx, node.init)
        if operation(node.op) === literal && node.op.val === auto
            :($tns .= $init)
        else
            op = finch_unparse_program(ctx, node.op)
            :(@declare($tns, $init, $op))
        end
    elseif operation(node) === freeze
        tns = finch_unparse_program(ctx, node.tns)
        op = finch_unparse_program(ctx, node.op)
        :(@freeze($tns, $op))
    elseif operation(node) === thaw
        tns = finch_unparse_program(ctx, node.tns)
        :(@thaw($tns))
    elseif operation(node) === yieldbind
        args = map(x -> finch_unparse_program(ctx, x), node.args)
        :(return ($(args...)))
    elseif operation(node) === block
        bodies = map(x -> finch_unparse_program(ctx, x), node.bodies)
        Expr(:block, bodies...)
    else
        error("unimplemented")
    end
end
