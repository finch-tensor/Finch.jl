struct EinsumEagerStyle end
struct EinsumLazyStyle end
combine_style(::EinsumEagerStyle, ::EinsumEagerStyle) = EinsumEagerStyle()
combine_style(::EinsumLazyStyle, ::EinsumLazyStyle) = EinsumLazyStyle()
combine_style(::EinsumEagerStyle, ::EinsumLazyStyle) = EinsumLazyStyle()

einsum_style(arg) = EinsumEagerStyle()
einsum_style(::LazyTensor) = EinsumLazyStyle()

struct EinsumTensor{Style,Arg<:LazyTensor}
    style::Style
    arg::Arg
end

einsum_tensor(tns) = EinsumTensor(einsum_style(tns), lazy(tns))

struct EinsumArgument{Vf,Tv,Style}
    style::Style
    data::LogicNode
    shape::Dict{Symbol,Any}
end

function EinsumArgument{Vf,Tv}(style::Style, data, shape) where {Vf,Tv,Style}
    EinsumArgument{Vf,Tv,Style}(style, data, shape)
end

Base.eltype(::EinsumArgument{Vf,Tv}) where {Vf,Tv} = Tv
fill_value(::EinsumArgument{Vf}) where {Vf} = Vf

function einsum_access(tns::EinsumTensor, idxs...)
    EinsumArgument{fill_value(tns.arg),eltype(tns.arg)}(
        tns.style,
        relabel(tns.arg.data, map(field, idxs)...),
        Dict(idx => idx_shape for (idx, idx_shape) in zip(idxs, tns.arg.shape)),
    )
end

function einsum_op(op, args::EinsumArgument...)
    EinsumArgument{
        op((fill_value(arg) for arg in args)...),
        return_type(DefaultAlgebra(), op, map(eltype, args)...),
    }(
        reduce(result_style, [arg.style for arg in args]; init=EinsumEagerStyle()),
        mapjoin(op, (arg.data for arg in args)...),
        merge((arg.shape for arg in reverse(args))...),
    )
end

function einsum_immediate(val)
    EinsumArgument{val,typeof(val)}(EinsumEagerStyle(), immediate(val), Dict())
end

struct EinsumProgram{Style,Arg<:LazyTensor}
    style::Style
    arg::Arg
end

function einsum(
    ::typeof(overwrite), arg::EinsumArgument{Vf}, idxs...; init=nothing
) where {Vf}
    einsum(initwrite(Vf), arg, idxs...; init=Vf)
end

function einsum(
    op, arg::EinsumArgument{Vf,Tv}, idxs...; init=initial_value(op, Tv)
) where {Vf,Tv}
    shape = ntuple(n -> arg.shape[idxs[n]], length(idxs))
    data = reorder(
        aggregate(
            immediate(op),
            immediate(init),
            arg.data,
            map(field, setdiff(collect(keys(arg.shape)), idxs))...,
        ),
        map(field, idxs)...,
    )
    einsum_execute(arg.style, LazyTensor{init,typeof(init)}(data, shape))
end

function einsum_execute(::EinsumEagerStyle, arg)
    compute(arg)
end

function einsum_execute(::EinsumLazyStyle, arg)
    arg
end

struct EinsumArgumentParserVisitor
    preamble
    space
    output
    inputs
end

function (ctx::EinsumArgumentParserVisitor)(ex)
    if @capture ex :ref(~tns, ~idxs...)
        tns isa Symbol ||
            ArgumentError("Einsum expressions must reference named tensor Symbols.")
        tns != ctx.output ||
            ArgumentError("Einsum expressions must not reference the output tensor.")
        for idx in idxs
            idx isa Symbol ||
                ArgumentError("Einsum expressions must use named index Symbols.")
        end
        my_tns = get!(ctx.inputs, tns) do
            res = freshen(ctx.space, tns)
            push!(ctx.preamble.args, :($res = $einsum_tensor($(esc(tns)))))
            res
        end
        return :($einsum_access($my_tns, $(map(QuoteNode, idxs)...)))
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
        return :($einsum_op($(esc(op)), $(map(ctx, args)...)))
    elseif ex isa Expr
        throw(FinchSyntaxError("Invalid einsum expression: $ex"))
    else
        return :($einsum_immediate($(esc(ex))))
    end
end

struct EinsumParserVisitor
    preamble
    space
    opts
end

function (ctx::EinsumParserVisitor)(ex)
    if ex isa Expr
        if (@capture ex (~op)(~lhs, ~rhs)) && haskey(incs, op)
            return ctx(:($lhs << $(incs[op]) >>= $rhs))
        elseif @capture ex :(=)(~lhs, ~rhs)
            return ctx(:($lhs << $overwrite >>= $rhs))
        elseif @capture ex :>>=(:call(:<<, :ref(~tns, ~idxs...), ~op), ~rhs)
            tns isa Symbol ||
                ArgumentError("Einsum expressions must reference named tensor Symbols.")
            for idx in idxs
                idx isa Symbol ||
                    ArgumentError("Einsum expressions must use named index Symbols.")
            end
            arg = EinsumArgumentParserVisitor(ctx.preamble, ctx.space, tns, Dict())(rhs)
            quote
                $(esc(tns)) = $einsum(
                    $(esc(op)), $arg, $(map(QuoteNode, idxs)...); $(map(esc, ctx.opts)...)
                )
            end
        else
            throw(FinchNotation.FinchSyntaxError("Invalid einsum expression: $ex"))
        end
    else
        throw(FinchNotation.FinchSyntaxError("Invalid einsum expression type: $ex"))
    end
end

"""
    @einsum tns[idxs...] <<op>>= ex...

Construct an einsum expression that computes the result of applying `op` to the
tensor `tns` with the indices `idxs` and the tensors in the expression `ex`.
The result is stored in the variable `tns`.

`ex` may be any pointwise expression consisting of function calls and tensor
references of the form `tns[idxs...]`, where `tns` and `idxs` are symbols.

The `<<op>>` operator can be any binary operator that is defined on the element
type of the expression `ex`.

The einsum will evaluate the pointwise expression `tns[idxs...] <<op>>= ex...`
over all combinations of index values in `tns` and the tensors in `ex`.

Here are a few examples:
```
@einsum C[i, j] += A[i, k] * B[k, j]
@einsum C[i, j, k] += A[i, j] * B[j, k]
@einsum D[i, k] += X[i, j] * Y[j, k]
@einsum J[i, j] = H[i, j] * I[i, j]
@einsum N[i, j] = K[i, k] * L[k, j] - M[i, j]
@einsum R[i, j] <<max>>= P[i, k] + Q[k, j]
@einsum x[i] = A[i, j] * x[j]
```
"""
macro einsum(opts_ex...)
    length(opts_ex) >= 1 ||
        throw(ArgumentError("Expected at least one argument to @finch(opts..., ex)"))
    (opts, ex) = (opts_ex[1:(end - 1)], opts_ex[end])
    preamble = Expr(:block)
    space = Namespace()
    res = EinsumParserVisitor(preamble, space, opts)(ex)
    quote
        $preamble
        $res
    end
end
