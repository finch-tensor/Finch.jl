@kwdef struct Switch
    cases
end

@kwdef struct Case
    cond
    body
end

Base.first(arg::Case) = arg.cond
Base.last(arg::Case) = arg.body

FinchNotation.finch_leaf(x::Switch) = virtual(x)

struct SwitchStyle end

get_style(ctx, ::Switch, root) = SwitchStyle()
combine_style(a::DefaultStyle, b::SwitchStyle) = SwitchStyle()
combine_style(a::LookupStyle, b::SwitchStyle) = SwitchStyle()
combine_style(a::ThunkStyle, b::SwitchStyle) = ThunkStyle()
combine_style(a::SimplifyStyle, b::SwitchStyle) = a
combine_style(a::RunStyle, b::SwitchStyle) = SwitchStyle()
combine_style(a::AcceptRunStyle, b::SwitchStyle) = SwitchStyle()
combine_style(a::SpikeStyle, b::SwitchStyle) = SwitchStyle()
combine_style(a::SwitchStyle, b::SwitchStyle) = SwitchStyle()

function get_switch_cases(ctx, node::FinchNode)
    if node.kind === virtual
        get_switch_cases(ctx, node.val)
    elseif istree(node)
        map(product(map(arg -> get_switch_cases(ctx, arg), arguments(node))...)) do case
            guards = map(first, case)
            bodies = map(last, case)
            return simplify(ctx, call(and, guards...)) =>
                similarterm(node, operation(node), collect(bodies))
        end
    else
        [(literal(true) => node)]
    end
end
get_switch_cases(ctx, node) = [(literal(true) => node)]
get_switch_cases(ctx, node::Switch) = node.cases

function lower(ctx::AbstractCompiler, stmt, ::SwitchStyle)
    cases = get_switch_cases(ctx, stmt)
    function nest(cases, inner=false)
        guard, body = cases[1]
        body = contain(ctx) do ctx_2
            (ctx_2)(body)
        end
        length(cases) == 1 && return body
        inner && return Expr(:elseif, ctx(guard), body, nest(cases[2:end], true))
        return Expr(:if, ctx(guard), body, nest(cases[2:end], true))
    end
    return nest(cases)
end

Base.show(io::IO, ex::Switch) = Base.show(io, MIME"text/plain"(), ex)
function Base.show(io::IO, mime::MIME"text/plain", ex::Switch)
    print(io, "Switch([...])")
end
