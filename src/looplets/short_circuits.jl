@kwdef struct ShortCircuitVisitor
    ctx
end

function (ctx::ShortCircuitVisitor)(node::FinchNode)
    if @capture node assign(access(~tns::isvirtual, ~m, ~i...), ~op, ~rhs)
        map(short_circuit_cases(ctx.ctx, tns.val, op)) do (guard, body)
            guard => assign(access(body, m, i...), op, rhs)
        end
    elseif istree(node)
        mapreduce(vcat, enumerate(arguments(node)); init=[]) do (n, arg)
            map(ctx(arg)) do (guard, body)
                args_2 = copy(arguments(node))
                args_2[n] = body
                guard => similarterm(node, operation(node), args_2)
            end
        end
    else
        []
    end
end

short_circuit_cases(ctx, node, op) = []
