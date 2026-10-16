function is_function_def(node)
    (@capture node :function(~args...)) ||
        (@capture node :->(~args...)) ||
        (@capture node (:(=))(:call(~f, ~args...), ~body)) ||
        (@capture node (:(=))(:where(:call(~f, ~args...), ~types), ~body))
end

has_function_def(root) = any(is_function_def, PostOrderDFS(root))

staged_defs = []

"""
    Galley.@staged

This function is used internally in Galley in lieu of @generated functions. It
ensures the first Galley invocation runs in the latest world, and leaves hooks so
that subsequent calls to [`Galley.refresh`](@ref) can update the world and
invalidate old versions. If the body contains closures, this macro uses an
eval and invokelatest strategy. Otherwise, it uses a generated function.
This macro does not support type parameters, varargs, or keyword arguments.
"""
macro staged(def)
    return esc(
        quote
            $(staged_maker(@__MODULE__, def))
            push!(staged_defs, (@__MODULE__, $(QuoteNode(def))))
        end,
    )
end

function staged_maker(mod, def)
    (@capture def :function(:call(~name, ~args...), ~body)) ||
        throw(ArgumentError("unrecognized function definition in @staged"))

    name_code = Symbol(name, :_code)
    name_generator = Symbol(name, :_generator)
    name_eval_invokelatest = gensym(Symbol(name, :_eval_invokelatest))

    return quote
        function $name_generator($(args...))
            $body
        end

        function $name_code($(args...))
            code = $name_generator($(map((arg) -> :(typeof($arg)), args)...))
        end

        function $name($(args...))
            $(Base.invokelatest)($name_eval_invokelatest, $(args...))
        end

        function $name_eval_invokelatest($(args...))
            code = $name_code($(args...))
            def = quote
                function $($(QuoteNode(name_eval_invokelatest)))(
                    $($(map(arg -> :(:($($(QuoteNode(arg)))::$(typeof($arg)))), args)...))
                )
                    $code
                end
            end
            ($mod).eval(def)
            $(Base.invokelatest)(($mod).$name_eval_invokelatest, $(args...))
        end
    end
end

"""
    Galley.refresh()

Galley caches the code for kernels as soon as they are run. If you modify the
Galley compiler after running a kernel, you'll need to invalidate the Galley
caches to reflect these changes by calling `Galley.refresh()`. This function
should only be called at global scope, and never during precompilation.
"""
function refresh()
    for (mod, def) in staged_defs
        mod.eval(staged_maker(mod, def))
    end
end
