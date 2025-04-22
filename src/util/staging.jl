function is_function_def(node)
    (@capture node :function(~args...)) ||
        (@capture node :->(~args...)) ||
        (@capture node (:(=))(:call(~f, ~args...), ~body)) ||
        (@capture node (:(=))(:where(:call(~f, ~args...), ~types), ~body))
end

has_function_def(root) = any(is_function_def, PostOrderDFS(root))

staged_defs = []

"""
    Finch.@staged

This function is used internally in Finch in lieu of @generated functions. It
ensures the first Finch invocation runs in the latest world, and leaves hooks so
that subsequent calls to [`Finch.refresh`](@ref) can update the world and
invalidate old versions. If the body contains closures, this macro uses an
eval and invokelatest strategy. Otherwise, it uses a generated function.
This macro does not support type parameters, varargs, or keyword arguments.
"""
macro staged(def)
    (@capture def :function(:call(~name, ~args...), ~body)) ||
        throw(ArgumentError("unrecognized function definition in @staged"))

    name_generator = gensym(Symbol(name, :_generator))
    name_invokelatest = gensym(Symbol(name, :_invokelatest))
    name_eval_invokelatest = gensym(Symbol(name, :_eval_invokelatest))

    def = quote
        function $name_generator($(args...))
            $body
        end

        function $name_invokelatest($(args...))
            $(Base.invokelatest)($name_eval_invokelatest, $(args...))
        end

        function $name_eval_invokelatest($(args...))
            code = $name_generator($(map((arg) -> :(typeof($arg)), args)...))
            def = quote
                function $($(QuoteNode(name_invokelatest)))(
                    $($(map(arg -> :(:($($(QuoteNode(arg)))::$(typeof($arg)))), args)...))
                )
                    $($(QuoteNode(name_eval_invokelatest)))($($(map(QuoteNode, args)...)))
                end
                function $($(QuoteNode(name_eval_invokelatest)))(
                    $($(map(arg -> :(:($($(QuoteNode(arg)))::$(typeof($arg)))), args)...))
                )
                    $code
                end
            end
            ($@__MODULE__).eval(def)
            $(Base.invokelatest)(($@__MODULE__).$name_eval_invokelatest, $(args...))
        end

        function $(Symbol(name, :_code))($(args...))
            code = $name_generator($(map((arg) -> :(typeof($arg)), args)...))
        end

        @generated function $name($(args...))
            # Taken from https://github.com/NHDaly/StagedFunctions.jl/blob/6fafbc560421f70b05e3df330b872877db0bf3ff/src/StagedFunctions.jl#L116
            body_2 =
                () -> begin
                    code = $name_generator($(args...))
                    if has_function_def(macroexpand($@__MODULE__, code))
                        :($($(name_invokelatest))($($(map(QuoteNode, args)...))))
                    else
                        quote
                            $code
                        end
                    end
                end
            Core._apply_pure(body_2, ())
        end
    end

    return esc(
        quote
            push!(staged_defs, $(QuoteNode(def)))
            $(def)
        end,
    )
end

"""
    Finch.refresh()

Finch caches the code for kernels as soon as they are run. If you modify the
Finch compiler after running a kernel, you'll need to invalidate the Finch
caches to reflect these changes by calling `Finch.refresh()`. This function
should only be called at global scope, and never during precompilation.
"""
function refresh()
    for def in staged_defs
        println(def)
        @eval $def
    end
end
