function fill_range!(arr, v, i, j)
    @simd for k in i:j
        arr[k] = v
    end
    arr
end

function resize_if_smaller!(arr, i)
    if length(arr) < i
        resize!(arr, i)
    end
end

"""
    scansearch(v, x, lo, hi)

return the first value of `v` greater than or equal to `x`, within the range
`lo:hi`. Return `hi+1` if all values are less than `x`. This implemantation uses an
exponential search strategy which involves two steps: 1) searching for binary search bounds
via exponential steps rightward 2) binary searching within those bounds.
"""
Base.@propagate_inbounds function scansearch(
    v, x, lo::T1, hi::T2
) where {T1<:Integer,T2<:Integer} # TODO types for `lo` and `hi` #406
    u = T1(1)
    d = T1(1)
    p = lo
    while p < hi && v[p] < x
        d <<= 0x01
        p += d
    end
    lo = p - d
    hi = min(p, hi) + u

    while lo < hi - u
        m = lo + ((hi - lo) >>> 0x01)
        if v[m] < x
            lo = m
        else
            hi = m
        end
    end
    return hi
end

Base.@propagate_inbounds function bin_scansearch(
    v, x, lo::T1, hi::T2
) where {T1<:Integer,T2<:Integer} # TODO types for `lo` and `hi` #406
    u = T1(1)
    lo = lo - u
    hi = hi + u
    while lo < hi - u
        m = lo + ((hi - lo) >>> 0x01)
        if v[m] < x
            lo = m
        else
            hi = m
        end
    end
    return hi
end

"""
    @barrier args... ex

Wrap `ex` in a let block that captures all free variables in `ex` that are bound in the arguments. This is useful for
ensuring that the variables in `ex` are not mutated by the arguments.
"""
macro barrier(args_ex...)
    (args, ex) = args_ex[1:(end - 1)], args_ex[end]
    f = gensym()
    esc(
        quote
            $f = Finch.@closure ($(args...),) -> $ex
            $f()
        end,
    )
end

# wrap_closure is taken from https://github.com/c42f/FastClosures.jl
#
# The FastClosures.jl package is licensed under the MIT "Expat" License:
# Copyright (c) 2017: Claire Foster.
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Wrap `closure_expression` in a `let` block to improve efficiency.
function wrap_closure(module_, ex)
    bound_vars = Symbol[]
    captured_vars = Symbol[]
    if @capture ex :->(:tuple(~args...), ~body)
    elseif @capture ex :function(:call(~f, ~args...), ~body)
        push!(bound_vars, f)
    else
        throw(
            ArgumentError(
                "Argument to @closure must be a closure!  (Got $closure_expression)"
            ),
        )
    end
    append!(bound_vars, [v for v in args])
    find_var_uses!(captured_vars, bound_vars, body)
    quote
        let $(map(var -> :($var = $var), captured_vars)...)
            $ex
        end
    end
end

"""
    @closure closure_expression

Wrap the closure definition `closure_expression` in a let block to encourage
the julia compiler to generate improved type information.  For example:

```julia
callfunc(f) = f()

function foo(n)
   for i=1:n
       if i >= n
           # Unlikely event - should be fast.  However, capture of `i` inside
           # the closure confuses the julia-0.6 compiler and causes it to box
           # the variable `i`, leading to a 100x performance hit if you remove
           # the `@closure`.
           callfunc(@closure ()->println("Hello \$i"))
       end
   end
end
```

There's nothing nice about this - it's a heuristic workaround for some
inefficiencies in the type information inferred by the julia 0.6 compiler.
However, it can result in large speedups in many cases, without the need to
restructure the code to avoid the closure.
"""
macro closure(ex)
    esc(wrap_closure(__module__, ex))
end

# Utility function - fill `varlist` with all accesses to variables inside `ex`
# which are not bound before being accessed.  Variables which were bound
# before access are returned in `bound_vars` as a side effect.
#
# With works with the surface syntax so it unfortunately has to reproduce some
# of the lowering logic (and consequently likely has bugs!)
function find_var_uses!(capture_vars, bound_vars, ex)
    if isa(ex, Symbol)
        if !(ex in bound_vars)
            #occursin("threadsfor", string(ex)) && error()
            ex ∈ capture_vars || push!(capture_vars, ex)
        end
        return capture_vars
    elseif isa(ex, Expr)
        if ex.head == :(=)
            find_var_uses_lhs!(capture_vars, bound_vars, ex.args[1])
            find_var_uses!(capture_vars, bound_vars, ex.args[2])
        elseif @capture ex :->(:tuple(~args...), ~body)
            body_vars = copy(bound_vars)
            for arg in args
                find_var_uses_lhs!(capture_vars, body_vars, arg)
            end
            find_var_uses!(capture_vars, body_vars, body)
        elseif ex.head == :kw
            find_var_uses!(capture_vars, bound_vars, ex.args[2])
        elseif ex.head == :for || ex.head == :while || ex.head == :let
            # New scopes
            inner_bindings = copy(bound_vars)
            find_var_uses!(capture_vars, inner_bindings, ex.args)
        elseif ex.head == :try
            # New scope + ex.args[2] is a new binding
            find_var_uses!(capture_vars, copy(bound_vars), ex.args[1])
            catch_bindings = copy(bound_vars)
            !isa(ex.args[2], Symbol) || push!(catch_bindings, ex.args[2])
            find_var_uses!(capture_vars, catch_bindings, ex.args[3])
            if length(ex.args) > 3
                finally_bindings = copy(bound_vars)
                find_var_uses!(capture_vars, finally_bindings, ex.args[4])
            end
        elseif ex.head == :call
            find_var_uses!(capture_vars, bound_vars, ex.args[2:end])
        elseif ex.head == :local
            foreach(ex.args) do e
                if !isa(e, Symbol)
                    find_var_uses!(capture_vars, bound_vars, e)
                end
            end
        elseif ex.head == :(::)
            find_var_uses_lhs!(capture_vars, bound_vars, ex)
        else
            find_var_uses!(capture_vars, bound_vars, ex.args)
        end
    end
    capture_vars
end

function find_var_uses!(capture_vars, bound_vars, exs::Vector)
    foreach(e -> find_var_uses!(capture_vars, bound_vars, e), exs)
end

# Find variable uses on the left hand side of an assignment.  Some of what may
# be variable uses turn into bindings in this context (cf. tuple unpacking).
function find_var_uses_lhs!(capture_vars, bound_vars, ex)
    if isa(ex, Symbol)
        ex ∈ bound_vars || push!(bound_vars, ex)
    elseif isa(ex, Expr)
        if ex.head == :tuple
            find_var_uses_lhs!(capture_vars, bound_vars, ex.args)
        elseif ex.head == :(::)
            find_var_uses!(capture_vars, bound_vars, ex.args[2])
            find_var_uses_lhs!(capture_vars, bound_vars, ex.args[1])
        else
            find_var_uses!(capture_vars, bound_vars, ex.args)
        end
    end
end

function find_var_uses_lhs!(capture_vars, bound_vars, exs::Vector)
    foreach(e -> find_var_uses_lhs!(capture_vars, bound_vars, e), exs)
end
