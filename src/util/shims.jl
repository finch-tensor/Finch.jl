function fill_range!(arr, v, i, j)
    @simd for k = i:j
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
Base.@propagate_inbounds function scansearch(v, x, lo::T1, hi::T2) where {T1<:Integer, T2<:Integer} # TODO types for `lo` and `hi` #406
    u = T1(1)
    d = T1(1)
    p = lo
    while p < hi && v[p] < x
        d <<= 0x01
        p +=  d
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

Base.@propagate_inbounds function bin_scansearch(v, x, lo::T1, hi::T2) where {T1<:Integer, T2<:Integer} # TODO types for `lo` and `hi` #406
    u = T1(1)
    stop = min(hi, lo + T1(32))
    while lo + u < stop && v[lo] < x
        lo += u
    end
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

# Taken from https://github.com/c42f/FastClosures.jl/blob/master/src/FastClosures.jl
#
# The FastClosures.jl package is licensed under the MIT "Expat" License:
#
#    Copyright (c) 2017: Claire Foster.
#
#    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

struct Var
    name::Symbol
    num_esc::Int
end

function makelet(v::Var)
    ex = v.name
    for i=1:v.num_esc
        ex = esc(ex)
    end
    :($ex=$ex)
end

"""
    @barrier args... ex

Wrap `ex` in a let block that captures all free variables in `ex` that are bound in the arguments. This is useful for
ensuring that the variables in `ex` are not mutated by the arguments.
"""
macro barrier(args_ex...)
    (args, ex) = args_ex[1:end-1], args_ex[end]
    bound_vars = Var[Var(v,0) for v in args]
    captured_vars = Var[]
    find_var_uses!(captured_vars, bound_vars, ex, 1)
    quote
        let $(map(makelet, captured_vars)...)
            $ex
        end
    end
end

# Utility function - fill `varlist` with all accesses to variables inside `ex`
# which are not bound before being accessed.  Variables which were bound
# before access are returned in `bound_vars` as a side effect.
#
# With works with the surface syntax so it unfortunately has to reproduce some
# of the lowering logic (and consequently likely has bugs!)
# Utility function - fill `varlist` with all accesses to variables inside `ex`
# which are not bound before being accessed.  Variables which were bound
# before access are returned in `bound_vars` as a side effect.
#
# With works with the surface syntax so it unfortunately has to reproduce some
# of the lowering logic (and consequently likely has bugs!)
function find_var_uses!(varlist, bound_vars, ex, num_esc)
    if isa(ex, Symbol)
        var = Var(ex,num_esc)
        if !(var in bound_vars)
            var ∈ varlist || push!(varlist, var)
        end
        return varlist
    elseif isa(ex, Expr)
        if ex.head == :quote || ex.head == :line || ex.head == :inbounds
            return varlist
        end
        if ex.head == :(=)
            find_var_uses_lhs!(varlist, bound_vars, ex.args[1], num_esc)
            find_var_uses!(varlist, bound_vars, ex.args[2], num_esc)
        elseif ex.head == :kw
            find_var_uses!(varlist, bound_vars, ex.args[2], num_esc)
        elseif ex.head == :for || ex.head == :while || ex.head == :comprehension || ex.head == :let
            # New scopes
            inner_bindings = copy(bound_vars)
            find_var_uses!(varlist, inner_bindings, ex.args, num_esc)
        elseif ex.head == :try
            # New scope + ex.args[2] is a new binding
            find_var_uses!(varlist, copy(bound_vars), ex.args[1], num_esc)
            catch_bindings = copy(bound_vars)
            !isa(ex.args[2], Symbol) || push!(catch_bindings, Var(ex.args[2],num_esc))
            find_var_uses!(varlist,catch_bindings,ex.args[3], num_esc)
            if length(ex.args) > 3
                finally_bindings = copy(bound_vars)
                find_var_uses!(varlist,finally_bindings,ex.args[4], num_esc)
            end
        elseif ex.head == :call
            find_var_uses!(varlist, bound_vars, ex.args[2:end], num_esc)
        elseif ex.head == :local
            foreach(ex.args) do e
                if !isa(e, Symbol)
                    find_var_uses!(varlist, bound_vars, e, num_esc)
                end
            end
        elseif ex.head == :(::)
            find_var_uses_lhs!(varlist, bound_vars, ex, num_esc)
        elseif ex.head == :escape
            # In the 0.7-DEV churn, escapes persist during recursive macro
            # expansion until all macros are expanded.  Therefore, we
            # need to need to keep track of the number of escapes we're
            # currently inside, and to replay these when we construct the let
            # expression. See https://github.com/JuliaLang/julia/issues/23221
            # for additional pain and gnashing of teeth ;-)
            find_var_uses!(varlist, bound_vars, ex.args[1], num_esc+1)
        else
            find_var_uses!(varlist, bound_vars, ex.args, num_esc)
        end
    end
    varlist
end

find_var_uses!(varlist, bound_vars, exs::Vector, num_esc) =
    foreach(e->find_var_uses!(varlist, bound_vars, e, num_esc), exs)

# Find variable uses on the left hand side of an assignment.  Some of what may
# be variable uses turn into bindings in this context (cf. tuple unpacking).
function find_var_uses_lhs!(varlist, bound_vars, ex, num_esc)
    if isa(ex, Symbol)
        var = Var(ex,num_esc)
        var ∈ bound_vars || push!(bound_vars, var)
    elseif isa(ex, Expr)
        if ex.head == :tuple
            find_var_uses_lhs!(varlist, bound_vars, ex.args, num_esc)
        elseif ex.head == :(::)
            find_var_uses!(varlist, bound_vars, ex.args[2], num_esc)
            find_var_uses_lhs!(varlist, bound_vars, ex.args[1], num_esc)
        else
            find_var_uses!(varlist, bound_vars, ex.args, num_esc)
        end
    end
end

find_var_uses_lhs!(varlist, bound_vars, exs::Vector, num_esc) = foreach(e->find_var_uses_lhs!(varlist, bound_vars, e, num_esc), exs)