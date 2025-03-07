```@meta
CurrentModule = Finch
```

# Optimization Tips for Finch

It's easy to ask Finch to run the same operation in different ways. However,
different approaches have different performance. The right approach really
depends on your particular situation. Here's a collection of general approaches
that help Finch generate faster code in most cases.

## Concordant Iteration

By default, Finch stores arrays in column major order (first index fast). When
the storage order of an array in a Finch expression corresponds to the loop
order, we call this
*concordant* iteration. For example, the following expression represents a
concordant traversal of a sparse matrix, as the outer loops access the higher
levels of the tensor tree:

```jldoctest example1; setup=:(using Finch)
A = Tensor(
    Dense(SparseList(Element(0.0))),
    fsparse([2, 3, 4, 1, 3], [1, 1, 1, 3, 3], [1.1, 2.2, 3.3, 4.4, 5.5], (4, 3)),
)
s = Scalar(0.0)
@finch for j in _, i in _
    s[] += A[i, j]
end

# output

NamedTuple()
```

We can investigate the generated code with `@finch_code`.  This code iterates
over only the nonzeros in order. If our matrix is `m × n` with `nnz` nonzeros,
this takes `O(n + nnz)` time.

```jldoctest example1
@finch_code for j in _, i in _
    s[] += A[i, j]
end

# output

quote
    s = (ex.bodies[1]).body.body.lhs.tns.bind
    s_val = s.val
    A_lvl = (ex.bodies[1]).body.body.rhs.tns.bind.lvl
    A_lvl_2 = A_lvl.lvl
    A_lvl_ptr = A_lvl_2.ptr
    A_lvl_idx = A_lvl_2.idx
    A_lvl_2_val = A_lvl_2.lvl.val
    for j_3 = 1:A_lvl.shape
        A_lvl_q = (1 - 1) * A_lvl.shape + j_3
        A_lvl_2_q = A_lvl_ptr[A_lvl_q]
        A_lvl_2_q_stop = A_lvl_ptr[A_lvl_q + 1]
        if A_lvl_2_q < A_lvl_2_q_stop
            A_lvl_2_i1 = A_lvl_idx[A_lvl_2_q_stop - 1]
        else
            A_lvl_2_i1 = 0
        end
        phase_stop = min(A_lvl_2_i1, A_lvl_2.shape)
        if phase_stop >= 1
            if A_lvl_idx[A_lvl_2_q] < 1
                A_lvl_2_q = Finch.scansearch(A_lvl_idx, 1, A_lvl_2_q, A_lvl_2_q_stop - 1)
            end
            while true
                A_lvl_2_i = A_lvl_idx[A_lvl_2_q]
                if A_lvl_2_i < phase_stop
                    A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
                    s_val = A_lvl_3_val + s_val
                    A_lvl_2_q += 1
                else
                    phase_stop_3 = min(phase_stop, A_lvl_2_i)
                    if A_lvl_2_i == phase_stop_3
                        A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
                        s_val += A_lvl_3_val
                        A_lvl_2_q += 1
                    end
                    break
                end
            end
        end
    end
    result = ()
    s.val = s_val
    result
end
```

When the loop order does not correspond to storage order, we call this
*discordant* iteration. For example, if we swap the loop order in the
above example, then Finch needs to randomly access each sparse column for each
row `i`. We end up needing to find each `(i, j)` pair because we don't know
whether it will be zero until we search for it. In all, this takes time
`O(n * m * log(nnz))`, much less efficient! We shouldn't randomly access sparse
arrays unless we really need to and they support it efficiently!

Note the double for loop in the following code

```jldoctest example1
@finch_code for i in _, j in _
    s[] += A[i, j]
end # DISCORDANT, DO NOT DO THIS

# output

quote
    s = (ex.bodies[1]).body.body.lhs.tns.bind
    s_val = s.val
    A_lvl = (ex.bodies[1]).body.body.rhs.tns.bind.lvl
    A_lvl_2 = A_lvl.lvl
    A_lvl_ptr = A_lvl_2.ptr
    A_lvl_idx = A_lvl_2.idx
    A_lvl_2_val = A_lvl_2.lvl.val
    @warn "Performance Warning: non-concordant traversal of A[i, j] (hint: most arrays prefer column major or first index fast, run in fast mode to ignore this warning)"
    for i_3 = 1:A_lvl_2.shape
        for j_3 = 1:A_lvl.shape
            A_lvl_q = (1 - 1) * A_lvl.shape + j_3
            A_lvl_2_q = A_lvl_ptr[A_lvl_q]
            A_lvl_2_q_stop = A_lvl_ptr[A_lvl_q + 1]
            if A_lvl_2_q < A_lvl_2_q_stop
                A_lvl_2_i1 = A_lvl_idx[A_lvl_2_q_stop - 1]
            else
                A_lvl_2_i1 = 0
            end
            phase_stop = min(i_3, A_lvl_2_i1)
            if phase_stop >= i_3
                if A_lvl_idx[A_lvl_2_q] < i_3
                    A_lvl_2_q = Finch.scansearch(A_lvl_idx, i_3, A_lvl_2_q, A_lvl_2_q_stop - 1)
                end
                while true
                    A_lvl_2_i = A_lvl_idx[A_lvl_2_q]
                    if A_lvl_2_i < phase_stop
                        A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
                        s_val = A_lvl_3_val + s_val
                        A_lvl_2_q += 1
                    else
                        phase_stop_3 = min(phase_stop, A_lvl_2_i)
                        if A_lvl_2_i == phase_stop_3
                            A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
                            s_val += A_lvl_3_val
                            A_lvl_2_q += 1
                        end
                        break
                    end
                end
            end
        end
    end
    result = ()
    s.val = s_val
    result
end
```

TL;DR: As a quick heuristic, if your array indices are all in alphabetical order, then
the loop indices should be reverse alphabetical.

## Appropriate Fill Values

The @finch macro requires the user to specify an output format. This is the most
flexibile approach, but can sometimes lead to densification unless the output
fill value is appropriate for the computation.

For example, if `A` is `m × n` with `nnz` nonzeros, the following Finch kernel will
densify `B`, filling it with `m * n` stored values:

```jldoctest example1
A = Tensor(
    Dense(SparseList(Element(0.0))),
    fsparse([2, 3, 4, 1, 3], [1, 1, 1, 3, 3], [1.1, 2.2, 3.3, 4.4, 5.5], (4, 3)),
)
B = Tensor(Dense(SparseList(Element(0.0)))) #DO NOT DO THIS, B has the wrong fill value
@finch (B .= 0;
for j in _, i in _
    B[i, j] = A[i, j] + 1
end;
return B)
countstored(B)

# output

12
```

Since `A` is filled with `0.0`, adding `1` to the fill value produces `1.0`. However, `B` can only represent a fill value of `0.0`. Instead, we should specify `1.0` for the fill.

```jldoctest example1
A = Tensor(
    Dense(SparseList(Element(0.0))),
    fsparse([2, 3, 4, 1, 3], [1, 1, 1, 3, 3], [1.1, 2.2, 3.3, 4.4, 5.5], (4, 3)),
)
B = Tensor(Dense(SparseList(Element(1.0))))
@finch (B .= 1;
for j in _, i in _
    B[i, j] = A[i, j] + 1
end;
return B)
countstored(B)

# output

5
```

## Static Versus Dynamic Values

In order to skip some computations, Finch must be able to determine the value of
program variables. Continuing our above example, if we obscure the value of `1`
behind a variable `x`, Finch can only determine that `x` has type `Int`, not that it is `1`.

```jldoctest example1
A = Tensor(
    Dense(SparseList(Element(0.0))),
    fsparse([2, 3, 4, 1, 3], [1, 1, 1, 3, 3], [1.1, 2.2, 3.3, 4.4, 5.5], (4, 3)),
)
B = Tensor(Dense(SparseList(Element(1.0))))
x = 1 #DO NOT DO THIS, Finch cannot see the value of x anymore
@finch (B .= 1;
for j in _, i in _
    B[i, j] = A[i, j] + x
end;
return B)
countstored(B)

# output

12
```

However, there are some situations where you may want a value to be dynamic. For example, consider the function `saxpy(x, a, y) = x .* a .+ y`. Because we do not know the value of `a` until we run the function, we should treat it as dynamic, and the following implementation is reasonable:

```julia
function saxpy(x, a, y)
    z = Tensor(SparseList(Element(0.0)))
    @finch (z .= 0;
    for i in _
        z[i] = a * x[i] + y[i]
    end;
    return z)
end
```

## Use Known Functions

Unless you declare the properties of your functions using Finch's [User-Defined Functions](@ref) interface, Finch doesn't know how they work. For example, using a lambda obscures
the meaning of `*`.

```jldoctest example1
A = Tensor(
    Dense(SparseList(Element(0.0))),
    fsparse([2, 3, 4, 1, 3], [1, 1, 1, 3, 3], [1.1, 2.2, 3.3, 4.4, 5.5], (4, 3)),
)
B = ones(4, 3)
C = Scalar(0.0)
f(x, y) = x * y # DO NOT DO THIS, Obscures *
@finch (C .= 0;
for j in _, i in _
    C[] += f(A[i, j], B[i, j])
end;
return C)

# output

(C = Scalar{0.0, Float64}(16.5),)
```

Checking the generated code, we see that this code is indeed densifying (notice the for-loop which repeatedly evaluates `f(B[i, j], 0.0)`).

```jldoctest example1
@finch_code (C .= 0;
for j in _, i in _
    C[] += f(A[i, j], B[i, j])
end;
return C)

# output

quote
    C = ((ex.bodies[1]).bodies[1]).tns.bind
    A_lvl = (((ex.bodies[1]).bodies[2]).body.body.rhs.args[1]).tns.bind.lvl
    A_lvl_2 = A_lvl.lvl
    A_lvl_ptr = A_lvl_2.ptr
    A_lvl_idx = A_lvl_2.idx
    A_lvl_2_val = A_lvl_2.lvl.val
    B_data = (((ex.bodies[1]).bodies[2]).body.body.rhs.args[2]).tns.bind
    sugar_1 = size((((ex.bodies[1]).bodies[2]).body.body.rhs.args[2]).tns.bind)
    B_mode1_stop = sugar_1[1]
    B_mode2_stop = sugar_1[2]
    B_mode1_stop == A_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(B_mode1_stop) != $(A_lvl_2.shape))"))
    B_mode2_stop == A_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(B_mode2_stop) != $(A_lvl.shape))"))
    C_val = 0
    for j_4 = 1:B_mode2_stop
        A_lvl_q = (1 - 1) * A_lvl.shape + j_4
        A_lvl_2_q = A_lvl_ptr[A_lvl_q]
        A_lvl_2_q_stop = A_lvl_ptr[A_lvl_q + 1]
        if A_lvl_2_q < A_lvl_2_q_stop
            A_lvl_2_i1 = A_lvl_idx[A_lvl_2_q_stop - 1]
        else
            A_lvl_2_i1 = 0
        end
        phase_stop = min(B_mode1_stop, A_lvl_2_i1)
        if phase_stop >= 1
            i = 1
            if A_lvl_idx[A_lvl_2_q] < 1
                A_lvl_2_q = Finch.scansearch(A_lvl_idx, 1, A_lvl_2_q, A_lvl_2_q_stop - 1)
            end
            while true
                A_lvl_2_i = A_lvl_idx[A_lvl_2_q]
                if A_lvl_2_i < phase_stop
                    for i_6 = i:-1 + A_lvl_2_i
                        val = B_data[i_6, j_4]
                        C_val = (Main).f(0.0, val) + C_val
                    end
                    A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
                    val_2 = B_data[A_lvl_2_i, j_4]
                    C_val += (Main).f(A_lvl_3_val, val_2)
                    A_lvl_2_q += 1
                    i = A_lvl_2_i + 1
                else
                    phase_stop_3 = min(phase_stop, A_lvl_2_i)
                    if A_lvl_2_i == phase_stop_3
                        for i_8 = i:-1 + phase_stop_3
                            val_3 = B_data[i_8, j_4]
                            C_val += (Main).f(0.0, val_3)
                        end
                        A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
                        val_4 = B_data[phase_stop_3, j_4]
                        C_val += (Main).f(A_lvl_3_val, val_4)
                        A_lvl_2_q += 1
                    else
                        for i_10 = i:phase_stop_3
                            val_5 = B_data[i_10, j_4]
                            C_val += (Main).f(0.0, val_5)
                        end
                    end
                    i = phase_stop_3 + 1
                    break
                end
            end
        end
        phase_start_3 = max(1, 1 + A_lvl_2_i1)
        if B_mode1_stop >= phase_start_3
            for i_12 = phase_start_3:B_mode1_stop
                val_6 = B_data[i_12, j_4]
                C_val += (Main).f(0.0, val_6)
            end
        end
    end
    C.val = C_val
    (C = C,)
end

```

## Type Stability

Julia code runs fastest when the compiler can [infer the
types](https://docs.julialang.org/en/v1/manual/performance-tips/#Write-%22type-stable%22-functions)
of all intermediate values.  Finch does not check that the generated code is
type-stable. In situations where tensors have nonuniform index or element types,
or the computation itself might involve multiple types, one should check that
the output of `@finch_kernel` code is type-stable with
[`@code_warntype`](https://docs.julialang.org/en/v1/stdlib/InteractiveUtils/#InteractiveUtils.@code_warntype).

## Dense Arrays

Finch is currently optimized for sparse code and does not implement traditional dense optimizations. We are currently adding these features, but if you need dense performance, you may want to look at [JuliaGPU](https://github.com/JuliaGPU) instead.
