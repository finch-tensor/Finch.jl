"""
    get_wrapper_rules(ctx, depth, alg)

Return the wrapperizing rule set for Finch, which converts expressions like `A[i
+ 1]` to array combinator expressions like `OffsetArray(A, (1,))`. The rules have
access to the algebra `alg` and the depth lookup `depth`` One can dispatch on
the `alg` trait to specialize the rule set for different algebras. These rules run
after simplification so one can expect constants to be folded.
"""
function get_wrapper_rules(ctx, depth, alg)
    return [
        (@rule access(~A, ~m, ~i1..., call(~proto::isliteral, ~j), ~i2...) =>
            if isprotocol(proto.val)
                protos = ([nothing for _ in i1]..., proto.val, [nothing for _ in i2]...)
                access(call(protocolize, A, protos...), m, i1..., j, i2...)
            end),
        (@rule call(protocolize, call(protocolize, ~A, ~protos_1...), ~protos_2...) =>
            begin
                protos_3 = map(protos_1, protos_2) do proto_1, proto_2
                    something(getval(proto_1), getval(proto_2), Some(nothing))
                end
                call(protocolize, A, protos_3...)
            end),
        (@rule call(protocolize, call(swizzle, ~A, ~sigma...), ~protos...) =>
            call(
                swizzle, call(protocolize, A, protos[invperm(getval.(sigma))]...), sigma...
            )),
        (@rule access(~A, ~m, ~i1..., call($(~), ~j), ~i2...) => begin
            dims = ([false for _ in i1]..., true, [false for _ in i2]...)
            access(call(permissive, A, dims...), m, i1..., j, i2...)
        end),
        (@rule call(permissive, call(permissive, ~A, ~dims_1...), ~dims_2...) => begin
            union_dims = getval.(dims_1) .| getval.(dims_2)
            call(permissive, A, union_dims...)
        end),
        (@rule call(permissive, call(swizzle, ~A, ~sigma...), ~dims...) =>
            call(swizzle, call(permissive, A, dims[invperm(getval.(sigma))]...), sigma...)),
        (@rule access(~A, ~m, ~i1..., call(-, ~j, ~k), ~i2...) =>
            access(A, m, i1..., call(+, j, call(-, k)), i2...)),
        (@rule access(~A, ~m, ~i1..., call(+, ~j), ~i2...) =>
            access(A, m, i1..., j, i2...)),
        (@rule access(~A, ~m, ~i1..., call(*, ~j1..., ~k, ~j2...), ~i2...) => begin
            if !isempty(j1) || !isempty(j2)
                if length(j1) == 1 && isempty(j2)
                    k_2 = j1[1]
                elseif isempty(j1) && length(j2) == 1
                    k_2 = j2[1]
                else
                    k_2 = call(*, ~j1..., ~j2...)
                end

                if depth(k_2) < depth(k) && depth(k_2) != 0
                    access(call(products, A, length(i1) + 1), m, i1..., k, k_2, i2...)
                end
            end
        end),
        (@rule access(~A, ~m, ~i1..., call(*, ~j1..., ~k, ~j2...), ~i2...) => begin
            if !isempty(j1) || !isempty(j2)
                if length(j1) == 1 && isempty(j2)
                    k_2 = j1[1]
                elseif isempty(j1) && length(j2) == 1
                    k_2 = j2[1]
                else
                    k_2 = call(*, ~j1..., ~j2...)
                end

                if depth(k_2) == 0
                    s1 = ([1 for _ in i1]..., k_2, [1 for _ in i2]...)
                    access(call(scale, A, s1...), m, i1..., k, i2...)
                end
            end
        end),
        (@rule call(scale, call(scale, ~A, ~factors_1...), ~factors_2...) => begin
            factors_3 = map(factors_1, factors_2) do factor_1, factor_2
                call(*, factor_1, factor_2)
            end
            call(scale, A, factors_3...)
        end),
        (@rule access(
            ~A, ~m, ~i1::(All(isindex))..., call(+, ~j1..., ~k, ~j2...), ~i2...
        ) => begin
            if (!isempty(j1) || !isempty(j2))
                k_2 = call(+, ~j1..., ~j2...)
                if depth(k_2) < depth(k) && depth(k_2) != 0
                    access(call(toeplitz, A, length(i1) + 1), m, i1..., k, k_2, i2...)
                end
            end
        end),
        (@rule call(<, ~i, ~j::isindex) => begin
            if depth(i) < depth(j)
                access(VirtualLoTriMask(), reader(), j, call(+, i, 1))
            end
        end),
        (@rule call(<, ~i::isindex, ~j) => begin
            if depth(i) > depth(j)
                access(VirtualUpTriMask(), reader(), i, call(-, j, 1))
            end
        end),
        (@rule call(<=, ~i, ~j::isindex) => begin
            if depth(i) < depth(j)
                access(VirtualLoTriMask(), reader(), j, i)
            end
        end),
        (@rule call(<=, ~i::isindex, ~j) => begin
            if depth(i) > depth(j)
                access(VirtualUpTriMask(), reader(), i, j)
            end
        end),
        (@rule call(>, ~i, ~j::isindex) => begin
            if depth(i) < depth(j)
                access(VirtualUpTriMask(), reader(), j, call(-, i, 1))
            end
        end),
        (@rule call(>, ~i::isindex, ~j) => begin
            if depth(i) > depth(j)
                access(VirtualLoTriMask(), reader(), i, call(+, j, 1))
            end
        end),
        (@rule call(>=, ~i, ~j::isindex) => begin
            if depth(i) < depth(j)
                access(VirtualUpTriMask(), reader(), j, i)
            end
        end),
        (@rule call(>=, ~i::isindex, ~j) => begin
            if depth(i) > depth(j)
                access(VirtualLoTriMask(), reader(), i, j)
            end
        end),
        (@rule call(==, ~i, ~j::isindex) => begin
            if depth(i) < depth(j)
                access(VirtualDiagMask(), reader(), j, i)
            end
        end),
        (@rule call(==, ~i::isindex, ~j) => begin
            if depth(i) > depth(j)
                access(VirtualDiagMask(), reader(), i, j)
            end
        end),
        (@rule call(!=, ~i, ~j::isindex) => begin
            if depth(i) < depth(j)
                call(!, access(VirtualDiagMask(), reader(), j, i))
            end
        end),
        (@rule call(!=, ~i::isindex, ~j) => begin
            if depth(i) > depth(j)
                call(!, access(VirtualDiagMask(), reader(), i, j))
            end
        end),
        (@rule call(toeplitz, call(swizzle, ~A, ~sigma...), ~dim...) => begin
            sigma = getval.(sigma)
            idim = invperm(sigma)[dim]
            call(
                swizzle,
                call(toeplitz, A, idim),
                sigma[1:(idim - 1)]...,
                sigma[idim],
                sigma[idim],
                sigma[(idim + 1):end]...,
            )
        end),
        (@rule access(~A, ~m, ~i1..., call(+, ~j1..., ~k, ~j2...), ~i2...) => begin
            if !isempty(j1) || !isempty(j2)
                k_2 = call(+, ~j1..., ~j2...)
                if depth(k_2) == 0
                    delta = ([0 for _ in i1]..., k_2, [0 for _ in i2]...)
                    access(call(offset, A, delta...), m, i1..., k, i2...)
                end
            end
        end),
        (@rule call(offset, call(offset, ~A, ~deltas_1...), ~deltas_2...) => begin
            deltas_3 = map(deltas_1, deltas_2) do delta_1, delta_2
                call(+, delta_1, delta_2)
            end
            call(offset, A, deltas_3...)
        end),
        (@rule call(offset, call(swizzle, ~A, ~sigma...), ~delta...) =>
            call(swizzle, call(offset, A, delta[invperm(getval.(sigma))]...), sigma...)),
        (@rule access(~A, ~m, ~i1..., call(call(extent, ~start, ~stop), ~k), ~i2...) =>
            begin
                A_2 = call(
                    window,
                    A,
                    [nothing for _ in i1]...,
                    call(extent, start, stop),
                    [nothing for _ in i2]...,
                )
                A_3 = call(
                    offset, A_2, [0 for _ in i1]..., call(-, start, 1), [0 for _ in i2]...
                )
                access(A_3, m, i1..., k, i2...)
            end),
        (@rule access(~A, ~m, ~i1..., call(~I::isvirtual, ~k), ~i2...) =>
            if I.val isa Extent
                A_2 = call(window, A, [nothing for _ in i1]..., I, [nothing for _ in i2]...)
                A_3 = call(
                    offset,
                    A_2,
                    [0 for _ in i1]...,
                    call(-, getstart(I), 1),
                    [0 for _ in i2]...,
                )
                access(A_3, m, i1..., k, i2...)
            end),
        (@rule assign(access(~a, updater(initwrite), ~i...), initwrite, ~rhs) => begin
            assign(
                access(a, updater(call(initwrite, call(fill_value, a))), i...),
                call(initwrite, call(fill_value, a)),
                rhs,
            ) #updater(auto)
        end),
        (@rule call(swizzle, call(swizzle, ~A, ~sigma_1...), ~sigma_2...) =>
            call(swizzle, A, sigma_1[getval.(sigma_2)]...)),
        (@rule access(call(swizzle, ~A, ~sigma...), ~m, ~i...) =>
            access(A, m, i[invperm(Vector{Int}(getval.(sigma)))]...)),
    ]
end

"""
    wrapperize(ctx, root)

Convert index expressions in the program `root` to wrapper arrays, according to
the rules in `get_wrapper_rules`. By default, the following transformations are
performed:

```julia
A[i - j] => A[i + (-j)]
A[3 * i] => ScaleArray(A, (3,))[i]
A[i * j] => ProductArray(A, 1)[i, j]
A[i + 1] => OffsetArray(A, (1,))[i]
A[i + j] => ToeplitzArray(A, 1)[i, j]
A[~i] => PermissiveArray(A, 1)[i]
```

The loop binding order may be used to determine which index comes first in an
expression like `A[i + j]`. Thus, `for i=:,j=:; ... A[i + j]` will result in
`ToeplitzArray(A, 1)[j, i]`, but `for j=:,i=:; ... A[i + j]` results in
`ToeplitzArray(A, 1)[i, j]`. `wrapperize` runs before dimensionalization, so
resulting raw indices may participate in dimensionalization according to the
semantics of the wrapper.
"""
function wrapperize(ctx::AbstractCompiler, root)
    depth = depth_calculator(root)
    root = unwrap_roots(ctx, root)
    root = Rewrite(
        Prewalk(
        (@rule loop(~idx, ~ext, ~body) => begin
            counts = OrderedDict()
            for node in PostOrderDFS(body)
                if @capture(node, access(~tn, reader(), ~idxs...))
                    counts[node] = get(counts, node, 0) + 1
                end
            end
            applied = false
            for (node, count) in counts
                if depth(idx) == depth(node)
                    if @capture(node, access(~tn, reader(), ~idxs...)) && count > 1
                        var = variable(
                            Symbol(
                                freshen(ctx, tn.val),
                                "_",
                                join([idx.val for idx in idxs]),
                            ),
                        )
                        body = Postwalk(@rule node => var)(body)
                        body = define(var, access(tn, reader(), idxs...), body)
                        applied = true
                    end
                end
            end
            if applied
                loop(idx, ext, body)
            end
        end)
    ),
    )(
        root
    )
    root = Rewrite(
        Fixpoint(
            Chain([
                Postwalk(Fixpoint(Chain(get_wrapper_rules(ctx, depth, ctx.algebra))))
            ]),
        ),
    )(
        root
    )
    evaluate_partial(ctx, root)
end

function unwrap(ctx, x, var)
    if x isa FinchNode && isvirtual(x)
        finch_leaf(unwrap(ctx, x.val, var))
    else
        if var != x
            set_binding!(ctx, var, finch_leaf(x))
        end
        var
    end
end

function unwrap_roots(ctx, root)
    tnss = unique(filter(
        !isnothing,
        map(PostOrderDFS(root)) do node
            if @capture(node, access(~A, ~m, ~i...))
                if getroot(A) === nothing
                    @info "Hi" (A)
                end
                getroot(A)
            elseif @capture(node, declare(~A, ~i, ~op))
                A
            elseif @capture(node, freeze(~A, ~op))
                A
            elseif @capture(node, thaw(~A, ~op))
                A
            end
        end,
    ))
    root = Rewrite(
        Postwalk(@rule access(~A, ~m, ~i...) => access(unwrap(ctx, A, getroot(A)), m, i...))
    )(
        root
    )
    for tns in tnss
        @assert isvariable(tns)
        @assert has_binding(ctx, tns) "root tensor variable $tns is not defined as a global binding"
        val = get_binding(ctx, tns)
        val_2 = unwrap(ctx, val, tns)
        if val_2 != tns
            #@info "Unwrapping" tns val val_2
            root = Rewrite(Postwalk(@rule tns => val_2))(root)
            root = Rewrite(
                Postwalk(
                    Chain([
                        (@rule declare(val_2, ~i, ~op) => declare(tns, i, op)),
                        (@rule freeze(val_2, ~op) => freeze(tns, op)),
                        (@rule thaw(val_2, ~op) => thaw(tns, op)),
                    ]),
                ),
            )(
                root
            )
        end
    end
    root
end
