struct FinchConcurrencyError
    msg
end

"""
    is_injective(ctx, tns)

Returns a vector of booleans, one for each dimension of the tensor, indicating
whether the access is injective in that dimension.  A dimension is injective if
each index in that dimension maps to a different memory space in the underlying
array.
"""
function is_injective end

"""
    is_atomic(ctx, tns)

    Returns a tuple (atomicities, overall) where atomicities is a vector, indicating which indices have an atomic that guards them,
    and overall is a boolean that indicates is the last level had an atomic guarding it.
"""
function is_atomic end

"""
    is_concurrent(ctx, tns)

    Returns a vector of booleans, one for each dimension of the tensor, indicating
    whether the index can be written to without any execution state. So if a matrix returns [true, false],
    then we can write to A[i, j] and A[i_2, j] without any shared execution state between the two, but
    we can't write to A[i, j] and A[i, j_2] without carrying over execution state.
"""
function is_concurrent end

"""
ensure_concurrent(root, ctx)

Ensures that all nonlocal assignments to the tensor root are consistently
accessed with the same indices and associative operator.  Also ensures that the
tensor is either atomic, or accessed by `i` and concurrent and injective on `i`.
"""
function ensure_concurrent(root, ctx)
    @assert @capture root loop(~idx, ~ext, ~body)

    #get local definitions
    locals = Set(filter(
        !isnothing,
        map(PostOrderDFS(body)) do node
            if @capture(node, declare(~tns, ~init, ~op))
                tns
            end
        end,
    ))

    #get nonlocal assignments and group by root
    nonlocal_assigns = Dict()
    for node in PostOrderDFS(body)
        if @capture(node, assign(~lhs, ~op, ~rhs)) && !(getroot(lhs.tns) in locals) &&
            getroot(lhs.tns) !== nothing #TODO remove the nothing check
            push!(get!(nonlocal_assigns, getroot(lhs.tns), []), node)
        end
    end

    # Get all indices in the parallel region.
    indices_in_region = Set([idx])
    for node in PostOrderDFS(body)
        if @capture node loop(~idx_2, ~ext, ~body)
            push!(indices_in_region, idx_2)
        end
    end

    for (root, agns) in nonlocal_assigns
        ops = map(agn -> (@capture agn assign(~lhs, ~op, ~rhs); op), agns)
        if !allequal(ops)
            throw(
                FinchConcurrencyError(
                    "Nonlocal assignments to $(root) are not all the same operator"
                ),
            )
        end

        accs = map(agn -> (@capture agn assign(~lhs, ~op, ~rhs); lhs), agns)
        acc = first(accs)
        # The operation must be associative.
        op = first(ops)
        if !(isassociative(ctx.algebra, op))
            if (length(ops) == 1)
                if (@capture(acc, access(~tns, ~mode, ~i...)))
                    injectivities = is_injective(ctx, tns)
                    concurrencies = is_concurrent(ctx, tns)
                    if !all(injectivities) || !all(concurrencies)
                        throw(
                            FinchConcurrencyError(
                                "Non-associative operations can only be parallelized in the case of a single injective acceses, but the injectivities is $(injectivities) and the concurrency is $(concurrencies)."
                            ),
                        )
                    else
                        continue # We pass via a single assignment that is completely injective.
                    end
                else
                    throw(FinchConcurrencyError("Assignment $(acc) is invalid!"))
                end
            end
            throw(
                FinchConcurrencyError(
                    "Nonlocal assignments to $(root) via $(op) are not associative"
                ),
            )
        end
        # If the acceses are different, then all acceses must be atomic.
        if !allequal(accs)
            for acc in accs
                (atomicities, _) = is_atomic(ctx, acc.tns)
                concurrencies = is_concurrent(ctx, acc.tns)
                if !all(atomicities) || !all(concurrencies)
                    throw(
                        FinchConcurrencyError(
                            "Nonlocal assignments to $(root) are not all the same access so concurrency and atomics are needed on all acceses!"
                        ),
                    )
                end
            end
            continue
        else
            #Since all operations/acceses are the same, a more fine grained analysis takes place:
            #Every access must be injective or they must all be atomic.
            if (@capture(acc, access(~tns, ~mode, ~i...)))
                injectivities::Vector{Bool} = is_injective(ctx, tns)
                concurrencies = is_concurrent(ctx, acc.tns)
                parallel_modes = findall(j -> j in indices_in_region, i)
                if length(parallel_modes) == 0
                    (atomicities, overall) = is_atomic(ctx, acc.tns)
                    if !([atomicities; overall])[1]
                        throw(
                            FinchConcurrencyError(
                                "Assignment $(acc) requires last level atomics!"
                            ),
                        )
                        # FIXME: we could do atomic operations here.
                    else
                        continue
                    end
                end

                #TODO If we could prove that some indices do not depend on the parallel index, we could exempt them from this somehow.
                if all(injectivities[parallel_modes]) && all(concurrencies[parallel_modes])
                    continue # We pass due to injectivity!
                end
                # FIXME: This could be more fine grained: atomics need to only protect the non-injectivity.
                (atomicities, _) = is_atomic(ctx, acc.tns)
                if all(atomicities[parallel_modes]) && all(concurrencies[parallel_modes])
                    continue # we pass due to atomics!
                else
                    throw(
                        FinchConcurrencyError(
                            "Assignment $(acc) requires injectivity or atomics in at least modes $(parallel_modes), but does not have them, due to injectivity=$(injectivities) and atomics=$(atomicities) and concurrency=$(concurrencies)."
                        ),
                    )
                end

                #TODO perhaps if the last access is the parallel index, we only need injectivity or atomics on the parallel one, and concurrency on that one only
            else
                throw(FinchConcurrencyError("Assignment $(acc) is invalid! "))
            end
        end
    end
    # we validated everything so we are done!
    return root
end
