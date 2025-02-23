"""
    dfs(edges; [source])

Calculate a depth-first search on the graph specified by the `edges` adjacency
matrix. Return the parents.
"""
function dfs(edges, source=5)
    (n, m) = size(edges)
    # 1 for all non-zero values else 0
    edges = pattern!(edges)

    @assert n == m

    # visited vector
    V = Tensor(Dense(Element(false)), n)

    # parent of each value
    P = Tensor(Dense(Element(0)), n)
    @finch P[source] = source

    stack = [source]

    while !isempty(stack)
        j = popfirst!(stack)

        if V[j]
            continue
        end

        @finch V[j] |= true

        _stack = []

        @finch begin
            for k in _
                if edges[k, j] && !(V[k])
                    P[k] = j
                    push!(_stack, k)
                end
            end
        end

        stack = vcat(_stack, stack)
    end
    return P
end
