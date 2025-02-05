"""
    bfs(edges; [source])

Calculate a breadth-first search on the graph specified by the `edges` adjacency
matrix. Return the node numbering.
"""
function bfs(edges, source=5)
    (n, m) = size(edges)
    edges = pattern!(edges)

    @assert n == m
    F = Tensor(SparseByteMap(Pattern()), n)
    _F = Tensor(SparseByteMap(Pattern()), n)
    @finch F[source] = true
    F_nnz = 1

    V = Tensor(Dense(Element(false)), n)
    @finch V[source] = true

    P = Tensor(Dense(Element(0)), n)
    @finch P[source] = source

    while F_nnz > 0
        @finch begin
            _F .= false
            for j in _, k in _
                if F[j] && edges[k, j] && !(V[k])
                    _F[k] |= true
                    P[k] << choose(0) >>= j #Only set the parent for this vertex
                end
            end
        end
        c = Scalar(0)
        @finch begin
            for k in _
                let _f = _F[k]
                    V[k] |= _f
                    c[] += _f
                end
            end
        end
        (F, _F) = (_F, F)
        F_nnz = c[]
    end
    return P
end
