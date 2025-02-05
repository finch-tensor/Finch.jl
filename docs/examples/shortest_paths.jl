"""
    bellmanford(adj, source=1)

Calculate the shortest paths from the vertex `source` in the graph specified by
an adjacency matrix `adj`, whose entries are edge weights. Weights should be
infinite when unconnected.

The output is given as a vector of distance, parent pairs for each node in the graph.
"""
function bellmanford(edges, source=1)
    (n, m) = size(edges)
    @assert n == m

    dists_prev = Tensor(Dense(Element((Inf, 0))), n)
    dists_prev[source] = (0.0, 0)
    dists = Tensor(Dense(Element((Inf, 0))), n)
    active_prev = Tensor(SparseByteMap(Pattern()), n)
    active_prev[source] = true
    active = Tensor(SparseByteMap(Pattern()), n)

    for iter in 1:n
        @finch for j in _
            if active_prev[j]
                dists[j] << minby >>= dists_prev[j]
            end
        end

        @finch begin
            active .= false
            for j in _
                if active_prev[j]
                    for i in _
                        let d = first(dists_prev[j]) + edges[i, j]
                            dists[i] << minby >>= (d, j)
                            active[i] |= d < first(dists_prev[i])
                        end
                    end
                end
            end
        end

        if !any(active)
            return dists
        end
        dists_prev, dists = dists, dists_prev
        active_prev, active = active, active_prev
    end

    return dists_prev
end
