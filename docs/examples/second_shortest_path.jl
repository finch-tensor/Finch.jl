using Finch
using SparseArrays
"""
    dijkstras(adj, source=1)

Calculate the shortest paths from the vertex `source` in the graph specified by
an adjacency matrix `adj`, whose entries are edge weights. Weights should be
infinite when unconnected.

The output is given as a vector of distance, parent pairs for each node in the graph.
"""
function dijkstras(edges, source=1)
    (n, m) = size(edges)
    @assert n == m

    dists = Tensor(Dense(Element((Inf, 0))), n)
    @finch dists[source] = (0.0, 0)
    dists_prev = Tensor(Dense(Element((Inf, 0))), n)
    @finch dists_prev[source] = (0.0, 0)

    pq = Tensor(Dense(Element(Inf)), n)
    @finch pq[source] = 0.0

    for iter = 1:n
        u = Scalar((Inf, 0))
        @finch begin
            for j = _
                if (pq[j] >= 0) u[] <<minby>>= (pq[j], j) end
            end
        end
        @finch pq[last(u[])] = -1
        @finch begin
            for j = _
                let d = first(u[]) + edges[j, last(u[])]
                    dists[j] <<minby>>= (d, last(u[]))
                    pq[j] <<choose(Inf)>>= min(d, first(dists_prev[j]))
                end
            end
        end

        dists_prev = dists
    end

    return dists
end

"""
    second_shortest_path(adj, source=1)

Calculate the second shortest paths from the vertex `source` in the graph specified by
an adjacency matrix `adj`, whose entries are edge weights. Weights should be
infinite when there does not exist a second shortest path.

The output is given as a vector of distance, parent pairs for each node in the graph.
"""
function second_shortest_path(edges, source = 1)
    (n, m) = size(edges)
    @assert n == m

    # Step 1: SSSP on original graph
    δ_G = dijkstras(edges, source)

    # Step 2: Create new graph
    edges_prime = Tensor(SparseByteMap(SparseByteMap(Element(Inf))), 2*n, 2*n)
    @finch begin
        edges_prime .= Inf
        for j=_,i=_
            if (edges[i, j] != Inf) 
                let d = edges[i, j]
                    edges_prime[min(2*n, 2*i) - 1,min(2*n, 2*j) - 1] = d
                    edges_prime[min(2*n, 2*i),min(2*n, 2*j)] = d
                    if (first(δ_G[i]) < first(δ_G[j]) + d) 
                        edges_prime[min(2*n, 2*i),min(2*n, 2*j) - 1] = d
                    end
                end
            end
        end
    end
    δ_G2 = dijkstras(edges_prime, 2 * source - 1)

    # Convert to correct format
    dists = Tensor(Dense(Element((Inf, 0))), n)
    for j = 1:n
        dists[j] = (first(δ_G2[2*j]), floor((last(δ_G2[2*j])+1)/2))
    end
    return dists
end