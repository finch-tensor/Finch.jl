using Finch;

function floyd_warshall_step(dist, k)
    n, m = size(dist)
    @assert n == m
    res = Tensor(Dense(SparseList(Element(Inf))), n, m)
    @finch begin
        res .= Inf
        for j in 1:m
            for i in 1:n
                res[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])
            end
        end
    end
    return res
end

"""
  floyd_warshall(adj)

Performs floyd-warshall on an adjacency matrix.
The distance from any node to itself should be zero,
and the distance from any node to any non-connected node should be infinite.
"""
function floyd_warshall(adj)
    n, m = size(adj)
    @assert n == m
    for i in 1:n
        @assert adj[i, i] == 0
    end
    for k in 1:n
        adj = floyd_warshall_step(adj, k)
    end
    return adj
end
