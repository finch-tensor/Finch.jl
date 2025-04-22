"""
    pagerank(adj; [nsteps], [damp])

Calculate `nsteps` steps of the page rank algorithm on the graph specified by
the adjacency matrix `adj`. `damp` is the damping factor.
"""
function pagerank(edges; nsteps=20, damp=0.85)
    (n, m) = size(edges)
    @assert n == m
    out_degree = Tensor(Dense(Element(0)))
    @finch begin
        out_degree .= 0
        for j in _, i in _
            out_degree[j] += edges[i, j]
        end
    end
    scaled_edges = Tensor(Dense(SparseList(Element(0.0))))
    @finch begin
        scaled_edges .= 0
        for j in _, i in _
            if out_degree[i] != 0
                scaled_edges[i, j] = edges[i, j] / out_degree[j]
            end
        end
    end
    r = Tensor(Dense(Element(0.0)), n)
    @finch begin
        r .= 0.0
        for j in _
            r[j] = 1.0 / n
        end
    end
    rank = Tensor(Dense(Element(0.0)), n)
    beta_score = (1 - damp) / n

    for step in 1:nsteps
        @finch begin
            rank .= 0
            for j in _, i in _
                rank[i] += scaled_edges[i, j] * r[j]
            end
        end
        @finch begin
            r .= 0.0
            for i in _
                r[i] = beta_score + damp * rank[i]
            end
        end
    end
    return r
end
