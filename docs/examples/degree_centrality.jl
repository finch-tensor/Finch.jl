"""
    degree_centrality(edges; [normalize])

Calculate the degree centrality of each node, optionally normalized 
"""
function degree_centrality(edges, normalize=true)
    (n, m) = size(edges)
    edges = pattern!(edges)

    @assert n == m

    P = Tensor(Dense(Element(0)), n)

    @finch begin
        for j in _
            for k in _
                if edges[k, j]
                    P[j] += 1
                    P[k] += 1
                end
            end
        end
    end

    if normalize
        P = P ./ (n - 1)
    end

    return P
end
