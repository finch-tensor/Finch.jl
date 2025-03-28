"""
    topological_sort(edges)

Calculate the order of the vertices in topological order

The output is given as a vector of vertices in the topological sorting order

# Arguments
- `edges`: `edge` must be Directed Acyclic Graph (DAG) such that
        `edge[i, j]` is the edge from j to i
"""
function topological_sort(edges)
    (n, m) = size(edges)
    edges = pattern!(edges)

    @assert n == m
    parent_count = Tensor(Dense(Element(0)), n)
    level = Tensor(Dense(Element(0)), n)
    active = Tensor(SparseByteMap(Element(false)), n)
    frontier = Tensor(SparseByteMap(Element(false)), n)
    step = Scalar(1)

    @finch begin
        for j in _, i in _
            if edges[i, j]
                parent_count[i] += 1
            end
        end
    end

    @finch begin
        active .= false
        for i in _
            if parent_count[i] == 0
                active[i] = true
            end
        end
    end

    while step[] <= n
        @finch begin
            frontier .= false
            for i in _
                if active[i] && (parent_count[i] == 0) && level[i] == 0
                    frontier[i] = true
                end
            end
        end
        @finch begin
            active .= false
            for j in _, i in _
                if frontier[j] && edges[i, j]
                    parent_count[i] += -1
                    active[i] = true
                end
            end
        end
        _step = step[]
        @finch begin
            for i in _
                if frontier[i]
                    level[i] = _step
                    step[] += 1
                end
            end
        end
    end
    return sortperm(Array(level))
end
