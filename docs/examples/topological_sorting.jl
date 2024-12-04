"""
    topological_sorting(edges)

Calculate the order of the vertices in topological order

The output is given as a vector of vertices in the topological sorting order

# Arguments
- `edges`: `edge` must be Directed Acyclic Graph (DAG) such that
        `edge[i, j]` is the edge from j to i
"""
function topological_sorting(edges)
        (n, m) = size(edges)
        edges = pattern!(edges)

        @assert n == m
        parent_count = Tensor(Dense(Element(0)), n)
        already_added = Tensor(Dense(Element(false)), n)
        to_add = Tensor(Dense(Element(false)), n)
        result = Tensor(Dense(Element(0)), n)
        index = 1

        @finch begin
                for j = _, i = _
                        if edges[i, j]
                                parent_count[i] += 1
                        end
                end
        end
        while index <= n
                @finch begin
                        to_add .= false
                        for i = _
                                if (parent_count[i] == 0) && !(already_added[i])
                                        to_add[i] = true
                                end
                        end
                end
                @finch begin
                        for j = _, i = _
                                if to_add[j] && edges[i, j]
                                        parent_count[i] += -1
                                end
                        end
                end
                for i = 1:n
                        if to_add[i]
                                already_added[i] = true
                                result[index] = i
                                index += 1
                        end
                end
        end
        return result
end
