"""
    tricount(adj)

Count the number of triangles in the graph specified by `adj`, which is assumed to be
symmetric. Requires edges to be 1 and non-edges 0.
"""
function tricount(edges)
    (n, m) = size(edges)
    @assert n == m

    #store lower triangles
    L = Tensor(Dense(SparseList(Element(0))), n, n)
    @finch begin
        L .= 0
        for j in _, i in _
            L[i, j] = lotrimask[i, j + 1] * edges[i, j]
        end
    end

    triangles = Scalar(0)
    @finch for j in _, k in _, i in _
        triangles[] += L[i, k] * L[k, j] * L[i, j]
    end

    return triangles[]
end
