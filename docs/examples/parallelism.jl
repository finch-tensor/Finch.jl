using Finch
using SparseArrays
using MatrixDepot
using BenchmarkTools

A = Tensor(Dense(SparseList(Element(0.0))), SparseMatrixCSC(matrixdepot("Boeing/ct20stif")))
(m, n) = size(A)
x = Tensor(Dense(Element(0.0)), randn(m))
y = Tensor(Dense(Element(0.0)))

println("serial")
@btime begin
    (A, x, y) = $((A, x, y))
    Finch.@finch begin
        y .= 0
        for j in _
            for i in _
                y[j] += A[walk(i), j] * x[i]
            end
        end
    end
end

println("parallel")
@btime begin
    (A, x, y) = $(A, x, y)
    Finch.@finch begin
        y .= 0
        for j in parallel(_)
            for i in _
                y[j] += A[walk(i), j] * x[i]
            end
        end
    end
end
