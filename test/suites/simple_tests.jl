@testitem "simple" begin
    using SparseArrays

    x = Tensor(Sparse(Element(0.0)))
    y = Tensor(SparseList(Element(0.0)), sprand(10, 0.5))
    z = Tensor(SparseList(Element(0.0)), sprand(10, 0.5))

    @finch begin
        x .= 0
        for i in _
            x[i] = y[i]
        end
    end

    @finch begin
        z .= 0
        for i in _
            z[i] = x[i]
        end
    end

    @test z == y
end
