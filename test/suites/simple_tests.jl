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

@testitem "write protocols" begin
    A = [0.0 2.0 0.0; 3.0 0.0 4.0; 0.0 5.0 0.0; 6.0 0.0 0.0]
    a = Tensor(Dense(Dense(Element(0.0))), A)
    x = Tensor(Dense(Element(0.0)), A[:, 2])

    # A sum over the columns writes each row once for each column.
    for fmt in (Dense, SparseByteMap, SparseHash, SparseDict)
        local y = Tensor(fmt(Element(0.0)))
        @finch begin
            y .= 0
            for j in _, i in _
                y[laminate(i)] += a[i, j]
            end
        end
        @test y == vec(sum(A; dims=2))
    end

    # A copy writes each index once, in order.
    for fmt in (Dense, SparseList, SparseByteMap, SparseHash, SparseDict)
        local y = Tensor(fmt(Element(0.0)))
        @finch begin
            y .= 0
            for i in _
                y[extrude(i)] = x[i]
            end
        end
        @test y == A[:, 2]
    end
end
