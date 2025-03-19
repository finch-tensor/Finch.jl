
@testitem "dims" begin
    u = lazy(fsprand(3, 1, 2, 1, 1, 0.2))
    v = compute(Finch.dropdims(u, [2, 4, 5]))
    u = compute(u)

    @test size(v) == (3, 2)

    not_equal = Scalar(0.0)
    @finch begin
        for i in _, j in _
            not_equal[] += (v[i, j] != u[i, 1, j, 1, 1])
        end
    end

    @test not_equal[] == 0.0

    # u = lazy(fsprand(3, 1, 2, 1, 1, 0.2))
    # v = compute(Finch.dropdims(u, 2))
    # u = compute(u)

    # @test size(v) == (3, 2, 1, 1)

    # not_equal = Scalar(0.0)
    # @finch begin
    #     for i in _, j in _
    #         not_equal[] += (v[i, j, 1, 1] != u[i, 1, j, 1, 1])
    #     end
    # end

    # @test not_equal[] == 0.0
end
