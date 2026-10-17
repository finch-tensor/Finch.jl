@testitem "interface_arrays" setup = [CheckOutput] begin
    using Finch: AsArray
    using SparseArrays

    A = Tensor(
        SparseList(Element(0.0)), fsparse([1, 3, 5, 7, 9], [2.0, 3.0, 4.0, 5.0, 6.0], (10,))
    )
    B = Tensor(SparseList(Element(0.0)), A)
    @test A == B

    A = [0.0 0.0 0.0 0.0; 1.0 0.0 0.0 1.0]
    B = Tensor(Dense(SparseList(Element(0.0))), A)
    C = Tensor(Dense(Dense(Element(0.0))), A)
    @test A == B

    A = [0 0; 0 0]
    B = Tensor(Dense(Dense(Element(0.0))), A)
    @test A == B

    A = Tensor(Dense(Element(0.0)), [0, 0, 0, 0])
    B = Tensor(Dense(Element(0.0)), [0, 0, 0, 0, 0])
    @test size(A) != size(B) && A != B

    A = [0 0 0 0 1 0 0 1]
    B = Tensor(Dense(SparseList(Element(0))), [0 0 0 0; 1 0 0 1])
    @test size(A) != size(B) && A != B

    A = Tensor(Dense(SparseList(Element(0.0))), [1 0 0 0; 1 1 0 0; 1 1 1 0])
    B = [0 0 0 0; 1 1 0 0; 1 1 1 0]
    @test size(A) == size(B) && A != B
    C = Tensor(Dense(SparseList(Element(0.0))), [0 0 0 0; 1 1 0 0; 1 1 1 0])
    @test B == C

    A = [NaN, 0.0, 3.14, 0.0]
    B = Tensor(SparseList(Element(0.0)), [NaN, 0.0, 3.14, 0.0])
    C = Tensor(SparseList(Element(0.0)), [NaN, 0.0, 3.14, 0.0])
    D = [1.0, 2.0, 4.0, 8.0]
    @test isequal(A, B)
    @test isequal(A, C)
    @test isequal(B, C)
    @test isequal(B, A)
    @test !isequal(A, D)
    @test A != B

    @testset "permutedims" begin
        io = IOBuffer()
        println(io, "permute tests")

        @repl io A = Tensor(Dense(Sparse(Element(0.0))), ones(1, 1))
        @repl io permutedims(A, (1, 2))
        @repl io permutedims(A, (2, 1))

        @repl io A = Tensor(Dense(Dense(Element(0.0))), ones(1, 1))
        @repl io permutedims(A, (1, 2))
        @repl io permutedims(A, (2, 1))

        @repl io A = Tensor(Sparse(Dense(Element(0.0))), ones(1, 1))
        @repl io permutedims(A, (1, 2))
        @repl io permutedims(A, (2, 1))

        @repl io A = Tensor(Sparse(Sparse(Element(0.0))), ones(1, 1))
        @repl io permutedims(A, (1, 2))
        @repl io permutedims(A, (2, 1))

        @repl io A = Tensor(Dense(Dense(Sparse(Element(0.0)))), ones(1, 1, 1))
        @repl io permutedims(A, (2, 1, 3))

        @repl io A = Tensor(Dense(Dense(Dense(Element(0.0)))), ones(1, 1, 1))
        @repl io permutedims(A, (2, 1, 3))

        @repl io A = Tensor(Dense(Sparse(Dense(Element(0.0)))), ones(1, 1, 1))
        @repl io permutedims(A, (2, 1, 3))

        @repl io A = Tensor(Dense(Sparse(Sparse(Element(0.0)))), ones(1, 1, 1))
        @repl io permutedims(A, (2, 1, 3))

        @repl io A = Tensor(Dense(Sparse(Dense(Element(0.0)))), ones(1, 1, 1))
        @repl io permutedims(A, (1, 3, 2))

        @repl io A = Tensor(Dense(Dense(Dense(Element(0.0)))), ones(1, 1, 1))
        @repl io permutedims(A, (1, 3, 2))

        @repl io A = Tensor(Sparse(Dense(Dense(Element(0.0)))), ones(1, 1, 1))
        @repl io permutedims(A, (1, 3, 2))

        @repl io A = Tensor(Sparse(Sparse(Dense(Element(0.0)))), ones(1, 1, 1))
        @repl io permutedims(A, (1, 3, 2))

        @test check_output("interface/permutedims.txt", String(take!(io)))
    end

    @testset "copy and collect" begin
        m = [1.0 0 0 2.0; 0 0 3.0 0; 0 0 0 0]
        for lvl in (
            Dense(Dense(Element(0.0))),
            Dense(SparseList(Element(0.0))),
            SparseCOO{2}(Element(0.0)),
        )
            A = Tensor(lvl, m)

            B = Tensor(A)
            @test typeof(B) == typeof(A)
            @test Array(B) == m
            @test countstored(B) == countstored(A)

            C = copy(A)
            @test typeof(C) == typeof(A)
            @test Array(C) == m

            @test collect(A) isa Array
            @test collect(A) == m
        end

        A = Tensor(Dense(Dense(Element(0.0))), m)
        sw = swizzle(A, 2, 1)
        @test typeof(copy(sw)) == typeof(sw)
        @test Array(copy(sw)) == permutedims(m)
        @test collect(sw) == permutedims(m)
    end

    @testset "reshape" begin
        io = IOBuffer()
        println(io, "reshape tests")

        @repl io A = Tensor(Dense(Sparse(Element(0))), LinearIndices((6, 6)))
        @repl io reshape(A, (3, 12))
        @test reshape(LinearIndices((6, 6)), (3, 12)) == reshape(A, (3, 12))
        @test reshape(LinearIndices((6, 6)), 3, 12) == reshape(A, 3, 12)
        @test reshape(LinearIndices((6, 6)), 3, :) == reshape(A, 3, :)
        @test Finch.reshape!(
            Tensor(Dense(Dense(Element(0))), 3, 12), A, (3, 12)
        ) == reshape(LinearIndices((6, 6)), (3, 12))

        @repl io reshape(A, (3, 2, 6))
        @test reshape(LinearIndices((6, 6)), (3, 2, 6)) == reshape(A, (3, 2, 6))

        @repl io reshape(A, (3, 2, 2, 3))
        @test reshape(LinearIndices((6, 6)), (3, 2, 2, 3)) == reshape(A, (3, 2, 2, 3))

        @repl io reshape(A, (9, 4))
        @test reshape(LinearIndices((6, 6)), (9, 4)) == reshape(A, (9, 4))

        @repl io reshape(A, :)
        @test reshape(LinearIndices((6, 6)), :) == reshape(A, :)

        @repl io A = Tensor(Dense(Element(0)), 1:36)
        @repl io reshape(A, (3, 12))
        @test reshape(1:36, (3, 12)) == reshape(A, (3, 12))

        @test check_output("interface/reshape.txt", String(take!(io)))

        A = [1]
        @test reshape(Tensor(A), 1, 1, 1, 1) == reshape(A, 1, 1, 1, 1)

        A = swizzle(Tensor(Dense(Sparse(Element(0))), LinearIndices((6, 6))), 2, 1)
        B = permutedims(LinearIndices((6, 6)))
        @test reshape(A, 3, 12) == reshape(B, 3, 12)

        T1 = swizzle(Tensor(Element(0, 0)))
        @test reshape(T1) == reshape(fill(0), ())
        @test reshape(T1, 1) == reshape(fill(0), 1)
        @test reshape(T1, 1, 1) == reshape(fill(0), 1, 1)

        T2 = swizzle(Tensor([0]), 1)
        @test reshape(T2) == reshape([0], ())
        @test reshape(T2, 1) == reshape([0], 1)
        @test reshape(T2, 1, 1) == reshape([0], 1, 1)
    end

    let
        io = IOBuffer()
        println(io, "getindex tests")

        A = Tensor(
            SparseList(
                Dense(
                    SparseList(
                        Element{0.0}(collect(1:30) .* 1.01),
                        5,
                        [1, 3, 6, 8, 12, 14, 17, 20, 24, 27, 27, 28, 31],
                        [
                            2,
                            3,
                            3,
                            4,
                            5,
                            2,
                            3,
                            1,
                            3,
                            4,
                            5,
                            2,
                            4,
                            2,
                            4,
                            5,
                            2,
                            3,
                            5,
                            1,
                            3,
                            4,
                            5,
                            2,
                            3,
                            4,
                            2,
                            1,
                            2,
                            3,
                        ],
                    ),
                    3,
                ),
                4,
                [1, 5],
                [1, 2, 3, 4],
            ),
        )

        print(io, "A = ")
        show(io, MIME("text/plain"), A)
        println(io)

        for inds in [(1, 2, 3), (1, 1, 1), (1, :, 3), (:, 1, 3), (:, :, 3), (:, :, :)]
            print(io, "A[")
            join(io, inds, ",")
            print(io, "] = ")
            show(io, MIME("text/plain"), A[inds...])
            println(io)
        end

        @test check_output("interface/getindex_default.txt", String(take!(io)))
    end

    let
        io = IOBuffer()
        println(io, "setindex! tests")

        @repl io A = Tensor(Dense(Dense(Element(0.0))), 10, 12)
        @repl io A[1, 4] = 3
        @repl io AsArray(A)
        @repl io A[4:6, 6] = 5:7
        @repl io AsArray(A)
        @repl io A[9, :] = 1:12
        @repl io AsArray(A)

        @test check_output("interface/setindex_default.txt", String(take!(io)))
    end

    let
        io = IOBuffer()
        println(io, "countstored tests")

        @repl io A = Tensor(
            Dense(SparseList(Element(0.0))),
            [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0],
        )
        @repl io countstored(A)
        @repl io A = Tensor(
            SparseCOO{2}(Element(0.0)),
            [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0],
        )
        @repl io countstored(A)
        @repl io A = Tensor(
            Dense(Dense(Element(0.0))),
            [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0],
        )
        @repl io countstored(A)
        @repl io A = Tensor(
            SparseList(Dense(Element(0.0))),
            [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0],
        )
        @repl io countstored(A)

        @test check_output("interface/countstored.txt", String(take!(io)))
    end
end

@testitem "interface_issues" setup = [CheckOutput] begin
    using Finch: AsArray
    using SparseArrays

    #https://github.com/finch-tensor/Finch.jl/issues/499
    let
        A = fspzeros(5, 6)
        @test A == zeros(5, 6)
        A = fspzeros(5, 6, 7)
        @test A == zeros(5, 6, 7)

        A = fspeye(5, 6)
        ref = [i == j for i in 1:5, j in 1:6]
        @test A == ref

        A = fspeye(5, 6, 7)
        ref = [i == j == k for i in 1:5, j in 1:6, k in 1:7]
        @test A == ref
    end

    #https://github.com/finch-tensor/Finch.jl/issues/530
    let
        A_tns = Tensor(Dense(Dense(Dense(Element(0.0)))), zeros(3, 3, 3))
        A_sw = swizzle(A_tns, 2, 3, 1)
        A_tns == A_sw #fails
    end

    #https://github.com/finch-tensor/Finch.jl/issues/527
    let
        tns_1 = swizzle(Tensor(ones(10, 10)), 1, 2)
        tns_1[:, :] # == tns_1 https://github.com/finch-tensor/Finch.jl/issues/530

        tns_2 = swizzle(Tensor(ones(10)), 1)
        tns_2[:]# == tns_2 https://github.com/finch-tensor/Finch.jl/issues/530
    end

    #https://github.com/finch-tensor/Finch.jl/issues/528
    let
        tns = swizzle(Tensor(ones(10, 10)), 1, 2)
        @test tns[:, :] == ones(10, 10)
    end

    #https://github.com/finch-tensor/Finch.jl/issues/457
    let
        A = zeros(2, 3, 3)
        A[1, :, :] = [1 2 3; 4 5 6; 7 8 9]
        A[2, :, :] = [1 1 1; 2 2 2; 3 3 3]
        perm = (2, 3, 1)
        A_t = permutedims(A, perm)

        A_tns = Tensor(Dense(Dense(Dense(Element(0.0)))), A)
        A_sw = swizzle(A_tns, perm...)

        @test Array(A_sw) == A_t
        @test permutedims(A_tns, perm) == A_t
    end

    begin
        A = Tensor(Dense(SparseList(Element(0.0))))
        B = dropfills!(
            swizzle(A, 2, 1),
            [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0],
        )
        @test B == swizzle(
            Tensor(
                Dense{Int64}(
                    SparseList{Int64}(
                        Element{0.0,Float64,Int64}([4.4, 1.1, 2.2, 5.5, 3.3]),
                        3,
                        [1, 2, 3, 5, 6],
                        [3, 1, 1, 3, 1],
                    ),
                    4,
                ),
            ),
            2,
            1,
        )
    end
end
