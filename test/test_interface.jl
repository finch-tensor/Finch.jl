using Finch: AsArray, JuliaContext
using Finch.FinchNotation: finch_unparse_program, @finch_program_instance

@testset "interface" begin

    @info "Testing Finch Interface"

    @testset "finch_unparse" begin
        prgm = @finch_program quote 
            A .= 0
            for i = _
                A[i] += 1
            end
        end
        @test prgm.val == @finch_program $(finch_unparse_program(JuliaContext(), prgm))
    end

    #https://github.com/finch-tensor/Finch.jl/issues/383
    let
        A = [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0]
        A_fbr = Tensor(Dense(Dense(Element(0.0))), A)

        -A # works
        -A_fbr # used to fail
    end

    @testset "permutedims" begin
        let
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
    end

    #https://github.com/finch-tensor/Finch.jl/issues/592
    let
        @test eltype(broadcast(Finch.fld_nothrow, Tensor(ones(Int16, 1)), Tensor(ones(Int8, 1)))) == Int16
        @test eltype(broadcast(Finch.fld_nothrow, Tensor(ones(Int16, 0)), Tensor(ones(Int8, 0)))) == Int16

        @test eltype(broadcast(Finch.fld_nothrow, Tensor(ones(Float64, 1)), Tensor(ones(Float32, 1)))) == Float64
        @test eltype(broadcast(Finch.fld_nothrow, Tensor(ones(Float64, 0)), Tensor(ones(Float32, 0)))) == Float64
    end

    #https://github.com/finch-tensor/Finch.jl/issues/578
    let
        @test maximum(Tensor(ones(Int8, 4,4))) === Int8(1)
        @test minimum(Tensor(ones(Int8, 4,4))) === Int8(1)
        @test extrema(Tensor(ones(Int8, 4,4))) === (Int8(1), Int8(1))
        @test compute(maximum(lazy(Tensor(ones(Int8, 4,4)))))[] === Int8(1)
        @test compute(minimum(lazy(Tensor(ones(Int8, 4,4)))))[] === Int8(1)
        @test compute(extrema(lazy(Tensor(ones(Int8, 4,4)))))[] === (Int8(1), Int8(1))
    end

    #https://github.com/finch-tensor/Finch.jl/issues/576
    let
        a = zeros(ComplexF64, 2, 1)
        a[1, 1] = 1.8 + 1.8im
        a[2, 1] = 4.8 + 4.8im

        b = [1 + 1im, 2 + 2im]

        a_tns = Tensor(a)
        b_tns = Tensor(b)

        res = Finch.tensordot(a_tns, b_tns, ((1,), (1,)))

        @test eltype(res) == ComplexF64
    end

    #https://github.com/finch-tensor/Finch.jl/issues/577
    let
        a = Tensor(ones(UInt8, 1))
        b = Tensor(ones(UInt8, 1))
        @test eltype(Finch.tensordot(a, b, 0)) == UInt8
    end

    for (key, scheduler) in [
            "default" => Finch.default_scheduler(),
            "interp" => Finch.DefaultLogicOptimizer(Finch.LogicInterpreter()),
            "galley" => Finch.galley_scheduler()
        ]
        Finch.with_scheduler(scheduler) do
            @info "Testing $scheduler"

            #https://github.com/finch-tensor/Finch.jl/issues/474
            let
                arr = [1 2 1 2; 2 1 2 1]
                arr2 = arr .+ 2

                tns = Tensor(Dense(Dense(Element(0))), arr)
                tns2 = Tensor(Dense(Dense(Element(0))), arr2)


                broadcast(/, tns, tns2)  # passes
                broadcast(Finch.fld_nothrow, tns, tns2)  # fails with RewriteTools.RuleRewriteError
                broadcast(Finch.rem_nothrow, tns, tns2)
            end

            #https://github.com/finch-tensor/Finch.jl/issues/520
            let
                A = rand(2, 2)
                x = rand(2)
                lx = lazy(x)
                y = compute(@einsum y[i] += A[i, j] * lx[j])
                @test norm(y .- A * x) < 1e-10
            end

            #https://github.com/finch-tensor/Finch.jl/issues/686
            let
                A = Tensor(Dense(SparseList(Element(0.0))), [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0])
                @test countstored(A - A) == 5
            end

            #https://github.com/finch-tensor/Finch.jl/issues/554
            let
                @test broadcast(trunc, swizzle(Tensor(ones(1)), 1)) == Tensor(ones(1))

                @test broadcast(trunc, swizzle(Tensor(ones(2)), 1)) == Tensor(ones(2))
            end

            #https://github.com/finch-tensor/Finch.jl/issues/533
            let
                A = lazy(fsprand(1, 1, 0.5))
                compute(sum(A .+ A)) #should not error
            end

            #https://github.com/finch-tensor/Finch.jl/issues/535
            let
                LEN = 10;
                a_raw = rand(LEN, LEN - 5) * 10;
                b_raw = rand(LEN, LEN - 5) * 10;
                c_raw = rand(LEN, LEN) * 10;

                a = lazy(swizzle(Tensor(a_raw), 1, 2));
                b = lazy(swizzle(Tensor(b_raw), 1, 2));
                c = lazy(swizzle(Tensor(c_raw), 1, 2));

                ref = reshape(c_raw, 10, 10, 1) .* reshape(a_raw, 10, 1, 5) .* reshape(b_raw, 1, 10, 5);

                plan = c[:, :, nothing] .* a[:, nothing, :] .* b[nothing, :, :];
                res = compute(plan)
                @test norm(res - ref)/norm(ref) < 0.01

                plan = broadcast(*, broadcast(*, c[:, :, nothing], a[:, nothing, :]), b[nothing, :, :]);
                res = compute(plan)
                @test norm(res - ref)/norm(ref) < 0.01
            end

            #https://github.com/finch-tensor/Finch.jl/issues/536
            let
                A = [1 2; 3 4]
                swizzle(lazy(A), 2, 1) == permutedims(A)
            end

            #https://github.com/finch-tensor/Finch.jl/issues/530
            let
                A_tns = Tensor(Dense(Dense(Dense(Element(0.0)))), zeros(3, 3, 3))
                A_sw = swizzle(A_tns, 2, 3, 1)
                A_tns == A_sw #fails
            end

            #https://github.com/finch-tensor/Finch.jl/issues/524
            let
                arr3d = rand(Int, 3, 2, 3) .% 10
                tns = Tensor(Dense(Dense(Dense(Element(0)))), arr3d)

                tns_l = lazy(tns)
                reduced = sum(tns_l, dims=(1, 2))

                plan = broadcast(+, tns_l, reduced)
                result = compute(plan)
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
                @test tns[nothing, :, :] == ones(1, 10, 10)
                @test tns[:, nothing, :] == ones(10, 1, 10)
                @test tns[:, :, nothing] == ones(10, 10, 1)
            end

            #https://github.com/finch-tensor/Finch.jl/issues/428
            let
                @testset "Verbose" begin
                    a = [1 2; 3 4]
                    b = [5 6; 7 8]
                    a_l = lazy(a)
                    b_l = lazy(b)

                    c = permutedims(broadcast(.+, permutedims(a_l, (2, 1)), permutedims(b_l, (2, 1))), (2, 1))
                    compute(c, verbose=true)
                end
            end

            let

                @testset "Einsum Tests" begin
                    # Test 0
                    A = [1 2; 3 4]
                    B = [5 6; 7 8]
                    s = Scalar(0)
                    @einsum s[] += abs(A[i, k] * B[k, j])
                    @test s[] == 134

                    # Test 1
                    A = [1 2; 3 4]
                    B = [5 6; 7 8]
                    @einsum C[i, j] += A[i, k] * B[k, j]
                    @test C == [19 22; 43 50]

                    # Test 2
                    A = Tensor(Dense(SparseList(Element(0))), fsprand(Int, 3, 5, 0.5))
                    B = Tensor(Dense(SparseList(Element(0))), fsprand(Int, 5, 3, 0.5))
                    @einsum C[i, j, k] += A[i, j] * B[j, k]

                    C_ref = zeros(Int, 3, 5, 3)
                    for i = 1:3, j = 1:5, k = 1:3
                        C_ref[i, j, k] += A[i, j] * B[j, k]
                    end
                    @test C == C_ref

                    # Test 3
                    X = Tensor(Dense(SparseList(Element(0))), fsprand(Int, 4, 6, 0.5))
                    Y = Tensor(Dense(SparseList(Element(0))), fsprand(Int, 6, 4, 0.5))
                    @einsum D[i, k] += X[i, j] * Y[j, k]

                    D_ref = zeros(Int, 4, 4)
                    for i = 1:4, j = 1:6, k = 1:4
                        D_ref[i, k] += X[i, j] * Y[j, k]
                    end
                    @test D == D_ref

                    # Test 4
                    H = Tensor(Dense(SparseList(Element(0))), fsprand(Int, 5, 5, 0.6))
                    I = Tensor(Dense(SparseList(Element(0))), fsprand(Int, 5, 5, 0.6))
                    @einsum J[i, j] = H[i, j] * I[i, j]

                    J_ref = zeros(Int, 5, 5)
                    for i = 1:5, j = 1:5
                        J_ref[i, j] = H[i, j] * I[i, j]
                    end
                    @test J == J_ref

                    # Test 5
                    K = Tensor(Dense(SparseList(Element(0))), fsprand(Int, 4, 4, 0.7))
                    L = Tensor(Dense(SparseList(Element(0))), fsprand(Int, 4, 4, 0.7))
                    M = Tensor(Dense(SparseList(Element(0))), fsprand(Int, 4, 4, 0.7))
                    @einsum N[i, j] += K[i, k] * L[k, j] - M[i, j]

                    N_ref = zeros(Int, 4, 4)
                    for i = 1:4, k = 1:4, j = 1:4
                        N_ref[i, j] += K[i, k] * L[k, j] - M[i, j]
                    end
                    @test N == N_ref

                    # Test 6
                    P = Tensor(Dense(SparseList(Element(-Inf))), fsprand(Int, 3, 3, 0.7)) # Adjacency matrix with probabilities
                    Q = Tensor(Dense(SparseList(Element(-Inf))), fsprand(Int, 3, 3, 0.7))
                    @einsum init=-Inf R[i, j] <<max>>= P[i, k] + Q[k, j]  # Max-plus product

                    R_ref = fill(-Inf, 3, 3)
                    for i = 1:3, j = 1:3
                        for k = 1:3
                            R_ref[i, j] = max(R_ref[i, j], P[i, k] + Q[k, j])
                        end
                    end
                    @test R == R_ref

                    # Test for Sparse Matrix-Vector Multiplication (SpMV)
                    # Define a sparse matrix `S` and a dense vector `v`
                    S = Tensor(Dense(SparseList(Element(0))), sprand(Int, 10, 10, 0.3))  # 10x10 sparse matrix with 30% density
                    v = Tensor(Dense(Element(0)), rand(Int, 10))              # Dense vector of size 10

                    # Perform matrix-vector multiplication using the @einsum macro
                    @einsum w[i] += S[i, k] * v[k]  # Compute the product

                    # Reference calculation using explicit loop for validation
                    w_ref = zeros(Int, 10)
                    for i = 1:10
                        for k = 1:10
                            w_ref[i] += S[i, k] * v[k]
                        end
                    end

                    # Test to ensure the results match
                    @test w == w_ref

                    # Test for Transposed Sparse Matrix-Vector Multiplication (SpMV)
                    # Define a sparse matrix `T` and a dense vector `u`
                    T = Tensor(Dense(SparseList(Element(0))), sprand(Int, 10, 10, 0.3))  # 10x10 sparse matrix with 30% density
                    u = Tensor(Dense(Element(0)), rand(Int, 10))              # Dense vector of size 10

                    # Perform transposed matrix-vector multiplication using the @einsum macro
                    @einsum x[k] += T[j, k] * u[j]  # Compute the product using the transpose of T

                    # Reference calculation using explicit loop for validation
                    x_ref = zeros(Int, 10)
                    for k = 1:10
                        for j = 1:10
                            x_ref[k] += T[j, k] * u[j]
                        end
                    end

                    # Test to ensure the results match
                    @test x == x_ref

                    # Test for Outer Product with Output Named A
                    # Define two vectors for outer product
                    v1 = Tensor(Dense(Element(0)), rand(Int, 5))  # Dense vector of size 5
                    v2 = Tensor(Dense(Element(0)), rand(Int, 7))  # Dense vector of size 7

                    # Perform outer product using the @einsum macro
                    @einsum A[i, j] = v1[i] * v2[j]  # Compute the outer product

                    # Reference calculation using explicit loop for validation
                    A_ref = zeros(Int, 5, 7)
                    for i = 1:5
                        for j = 1:7
                            A_ref[i, j] = v1[i] * v2[j]
                        end
                    end

                    # Test to ensure the results match
                    @test A == A_ref


                    # Test for multiplying a vector by a Scalar
                    v = Tensor(Dense(Element(0)), rand(Int, 5))
                    n = 7

                    #Perform scalar multiplcation
                    @einsum A[i] = n*v[i]

                    # Reference Calculation using explicit loop for validation
                    A_ref = Tensor(Dense(Element(0)), rand(Int, 5))
                    for i = 1:5
                        A_ref[i] = v[i]*n
                    end

                    #Test to ensure the results match
                    @test A == A_ref


                end
            end

            A = Tensor(SparseList(Element(0.0)), fsparse([1, 3, 5, 7, 9], [2.0, 3.0, 4.0, 5.0, 6.0], (10,)))
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

            let
                io = IOBuffer()
                println(io, "getindex tests")

                A = Tensor(SparseList(Dense(SparseList(Element{0.0}(collect(1:30).* 1.01), 5, [1, 3, 6, 8, 12, 14, 17, 20, 24, 27, 27, 28, 31], [2, 3, 3, 4, 5, 2, 3, 1, 3, 4, 5, 2, 4, 2, 4, 5, 2, 3, 5, 1, 3, 4, 5, 2, 3, 4, 2, 1, 2, 3]), 3), 4, [1, 5], [1, 2, 3, 4]))

                print(io, "A = ")
                show(io, MIME("text/plain"), A)
                println(io)

                for inds in [(1, 2, 3), (1, 1, 1), (1, :, 3), (:, 1, 3), (:, :, 3), (:, :, :)]
                    print(io, "A["); join(io, inds, ","); print(io, "] = ")
                    show(io, MIME("text/plain"), A[inds...])
                    println(io)
                end

                @test check_output("interface/getindex_$key.txt", String(take!(io)))
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

                @test check_output("interface/setindex_$key.txt", String(take!(io)))
            end

            let
                io = IOBuffer()
                println(io, "broadcast tests")

                @repl io A = Tensor(Dense(SparseList(Element(0.0))), [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0])
                @repl io B = [1, 2, 3, 4]
                @repl io C = A .+ B true
                @repl io AsArray(C)
                @repl io D = A .* B true
                @repl io AsArray(D)
                @repl io E = ifelse.(A .== 0, 1, 2)
                @repl io AsArray(E)

                @test check_output("interface/broadcast_$key.txt", String(take!(io)))
            end

            let
                io = IOBuffer()
                println(io, "reduce tests")

                @repl io A = Tensor(Dense(SparseList(Element(0.0))), [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0])
                @repl io reduce(+, A, dims=(1,))
                @repl io reduce(+, A, dims=1)
                @repl io reduce(+, A, dims=(2,))
                @repl io reduce(+, A, dims=2)
                @repl io reduce(+, A, dims=(1,2))
                @repl io reduce(+, A, dims=:)

                @test check_output("interface/reduce_$key.txt", String(take!(io)))
            end

            let
                io = IOBuffer()
                println(io, "countstored tests")

                @repl io A = Tensor(Dense(SparseList(Element(0.0))), [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0])
                @repl io countstored(A)
                @repl io A = Tensor(SparseCOO{2}(Element(0.0)), [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0])
                @repl io countstored(A)
                @repl io A = Tensor(Dense(Dense(Element(0.0))), [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0])
                @repl io countstored(A)
                @repl io A = Tensor(SparseList(Dense(Element(0.0))), [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0])
                @repl io countstored(A)

                @test check_output("interface/countstored_$key.txt", String(take!(io)))
            end

            let
                io = IOBuffer()
                println(io, "+,-, *, / tests")

                @repl io A = Tensor(Dense(SparseList(Element(0.0))), [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0])
                @repl io A + 1
                @repl io 1 + A
                @repl io A + A
                @repl io 2 * A
                @repl io A * 3
                @repl io A / 3
                @repl io 3 / A

                @test check_output("interface/asmd_$key.txt", String(take!(io)))
            end

            let
                A_ref = [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0]
                A_ref = A_ref * floatmax()/sum(A_ref)
                A = Tensor(Dense(SparseList(Element(0.0))), A_ref)
                
                @test sum(A) == sum(A_ref)
                @test minimum(A) == minimum(A_ref)
                @test maximum(A) == maximum(A_ref)
                @test extrema(A) == extrema(A_ref)
                @test norm(A) == norm(A_ref)
                @test norm(A, -Inf) == norm(A_ref, -Inf)
                @test norm(A, 0) == norm(A_ref, 0)
                @test norm(A, 1) == norm(A_ref, 1)
                @test norm(A, 1.5) == norm(A_ref, 1.5)
                @test norm(A, Inf) == norm(A_ref, Inf)
            end

            let
                A = Tensor(Dense(SparseList(Element(0.0))), [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0])
                B = Tensor(Dense(SparseList(Element(0.0))), [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0])
                C = lazy(A)
                D = lazy(B)
                E = (C + D) * 0.5
                F = compute(E)
                @test F == A
            end

            let
                A = Tensor(Dense(SparseList(Element(0))), [0 0 44; 11 0 0; 22 00 55; 33 0 0])
                B = Tensor(Dense(SparseList(Element(0))), [0 0 44; 11 0 0; 22 00 55; 33 0 0])
                c_correct = Tensor(Dense(Dense(Element(0))), [1936 0 2420 0; 0 121 242 363; 2420 242 3509 726; 0 363 726 1089])
                c = compute(tensordot(lazy(A), lazy(B), ((2, ), (2,)), init=0))
                @test c == c_correct
            end

            let
                A = lazy(Tensor(Dense(SparseList(Element(0))), [0 0 44; 11 0 0; 22 00 55; 33 0 0]))
                B = lazy(Tensor(Dense(SparseList(Element(0))), [0 0 44; 11 0 0; 22 00 55; 33 0 0]'))
                c_correct = Tensor(Dense(Dense(Element(0))), [1936 0 2420 0; 0 121 242 363; 2420 242 3509 726; 0 363 726 1089])
                c = compute(sum(A[:, :, nothing] .* B[nothing, :, :], dims=[2]))
                @test c == c_correct
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
                A_lazy = lazy(A_sw)

                A_result = compute(A_lazy)

                @test Array(A_result) == A_t
                @test permutedims(A_tns, perm) == A_t
            end

            #https://github.com/finch-tensor/Finch.jl/pull/477
            let
                A = zeros(2, 3, 3)
                A_tns = Tensor(Dense(Dense(Dense(Element(0.0)))), A)

                @test compute(A) == A #If the scheduler improves, we can change this to ===
                @test compute(A_tns) == A_tns #If the scheduler improves, we can change this to ===
            end

            #https://github.com/finch-tensor/Finch.jl/issues/481
            let
                r = fsprand(1, 10, 10, 0.01)
                r_tns = Tensor(Dense(Dense(Dense(Element(0.0)))), r)
                @test r_tns + r_tns == 2 * r_tns
            end

            #https://github.com/finch-tensor/Finch.jl/issues/487
            let
                a = fsprand(10, 1, 0.8)
                b = fsprand(10, 1, 0.8)

                permutedims(broadcast(+, permutedims(a, (2, 1)), permutedims(b, (2, 1))), (2, 1))  # passes

                a_l = lazy(a)
                b_l = lazy(b)

                plan = permutedims(broadcast(+, permutedims(a_l, (2, 1)), permutedims(b_l, (2, 1))), (2, 1))
                compute(plan)  # fails
            end
        end
    end


    @testset "concordize" begin
        using Finch.FinchLogic
        A = alias(:A)
        B = alias(:B)
        C = alias(:C)
        i = field(:i)
        j = field(:j)
        k = field(:k)
        prgm_in = plan(
            query(A, table(0, i, j)),
            query(B, table(0, i, j)),
            query(C, aggregate(+, 0, mapjoin(*,
                reorder(relabel(A, i, k), i, k, j),
                reorder(relabel(B, j, k), i, k, j)
            ))),
            produces(C))
        B_2 = alias(:B_2)
        prgm_out = plan(
            query(A, table(0, i, j)),
            query(B, table(0, i, j)),
            query(B_2, reorder(relabel(B, i, j), j, i)),
            query(C, aggregate(+, 0, mapjoin(*,
                reorder(relabel(A, i, k), i, k, j),
                reorder(relabel(B_2, k, j), i, k, j)
            ))),
            produces(C))
        @test Finch.concordize(prgm_in) == prgm_out

        prgm_in = plan(produces())
        prgm_out = plan(produces())
        @test Finch.concordize(prgm_in) == prgm_out

        prgm_in = plan(
            query(A, table(0, i, j)),
            query(B, table(0, i, j)),
            query(C, mapjoin(+,
                reorder(relabel(A, i, j), j, i),
                reorder(relabel(B, j, i), i, j)
            )),
            produces(C)
        )
        A_2 = alias(:A_2)
        prgm_out = plan(
            query(A, table(0, i, j)),
            query(A_2, reorder(relabel(A, i, j), j, i)),
            query(B, table(0, i, j)),
            query(B_2, reorder(relabel(B, i, j), j, i)),
            query(C, mapjoin(+,
                reorder(relabel(A_2, j, i), j, i),
                reorder(relabel(B_2, i, j), i, j)
            )),
            produces(C)
        )
        @test Finch.concordize(prgm_in) == prgm_out

        prgm_in = plan(
            query(A, table(0, i, j)),
            query(B, reorder(relabel(A, i, j), i, j)),
            produces(B)
        )
        prgm_out = plan(
            query(A, table(0, i, j)),
            query(B, reorder(relabel(A, i, j), i, j)),
            produces(B)
        )
        @test Finch.concordize(prgm_in) == prgm_out

        D = alias(:D)
        prgm_in = plan(
            query(A, table(0, i, j)),
            query(B, table(0, i, j)),
            query(C, reorder(relabel(A, i, j), j, i)),
            query(D, reorder(relabel(B, j, i), i, j)),
            produces(C, D)
        )
        prgm_out = plan(
            query(A, table(0, i, j)),
            query(A_2, reorder(relabel(A, i, j), j, i)),
            query(B, table(0, i, j)),
            query(B_2, reorder(relabel(B, i, j), j, i)),
            query(C, reorder(relabel(A_2, j, i), j, i)),
            query(D, reorder(relabel(B_2, i, j), i, j)),
            produces(C, D)
        )
        @test Finch.concordize(prgm_in) == prgm_out

        prgm_in = plan(
            query(A, table(0, i, j)),
            query(B, table(0, i, j)),
            query(C, mapjoin(+,
                reorder(relabel(A, i, k), k, i),
                reorder(relabel(B, k, j), j, k)
            )),
            produces(C)
        )
        C_2 = alias(:C_2)
        prgm_out = plan(
            query(A, table(0, i, j)),
            query(A_2, reorder(relabel(A, i, j), j, i)),
            query(B, table(0, i, j)),
            query(B_2, reorder(relabel(B, i, j), j, i)),
            query(C, mapjoin(+,
                reorder(relabel(A_2, k, i), k, i),
                reorder(relabel(B_2, j, k), j, k)
            )),
            produces(C)
        )
        @test Finch.concordize(prgm_in) == prgm_out

        prgm_in = plan(
            query(A, table(0)),
            query(B, reorder(relabel(A, ), )),
            produces(B)
        )
        prgm_out = plan(
            query(A, table(0)),
            query(B, reorder(relabel(A, ), )),
            produces(B)
        )
        @test Finch.concordize(prgm_in) == prgm_out

        prgm_in = plan(
            query(A, table(0, i, j, k)),
            query(B, reorder(relabel(A, i, j, k), k, j, i)),
            query(C, reorder(relabel(A, i, j, k), j, k, i)),
            query(D, mapjoin(*,
                reorder(relabel(B, k, j, i), i, j, k),
                reorder(relabel(C, j, k, i), i, j, k)
            )),
            produces(D)
        )
        A_3 = alias(:A_3)
        C_2 = alias(:C_2)
        prgm_out = plan(
            query(A, table(0, i, j, k)),
            query(A_2, reorder(relabel(A, i, j, k), k, j, i)),
            query(A_3, reorder(relabel(A, i, j, k), j, k, i)),
            query(B, reorder(relabel(A_2, k, j, i), k, j, i)),
            query(B_2, reorder(relabel(B, k, j, i), i, j, k)),
            query(C, reorder(relabel(A_3, j, k, i), j, k, i)),
            query(C_2, reorder(relabel(C, j, k, i), i, j, k)),
            query(D, mapjoin(*,
                reorder(relabel(B_2, i, j, k), i, j, k),
                reorder(relabel(C_2, i, j, k), i, j, k)
            )),
            produces(D)
        )
        @test Finch.concordize(prgm_in) == prgm_out
    end

    @testset "push_fields" begin
        using Finch.FinchLogic
        A = alias(:A)
        i = field(:i)
        j = field(:j)
        k = field(:k)

        # Test 1: Simple reorder and relabel on a table
        expr_in = reorder(relabel(table(A, i, j, k), k, j, i), i, j, k)
        expr_out = reorder(table(A, k, j, i), i, j, k)  # After push_fields, reorder and relabel should be absorbed
        @test Finch.push_fields(expr_in) == expr_out

        # Test 2: Nested reorders and relabels on a table
        expr_in = reorder(relabel(reorder(relabel(table(A, i, j, k), j, i, k), k, j, i), i, k, j), j, i, k)
        expr_out = reorder(table(A, k, j, i), j, i, k)
        @test Finch.push_fields(expr_in) == expr_out

        # Test 3: Mapjoin with internal reordering and relabeling
        expr_in = mapjoin(+,
                    reorder(relabel(table(A, i, j), j, i), i, j),
                    reorder(relabel(table(A, j, i), i, j), j, i))
        expr_out = mapjoin(+,
                    reorder(table(A, j, i), i, j),
                    reorder(table(A, i, j), j, i))
        @test Finch.push_fields(expr_in) == expr_out

        # Test 4: Immediate values absorbing relabel and reorder
        expr_in = reorder(relabel(immediate(42)), i)
        expr_out = reorder(immediate(42), i)
        @test Finch.push_fields(expr_in) == expr_out

        # Test 5: Complex nested structure with mapjoin and aggregates
        expr_in = mapjoin(+,
                    reorder(relabel(mapjoin(*,
                        reorder(relabel(table(A, i, j, k), k, j, i), i, j, k),
                        table(A, j, i, k)), i, k, j), j, i, k),
                    mapjoin(*,
                        reorder(relabel(table(A, i, j, k), j, i, k), k, j, i)))
        expr_out = mapjoin(+,
                    mapjoin(*,
                        reorder(table(A, j, k, i), j, i, k),
                        reorder(table(A, k, i, j), j, i, k)),
                    mapjoin(*,
                        reorder(table(A, j, i, k), k, j, i)))
        @test Finch.push_fields(expr_in) == expr_out

        #=
        query(A1, table(0, i0, i1))
        query(A2, table(1, i2, i3))
        query(A5,
            aggregate(+, 0.0, relabel(
                mapjoin(*,
                    reorder(relabel(relabel(A2, i2, i3), i7, i8), i7, i8, i9),
                    reorder(relabel(relabel(A0, i0, i1), i8, i9), i7, i8, i9)
                ), i13, i14, i15), i14))
        =#
    end

    begin
        A = Tensor(Dense(SparseList(Element(0.0))))
        B = dropfills!(swizzle(A, 2, 1), [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0])
        @test B == swizzle(Tensor(Dense{Int64}(SparseList{Int64}(Element{0.0, Float64, Int64}([4.4, 1.1, 2.2, 5.5, 3.3]), 3, [1, 2, 3, 5, 6], [3, 1, 1, 3, 1]), 4)), 2, 1)
    end


end
