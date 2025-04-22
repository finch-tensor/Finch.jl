@testitem "issues" setup = [CheckOutput] begin
    using SparseArrays
    using Finch: Structure

    #https://github.com/finch-tensor/Finch.jl/issues/603
    let
        # Test DenseFormat
        fmt_dense = DenseFormat(2, 0.0)
        tensor_dense = Tensor(fmt_dense, undef, 4, 4)
        @test size(tensor_dense) == (4, 4)
        @test fill_value(tensor_dense) == 0.0

        # Test CSFFormat
        fmt_csf = CSFFormat(3, 0.0)
        tensor_csf = Tensor(fmt_csf, undef, 4, 4, 4)
        @test size(tensor_csf) == (4, 4, 4)
        @test fill_value(tensor_csf) == 0.0

        # Test CSCFormat
        fmt_csc = CSCFormat(0.0)
        tensor_csc = Tensor(fmt_csc, undef, 4, 4)
        @test size(tensor_csc) == (4, 4)
        @test fill_value(tensor_csc) == 0.0

        # Test DCSFFormat
        fmt_dcsf = DCSFFormat(2, 0.0)
        tensor_dcsf = Tensor(fmt_dcsf, undef, 4, 4)
        @test size(tensor_dcsf) == (4, 4)
        @test fill_value(tensor_dcsf) == 0.0

        # Test HashFormat
        fmt_hash = HashFormat(2, 0.0)
        tensor_hash = Tensor(fmt_hash, undef, 4, 4)
        @test size(tensor_hash) == (4, 4)
        @test fill_value(tensor_hash) == 0.0

        # Test ByteMapFormat
        fmt_bytemap = ByteMapFormat(2, 0.0)
        tensor_bytemap = Tensor(fmt_bytemap, undef, 4, 4)
        @test size(tensor_bytemap) == (4, 4)
        @test fill_value(tensor_bytemap) == 0.0

        # Test DCSCFormat
        fmt_dcsc = DCSCFormat(0.0)
        tensor_dcsc = Tensor(fmt_dcsc, undef, 4, 4)
        @test size(tensor_dcsc) == (4, 4)
        @test fill_value(tensor_dcsc) == 0.0

        # Test COOFormat
        fmt_coo = COOFormat(2, 0.0)
        tensor_coo = Tensor(fmt_coo, undef, 4, 4)
        @test size(tensor_coo) == (4, 4)
        @test fill_value(tensor_coo) == 0.0
    end

    #https://github.com/finch-tensor/Finch.jl/issues/579
    for fmt in [
        Tensor(Dense(SparseDict(Element(0.0))))
        Tensor(Dense(SparseByteMap(Element(0.0))))
    ]
        arr_1 = fsprand(10, 10, 0.5)
        fmt = copyto!(fmt, arr_1)
        arr_2 = fsprand(10, 10, 0.5)
        check_output(
            "representation/increment_to_$(summary(fmt)).jl",
            @finch_code begin
                for j in _
                    for i in _
                        fmt[i, j] += arr_2[i, j]
                    end
                end
            end
        )
        @finch begin
            for j in _
                for i in _
                    fmt[i, j] += arr_2[i, j]
                end
            end
        end
        ref = arr_1 .+ arr_2
        @test size(fmt) == size(ref)
        @test axes(fmt) == axes(ref)
        @test ndims(fmt) == ndims(ref)
        @test eltype(fmt) == eltype(ref)
        @test fmt == ref
        @test isequal(fmt, ref)

        arr_1 = fsprand(1, 10, 10, 0.5)
        arr_2 = fsprand(1, 10, 10, 0.5)

        @finch begin
            fmt .= 0
            for i in _
                for j in _
                    for k in _
                        fmt[j, i] += arr_1[k, j, i]
                        fmt[j, i] += arr_2[k, j, i]
                    end
                end
            end
        end

        ref = dropdims(sum(arr_1 .+ arr_2; dims=1); dims=1)
        @test size(fmt) == size(ref)
        @test axes(fmt) == axes(ref)
        @test ndims(fmt) == ndims(ref)
        @test eltype(fmt) == eltype(ref)
        @test fmt == ref
        @test isequal(fmt, ref)
    end

    #https://github.com/finch-tensor/Finch.jl/issues/320
    let
        A = sprand(4, 4, 0.9)
        B = sprand(5, 5, 0.9)
        check_output("issues/SparseMatrixCSC_copy.jl", @finch_code begin
            B .= 0
            for j in _, i in _
                B[i, j] = A[i, j]
            end
        end)

        A = sprand(4, 0.9)
        B = sprand(5, 0.9)
        check_output("issues/SparseVector_copy.jl", @finch_code begin
            B .= 0
            for i in _
                B[i] = A[i]
            end
        end)
    end

    #https://github.com/finch-tensor/Finch.jl/issues/290
    @test true skip = true
    #=
    let
        A = Tensor(Dense(SparseList(Element(0.0))))
        B = Tensor(Dense(SparseList(Element(0.0))))
        C = Tensor(Dense(SparseList(Element(0.0))))
        w = Tensor(SparseByteMap(Element(0.0)))
        @test_throws Finch.FinchNotation.FinchSyntaxError begin
            @finch_kernel function foo(A, B, C)
                C .= 0
                for j in _
                    w .= 0
                    for k in _, i in _
                        w[i] += A[i, k] * B[k, j]
                    end
                    for i in _
                        C[i, j] = w[i]
                    end
                end
            end
        end
    end
    =#

    #https://github.com/finch-tensor/Finch.jl/issues/358
    let
        A = Tensor(
            Dense(SparseList(Element(0))),
            [
                0 0 0 0 0;
                1 0 0 0 0;
                0 0 0 0 0
            ],
        )
        B = Tensor(
            Dense(SparseList(Element(0))),
            [
                0 0 0 0 0;
                0 0 1 0 0;
                1 0 0 0 0
            ],
        )
        C = Tensor(
            Dense(SparseList(Element(0))),
            [
                0 0 0 0 0;
                0 0 0 1 0;
                0 0 0 0 0
            ],
        )
        D_ref = A .+ B
        E_ref = D_ref .+ C
        for D in [
            Tensor(Dense(SparseList(Element(0)))),
            Tensor(Dense(SparseDict(Element(0)))),
            Tensor(Dense(SparseByteMap(Element(0)))),
            Tensor(Dense(Dense(Element(0)))),
        ]
            E = deepcopy(D)
            @finch mode = :fast begin
                D .= 0
                E .= 0
                for j in _, i in _
                    E[i, j] += A[i, j]
                end
                for j in _, i in _
                    E[i, j] += B[i, j]
                end
                for j in _, i in _
                    D[i, j] += E[i, j]
                end
                for j in _, i in _
                    E[i, j] += C[i, j]
                end
            end
            @test D == D_ref
            @test E == E_ref
        end
    end

    #https://github.com/finch-tensor/Finch.jl/issues/51
    let
        x = Tensor(Dense(Element(0.0)), [1, 2, 3])
        y = Scalar{0.0}()
        @finch for i in _, j in _
            y[] += min(x[i], x[j])
        end
        @test y[] == 14
    end

    #https://github.com/finch-tensor/Finch.jl/issues/53
    let
        x = Tensor(
            SparseList(Pattern()), fsparse([1, 3, 7, 8], [true, true, true, true], (10,))
        )
        y = Scalar{0.0}()
        @finch for i in _
            y[] += ifelse(x[i], 3, -1)
        end
        @test y[] == 6
        a = 3
        b = -1
    end

    #https://github.com/finch-tensor/Finch.jl/issues/59
    let
        B = Tensor(Dense(Element(0)), [2, 4, 5])
        A = Tensor(Dense(Element(0)), 6)
        @finch (A .= 0;
        for i in _
            A[B[i]] = i
        end)
        @test A == [0, 1, 0, 2, 3, 0]
    end

    #https://github.com/finch-tensor/Finch.jl/issues/61
    I = copyto!(Tensor(RunList(Element(0))), [1, 1, 9, 3, 3])
    A = [
        11 12 13 14 15;
        21 22 23 24 25;
        31 32 33 34 35;
        41 42 43 44 45;
        51 52 53 54 55;
        61 62 63 64 65;
        71 72 73 74 75;
        81 82 83 84 85;
        91 92 93 94 95]
    A = copyto!(Tensor(Dense(Dense(Element(0)))), A)
    B = Tensor(Dense(Element(0)))

    @test check_output(
        "issues/fiber_as_idx.jl", @finch_code (B .= 0;
        for i in _
            B[i] = A[I[i], i]
        end)
    )
    @finch (B .= 0;
    for i in _
        B[i] = A[I[i], i]
    end)

    @test B == [11, 12, 93, 34, 35]

    #https://github.com/finch-tensor/Finch.jl/issues/101
    let
        t = Tensor(SparseList(SparseList(Element(0.0))))
        X = Tensor(SparseList(SparseList(Element(0.0))))
        A = Tensor(
            SparseList(SparseList(Element(0.0))),
            SparseMatrixCSC([0 0 0 0; -1 -1 -1 -1; -2 -2 -2 -2; -3 -3 -3 -3]),
        )
        @test_throws DimensionMismatch @finch (t .= 0;
        for j in _, i in _
            t[i, j] = min(X[i, j], A[i, j])
        end)
        X = Tensor(SparseList(SparseList(Element(0.0))), 4, 4)
        @finch (t .= 0;
        for j in _, i in _
            t[i, j] = min(X[i, j], A[i, j])
        end)
        @test t == A
    end

    #https://github.com/finch-tensor/Finch.jl/issues/115

    let
        function f(a::Float64, b::Float64, c::Float64)
            return a + b + c
        end
        struct MyAlgebra115 <: Finch.AbstractAlgebra end
        t = Tensor(SparseList(SparseList(Element(0.0))))
        B = SparseMatrixCSC([0 0 0 0; -1 -1 -1 -1; -2 -2 -2 -2; -3 -3 -3 -3])
        A = dropfills(copyto!(Tensor(SparseList(SparseList(Element(0.0)))), B))
        @finch algebra = MyAlgebra115() (t .= 0;
        for j in _, i in _
            t[i, j] = f(A[i, j], A[i, j], A[i, j])
        end)
        @test t == B .* 3
    end

    #https://github.com/finch-tensor/Finch.jl/issues/115
    let
        t = Tensor(SparseList(SparseList(Element(0.0))))
        B = SparseMatrixCSC([0 0 0 0; -1 -1 -1 -1; -2 -2 -2 -2; -3 -3 -3 -3])
        A = dropfills(copyto!(Tensor(SparseList(SparseList(Element(0.0)))), B))
        @test_logs (
            :warn,
            "Performance Warning: non-concordant traversal of t[i, j] (hint: most arrays prefer column major or first index fast, run in fast mode to ignore this warning)",
        ) match_mode = :any @test_throws Finch.FinchProtocolError @finch (
            t .= 0;
            for i in _, j in _
                t[i, j] = A[i, j]
            end
        )
    end

    let
        t = Tensor(Dense(SparseList(Element(0.0))))
        B = SparseMatrixCSC([0 0 0 0; -1 -1 -1 -1; -2 -2 -2 -2; -3 -3 -3 -3])
        A = dropfills(copyto!(Tensor(Dense(SparseList(Element(0.0)))), B))
        @test_logs (
            :warn,
            "Performance Warning: non-concordant traversal of t[i, j] (hint: most arrays prefer column major or first index fast, run in fast mode to ignore this warning)",
        ) match_mode = :any @test_throws Finch.FinchProtocolError @finch (
            t .= 0;
            for i in _, j in _
                t[i, j] = A[i, j]
            end
        )
    end

    #https://github.com/finch-tensor/Finch.jl/issues/129
    let
        a = Tensor(Dense(Element(0)), [1, 3, 7, 2])

        x = Scalar((0, 0))
        @finch for i in _
            x[] << maxby >>= (a[i], i)
        end
        @test x[][2] == 3

        y = Scalar(0 => 0)
        @finch for i in _
            y[] << maxby >>= a[i] => i
        end
        @test y[][2] == 3
    end

    #https://github.com/finch-tensor/Finch.jl/issues/124

    let
        A = sparse([3, 4, 3, 4], [1, 2, 3, 3], [1.1, 2.2, 3.3, 4.4], 4, 3)

        B = Tensor(Dense(SparseList(Element(0.0))))

        @finch (B .= 0;
        for j in _, i in _
            B[i, j] = A[i, j]
        end)

        @test Structure(B) == Structure(Tensor(A))

        v = SparseVector(10, [1, 6, 7, 9], [1.1, 2.2, 3.3, 4.4])

        w = Tensor(SparseList(Element(0.0)))

        @finch (w .= 0;
        for i in _
            w[i] = v[i]
        end)

        @test Structure(w) == Structure(Tensor(v))
    end

    #https://github.com/finch-tensor/Finch.jl/issues/99
    let
        m = 4
        n = 3
        ptr_c = [0, 3, 3, 5]
        idx_c = [1, 2, 3, 0, 2]
        val_c = [1.1, 2.2, 3.3, 4.4, 5.5]

        ptr_jl = PlusOneVector(ptr_c)
        idx_jl = PlusOneVector(idx_c)
        A = Tensor(
            Dense(SparseList{Int}(Element{0.0,Float64,Int}(val_c), m, ptr_jl, idx_jl), n)
        )

        @test A == [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0]
    end

    #https://github.com/finch-tensor/Finch.jl/issues/121
    let
        io = IOBuffer()
        y = [2.0, Inf, Inf, 1.0, 3.0, Inf]
        yf = Tensor(SparseList(Element(Inf)), y)
        println(io, "Tensor(SparseList(Element(Inf)), $y):")
        println(io, yf)

        x = Scalar(Inf)

        @test check_output(
            "issues/specialvals_minimum_inf.jl", @finch_code (
                for i in _
                    x[] << min >>= yf[i]
                end
            )
        )
        @finch for i in _
            x[] << min >>= yf[i]
        end
        @test x[] == 1.0

        @test check_output("issues/specialvals_repr_inf.txt", String(take!(io)))

        io = IOBuffer()
        y = [2.0, NaN, NaN, 1.0, 3.0, NaN]
        yf = Tensor(SparseList(Element(NaN)), y)
        println(io, "Tensor(SparseList(Element(NaN)), $y):")
        println(io, yf)

        x = Scalar(Inf)

        @test check_output(
            "issues/specialvals_minimum_nan.jl", @finch_code (
                for i in _
                    x[] << min >>= yf[i]
                end
            )
        )
        @finch for i in _
            x[] << min >>= yf[i]
        end
        @test isequal(x[], NaN)

        @test check_output("issues/specialvals_repr_nan.txt", String(take!(io)))

        io = IOBuffer()
        y = [2.0, missing, missing, 1.0, 3.0, missing]
        yf = Tensor(SparseList(Element{missing,Union{Float64,Missing}}()), y)
        println(io, "Tensor(SparseList(Element(missing)), $y):")
        println(io, yf)

        x = Scalar(Inf)

        @test check_output(
            "issues/specialvals_minimum_missing.jl", @finch_code (
                for i in _
                    x[] << min >>= yf[i]
                end
            )
        )
        @finch for i in _
            x[] << min >>= coalesce(yf[i], missing, Inf)
        end
        @test x[] == 1.0

        @test check_output("issues/specialvals_repr_missing.txt", String(take!(io)))

        io = IOBuffer()
        y = [2.0, nothing, nothing, 1.0, 3.0, Some(1.0), nothing]
        yf = Tensor(SparseList(Element{nothing,Union{Float64,Nothing,Some{Float64}}}()), y)
        println(io, "Tensor(SparseList(Element(nothing)), $y):")
        println(io, yf)

        x = Scalar(Inf)

        @test check_output(
            "issues/specialvals_minimum_nothing.jl",
            @finch_code (
                for i in _
                    x[] << min >>= something(yf[i], nothing, Inf)
                end
            )
        )
        @finch for i in _
            x[] << min >>= something(yf[i], nothing, Inf)
        end
        @test x[] == 1.0

        @test check_output("issues/specialvals_repr_nothing.txt", String(take!(io)))
    end

    #https://github.com/finch-tensor/Finch.jl/issues/118

    let
        io = IOBuffer()
        A = [0.0 1.0 0.0 2.0; 0.0 1.0 0.0 3.0; 0.0 0.0 2.0 0.0]
        B = Tensor(Dense(SparseList(Element(0.0))), A)
        C = Tensor(Dense(SparseList(Element(Inf))))
        @finch (C .= Inf;
        for j in _, i in _
            C[i, j] = ifelse(B[i, j] == 0, Inf, B[i, j])
        end)

        println(io, "A :", A)
        println(io, "C :", C)
        println(io, "set_fill_value!(B, Inf) :", set_fill_value!(B, Inf))
        println(io, set_fill_value!(B, Inf))
        println(io, C)
        @test Structure(C) == Structure(set_fill_value!(B, Inf))
        @test check_output("issues/issue118.txt", String(take!(io)))
    end

    #https://github.com/finch-tensor/Finch.jl/issues/97

    let
        @test_throws DimensionMismatch A = Tensor(Dense(SparseList(Element(0.0))), [0, 1])
        A = fsprand(10, 11, 0.5)
        B = Tensor(Dense(SparseList(Element(0.0))))
        C = fsprand(10, 10, 0.5)
        @test_throws DimensionMismatch @finch (A .= 0;
        for j in _, i in _
            A[i, j] = B[i]
        end)
        @test_throws DimensionMismatch @finch (A .= 0;
        for j in _, i in _
            A[i] = B[i, j]
        end)
        @test_throws DimensionMismatch @finch (A .= 0;
        for j in _, i in _
            A[i, j] = B[i, j] + C[i, j]
        end)
        @test_throws DimensionMismatch copyto!(Tensor(SparseList(Element(0.0))), A)
        @test_throws DimensionMismatch dropfills!(Tensor(SparseList(Element(0.0))), A)

        A = fsprand(10, 11, 0.5)
        B = fsprand(10, 10, 0.5)
        @test_throws Finch.FinchProtocolError @finch for j in _, i in _
            A[i, j] = B[i, follow(j)]
        end
        @test_throws ArgumentError Tensor(SparseCOO(Element(0.0)))
        @test_throws ArgumentError Tensor(SparseList(Element("hello")))
    end

    #https://github.com/finch-tensor/Finch.jl/issues/70

    let
        A = Tensor(Dense(SparseList(Element(0.0))))
        B = Tensor(Dense(SparseList(Element(0.0))))
        eval(@finch_kernel function copy_array(A, B)
            A .= 0
            for j in _, i in _
                A[i, j] = B[i, j]
            end
            return A
        end)
        C = Tensor(Dense(SparseList(Element(0.0))))
        D = Tensor(Dense(SparseList(Element(0.0))), fsprand(5, 5, 0.5))
        C = copy_array(C, D).A
        @test C == D
    end

    #https://github.com/finch-tensor/Finch.jl/issues/243

    let
        @test_throws Finch.ScopeError (@finch begin
            let x = 0
                let x = 0
                end
            end
        end)
    end

    #https://github.com/finch-tensor/Finch.jl/issues/278

    let
        A = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
        x = Scalar{0.0}()
        @finch (x .= 0;
        for i in _
            x[] += A[i, i]
        end)
        @test x[] == 15.0
    end

    #https://github.com/finch-tensor/Finch.jl/issues/267
    let
        A = ones(3, 3)
        B = ones(3, 3)
        C = zeros(3, 3)
        alpha = beta = 1
        @finch begin
            for j in _
                for i in _
                    C[i, j] *= beta
                end
            end
            for j in _
                for k in _
                    let foo = alpha * B[k, j]
                        for i in _
                            C[i, j] += foo * A[i, k]
                        end
                    end
                end
            end
        end
        @test C == A * B
    end

    #https://github.com/finch-tensor/Finch.jl/issues/284
    let
        C = Tensor(Dense(Dense(Element(0.0))), [1 0; 0 1])
        w = Tensor(Dense(Dense(Element(0.0))), [0 0; 0 0])
        @finch mode = :fast begin
            for j in _, i in _
                C[i, j] += 1
            end
            for j in _, i in _
                w[j, i] = C[i, j]
            end
            for i in _, j in _
                C[j, i] = w[j, i]
            end
        end
        @test C == [2.0 1.0; 1.0 2.0]
    end

    #https://github.com/finch-tensor/Finch.jl/issues/291
    let
        A = [1 2 3; 4 5 6; 7 8 9]
        x = Scalar(0.0)
        @finch mode = :fast for j in _, i in _
            if i < j
                x[] += A[i, j]
            end
        end
        @test x[] == 11.0

        @finch mode = :fast (x .= 0;
        for i in _, j in _
            if i < j
                x[] += A[j, i]
            end
        end)
        @test x[] == 19.0

        @finch mode = :fast (x .= 0;
        for j in _, i in _
            if i <= j
                x[] += A[i, j]
            end
        end)
        @test x[] == 26.0

        @finch mode = :fast (x .= 0;
        for i in _, j in _
            if i <= j
                x[] += A[j, i]
            end
        end)
        @test x[] == 34.0

        @finch mode = :fast (x .= 0;
        for j in _, i in _
            if i > j
                x[] += A[i, j]
            end
        end)
        @test x[] == 19.0

        @finch mode = :fast (x .= 0;
        for i in _, j in _
            if i > j
                x[] += A[j, i]
            end
        end)
        @test x[] == 11.0

        @finch mode = :fast (x .= 0;
        for j in _, i in _
            if i >= j
                x[] += A[i, j]
            end
        end)
        @test x[] == 34.0

        @finch mode = :fast (x .= 0;
        for i in _, j in _
            if i >= j
                x[] += A[j, i]
            end
        end)
        @test x[] == 26.0

        @finch mode = :fast (x .= 0;
        for j in _, i in _
            if i == j
                x[] += A[i, j]
            end
        end)
        @test x[] == 15.0

        @finch mode = :fast (x .= 0;
        for i in _, j in _
            if i == j
                x[] += A[j, i]
            end
        end)
        @test x[] == 15.0

        @finch mode = :fast (x .= 0;
        for j in _, i in _
            if i != j
                x[] += A[i, j]
            end
        end)
        @test x[] == 30.0

        @finch mode = :fast (x .= 0;
        for i in _, j in _
            if i != j
                x[] += A[j, i]
            end
        end)
        @test x[] == 30.0
    end

    #https://github.com/finch-tensor/Finch.jl/issues/286
    let
        A = [1 0; 0 1]
        #note that A[i, j] is ignored here, as the temp local is never used
        @finch (
            for j in _, i in _
                let temp = A[i, j]
                end
            end
        )
    end

    #https://github.com/finch-tensor/Finch.jl/issues/288
    let
        A = zeros(3, 3, 3)
        C = zeros(3, 3, 3)
        X = zeros(3, 3)
        @test check_output(
            "issues/issue288_concordize_let.jl",
            @finch_code mode = :fast begin
                for k in _, j in _, i in _
                    let temp1 = X[i, j]
                        for l in _
                            let temp3 = A[i, l, k]
                                if uptrimask[i + 1, l]
                                    C[i, j, k] += temp1 * temp3
                                end
                            end
                        end
                    end
                end
            end
        )
        @test check_output(
            "issues/issue288_concordize_double_let.jl",
            @finch_code mode = :fast begin
                for k in _, j in _, i in _
                    let temp1 = X[i, j]
                        for l in _
                            let temp3 = A[i, l, k]
                                if uptrimask[i + 1, l]
                                    C[i, j, k] += temp1 * temp3
                                end
                            end
                            let temp4 = A[i, l, k]
                                if uptrimask[i + 1, l]
                                    C[i, j, k] += temp1 * temp4
                                end
                            end
                        end
                    end
                end
            end
        )
    end

    let
        A = Tensor(Dense(Dense(SparseList(Element(0.0)))), fsprand(10, 10, 10, 0.1))
        C = Tensor(Dense(Dense(Dense(Element(0.0)))), zeros((10, 10, 10)))
        X = Tensor(Dense(Dense(Element(0.0))), rand(10, 10))
        temp2 = Scalar(0.0)
        @finch begin
            for l in _, j in _, i in _
                let temp1 = X[i, j]
                    temp2 .= 0
                    for k in _
                        if uptrimask[k + 1, i]
                            C[k, j, l] += temp1 * A[k, i, l]
                        end
                        if uptrimask[k, l]
                            temp2[] += X[k, j] * A[k, i, l]
                        end
                    end
                    C[i, j, l] += temp2[]
                end
            end
        end
    end

    #https://github.com/finch-tensor/Finch.jl/issues/52
    let
        s = ShortCircuitScalar(false)
        x = Tensor(SparseList(Element(false)), [false, true, true, false])
        y = Tensor(SparseList(Element(false)), [false, true, false, true])
        @test check_output("issues/short_circuit.jl", @finch_code begin
            for i in _
                s[] |= x[i] && y[i]
            end
        end)

        c = Scalar(0)
        @test check_output("issues/short_circuit_sum.jl", @finch_code begin
            for i in _
                let x_i = x[i]
                    s[] |= x_i && y[i]
                    c[] += x_i
                end
            end
        end)

        A = Tensor(
            Dense(SparseList(Element(false))),
            [false true true false; false true false false]',
        )

        p = ShortCircuitScalar(0)
        P = Tensor(Dense(Element(0)))

        @test check_output(
            "issues/short_circuit_bfs.jl",
            @finch_code begin
                P .= false
                for j in _
                    p .= false
                    for i in _
                        p[] << choose(0) >>= ifelse(A[i, j] && y[i], i, 0)
                    end
                    P[j] = p[]
                end
            end
        )
    end

    #https://github.com/finch-tensor/Finch.jl/issues/313
    let
        edge_matrix = Tensor(SparseList(SparseList(Element(0.0))), 254, 254)
        edge_values = fsprand(254, 254, 0.001)
        @finch (edge_matrix .= 0;
        for j in _, i in _
            edge_matrix[i, j] = edge_values[i, j]
        end)
        output_matrix = Tensor(SparseDict(SparseDict(Element(0.0))), 254, 254)
        @finch (
            for v_4 in _, v_3 in _, v_2 in _, v_5 in _
                output_matrix[v_2, v_5] +=
                    edge_matrix[v_5, v_4] * edge_matrix[v_2, v_3] * edge_matrix[v_3, v_4]
            end
        )

        a_matrix = [1 0; 0 1]
        a_fiber = Tensor(SparseList(SparseList(Element(0.0))), 2, 2)
        copyto!(a_fiber, a_matrix)
        b_matrix = [0 1; 1 0]
        b_fiber = Tensor(SparseList(SparseList(Element(0.0))), 2, 2)
        copyto!(b_fiber, b_matrix)
        output_tensor = Tensor(SparseDict(SparseDict(Element(0.0))), 2, 2)

        @finch (
            output_tensor .= 0;
            for j in _, i in _, k in _
                output_tensor[i, k] += a_fiber[i, j] * b_fiber[k, j]
            end
        )
    end

    #https://github.com/finch-tensor/Finch.jl/issues/321
    let
        A = fsprand(10, 10, 0.1)
        B = sparse(A)
        @test B isa SparseMatrixCSC
        @test B == A
        B = SparseMatrixCSC(A)
        @test B isa SparseMatrixCSC
        @test B == A
        A = fsprand(10, 0.1)
        B = sparse(A)
        @test B isa SparseVector
        @test B == A
        B = SparseVector(A)
        @test B isa SparseVector
        @test B == A
        A = fsprand(10, 10, 0.1)
        B = Array(A)
        @test B isa Array
        @test B == A
    end

    #https://github.com/finch-tensor/Finch.jl/issues/339
    let
        Output = Tensor(SparseList(Dense(Element(0), 1), 10))
        Point = Tensor(SparseList(Element{0}([1]), 10, [1, 2], [1]))
        Kernel = Tensor(SparseList(Dense(Element{0}([1]), 1), 10, [1, 2], [2]))

        eval(@finch_kernel function test(Output, Point, Kernel)
            Output .= 0
            for x in _
                for xx in _
                    for m in _
                        Output[m, x] += Point[x] * Kernel[m, xx]
                    end
                end
            end
        end)

        test(Output, Point, Kernel)
        Ans = Tensor(
            SparseList{Int64}(Dense{Int64}(Element{0,Int64,Int64}([1]), 1), 10, [1, 2], [1])
        )
        @test Ans == Output
    end

    #https://github.com/finch-tensor/Finch.jl/issues/385
    let
        c = Scalar(0)
        @finch let a = 1, b = 2
            c[] += a + b
        end
        @test c[] == 3
    end

    #https://github.com/finch-tensor/Finch.jl/issues/387

    A = zeros(2, 4, 3)
    A[1, :, :] = [0.0 0.0 4.4; 1.1 0.0 0.0; 0.0 0.0 0.0; 3.3 0.0 0.0]
    A[2, :, :] = [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 1.0 0.0; 3.3 0.0 0.0]

    permutation = (3, 1, 2)

    new_shape_1 = size(permutedims(A, permutation))

    t = Tensor(Dense(SparseList(SparseList(Element(0.0)))), A)
    st = swizzle(t, permutation...)
    # materialize swizzle
    new_shape_2 = size(Tensor(Dense(SparseList(SparseList(Element(0.0)))), st))

    @test new_shape_1 == new_shape_2

    @test swizzle(swizzle(zeros(3, 3, 3), 3, 1, 2), 3, 2, 1) isa
        Finch.SwizzleArray{(2, 1, 3),<:Array}

    #https://github.com/finch-tensor/Finch.jl/issues/134
    let
        A = Tensor(Dense(Dense(Element(0.0))), rand(3, 3))
        x = Tensor(Dense(Element(0.0)), rand(3))
        y = Tensor(Dense(Element(0.0)), rand(3))

        @test check_output("issues/cse_symv.jl", @finch_code begin
            for i in _, j in _
                y[i] += A[i, j] * x[j]
                y[j] += A[i, j] * x[i]
            end
        end)
    end

    #https://github.com/finch-tensor/Finch.jl/issues/397
    let
        A = AsArray(
            swizzle(Tensor(Dense(Dense(Element(0.0))), [1 2 3; 4 5 6; 7 8 9]), 2, 1)
        )
        @test check_output(
            "issues/print_swizzle_as_array.txt", sprint(show, MIME"text/plain"(), A)
        )
    end

    #https://github.com/finch-tensor/Finch.jl/issues/427
    let
        a = [1, 2, 0, 0, 1]
        a_fbr = Tensor(Dense(Element(0)), a)
        a_sw = swizzle(a_fbr, 1)
        idx = [1, 1, 3]

        @test a[idx] == a_fbr[idx]
        @test a[idx] == a_sw[idx]
    end

    #https://github.com/finch-tensor/Finch.jl/pull/433
    input = Tensor(Dense(Dense(Element(Float64(0)))))
    output = Tensor(Dense(Dense(Element(Float64(0)))))

    eval(
        Finch.@finch_kernel function blurSimple(input, output)
            output .= 0
            for x in _
                for c in _
                    output[c, x] +=
                        (
                            coalesce(input[c, ~(x - 1)]) + coalesce(input[c, x], 0) +
                            coalesce(input[c, ~(x + 1)], 0)
                        ) / 3
                end
            end
        end
    )

    #https://github.com/finch-tensor/Finch.jl/pull/442
    let
        I = [6, 6, 6, 9]
        J = [1, 3, 10, 5]
        V = [1, 1, 1, 1]
        fsparse(I, J, V) == sparse(I, J, V)
    end

    #https://github.com/finch-tensor/Finch.jl/pull/450
    let
        #there was a bug with let statements and scoping
        input = Tensor(Dense(Dense(Element(UInt(0)))))
        output = Tensor(Dense(Dense(Element(UInt(0)))))
        tmp = Tensor(Dense(Dense(Element(UInt(0)))))

        eval(
            Finch.@finch_kernel function erode_finch_bits_kernel(output, input, tmp)
                tmp .= 0
                for y in _
                    for x in _
                        tmp[x, y] =
                            coalesce(input[x, ~(y - 1)], ~(UInt(0))) & input[x, y] &
                            coalesce(input[x, ~(y + 1)], ~(UInt(0)))
                    end
                end
                output .= 0
                for y in _
                    for x in _
                        let tl = coalesce(tmp[~(x - 1), y], ~(UInt(0))), t = tmp[x, y],
                            tr = coalesce(tmp[~(x + 1), y], ~(UInt(0)))

                            output[x, y] =
                                ((tr << (8 * sizeof(UInt) - 1)) | (t >> 1)) & t &
                                ((t << 1) | (tl >> (8 * sizeof(UInt) - 1)))
                        end
                    end
                end
            end
        )
    end

    let
        A = zeros(2, 3, 3)
        A[1, :, :] = [1 2 3; 4 5 6; 7 8 9]
        A[2, :, :] = [1 1 1; 2 2 2; 3 3 3]
        perm = (2, 3, 1)
        A_t = permutedims(A, perm)

        A_tns = Tensor(Dense(Dense(Dense(Element(0.0)))), A)
        A_sw = swizzle(A_tns, perm...)

        actual = broadcast(.*, A_t, A_t)
        expected = broadcast(.*, A_sw, A_sw)

        @test actual == expected

        B = zeros(size(A_sw)...)
        copyto!(B, A_sw)

        @test B == A_t
    end

    #https://github.com/finch-tensor/Finch.jl/issues/474
    let
        A = [1 2 3; 4 5 6; 7 8 9]
        A_t = permutedims(A, (2, 1))
        A_tns = Tensor(Dense(Dense(Element(0.0))), A)
        A_sw = swizzle(A_tns, 2, 1)

        @test prod(A_t; dims=2) == prod(A_sw; dims=2)
        @test prod(A_t; dims=1) == prod(A_sw; dims=1)
        @test prod(A_t) == prod(A_sw)

        @test tensordot(A_sw, A_sw, (2, 1)) == transpose(A * A)
    end

    #https://github.com/finch-tensor/Finch.jl/issues/483
    let
        D = Tensor(
            Dense{Int64}(
                SparseDict{Int64}(
                    Element{0.0,Float64,Int64}([
                        0.4550140005294436,
                        0.6850107897370502,
                        0.4426087218562376,
                        0.24735950954269276,
                        0.05888493126676229,
                        0.5646016165775413,
                        0.0316029017824437,
                    ]),
                    5,
                    [1, 3, 5, 8, 8, 8],
                    [2, 5, 2, 3, 1, 2, 3],
                    [1, 2, 3, 4, 5, 6, 7],
                    Dict(
                        (3, 2) => 6,
                        (1, 2) => 1,
                        (3, 1) => 5,
                        (3, 3) => 7,
                        (2, 2) => 3,
                        (2, 3) => 4,
                        (1, 5) => 2,
                    ),
                    [],
                ),
                5,
            ),
        )
        @test countstored(D) == 7
    end

    #https://github.com/finch-tensor/Finch.jl/issues/488
    let
        M = Tensor(Dense(SparseList(Element(0.0))), fsprand(1_000_000, 1_000_000, 1e-05))

        expected(f) = sum([f() for _ in 1:100_000]) / 100_000.0

        t = 1_000_000
        @test 0.5 < expected(() -> countstored(fsprand(t, t, t, t, 1.0 / t / t / t / t))) <
            1.5
        @test 8.5 < expected(() -> countstored(fsprand(t, t, t, t, 9.0 / t / t / t / t))) <
            9.5
        @test 10.5 <
            expected(() -> countstored(fsprand(t, t, t, t, 11.0 / t / t / t / t))) < 11.5

        @test 99.5 < expected(() -> countstored(fsprand(10, 10, 1.0))) < 100.5
        @test 49.5 < expected(() -> countstored(fsprand(10, 10, 0.5))) < 50.5

        @test countstored(fsprand(10, 10, 50)) == 50
        @test countstored(fsprand(10, 10, 100)) == 100
        @test countstored(fsprand(t, t, t, t, 100)) == 100
    end

    #https://github.com/finch-tensor/Finch.jl/issues/480
    let
        A = fsprand(10, 10, 0.5)
        @test size(A[:, nothing, :]) == (10, 1, 10)

        copyto!(Tensor(Element(0.0)), swizzle(Tensor(Element(0.0, [1.0]))))
    end

    #https://github.com/finch-tensor/Finch.jl/issues/485
    let
        a = zeros(1, 1, 1)
        a_tns = Tensor(Dense(SparseList(SparseList(Element(0.0)))), a)

        sum(a; dims=(1, 2))
        sum(a_tns; dims=(1, 2))
    end

    #https://github.com/finch-tensor/Finch.jl/issues/546
    let
        n = 100
        A = Tensor(Dense(SparseList(Element(0.0))), fsprand(n, n, 0.001))
        B = Tensor(Dense(SparseList(Element(0.0))), fsprand(n, n, 0.001))
        C = Tensor(SparseList(Element(0.0)), fsprand(n, 0.1))
        D = Scalar(0.0)

        check_output("issues/protocol_issue.jl",
            @finch_code begin
                D .= 0
                for i in _
                    for j in _
                        for k in _
                            D[] +=
                                C[gallop(j)] * B[walk(k), follow(j)] *
                                A[gallop(j), follow(i)]
                        end
                    end
                end
            end)
    end

    #https://github.com/finch-tensor/Finch.jl/issues/580
    let
        n = 100
        data = fsprand(Int, n, n, 0.01) .% 100
        A_sbm = Tensor(Dense(SparseByteMap(Element(0))), data)
        A_d = Tensor(Dense(Dense(Element(0))), data)
        C1 = Scalar(0)
        C2 = Scalar(0)
        @finch begin
            C1 .= 0
            for i in _
                for j in _
                    for k in _
                        C1[] += A_sbm[follow(j), i] * A_sbm[k, j]
                    end
                end
            end

            C2 .= 0
            for i in _
                for j in _
                    for k in _
                        C2[] += A_d[j, i] * A_d[k, j]
                    end
                end
            end
        end
        @test C1 == C2
    end

    # Basic Sparse Follow Test
    let
        n = 1000
        A = Tensor(SparseList(Element(0.0)), fsprand(n, 100))
        B = Tensor(SparseList(Element(0.0)), fsprand(n, 100))
        C_follow = Tensor(SparseList(Element(0.0)))
        @finch begin
            C_follow .= 0
            for i in _
                C_follow[i] = A[follow(i)] * B[i]
            end
        end
        C_walk = Tensor(SparseList(Element(0.0)))
        @finch begin
            C_walk .= 0
            for i in _
                C_walk[i] = A[i] * B[i]
            end
        end
        @test C_follow == C_walk
    end
end
