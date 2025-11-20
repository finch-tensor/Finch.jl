@testitem "kernels" setup = [CheckOutput] begin
    using SparseArrays
    using MatrixMarket
    using LinearAlgebra
    matrices = ["LPnetlib/lpi_itest6", "HB/west0132", "LPnetlib/lp_blend"]

    seen = false
    for mtx in matrices
        global seen
        A_ref = SparseMatrixCSC(mmread(joinpath(@__DIR__, "../data", "$mtx.mtx")))
        m, n = size(A_ref)
        B_ref = transpose(A_ref) * A_ref
        A = Tensor(A_ref)
        B = Tensor(Dense(SparseList(Element(0.0))), m, m)

        if !seen
            check_output(
                "kernels/innerprod.jl",
                @finch_code (B .= 0;
                for j in _, i in _, k in _
                    B[i, j] += A[k, i] * A[k, j]
                end)
            )
            seen = true
        end
        @finch (B .= 0;
        for j in _, i in _, k in _
            B[i, j] += A[k, i] * A[k, j]
        end)
        @test B == B_ref
    end

    seen = false
    for mtx in matrices
        global seen
        A_ref = SparseMatrixCSC(mmread(joinpath(@__DIR__, "../data", "$mtx.mtx")))
        m, n = size(A_ref)
        if m == n
            A = Tensor(A_ref)
            B = Finch.Scalar{0.0}()
            if !seen
                check_output(
                    "kernels/triangle.jl",
                    @finch_code (B .= 0;
                    for i in _, j in _, k in _
                        B[] += A[k, i] * A[j, i] * A[k, j]
                    end)
                )
                seen = true
            end
            @finch (B .= 0;
            for i in _, j in _, k in _
                B[] += A[k, i] * A[j, i] * A[k, j]
            end)
            @test B() ≈ sum(A_ref .* (A_ref * transpose(A_ref)))
        end
    end

    for trial in 1:10
        n = 100
        p = q = 0.1

        A_ref = sprand(n, p)
        B_ref = sprand(n, q)
        A = Tensor(A_ref)
        B = Tensor(B_ref)
        C = Tensor(SparseList(Element(0.0)))
        d = Scalar{0.0}()
        a = Scalar{0.0}()
        b = Scalar{0.0}()

        @finch begin
            C .= 0
            d .= 0
            for i in _
                a .= 0
                b .= 0
                a[] = A[i]
                b[] = B[i]
                C[i] = a[] - b[]
                d[] += a[] * b[]
            end
        end

        @test C == A_ref .- B_ref
        @test d[] ≈ dot(A_ref, B_ref)
    end

    seen = false
    for mtx in matrices
        global seen
        A_ref = SparseMatrixCSC(mmread(joinpath(@__DIR__, "../data", "$mtx.mtx")))
        m, n = size(A_ref)
        if m == n
            A = Tensor(A_ref)
            B = Tensor(Dense(SparseList(Element(0.0))))
            w = Tensor(SparseByteMap(Element(0.0)))

            if !seen
                code = @finch_code begin
                    B .= 0
                    for j in _
                        w .= 0
                        for k in _, i in _
                            w[i] += A[i, k] * A[k, j]
                        end
                        for i in _
                            B[i, j] = w[i]
                        end
                    end
                end
                check_output("kernels/gustavsons.jl", code)
                seen = true
            end
            @finch begin
                B .= 0
                for j in _
                    w .= 0
                    for k in _, i in _
                        w[i] += A[i, k] * A[k, j]
                    end
                    for i in _
                        B[i, j] = w[i]
                    end
                end
            end
            B_ref = A_ref * A_ref
            @test B == B_ref
        end
    end
end
