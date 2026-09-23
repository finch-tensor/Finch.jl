@testitem "fileio" setup = [CheckOutput] begin
    using MatrixMarket
    using Pkg
    using HDF5
    using Finch: Structure
    @testset "h5 binsparse" begin
        let f = mktempdir()
            A = [0.0 1.0 2.0 2.0;
                0.0 0.0 0.0 0.0;
                1.0 1.0 2.0 0.0;
                0.0 0.0 0.0 1.0]
            A_COO = Tensor(SparseCOO{2}(Element(0.0)), A)
            A_COO_fname = joinpath(f, "A_COO.bsp.h5")
            fwrite(A_COO_fname, A_COO)
            A_COO_test = fread(A_COO_fname)
            @test A_COO_test == A_COO

            for (iA, A) in enumerate([
                [false true false false;
                    true true true true],
                [0 1 2 2;
                    0 0 0 0;
                    1 1 2 0;
                    0 0 0 0],
                [0.0 1.0 2.0 2.0;
                    0.0 0.0 0.0 0.0;
                    1.0 1.0 2.0 0.0;
                    0.0 0.0 0.0 0.0],
                [0+1im 1+0im 0+0im;
                    0+0im 1+0im 0+0im],
            ])
                @testset "$(typeof(A))" begin
                    for (iD, Vf) in [
                        0 => zero(eltype(A)),
                        1 => one(eltype(A)),
                    ]
                        elem = Element{Vf,eltype(A),Int}()
                        for (name, fmt) in [
                            "A_dense" =>
                                swizzle(Tensor(Dense{Int}(Dense{Int}(elem))), 2, 1),
                            "A_denseC" => Tensor(Dense{Int}(Dense{Int}(elem))),
                            "A_CSC" => Tensor(Dense{Int}(SparseList{Int}(elem))),
                            "A_CSR" =>
                                swizzle(Tensor(Dense{Int}(SparseList{Int}(elem))), 2, 1),
                            "A_COO" =>
                                swizzle(Tensor(SparseCOO{2,Tuple{Int,Int}}(elem)), 2, 1),
                            "A_COOC" => Tensor(SparseCOO{2,Tuple{Int,Int}}(elem)),
                        ]
                            @testset "binsparse $name($Vf)" begin
                                fmt = copyto!(fmt, A)
                                fname = joinpath(f, "foo.bsp.h5")
                                bspwrite(fname, fmt)
                                @test Structure(fmt) == Structure(bspread(fname))
                            end
                        end
                    end
                end
            end

            B = fsprand(100, 100, 100, 0.1)
            @testset "binsparse COO3" begin
                fname = joinpath(f, "foo.bsp.h5")
                bspwrite(fname, B)
                @test Structure(B) == Structure(bspread(fname))
            end
        end
    end

    if haskey(Pkg.project().dependencies, "NPZ")
        using NPZ
        @testset "npy binsparse" begin
            let f = mktempdir()
                A = [0.0 1.0 2.0 2.0;
                    0.0 0.0 0.0 0.0;
                    1.0 1.0 2.0 0.0;
                    0.0 0.0 0.0 1.0]
                A_COO = Tensor(SparseCOO{2}(Element(0.0)), A)
                A_COO_fname = joinpath(f, "A_COO.bspnpy")
                fwrite(A_COO_fname, A_COO)
                A_COO_test = fread(A_COO_fname)
                @test A_COO_test == A_COO

                for (iA, A) in enumerate([
                    [false true false false;
                        true true true true],
                    [0 1 2 2;
                        0 0 0 0;
                        1 1 2 0;
                        0 0 0 0],
                    [0.0 1.0 2.0 2.0;
                        0.0 0.0 0.0 0.0;
                        1.0 1.0 2.0 0.0;
                        0.0 0.0 0.0 0.0],
                    [0+1im 1+0im 0+0im;
                        0+0im 1+0im 0+0im],
                ])
                    @testset "$(typeof(A))" begin
                        for (iD, Vf) in [
                            0 => zero(eltype(A)),
                            1 => one(eltype(A)),
                        ]
                            elem = Element{Vf,eltype(A),Int}()
                            for (name, fmt) in [
                                "A_dense" =>
                                    swizzle(Tensor(Dense{Int}(Dense{Int}(elem))), 2, 1),
                                "A_denseC" => Tensor(Dense{Int}(Dense{Int}(elem))),
                                "A_CSC" => Tensor(Dense{Int}(SparseList{Int}(elem))),
                                "A_CSR" => swizzle(
                                    Tensor(Dense{Int}(SparseList{Int}(elem))), 2, 1
                                ),
                                "A_COO" => swizzle(
                                    Tensor(SparseCOO{2,Tuple{Int,Int}}(elem)), 2, 1
                                ),
                                "A_COOC" => Tensor(SparseCOO{2,Tuple{Int,Int}}(elem)),
                            ]
                                @testset "binsparse $name($Vf)" begin
                                    fmt = copyto!(fmt, A)
                                    fname = joinpath(f, "A$(iA)_D$(iD)_$name.bspnpy")
                                    bspwrite(fname, fmt)
                                    @test Structure(fmt) == Structure(bspread(fname))
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    if haskey(Pkg.project().dependencies, "TensorMarket")
        using TensorMarket
        A = [0.0 1.0 2.0 2.0;
            0.0 0.0 0.0 0.0;
            1.0 1.0 2.0 0.0;
            0.0 0.0 0.0 1.0]
        let f = mktempdir()
            A_COO = Tensor(SparseCOO{2}(Element(0.0)), A)
            A_COO_fname = joinpath(f, "A_COO.ttx")
            fttwrite(A_COO_fname, A_COO)
            A_COO_test = fttread(A_COO_fname)
            @test Structure(A_COO_test) == Structure(A_COO)

            A_COO_fname2 = joinpath(f, "A_COO.ttx")
            fwrite(A_COO_fname2, A_COO)
            A_COO_test = fread(A_COO_fname2)
            @test A_COO_test == A_COO

            A_COO_fname2 = joinpath(f, "A_COO.mtx")
            fwrite(A_COO_fname2, A_COO)
            A_COO_test = fread(A_COO_fname2)
            @test A_COO_test == A_COO

            A_COO = Tensor(SparseCOO{2}(Element(0.0)), A)
            A_COO_fname = joinpath(f, "A_COO.tns")
            ftnswrite(A_COO_fname, A_COO)
            A_COO_test = ftnsread(A_COO_fname)
            @test Structure(A_COO_test) == Structure(A_COO)

            A_COO_fname2 = joinpath(f, "A_COO.tns")
            fwrite(A_COO_fname2, A_COO)
            A_COO_test = fread(A_COO_fname2)
            @test A_COO_test == A_COO

            #A test to ensure some level of canonical interpretation.
            A_ref = mmread(joinpath(@__DIR__, "../data/JGD_Kocay/Trec4.mtx"))
            fwrite(joinpath(f, "test.ttx"), Tensor(A_ref))
            str = String(read(joinpath(f, "test.ttx")))
            @test check_output("fileio/Trec4.ttx", str)
        end
    end

    #https://github.com/finch-tensor/Finch.jl/issues/500
    let
        using NPZ
        f = mktempdir(; prefix="finch-issue-500")
        cd(f) do
            A = Tensor(Dense(Element(0.0)), rand(4))
            fwrite("test.bspnpy", A)
            B = fread("test.bspnpy")
            @test A == B
        end
    end
end

@testitem "fileio_mtx" begin
    using MatrixMarket

    mktempdir() do dir
        fname = joinpath(dir, "matrix.mtx")
        write(
            fname,
            """
            %%MatrixMarket matrix array real general
            2 3
            1.0
            2.0
            3.0
            4.0
            5.0
            6.0
            """,
        )
        # MatrixMarket arrays store entries in column-major order.
        expected = [1.0 3.0 5.0; 2.0 4.0 6.0]
        @test Array(Finch.fmmread(fname)) == expected
        @test Array(fread(fname)) == expected

        for expected in (
            [0.0 3.0 0.0 0.0; 2.0 0.0 6.0 0.0; 0.0 0.0 0.0 0.0],
            Int64[0 3 0; 2 0 6],
            [false true false; true false true],
            ComplexF64[0+0im 3+2im 0+0im; 2-1im 0+0im 6+0im],
            ComplexF64[1 2+3im; 2-3im 4],
            zeros(2, 3),
            zeros(0, 3),
        )
            @testset "roundtrip $(eltype(expected)) $(size(expected))" begin
                for tensor in (
                    Tensor(Dense(SparseList(Element(zero(eltype(expected))))), expected),
                    Tensor(SparseCOO{2}(Element(zero(eltype(expected)))), expected),
                    Tensor(expected),
                )
                    fwrite(fname, tensor)
                    actual = MatrixMarket.mmread(fname)
                    @test actual == expected
                    @test eltype(actual) == eltype(expected)
                    @test Array(fread(fname)) == expected
                end
            end
        end

        expected = [0.0 3.0 0.0; 2.0 0.0 6.0]
        tensor = swizzle(Tensor(SparseCOO{2}(Element(0.0)), permutedims(expected)), 2, 1)
        fwrite(fname, tensor)
        @test Array(fread(fname)) == expected

        @test_throws ArgumentError fwrite(fname, Tensor([1.0, 2.0]))
        @test_throws ArgumentError fwrite(
            fname, Tensor(Dense(SparseList(Element(1.0))), ones(2, 3))
        )
    end
end

@testitem "fileio_ttx" begin
    using TensorMarket

    mktempdir() do dir
        fname = joinpath(dir, "matrix.ttx")
        write(
            fname,
            """
            %%MatrixMarket matrix array real general
            2 3
            1.0
            2.0
            3.0
            4.0
            5.0
            6.0
            """,
        )
        # TensorMarket arrays store entries in row-major order.
        expected = [1.0 2.0 3.0; 4.0 5.0 6.0]
        @test Array(fttread(fname)) == expected
        @test Array(fread(fname)) == expected

        fname = joinpath(dir, "tensor.ttx")
        write(
            fname,
            """
            %%MatrixMarket tensor coordinate real general
            2 3 4 2
            1 2 3 4.5
            2 1 4 -2.0
            """,
        )
        expected = zeros(2, 3, 4)
        expected[1, 2, 3] = 4.5
        expected[2, 1, 4] = -2.0
        @test Array(fttread(fname)) == expected
        @test Array(fread(fname)) == expected
    end
end

@testitem "binsparse_compliance" skip = (!Sys.isunix()) begin
    harness_tests = normpath(joinpath(@__DIR__, "..", "compliance", "test_harness.py"))
    @test success(pipeline(`python3 $harness_tests`; stdout=stdout, stderr=stderr))
    script = normpath(joinpath(@__DIR__, "..", "compliance", "run-binsparse-tests.sh"))
    @test success(pipeline(`bash $script`; stdout=stdout, stderr=stderr))
end
