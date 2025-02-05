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
