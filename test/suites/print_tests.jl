@testitem "print" setup = [CheckOutput] begin
    @testset "SparseDict snapshot order" begin
        normalize = CheckOutput.normalize_sparse_dicts
        for prefix in
            ("Dict", "Dict{Tuple{Int32, Int16}, Int32}", "Dict{Tuple{Int64, Int16}, Int64}")
            ordered = "$prefix((1, 2) => 3, (4, 5) => 6)"
            reversed = "$prefix((4, 5) => 6, (1, 2) => 3)"
            @test normalize(ordered) == normalize(reversed)
            @test normalize(ordered) != normalize("$prefix((1, 2) => 6, (4, 5) => 3)")
            @test normalize(ordered) != normalize("$prefix((1, 3) => 3, (4, 5) => 6)")
            @test normalize("Tensor($reversed, [2, 1])") == "Tensor($ordered, [2, 1])"
        end
    end

    A = Tensor([(i + j) % 3 for i in 1:5, j in 1:10])

    formats = [
        "list" => SparseList,
        "byte" => SparseByteMap,
        "dict" => SparseDict,
        "coo1" => SparseCOO{1},
    ]

    for (rown, rowf) in formats
        @testset "print $rown d" begin
            B = dropfills!(Tensor(rowf(Dense(Element{0.0}()))), A)
            @test check_output("print/print_$(rown)_dense.txt", sprint(show, B))
            @test check_output(
                "print/print_$(rown)_dense_small.txt",
                sprint(show, B; context=:compact => true),
            )
            @test check_output(
                "print/display_$(rown)_dense.txt", sprint(show, MIME"text/plain"(), B)
            )
            @test check_output("print/summary_$(rown)_dense.txt", summary(B))
        end
    end

    for (coln, colf) in formats
        @testset "print d $coln" begin
            B = dropfills!(Tensor(Dense(colf(Element{0.0}()))), A)
            @test check_output("print/print_dense_$coln.txt", sprint(show, B))
            @test check_output(
                "print/print_dense_$(coln)_small.txt",
                sprint(show, B; context=:compact => true),
            )
            @test check_output(
                "print/display_dense_$(coln).txt", sprint(show, MIME"text/plain"(), B)
            )
            @test check_output("print/summary_dense_$(coln).txt", summary(B))
        end
    end

    formats = [
        "coo2" => SparseCOO{2}
    ]

    for (rowcoln, rowcolf) in formats
        @testset "print $rowcoln" begin
            B = dropfills!(Tensor(rowcolf(Element{0.0}())), A)
            @test check_output("print/print_$rowcoln.txt", sprint(show, B))
            @test check_output(
                "print/print_$(rowcoln)_small.txt",
                sprint(show, B; context=:compact => true),
            )
            @test check_output(
                "print/display_$(rowcoln).txt", sprint(show, MIME"text/plain"(), B)
            )
            @test check_output("print/summary_$(rowcoln).txt", summary(B))
        end
    end

    A = Tensor([fld(i + j, 3) for i in 1:5, j in 1:10])
end
