@testitem "reference_dict_order" setup = [CheckOutput] begin
    a = "SparseDict(Dict((1, 2) => 3, (1, 1) => 2), [2, 3])"
    b = "SparseDict(Dict((1, 1) => 2, (1, 2) => 3), [2, 3])"
    @test CheckOutput.normalize_dict_order(a) == CheckOutput.normalize_dict_order(b)
    @test CheckOutput.normalize_dict_order(a) !=
        CheckOutput.normalize_dict_order(replace(b, "=> 3" => "=> 4"))
    @test CheckOutput.normalize_dict_order(a) !=
        CheckOutput.normalize_dict_order(replace(b, "[2, 3]" => "[3, 2]"))
    typed_a = replace(a, "Dict(" => "Dict{Tuple{Int64, Int16}, Int64}(")
    typed_b = replace(b, "Dict(" => "Dict{Tuple{Int64, Int16}, Int64}(")
    @test CheckOutput.normalize_dict_order(typed_a) ==
        CheckOutput.normalize_dict_order(typed_b)
    @test CheckOutput.normalize_dict_order(typed_a) != CheckOutput.normalize_dict_order(a)
    @test CheckOutput.normalize_dict_order(typed_a) !=
        CheckOutput.normalize_dict_order(replace(typed_b, "Int16" => "Int32"))
end

@testitem "print" setup = [CheckOutput] begin
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
