@testset "print" begin
    @info "Testing Fiber Printing"

    A = fiber([(i + j) % 3 for i = 1:5, j = 1:10])

    formats = [
        "list" => SparseList{Int64, Int64},
        "byte" => SparseByteMap{Int64, Int64},
        "hash1" => SparseHash{1, Tuple{Int64}, Int64},
        "coo1" => SparseCOO{1, Tuple{Int64}, Int64},
        "dense" => Dense{Int64},
    ]

    for (rown, rowf) in formats
        @testset "print $rown d" begin
            B = dropdefaults!(Fiber!(rowf(Dense{Int64}(Element{0.0}()))), A)
            @test check_output("print_$(rown)_dense.txt", sprint(show, B))
            @test check_output("print_$(rown)_dense_small.txt", sprint(show, B, context=:compact=>true))
            @test check_output("display_$(rown)_dense.txt", sprint(show, MIME"text/plain"(), B))
            @test check_output("summary_$(rown)_dense.txt", summary(B))
        end
    end

    for (coln, colf) in formats
        @testset "print d $coln" begin
            B = dropdefaults!(Fiber!(Dense{Int64}(colf(Element{0.0}()))), A)
            @test check_output("print_dense_$coln.txt", sprint(show, B))
            @test check_output("print_dense_$(coln)_small.txt", sprint(show, B, context=:compact=>true))
            @test check_output("display_dense_$(coln).txt", sprint(show, MIME"text/plain"(), B))
            @test check_output("summary_dense_$(coln).txt", summary(B))
        end
    end

    formats = [
        "hash2" => SparseHash{2, Tuple{Int64, Int64}, Int64},
        "coo2" =>SparseCOO{2, Tuple{Int64, Int64}, Int64},
        ]

    for (rowcoln, rowcolf) in formats
        @testset "print $rowcoln" begin
            B = dropdefaults!(Fiber!(rowcolf(Element{0.0}())), A)
            @test check_output("print_$rowcoln.txt", sprint(show, B))
            @test check_output("print_$(rowcoln)_small.txt", sprint(show, B, context=:compact=>true))
            @test check_output("display_$(rowcoln).txt", sprint(show, MIME"text/plain"(), B))
            @test check_output("summary_$(rowcoln).txt", summary(B))
        end
    end

    A = fiber([fld(i + j, 3) for i = 1:5, j = 1:10])

    formats = [
        "rle" => RepeatRLE{0.0, Int64, Int64},
    ]

    for (coln, colf) in formats
        @testset "print d $coln" begin
            B = dropdefaults!(Fiber!(Dense{Int64}(colf())), A)
            @test check_output("print_dense_$coln.txt", sprint(show, B))
            @test check_output("print_dense_$(coln)_small.txt", sprint(show, B, context=:compact=>true))
            @test check_output("display_dense_$(coln).txt", sprint(show, MIME"text/plain"(), B))
            @test check_output("summary_dense_$(coln).txt", summary(B))
        end
    end
end