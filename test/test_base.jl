using Finch: AsArray

@testset "base" begin
    @info "Testing Julia Base Functions"
    A = @fiber(sl(e(0.0)), fsparse(([1, 3, 5, 7, 9],), [2.0, 3.0, 4.0, 5.0, 6.0], (10,)))
    B = @fiber(sl(e(0.0)), A)
    @test A == B

    A = [0.0 0.0 0.0 0.0; 1.0 0.0 0.0 1.0]
    B = @fiber(d(sl(e(0.0))), A)
    C = @fiber(d(d(e(0.0))), A)
    @test A == B

    A = [0 0; 0 0]
    B = @fiber(d(d(e(0.0))), A)
    @test A == B

    A = @fiber(d(e(0.0)), [0, 0, 0, 0])
    B = @fiber(d(e(0.0)), [0, 0, 0, 0, 0])
    @test size(A) != size(B) && A != B
        
    A = [0 0 0 0 1 0 0 1]
    B = @fiber(d(sl(e(0))), [0 0 0 0; 1 0 0 1])
    @test size(A) != size(B) && A != B

    A = @fiber(d(sl(e(0.0))), [1 0 0 0; 1 1 0 0; 1 1 1 0])
    B = [0 0 0 0; 1 1 0 0; 1 1 1 0]
    @test size(A) == size(B) && A != B
    C = @fiber(d(sl(e(0.0))), [0 0 0 0; 1 1 0 0; 1 1 1 0])
    @test B == C
    
    A = [NaN, 0.0, 3.14, 0.0]
    B = @fiber(sl(e(0.0)), [NaN, 0.0, 3.14, 0.0])
    C = @fiber(sl(e(0.0)), [NaN, 0.0, 3.14, 0.0])
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

        A = Fiber(SparseList(Dense(SparseList(Element{0.0, Float64}(collect(1:30).* 1.01), 5, [1, 3, 6, 8, 12, 14, 17, 20, 24, 27, 27, 28, 31], [2, 3, 3, 4, 5, 2, 3, 1, 3, 4, 5, 2, 4, 2, 4, 5, 2, 3, 5, 1, 3, 4, 5, 2, 3, 4, 2, 1, 2, 3]), 3), 4, [1, 5], [1, 2, 3, 4]))

        print(io, "A = ")
        show(io, MIME("text/plain"), A)
        println(io)

        for inds in [(1, 2, 3), (1, 1, 1), (1, :, 3), (:, 1, 3), (:, :, 3), (:, :, :)]
            print(io, "A["); join(io, inds, ","); print(io, "] = ")
            show(io, MIME("text/plain"), A[inds...])
            println(io)
        end
        
        @test check_output("getindex.txt", String(take!(io)))
    end

    let
        io = IOBuffer()
        println(io, "setindex! tests")

        @repl io A = @fiber(d(d(e(0.0), 10), 12))
        @repl io A[1, 4] = 3
        @repl io AsArray(A)
        @repl io A[4:6, 6] = 5:7
        @repl io AsArray(A)
        @repl io A[9, :] = 1:12
        @repl io AsArray(A)
        
        @test check_output("setindex.txt", String(take!(io)))
    end

    let
        io = IOBuffer()
        println(io, "broadcast tests")

        @repl io A = @fiber(d(sl(e(0.0))), [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0])
        @repl io B = [1, 2, 3, 4]
        @repl io C = A .+ B true
        @repl io AsArray(C)
        @repl io D = A .* B true
        @repl io AsArray(D)
        @repl io E = ifelse.(A .== 0, 1, 2)
        @repl io AsArray(E)
        
        @test check_output("broadcast.txt", String(take!(io)))
    end

    let
        io = IOBuffer()
        println(io, "reduce tests")

        @repl io A = @fiber(d(sl(e(0.0))), [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0])
        @repl io reduce(+, A, dims=(1,))
        @repl io reduce(+, A, dims=1)
        @repl io reduce(+, A, dims=(2,))
        @repl io reduce(+, A, dims=2)
        @repl io reduce(+, A, dims=(1,2))
        @repl io reduce(+, A, dims=:)
        
        @test check_output("reduce.txt", String(take!(io)))
    end

    let
        io = IOBuffer()
        println(io, "countstored tests")

        @repl io A = @fiber(d(sl(e(0.0))), [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0])
        @repl io countstored(A)
        @repl io A = @fiber(sc{2}(e(0.0)), [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0])
        @repl io countstored(A)
        @repl io A = @fiber(d(d(e(0.0))), [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0])
        @repl io countstored(A)
        @repl io A = @fiber(sl(d(e(0.0))), [0.0 0.0 4.4; 1.1 0.0 0.0; 2.2 0.0 5.5; 3.3 0.0 0.0])
        @repl io countstored(A)
        
        @test check_output("countstored.txt", String(take!(io)))
    end
end