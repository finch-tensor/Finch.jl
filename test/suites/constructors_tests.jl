@testitem "constructors" setup = [CheckOutput] begin
    using Base.Meta
    using Finch: Structure

    basic_levels = [
        ("Dense", Dense, (;), [[0.0, 2.0, 2.0, 0.0, 3.0, 3.0]]),
        ("RunList", RunList, (;), [[0.0, 2.0, 2.0, 0.0, 3.0, 3.0]]),
        ("RunListlazy", RunList, (; merge=false), [[0.0, 2.0, 2.0, 0.0, 3.0, 3.0]]),
        ("SparseList", SparseList, (;), [[0.0, 2.0, 2.0, 0.0, 3.0, 3.0]]),
        ("SparseBlockList", SparseBlockList, (;), [[0.0, 2.0, 2.0, 0.0, 3.0, 3.0]]),
        ("SparseBand", SparseBand, (;), [[0.0, 2.0, 2.0, 0.0, 3.0, 0.0]]),
        ("SparseByteMap", SparseByteMap, (;), [[0.0, 2.0, 2.0, 0.0, 3.0, 3.0]]),
        ("SparseRunList", SparseRunList, (;), [[0.0, 2.0, 2.0, 0.0, 3.0, 3.0]]),
        (
            "SparseRunListLazy",
            SparseRunList,
            (; merge=false),
            [[0.0, 2.0, 2.0, 0.0, 3.0, 3.0]],
        ),
        ("SparseDict", SparseDict, (;), [[0.0, 2.0, 2.0, 0.0, 3.0, 3.0]]),
        ("SparsePoint", SparsePoint, (;), [[0.0, 0.0, 2.0, 0.0, 0.0, 0.0]]),
        ("SparseInterval", SparseInterval, (;), [[0.0, 0.0, 2.0, 0.0, 0.0, 0.0]]),
    ]

    for (key, Lvl, flags, arrs) in basic_levels
        @testset "Construct $key" begin
            io = IOBuffer()
            println(io, "Tensor($key(Element(0))) constructors:")

            for arr in arrs
                fbr = dropfills!(Tensor(Lvl(Element(zero(eltype(arr))); flags...)), arr)
                println(io, "initialized tensor: ", fbr)
                lvl = fbr.lvl
                props = map(name -> getproperty(lvl, name), propertynames(lvl))
                @test Structure(fbr) == Structure(Tensor(Lvl(props...; flags...)))
                @test Structure(fbr) == Structure(Tensor(Lvl{Int}(props...; flags...)))

                fbr = dropfills!(
                    Tensor(Lvl{Int16}(Element(zero(eltype(arr))); flags...)), arr
                )
                println(io, "initialized tensor: ", fbr)
                lvl = fbr.lvl
                props = map(name -> getproperty(lvl, name), propertynames(lvl))
                @test Structure(fbr) == Structure(Tensor(Lvl{Int16}(props...; flags...)))

                fbr = Tensor(Lvl(Element(0.0), 7; flags...))
                println(io, "sized tensor: ", fbr)
                lvl = fbr.lvl
                @test Structure(fbr) == Structure(Tensor(Lvl(Element(0.0), 7; flags...)))
                @test Structure(fbr) ==
                    Structure(Tensor(Lvl{Int}(Element(0.0), 7; flags...)))

                fbr = Tensor(Lvl{Int16}(Element(0.0), 7; flags...))
                println(io, "sized tensor: ", fbr)
                lvl = fbr.lvl
                @test Structure(fbr) ==
                    Structure(Tensor(Lvl(Element(0.0), Int16(7); flags...)))
                @test Structure(fbr) ==
                    Structure(Tensor(Lvl{Int16}(Element(0.0), 7; flags...)))

                fbr = Tensor(Lvl(Element(0.0); flags...))
                println(io, "empty tensor: ", fbr)
                lvl = fbr.lvl
                @test Structure(fbr) == Structure(Tensor(Lvl(Element(0.0); flags...)))
                @test Structure(fbr) == Structure(Tensor(Lvl{Int}(Element(0.0); flags...)))
                @test Structure(fbr) == Structure(Tensor(Lvl(Element(0.0), 0; flags...)))
                @test Structure(fbr) ==
                    Structure(Tensor(Lvl{Int}(Element(0.0), 0; flags...)))

                fbr = Tensor(Lvl{Int16}(Element(0.0); flags...))
                println(io, "empty tensor: ", fbr)
                lvl = fbr.lvl
                @test Structure(fbr) ==
                    Structure(Tensor(Lvl{Int16}(Element(0.0); flags...)))
                @test Structure(fbr) ==
                    Structure(Tensor(Lvl(Element(0.0), Int16(0); flags...)))
                @test Structure(fbr) ==
                    Structure(Tensor(Lvl{Int16}(Element(0.0), 0; flags...)))

                fbr = Tensor(
                    Dense(Lvl(Element(Int64(0)); flags...)), [0 0 0 1; 0 1 0 0; 0 0 0 0]
                )
                res = similar(fbr)
                @test size(res) == size(fbr)
                @test fill_value(res) == 0 && eltype(res) == Int64

                res = similar(fbr, (10, 5))
                @test size(res) == (10, 5)
                @test fill_value(res) == 0 && eltype(res) == Int64

                res = similar(fbr, Float64)
                @test size(res) == size(fbr)
                @test fill_value(res) == 0 && eltype(res) == Float64

                res = similar(fbr, 1, Float64)
                @test size(res) == size(fbr)
                @test fill_value(res) == 1 && eltype(res) == Float64

                res = similar(fbr, ComplexF32, (10, 5))
                @test size(res) == (10, 5)
                @test fill_value(res) == 0 && eltype(res) == ComplexF32

                res = similar(fbr, 2, ComplexF64, (10, 5))
                @test size(res) == (10, 5)
                @test fill_value(res) == 2 && eltype(res) == ComplexF64

                if key == "SparsePoint" || key == "SparseInterval"
                    continue  # don't test copyto! for Single*
                end

                res = copyto!(similar(fbr, -1, Float64), fbr)
                @test res == fbr
                @test fill_value(res) == -1 && eltype(res) == Float64
            end

            @test check_output("constructors/format_$key.txt", String(take!(io)))
        end
    end

    multi_levels = [
        (
        "SparseCOO",
        SparseCOO,
        (;),
        [
            [0.0, 2.0, 2.0, 0.0, 3.0, 3.0],
            [0.0 2.0 2.0; 0.0 3.0 3.0],
        ],
    ),
    ]

    for (key, Lvl, flags, arrs) in multi_levels
        @testset "Tensor($key{?}(Element(0)))" begin
            io = IOBuffer()
            for arr in arrs
                N = ndims(arr)
                println(io, "Tensor($key{$N}(Element(0))) constructors:")

                fbr = dropfills!(Tensor(Lvl{N}(Element(zero(eltype(arr))); flags...)), arr)
                println(io, "initialized tensor: ", fbr)
                lvl = fbr.lvl
                props = map(name -> getproperty(lvl, name), propertynames(lvl))
                @test Structure(fbr) == Structure(Tensor(Lvl{N}(props...; flags...)))
                @test Structure(fbr) ==
                    Structure(Tensor(Lvl{N,NTuple{N,Int}}(props...; flags...)))

                fbr = dropfills!(
                    Tensor(Lvl{N,NTuple{N,Int16}}(Element(zero(eltype(arr))); flags...)),
                    arr,
                )
                println(io, "initialized tensor: ", fbr)
                lvl = fbr.lvl
                props = map(name -> getproperty(lvl, name), propertynames(lvl))
                @test Structure(fbr) ==
                    Structure(Tensor(Lvl{N,NTuple{N,Int16}}(props...; flags...)))

                fbr = Tensor(Lvl{N}(Element(0.0), size(arr); flags...))
                println(io, "sized tensor: ", fbr)
                lvl = fbr.lvl
                @test Structure(fbr) ==
                    Structure(Tensor(Lvl{N}(Element(0.0), size(arr); flags...)))
                @test Structure(fbr) == Structure(
                    Tensor(Lvl{N,NTuple{N,Int}}(Element(0.0), size(arr); flags...))
                )

                fbr = Tensor(
                    Lvl{N,NTuple{N,Int16}}(Element(0.0), Int16.(size(arr)); flags...)
                )
                println(io, "sized tensor: ", fbr)
                lvl = fbr.lvl
                @test Structure(fbr) ==
                    Structure(Tensor(Lvl{N}(Element(0.0), Int16.(size(arr)); flags...)))
                @test Structure(fbr) == Structure(
                    Tensor(
                        Lvl{N,NTuple{N,Int16}}(Element(0.0), Int16.(size(arr)); flags...)
                    ),
                )

                zerodim = size(arr) .- size(arr)

                fbr = Tensor(Lvl{N}(Element(0.0); flags...))
                println(io, "empty tensor: ", fbr)
                lvl = fbr.lvl
                @test Structure(fbr) == Structure(Tensor(Lvl{N}(Element(0.0); flags...)))
                @test Structure(fbr) ==
                    Structure(Tensor(Lvl{N,NTuple{N,Int}}(Element(0.0); flags...)))
                @test Structure(fbr) ==
                    Structure(Tensor(Lvl{N}(Element(0.0), zerodim; flags...)))
                @test Structure(fbr) == Structure(
                    Tensor(Lvl{N,NTuple{N,Int}}(Element(0.0), zerodim; flags...))
                )

                fbr = Tensor(Lvl{N,NTuple{N,Int16}}(Element(0.0); flags...))
                println(io, "empty tensor: ", fbr)
                lvl = fbr.lvl
                @test Structure(fbr) ==
                    Structure(Tensor(Lvl{N,NTuple{N,Int16}}(Element(0.0); flags...)))
                @test Structure(fbr) ==
                    Structure(Tensor(Lvl{N}(Element(0.0), Int16.(zerodim); flags...)))
                @test Structure(fbr) == Structure(
                    Tensor(Lvl{N,NTuple{N,Int16}}(Element(0.0), Int16.(zerodim); flags...))
                )

                fbr = Tensor(Lvl{2}(Element(0); flags...), Matrix(reshape(1:25, (5, 5))))
                res = copyto!(similar(fbr, -1, Float64), fbr)
                @test res == fbr
                @test fill_value(res) == -1 && eltype(res) == Float64
            end
            @test check_output("constructors/format_$(key).txt", String(take!(io)))
        end
    end

    @testset "Tensor(Dense(Separate(Dense(Element(0)))))" begin
        io = IOBuffer()
        arr = [0.0 2.0 2.0 0.0 3.0 3.0;
            1.0 0.0 7.0 1.0 0.0 0.0;
            0.0 0.0 0.0 0.0 0.0 9.0]

        println(io, "Tensor(Dense(Separate(Dense(Element(0))))):")

        fbr = dropfills!(Tensor(Dense(Separate(Dense(Element(0))))), arr)

        # sublvl = Tensor(Dense(Element(0)), [])
        # col1 = dropfills!(Tensor((Dense(Element(0)))), arr[:, 1])
        # col2 = dropfills!(Tensor((Dense(Element(0)))), arr[:, 2])
        # col3 = dropfills!(Tensor((Dense(Element(0)))), arr[:, 3])
        # col4 = dropfills!(Tensor((Dense(Element(0)))), arr[:, 4])
        # col5 = dropfills!(Tensor((Dense(Element(0)))), arr[:, 5])
        # col6 = dropfills!(Tensor((Dense(Element(0)))), arr[:, 6])
        # vals = [col1, col2, col3, col4, col5, col6]

        println(io, "initialized tensor: ", fbr)
        @test Structure(fbr) ==
            Structure(Tensor(Dense(Separate(fbr.lvl.lvl.lvl, fbr.lvl.lvl.val), 6)))
        @test Structure(fbr) == Structure(
            Tensor(
                Dense(
                    Separate{typeof(fbr.lvl.lvl.lvl),typeof(fbr.lvl.lvl.val)}(
                        fbr.lvl.lvl.lvl, fbr.lvl.lvl.val
                    ),
                    6,
                ),
            ),
        )

        fbr = Tensor(Dense(Separate(Dense(Element(0), 3)), 6))
        println(io, "sized tensor: ", fbr)
        @test Structure(fbr) == Structure(Tensor(Dense(Separate(Dense(Element(0), 3)), 6)))

        fbr = Tensor(Dense(Separate(Dense(Element(0)))))
        println(io, "empty tensor: ", fbr)
        @test Structure(fbr) == Structure(Tensor(Dense(Separate(Dense(Element(0))))))

        fbr = Tensor(Dense(Separate(Dense(Element(0)))), Matrix(reshape(1:25, (5, 5))))
        res = copyto!(similar(fbr, -1, Float64), fbr)
        @test res == fbr
        @test fill_value(res) == -1 && eltype(res) == Float64

        @test check_output("constructors/format_d_p_d_e.txt", String(take!(io)))
    end

    @testset "Tensor(Dense(Mutex(Dense(Element(0)))))" begin
        io = IOBuffer()
        arr = [0.0 2.0 2.0 0.0 3.0 3.0;
            1.0 0.0 7.0 1.0 0.0 0.0;
            0.0 0.0 0.0 0.0 0.0 9.0]

        fbr = dropfills!(Tensor(Dense(Mutex(Dense(Element(0))))), arr)

        println(io, "initialized tensor: ", fbr)
        @test Structure(fbr) ==
            Structure(Tensor(Dense(Mutex(fbr.lvl.lvl.lvl, fbr.lvl.lvl.locks), 6)))
        @test Structure(fbr) == Structure(
            Tensor(
                Dense(
                    Mutex{Vector{Base.Threads.SpinLock},typeof(fbr.lvl.lvl.lvl)}(
                        fbr.lvl.lvl.lvl, fbr.lvl.lvl.locks
                    ),
                    6,
                ),
            ),
        )

        fbr = Tensor(Dense(Mutex(Dense(Element(0), 3)), 6))
        println(io, "sized tensor: ", fbr)
        @test Structure(fbr) == Structure(Tensor(Dense(Mutex(Dense(Element(0), 3)), 6)))

        fbr = Tensor(Dense(Mutex(Dense(Element(0)))))
        println(io, "empty tensor: ", fbr)
        @test Structure(fbr) == Structure(Tensor(Dense(Mutex(Dense(Element(0))))))

        fbr = Tensor(Dense(Mutex(Dense(Element(0)))), Matrix(reshape(1:25, (5, 5))))
        res = copyto!(similar(fbr, -1, Float64), fbr)
        @test res == fbr
        @test fill_value(res) == -1 && eltype(res) == Float64

        @test check_output("constructors/format_d_a_d_e.txt", String(take!(io)))
    end

    @testset "PlusOneVector" begin
        # test off-by-one
        v = Vector([1, 0, 2, 3])
        obov = PlusOneVector(v)
        @test obov == v .+ 1
        @test obov.data == v

        # test off-by-one in a tensor
        coo = Tensor(
            SparseCOO{2}(
                Element(0, Vector([1, 2, 3])),  # data
                (3, 3),  # shape
                Vector([1, 4]),  # ptr
                (
                    PlusOneVector(Vector([0, 0, 2])),
                    PlusOneVector(Vector([0, 2, 2])),
                ),  # off-by-one indices
            ),
        )
        @test Array(Tensor(Dense(Dense(Element(0))), coo)) == [1 0 2; 0 0 0; 0 0 3]

        # test off-by-one write operation
        val = 10
        obov[2] = val
        @test obov == [2, val, 3, 4] && obov.data == [1, val - 1, 2, 3]
        obov[1:3] .= val
        @test obov == [val, val, val, 4] && obov.data == [val - 1, val - 1, val - 1, 3]
    end

    @testset "ShardLevel" begin
        #Test shard ShardLevel
        ncpu = cpu(4)
        A = Tensor(Dense(Shard(ncpu, Element(0.0))), 4)
        B = Tensor(Dense(Shard(ncpu, Sparse(Element(0.0)))), 4, 4)
        C = Tensor(Dense(Shard(ncpu, Dense(Element(0.0)))), 4, 4)

        @finch begin
            A .= 0
            for i in parallel(1:4, ncpu)
                let j = i
                    A[i] = j
                end
            end
        end

        @test A[1] == 1
        @test A[4] == 4

        @finch begin
            B .= 0
            for j in parallel(1:4, ncpu)
                let q = j
                    for i in 1:4
                        let r = i
                            B[i, j] = q + r
                        end
                    end
                end
            end
        end

        @test B[4, 4] == 8

        @finch begin
            C .= 0
            for j in parallel(1:4, ncpu)
                let q = j
                    for i in 1:4
                        let r = i
                            C[i, j] = B[i, j] + A[j]
                        end
                    end
                end
            end
        end

        @test C[4, 4] == 12
    end
end
