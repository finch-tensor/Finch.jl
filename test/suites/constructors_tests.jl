@testitem "sparse_hash" begin
    @testset "uniform parent shifts" begin
        h = Finch.sparse_hash_hash
        a = h(1, 0) - h(0, 0)
        @test isodd(a)
        for i in (1, 37, typemax(Int))
            @test h(0, i) == hash(i)
            for p in (UInt(0), UInt(17), typemax(UInt) - UInt(1)),
                delta in (UInt(0), UInt(1), UInt(19), typemax(UInt))

                @test h(p + delta, i) - h(p, i) == a * delta
            end
        end
        @test h(Int32(17), Int32(37)) == h(17, 37)
    end

    @testset "collisions, wraparound, and resizing" for subtables in (1, 4)
        cap = 64 * subtables
        ctrl = fill(Finch.SPARSE_HASH_CTRL_EMPTY, cap)
        tbl = Vector{NTuple{3,Int}}(undef, cap)
        # Choose colliding keys at the last slot of the first subtable, so
        # insertion must wrap without crossing into the next subtable.
        p0 = findfirst(1:cap) do p
            Finch.sparse_hash_hash_slot_parts(Finch.sparse_hash_hash(p, 7), cap, subtables) ==
                (1, 63, 63)
        end
        entries = [(p0 + k * cap, 7, k + 1) for k in 0:11]
        for (p, i, q) in entries
            Finch.sparse_hash_table_insert_noresize!(ctrl, tbl, p, i, q, subtables)
        end
        @test ctrl[1] != Finch.SPARSE_HASH_CTRL_EMPTY
        for newcap in (cap, 2cap, 4cap)
            if newcap != cap
                Finch.sparse_hash_table_resize!(ctrl, tbl, newcap, subtables)
            end
            for (p, i, q) in entries
                @test Finch.sparse_hash_table_lookup(ctrl, tbl, p, i, subtables) == q
            end
            @test Finch.sparse_hash_table_lookup(ctrl, tbl, p0 + 12cap, 7, subtables) == 0
        end
        Finch.sparse_hash_table_insert_noresize!(ctrl, tbl, p0, 7, 99, subtables)
        @test Finch.sparse_hash_table_lookup(ctrl, tbl, p0, 7, subtables) == 99
        @test count(!=(Finch.SPARSE_HASH_CTRL_EMPTY), ctrl) == length(entries)
    end

    @testset "tensor assembly and updates" for single_writer in (true, false)
        data = [mod(i + 3j, 5) == 0 ? i + j : 0 for i in 1:17, j in 1:9]
        input = Tensor(Dense(Dense(Element(0))), data)
        tensor = Tensor(Dense(SparseHash{Int,single_writer}(Element(0))), data)
        @test Array(tensor) == data
        @finch begin
            for j in _, i in _
                tensor[i, j] += input[i, j]
            end
        end
        @test Array(tensor) == 2data
    end
end

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
        ncpu = cpu(:t, 4)
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

    @testset "CoalesceLevel" begin
        ncpu = cpu(:t, 2)
        tens = Tensor(Dense(Coalesce(ncpu, SparseList(Element(0.0)))), 2, 2)

        acc = Tensor(Dense(SparseList(Element(0.0))), [1 0; 2 0])
        @finch begin
            tens .= 0
            for j in parallel(_, ncpu)
                for i in _
                    tens[i, j] = acc[i, j] + acc[i, j]
                end
            end
        end

        @test tens[1, 1] == 2
        @test tens[1, 2] == 0
        @test tens[2, 1] == 4
        @test tens[2, 2] == 0

        acc = Tensor(Dense(SparseList(Element(0.0))), [0 1; 0 2])
        @finch begin
            tens .= 0
            for j in parallel(_, ncpu)
                for i in _
                    tens[i, j] = acc[i, j] + acc[i, j]
                end
            end
        end

        @test tens[1, 1] == 0
        @test tens[1, 2] == 2
        @test tens[2, 1] == 0
        @test tens[2, 2] == 4

        acc = Tensor(Dense(SparseList(Element(0.0))), [1 2; 3 4])
        @finch begin
            tens .= 0
            for j in parallel(_, ncpu)
                for i in _
                    tens[i, j] = acc[i, j] + acc[i, j]
                end
            end
        end

        @test tens[1, 1] == 2
        @test tens[1, 2] == 4
        @test tens[2, 1] == 6
        @test tens[2, 2] == 8
    end
end
