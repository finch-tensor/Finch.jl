@testitem "coalesce_hash_subtables" begin
    hash_counts(lvl::Finch.AbstractLevel) =
        hasproperty(lvl, :lvl) ? hash_counts(lvl.lvl) : Int[]
    hash_counts(lvl::Finch.SparseHashLevel) = [lvl.subtables; hash_counts(lvl.lvl)]

    @testset "destination follows configured merge workers" begin
        for P in (1, 2, 3, 5, 8), mode in (:normalize, :fast)
            for fmt in (
                () -> SparseHash(Element(0), 7, 16),
                () -> Dense(SparseHash(Element(0), 7, 16), 3),
                () -> SparseList(SparseHash(Element(0), 7, 16), 3),
                () -> SparseByteMap(SparseHash(Element(0), 7, 16), 3),
                () -> SparseHash(SparseHash{Int32,false}(Element(0), 7, 16), 3, 16),
            )
                original = fmt()
                lvl = Coalesce(cpu(:hash, P), original; mode)
                @test hash_counts(lvl.coalescent) ==
                    fill(nextpow(2, P), length(hash_counts(original)))
                @test all(==(16), hash_counts(lvl.lvl))
                @test all(==(16), hash_counts(original))
                if mode == :fast
                    @test lvl.accumulator === nothing
                end
            end
        end
        # Generated hash accumulators are task-local, not the merge destination.
        lvl = Coalesce(cpu(:hash, 5), SparseList(SparseList(Element(0))))
        @test hash_counts(lvl.accumulator) == [1, 1]
    end

    @testset "reject non-power-of-two bucket counts" begin
        for P in (0, 3, 5, 6)
            @test_throws ArgumentError SparseHash(Element(0), 7, P)
        end
        for mode in (:normalize, :fast)
            @test_throws ArgumentError Coalesce(cpu(:hash, 0), SparseHash(Element(0)); mode)
            @test_throws ArgumentError Coalesce(
                cpu(:hash, 0), Dense(SparseHash(Element(0))); mode
            )
        end
        @test_throws ArgumentError SparseHash(Element(0), 7, 1.5)
    end
end

@testitem "coalesce_merges" begin
    # Split `data` into shards as the normalizing merge sees them: shard p stores
    # `data` restricted to the flat (column-major) index range `cuts[p]`.
    function band_shards(fmt, data, cuts)
        P = length(cuts)
        mem = Finch.MultiChannelMemory(cpu(:t, P), P)
        chans(xs) = Finch.MultiChannelBuffer(mem, xs)
        shards = map(cuts) do cut
            masked = zero(data)
            masked[cut] = data[cut]
            Tensor(fmt(), masked).lvl
        end
        function stack(lvls)
            lvl = first(lvls)
            if lvl isa Finch.ElementLevel
                Element(zero(eltype(data)), chans([l.val for l in lvls]))
            elseif lvl isa Finch.DenseLevel
                Dense(stack([l.lvl for l in lvls]), lvl.shape)
            elseif lvl isa Finch.SparseListLevel
                SparseList{Int}(
                    stack([l.lvl for l in lvls]), lvl.shape,
                    chans([l.ptr for l in lvls]), chans([l.idx for l in lvls]),
                )
            else
                SparseByteMap{Int}(
                    stack([l.lvl for l in lvls]), lvl.shape,
                    chans([l.ptr for l in lvls]), chans([l.tbl for l in lvls]),
                    chans([l.srt for l in lvls]),
                )
            end
        end
        return stack(shards)
    end

    # Merging must give exactly the storage of a tensor built directly, whether
    # or not the merge knows each shard's band.
    function check_merge(fmt, data, cuts)
        src = band_shards(fmt, data, cuts)
        for bands in (cuts, nothing)
            # Destinations arrive cleared, as assemble_level! leaves them.
            dst = Tensor(fmt(), zero(data)).lvl
            Finch.coalesce_shards!(src, dst, length(cuts), 1, bands)
            @test Finch.isstructequal(dst, Tensor(fmt(), data).lvl)
            @test Array(Tensor(dst)) == data
        end
    end

    formats_2d = [
        () -> Dense(Dense(Element(0))),
        () -> Dense(SparseList(Element(0))),
        () -> Dense(SparseByteMap(Element(0))),
        () -> SparseList(Dense(Element(0))),
        () -> SparseList(SparseList(Element(0))),
        () -> SparseList(SparseByteMap(Element(0))),
        () -> SparseByteMap(Dense(Element(0))),
        () -> SparseByteMap(SparseList(Element(0))),
        () -> SparseByteMap(SparseByteMap(Element(0))),
    ]

    # Columns hold flat indices 1:5, 6:10, 11:15, and 16:20; column 3 is empty.
    data = zeros(Int, 5, 4)
    data[[1, 3, 4], 1] .= [1, 2, 3]
    data[[1, 5], 2] .= [4, 5]
    data[[2, 5], 4] .= [6, 7]
    @testset "2d $(summary(fmt()))" for fmt in formats_2d
        # One shard.
        check_merge(fmt, data, [1:20])
        # Bands along column boundaries.
        check_merge(fmt, data, [1:5, 6:10, 11:15, 16:20])
        # Column 1 split between shards, and a band covering only the empty column.
        check_merge(fmt, data, [1:3, 4:12, 13:20])
        # Column 2 spans three shards and the middle one holds none of it.
        check_merge(fmt, data, [1:7, 8:9, 10:20])
        # Empty bands, as when there are more tasks than indices.
        check_merge(fmt, data, [1:5, 6:5, 6:20, 21:20])
    end

    formats_3d = [
        () -> Dense(Dense(Dense(Element(0)))),
        () -> SparseList(Dense(SparseList(Element(0)))),
        () -> SparseByteMap(SparseList(Dense(Element(0)))),
        () -> Dense(SparseByteMap(SparseList(Element(0)))),
        () -> SparseList(SparseByteMap(SparseByteMap(Element(0)))),
    ]
    data_3d = zeros(Int, 3, 2, 3)
    data_3d[[1, 3, 4, 8, 9, 13, 16, 18]] .= 1:8
    @testset "3d $(summary(fmt()))" for fmt in formats_3d
        check_merge(fmt, data_3d, [1:18])
        check_merge(fmt, data_3d, [1:4, 5:8, 9:14, 15:18])
        # The middle bands hold only parts of slices.
        check_merge(fmt, data_3d, [1:7, 8:8, 9:10, 11:18])
    end
end

@testitem "coalesce_end_to_end" begin
    using Random

    function outer!(C, A, BT, device)
        return (@finch begin
            C .= 0
            for k in parallel(_, device), j in _, i in _
                C[i, j] += A[i, k] * BT[j, k]
            end
            return C
        end).C
    end

    function copy_columns!(C, X, device)
        return (@finch begin
            C .= 0
            for j in parallel(_, device), i in _
                C[i, j] = X[i, j]
            end
            return C
        end).C
    end

    formats = [
        () -> Dense(Dense(Element(0))),
        () -> Dense(SparseList(Element(0))),
        () -> Dense(SparseByteMap(Element(0))),
        () -> SparseList(Dense(Element(0))),
        () -> SparseList(SparseList(Element(0))),
        () -> SparseList(SparseByteMap(Element(0))),
        () -> SparseByteMap(Dense(Element(0))),
        () -> SparseByteMap(SparseList(Element(0))),
        () -> SparseByteMap(SparseByteMap(Element(0))),
    ]

    rng = MersenneTwister(1)
    m, k, n = 37, 5, 29
    a = rand(rng, 1:9, m, k) .* (rand(rng, m, k) .< 0.3)
    bt = rand(rng, 1:9, n, k) .* (rand(rng, n, k) .< 0.3)
    a[:, end] .= 0
    BT = Tensor(Dense(SparseList(Element(0))), bt)
    device = cpu(:k, k)

    @testset "$(summary(fmt()))" for fmt in formats
        # Overlapping outer products are summed into band-split shards and
        # merged; the odd sizes split columns between shards. Reusing the
        # output checks that the merge leaves it resettable.
        C = Tensor(Coalesce(device, fmt()))
        for a_i in (a, circshift(a, (1, 0)))
            C = outer!(C, Tensor(Dense(SparseList(Element(0))), a_i), BT, device)
            expected = a_i * transpose(bt)
            @test Array(C) == expected
            @test Finch.isstructequal(C.lvl.coalescent, Tensor(fmt(), expected).lvl)
        end

        # In :fast mode, tasks write disjoint column blocks directly.
        X = Tensor(Dense(SparseList(Element(0))), a * transpose(bt))
        F = Tensor(Coalesce(cpu(:k, 4), fmt(); mode=:fast))
        @test Array(copy_columns!(F, X, cpu(:k, 4))) == Array(X)
    end

    @testset "more tasks than work" begin
        for fmt in formats[[1, 3, 5]]
            # Fewer indices than tasks, and a single nonzero, leave bands empty.
            for (a_i, bt_i) in (
                (ones(Int, 2, 8), ones(Int, 2, 8)),
                (
                    [Int(i == 3 && k == 1) for i in 1:6, k in 1:8],
                    [Int(j == 2 && k == 1) for j in 1:5, k in 1:8],
                ),
            )
                A_i = Tensor(Dense(SparseList(Element(0))), a_i)
                BT_i = Tensor(Dense(SparseList(Element(0))), bt_i)
                C = outer!(Tensor(Coalesce(cpu(:k, 8), fmt())), A_i, BT_i, cpu(:k, 8))
                @test Array(C) == a_i * transpose(bt_i)
            end
        end
    end
end
