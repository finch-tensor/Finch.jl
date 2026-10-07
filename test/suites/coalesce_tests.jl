@testitem "coalesce_hash_subtables" begin
    hash_counts(lvl::Finch.AbstractLevel) =
        hasproperty(lvl, :lvl) ? hash_counts(lvl.lvl) : Int[]
    hash_counts(lvl::Finch.SparseHashLevel) = [lvl.subtables; hash_counts(lvl.lvl)]

    @testset "all hashes follow configured merge workers" begin
        for P in (1, 2, 3, 5, 8), mode in (:normalize, :fast)
            for fmt in (
                () -> SparseHash(Element(0), 7, 16),
                () -> Dense(SparseHash(Element(0), 7, 16), 3),
                () -> SparseList(SparseHash(Element(0), 7, 16), 3),
                () -> SparseByteMap(SparseHash(Element(0), 7, 16), 3),
                () -> SparseHash(SparseHash{Int32}(Element(0), 7, 16), 3, 16),
            )
                original = fmt()
                lvl = Coalesce(cpu(:hash, P), original; mode)
                @test hash_counts(lvl.coalescent) ==
                    fill(nextpow(2, P), length(hash_counts(original)))
                @test hash_counts(lvl.lvl) == hash_counts(lvl.coalescent)
                @test all(==(16), hash_counts(original))
                if mode == :fast
                    @test lvl.accumulator === nothing
                else
                    @test all(==(nextpow(2, P)), hash_counts(lvl.accumulator))
                end
            end
        end
        lvl = Coalesce(cpu(:hash, 5), SparseList(SparseList(Element(0))))
        @test hash_counts(lvl.accumulator) == [8, 8]
    end

    @testset "similar levels start empty" begin
        data = [mod(i + 3j, 5) == 0 ? i + j : 0 for i in 1:17, j in 1:9]
        for B in (1, 16), mode in (:normalize, :fast)
            original = Tensor(Dense(SparseHash(Element(0), 17, B)), data)
            lvl = Coalesce(cpu(:hash, 5), original.lvl; mode)
            @test Array(original) == data
            src = lvl.lvl.lvl
            @test src.subtables == 8
            @test all(isempty, src.tbl.data)
            @test all(isempty, src.tbl_ctrl.data)
            @test all(isempty, src.perm.data)
            @test all(isempty, src.lvl.val.data)
        end
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
            elseif lvl isa Finch.SparseHashLevel
                SparseHash{Int}(
                    stack([l.lvl for l in lvls]), lvl.shape, lvl.subtables,
                    (chans([getfield(l, f) for l in lvls]) for f in Finch.SPARSE_HASH_BUFFERS)...,
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

    has_hash(lvl) = lvl isa Finch.SparseHashLevel ||
        (hasproperty(lvl, :lvl) && has_hash(lvl.lvl))
    function check_hash_storage(lvl)
        if lvl isa Finch.SparseHashLevel
            @test issorted(lvl.key[lvl.perm])
            # The first block holds the bucket totals.
            totals = lvl.tbl_count[1:(lvl.subtables)]
            @test sum(totals) == length(lvl.perm)
            live = Set(lvl.perm)
            @test all(q -> lvl.key[q][3] == (q in live ?
                Finch.SPARSE_HASH_KEY_RETAINED : Finch.SPARSE_HASH_KEY_FREE), eachindex(lvl.key))
            width = length(lvl.tbl) ÷ lvl.subtables
            @test totals == [count(!=(Finch.SPARSE_HASH_CTRL_EMPTY),
                view(lvl.tbl_ctrl, ((b - 1) * width + 1):(b * width))) for b in 1:lvl.subtables]
            for r in eachindex(lvl.perm)
                q = lvl.perm[r]
                p, i = lvl.key[q]
                @test lvl.ptr[p] <= r < lvl.ptr[p + 1]
                @test Finch.sparse_hash_lookup(lvl.tbl_ctrl, lvl.tbl, lvl.key, p, i, lvl.subtables) == q
            end
        end
        hasproperty(lvl, :lvl) && check_hash_storage(lvl.lvl)
    end

    # Merging must give exactly the storage of a tensor built directly, whether
    # or not the merge knows each shard's band.
    function check_merge(fmt, data, cuts)
        src = band_shards(fmt, data, cuts)
        for bands in (cuts, nothing)
            # Destinations arrive cleared, as assemble_level! leaves them.
            dst = Tensor(fmt(), zero(data)).lvl
            Finch.coalesce_shards!(src, dst, length(cuts), 1, bands)
            if has_hash(dst)
                check_hash_storage(dst)
            else
                @test Finch.isstructequal(dst, Tensor(fmt(), data).lvl)
            end
            @test Array(Tensor(dst)) == data
        end
    end

    @testset "shared positions own metadata, not children" begin
        data = zeros(Int, 2, 6)
        data[1, 1] = 1
        data[:, 4] .= [2, 3]
        data[2, 6] = 4
        cuts = [1:7, 8:12]
        for (fmt, shared, shared_dst) in (
            (() -> SparseList(Dense(Element(0))), 1, 2),
            (() -> SparseByteMap(Dense(Element(0))), 4, 4),
        )
            src = band_shards(fmt, data, cuts)
            dst = Tensor(fmt(), zero(data)).lvl
            plan = Finch.setup_coalesce!(src, 1, dst, 2, [0, 0], true)
            @test eltype(plan.shared) == Int
            @test plan.shared == [0, shared]
            @test plan.shared_dst == [0, shared_dst]
            fmt() isa SparseListLevel && @test plan.off == [0, 2]
            @test plan.nnz == 3
            @test plan.child.child.overlap
            for (buffer, start, value) in plan.init
                fill!(view(buffer, start:length(buffer)), value)
            end
            for tid in 1:2
                Finch.coalesce_shard!(tid, plan, src, dst, (cuts[tid],))
            end
            @test Array(Tensor(dst)) == data
            @test Finch.isstructequal(dst, Tensor(fmt(), data).lvl)
        end
    end

    @testset "initialize overlapping values with their fill value" begin
        for P in (1, 2, 5), old_length in (0, 2, 8)
            mem = Finch.MultiChannelMemory(cpu(:init, P), P)
            values = [fill(7, 3) for _ in 1:P]
            values[1][1] = 1
            values[end][3] = 3
            src = Element(7, Finch.MultiChannelBuffer(mem, values))
            dst = Element(7, fill(7, old_length))
            Finch.coalesce_shards!(src, dst, P, 3, nothing)
            @test dst.val == [1, 7, 3]
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
        () -> Dense(SparseHash(Element(0), 0, 8)),
        () -> SparseList(SparseHash(Element(0), 0, 8)),
        () -> SparseByteMap(SparseHash(Element(0), 0, 8)),
        () -> SparseHash(Dense(Element(0)), 0, 8),
        () -> SparseHash(SparseByteMap(Element(0)), 0, 8),
        () -> SparseHash(SparseHash(Element(0), 0, 8), 0, 8),
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
        # No entries will be copied, so initialization must finish the output.
        check_merge(fmt, zero(data), [1:5, 6:5, 6:20, 21:20])
    end

    @testset "lists cannot merge below a hash" begin
        fmt = () -> SparseHash(SparseList(Element(0)), 0, 8)
        src = band_shards(fmt, data, [1:3, 4:12, 13:20])
        dst = Tensor(fmt(), zero(data)).lvl
        @test_throws ArgumentError Finch.coalesce_shards!(src, dst, 3, 1, nothing)
    end

    @testset "thaw reuses merged child holes" for child in (
        () -> Dense(Element(0)), () -> SparseByteMap(Element(0)),
        () -> SparseHash(Element(0), 0, 8),
    )
        fmt = () -> SparseHash(child(), 0, 8)
        src = band_shards(fmt, data, [1:3, 4:12, 13:20])
        dst = Tensor(fmt(), zero(data)).lvl
        Finch.coalesce_shards!(src, dst, 3, 1, nothing)
        @test length(dst.key) > length(dst.perm)
        extent = length(dst.key)
        tensor = Tensor(dst)
        input = Tensor(Dense(Dense(Element(0))), ones(Int, size(data)))
        @finch for j in _, i in _
            tensor[i, j] += input[i, j]
        end
        @test Array(tensor) == data .+ 1
        @test length(tensor.lvl.key) == extent
        check_hash_storage(tensor.lvl)
    end

    formats_3d = [
        () -> Dense(Dense(Dense(Element(0)))),
        () -> SparseList(Dense(SparseList(Element(0)))),
        () -> SparseByteMap(SparseList(Dense(Element(0)))),
        () -> Dense(SparseByteMap(SparseList(Element(0)))),
        () -> SparseList(SparseByteMap(SparseByteMap(Element(0)))),
        () -> SparseHash(Dense(SparseHash(Element(0), 0, 8)), 0, 8),
        () -> SparseHash(SparseByteMap(SparseHash(Element(0), 0, 8)), 0, 8),
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

    has_hash(lvl) = lvl isa Finch.SparseHashLevel ||
        (hasproperty(lvl, :lvl) && has_hash(lvl.lvl))

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
        () -> Dense(SparseHash(Element(0))),
        () -> SparseList(SparseHash(Element(0))),
        () -> SparseByteMap(SparseHash(Element(0))),
        () -> SparseHash(Dense(Element(0))),
        () -> SparseHash(SparseByteMap(Element(0))),
        () -> SparseHash(SparseHash(Element(0))),
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
            if !has_hash(C.lvl.coalescent)
                @test Finch.isstructequal(C.lvl.coalescent, Tensor(fmt(), expected).lvl)
            end
        end

        # In :fast mode, tasks write disjoint column blocks directly.
        X = Tensor(Dense(SparseList(Element(0))), a * transpose(bt))
        F = Tensor(Coalesce(cpu(:k, 4), fmt(); mode=:fast))
        @test Array(copy_columns!(F, X, cpu(:k, 4))) == Array(X)
    end

    @testset "more tasks than work" begin
        for fmt in formats[[1, 3, 5, 10, 13, 15]]
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


@testitem "coalesce_hash_setup" begin
    # Frozen hash shards holding entries (p, i, q), one list per shard.
    function hash_shards(entries; B=nextpow(2, length(entries)))
        P = length(entries)
        mem = Finch.MultiChannelMemory(cpu(:setup, P), P)
        channels(data) = Finch.MultiChannelBuffer(mem, data)
        shards = map(entries) do es
            counts = zeros(Int, B)
            for (p, i, _) in es
                counts[Finch.sparse_hash_hash_subtable(Finch.sparse_hash_hash(p, i), B)] += 1
            end
            ctrl = fill(Finch.SPARSE_HASH_CTRL_EMPTY,
                        B * Finch.sparse_hash_table_capacity(maximum(counts)))
            tbl = zeros(Int, length(ctrl))
            key = fill((0, 0, Finch.SPARSE_HASH_KEY_FREE), maximum(e -> e[3], es; init=0))
            for (p, i, q) in es
                x = Finch.sparse_hash_hash(p, i)
                h = Finch.sparse_hash_vacancy(ctrl, x, B)
                ctrl[h], tbl[h], key[q] = Finch.sparse_hash_hash_ctrl(x), q, (p, i, Finch.SPARSE_HASH_KEY_RETAINED)
            end
            ptr, perm = Int[], Int[]
            Finch.sparse_hash_freeze!(ptr, perm, key, maximum(first, es; init=0))
            Finch.sparse_hash_count_buckets!(counts, perm, key, B)
            (; ptr, tbl_ctrl=ctrl, tbl, key, perm, tbl_count=counts)
        end
        return SparseHash{Int}(
            Element(0, channels([zeros(Int, length(s.key)) for s in shards])), 1000, B,
            (channels([getfield(s, f) for s in shards]) for f in Finch.SPARSE_HASH_BUFFERS)...,
        )
    end
    entries_of(lvl) = [(lvl.key[q][1], lvl.key[q][2], q) for q in lvl.perm]

    function initialize!(plan)
        for (buffer, start, value) in plan.init
            fill!(view(buffer, start:length(buffer)), value)
        end
    end

    @testset "shared boundaries, shifts, and child positions" begin
        entries = [
            [(1, 2, 4), (2, 5, 1)], Tuple{Int,Int,Int}[],
            [(1, 5, 3)], [(1, 5, 2), (1, 7, 4)], [(1, 7, 1)],
        ]
        src = hash_shards(entries)
        dst = SparseHash{Int}(Element(0), 1000, 8)
        shift = [0, 0, 1, 1, 2]
        plan = Finch.setup_coalesce!(src, 4, dst, 5, shift, true)
        initialize!(plan)
        @test plan.shift == shift
        @test eltype(plan.shared) == Int
        @test plan.shared == [0, 0, 3, 2, 0]
        @test plan.shared_dst == [0, 0, 1, 1, 0]
        @test plan.nnz == 4
        @test plan.child_shift == [0, 4, 4, 7, 11]
        @test plan.max_child_pos == 12
        @test length(dst.ptr) == 5
        @test length(dst.perm) == 4
        @test length(dst.tbl) == length(dst.tbl_ctrl)
        @test all(==(Finch.SPARSE_HASH_CTRL_EMPTY), dst.tbl_ctrl)
        @test sum(plan.bucket_counts) == plan.nnz
        @test dst.tbl_count == plan.bucket_counts
        @test length(dst.key) == plan.max_child_pos
        @test plan.bucket_shift == [Int((Finch.SPARSE_HASH_POS_MULTIPLIER * (s % UInt)) & UInt(7)) for s in shift]
        @test src.key.data[1][src.perm.data[1][1]] == (1, 2, Finch.SPARSE_HASH_KEY_RETAINED)

        # Traversal ranks are concatenated by shard; bucket owners publish the
        # final table slots into those ranks without resizing or sorting.
        src.lvl.val.data[1][4] = 12
        src.lvl.val.data[3][3] = 25
        src.lvl.val.data[4][4] = 27
        src.lvl.val.data[5][1] = 37
        Threads.@threads for tid in 1:5
            Finch.coalesce_shard!(tid, plan, src, dst, ())
        end
        @test entries_of(dst) == [(1, 2, 4), (2, 5, 1), (2, 7, 11), (3, 7, 12)]
        @test dst.ptr == [1, 2, 4, 5, 5]
        @test dst.lvl.val[[4, 1, 11, 12]] == [12, 25, 27, 37]
        for (p, i, q) in [(1, 2, 4), (2, 5, 1), (2, 7, 11), (3, 7, 12)]
            @test Finch.sparse_hash_lookup(dst.tbl_ctrl, dst.tbl, dst.key, p, i, 8) == q
        end
        for tid in (3, 4)
            p, i = src.key.data[tid][first(src.perm.data[tid])]
            @test Finch.sparse_hash_lookup(
                dst.tbl_ctrl, dst.tbl, dst.key, p + shift[tid], i, 8
            ) == plan.shared_dst[tid]
        end

        # Repeated setup schedules control-byte initialization.
        plan = Finch.setup_coalesce!(src, 4, dst, 5, shift, true)
        @test (dst.tbl_ctrl, 1, Finch.SPARSE_HASH_CTRL_EMPTY) in plan.init
        initialize!(plan)
        @test all(==(Finch.SPARSE_HASH_CTRL_EMPTY), dst.tbl_ctrl)
    end

    @testset "nested hash with an interior shared parent" begin
        outer = hash_shards([[(1, 2, 3), (1, 5, 1)], [(1, 5, 2), (1, 7, 1)]]; B=8)
        inner = hash_shards([[(3, 1, 2), (1, 2, 1)], [(2, 3, 2), (1, 4, 1)]]; B=8)
        inner.lvl.val.data[1] .= [25, 12]
        inner.lvl.val.data[2] .= [47, 35]
        src = SparseHash{Int}(
            inner, outer.shape, outer.subtables,
            (getfield(outer, f) for f in Finch.SPARSE_HASH_BUFFERS)...,
        )
        dst = SparseHash(SparseHash(Element(0), 1000, 8), 1000, 8)
        Finch.coalesce_shards!(src, dst, 2, 1, nothing)
        result = Tensor(dst)
        @test [result[1, 2], result[2, 5], result[3, 5], result[4, 7]] == [12, 25, 35, 47]
        @test result[1, 5] == result[3, 2] == 0
        @test issorted(dst.lvl.key[dst.lvl.perm])
        for lvl in (dst, dst.lvl)
            width = length(lvl.tbl) ÷ lvl.subtables
            @test lvl.tbl_count == [count(!=(Finch.SPARSE_HASH_CTRL_EMPTY),
                view(lvl.tbl_ctrl, ((b - 1) * width + 1):(b * width))) for b in 1:8]
        end
    end

    @testset "moved blocks count buckets from checkpoints" begin
        # Shard 2's first outer entry, (1, 5), is shard 1's last. Its inner block,
        # under shard 2's child 3, spans several checkpoints and sits between
        # other parents' entries.
        outer = hash_shards([[(1, 2, 2), (1, 5, 1)], [(1, 5, 3), (1, 7, 1), (1, 9, 2)]]; B=8)
        inner_keys = [
            [[(1, i) for i in 1:3]; [(2, i) for i in 1:5]],
            [[(1, i) for i in 1:4]; [(3, i) for i in 4:30]; [(2, i) for i in 2:3]],
        ]
        inner = hash_shards([[(p, i, q) for (q, (p, i)) in enumerate(reverse(ks))]
                             for ks in inner_keys]; B=8)
        foreach(v -> fill!(v, 1), inner.lvl.val.data)
        src = SparseHash{Int}(
            inner, outer.shape, outer.subtables,
            (getfield(outer, f) for f in Finch.SPARSE_HASH_BUFFERS)...,
        )
        dst = SparseHash(SparseHash(Element(0), 1000, 8), 1000, 8)
        Finch.coalesce_shards!(src, dst, 2, 1, nothing)
        expected = zeros(Int, 1000, 1000)
        expected[1:30, 5] .= 1
        expected[1:5, 2] .= 1
        expected[1:4, 7] .= 1
        expected[2:3, 9] .= 1
        @test Array(Tensor(dst)) == expected
        for lvl in (dst, dst.lvl)
            width = length(lvl.tbl) ÷ lvl.subtables
            @test lvl.tbl_count == [count(!=(Finch.SPARSE_HASH_CTRL_EMPTY),
                view(lvl.tbl_ctrl, ((b - 1) * width + 1):(b * width))) for b in 1:8]
        end
    end

    @testset "skewed buckets" begin
        indices = filter(1:1000) do i
            Finch.sparse_hash_hash_subtable(Finch.sparse_hash_hash(1, i), 8) == 1
        end[1:20]
        src = hash_shards([[(1, i, q) for (q, i) in enumerate(indices)]]; B=8)
        dst = SparseHash(Element(0), 1000, 8)
        plan = Finch.setup_coalesce!(src, 1, dst, 1, [0], false)
        initialize!(plan)
        @test plan.bucket_counts == [20, 0, 0, 0, 0, 0, 0, 0]
        @test length(dst.tbl) == 8 * 64
        for (q, i) in enumerate(indices)
            x = Finch.sparse_hash_hash(1, i)
            h = Finch.sparse_hash_vacancy(dst.tbl_ctrl, x, 8)
            dst.tbl_ctrl[h], dst.tbl[h], dst.key[q] = Finch.sparse_hash_hash_ctrl(x), q, (1, i, Finch.SPARSE_HASH_KEY_RETAINED)
        end
        @test all(enumerate(indices)) do (q, i)
            Finch.sparse_hash_lookup(dst.tbl_ctrl, dst.tbl, dst.key, 1, i, 8) == q
        end
    end

    @testset "empty shards" begin
        src = hash_shards([Tuple{Int,Int,Int}[] for _ in 1:3])
        dst = SparseHash(Element(0), 1000, 4)
        plan = Finch.setup_coalesce!(src, 3, dst, 3, zeros(Int, 3), true)
        @test (dst.ptr, 1, 1) in plan.init
        initialize!(plan)
        @test plan.nnz == 0
        @test all(iszero, plan.shared)
        @test all(iszero, plan.shared_dst)
        @test all(iszero, plan.child_shift)
        @test plan.max_child_pos == 0
        @test dst.ptr == ones(Int, 4)
        @test isempty(dst.perm)
        @test length(dst.tbl) == length(dst.tbl_ctrl) == 16
        @test all(==(Finch.SPARSE_HASH_CTRL_EMPTY), dst.tbl_ctrl)
    end

    @testset "rotated counts match entries" begin
        for P in (1, 3, 5)
            B = nextpow(2, P)
            shift = [3t - 7 for t in 1:P]
            entries = [[(8, i, 2q) for (q, i) in enumerate((3, 9, 17, 35))] for _ in 1:P]
            src = hash_shards(entries; B)
            dst = SparseHash(Element(0), 1000, B)
            plan = Finch.setup_coalesce!(src, 3P + 1, dst, P, shift, false)
            expected = zeros(Int, B)
            for t in 1:P, (p, i, _) in entries[t]
                expected[Finch.sparse_hash_hash_subtable(Finch.sparse_hash_hash(p + shift[t], i), B)] += 1
            end
            @test plan.bucket_counts == expected
            @test plan.child_shift == collect(0:8:(8(P - 1)))
            @test plan.nnz == 4P
        end
    end

    struct SetupReadGuard{T} <: AbstractVector{T}
        data::Vector{T}
        reads::Base.RefValue{Int}
    end
    Base.size(v::SetupReadGuard) = size(v.data)
    function Base.getindex(v::SetupReadGuard, i::Int)
        v.reads[] += 1
        v.reads[] <= 2 || error("Hash setup scanned entries beyond the boundaries")
        v.data[i]
    end

    @testset "setup only reads boundary entries" begin
        entries = [[(t, i, 2i) for i in 1:1000] for t in 1:3]
        src = hash_shards(entries)
        keys = [SetupReadGuard(v, Ref(0)) for v in src.key.data]
        perms = [SetupReadGuard(v, Ref(0)) for v in src.perm.data]
        guarded = SparseHash{Int}(
            src.lvl, src.shape, src.subtables, src.ptr, src.tbl_ctrl, src.tbl,
            Finch.MultiChannelBuffer(src.key.device, keys),
            Finch.MultiChannelBuffer(src.perm.device, perms), src.tbl_count,
        )
        dst = SparseHash(Element(0), 1000, 4)
        plan = Finch.setup_coalesce!(guarded, 3, dst, 3, [0, 0, 0], false)
        @test all(v -> v.reads[] <= 2, keys)
        @test all(v -> v.reads[] <= 2, perms)
        @test plan.nnz == 3000
        @test plan.max_child_pos == 6000
        @test sum(plan.bucket_counts) == 3000
        @test_throws ArgumentError Finch.setup_coalesce!(src, 3, SparseHash(Element(0)), 3, [0, 0, 0], false)
    end
end

@testitem "coalesce_hash_sampling" begin
    using Random

    device = cpu(:sample, 2)
    mem = Finch.MultiChannelMemory(device, 2)
    channels(data) = Finch.MultiChannelBuffer(mem, data)
    # Frozen hash shards with one bucket, holding entries (p, i, q) and unused
    # child positions.
    function hash_shards(child, entries; parents=2, extents=[0, 0])
        shards = map(entries, extents) do es, extent
            ctrl = fill(Finch.SPARSE_HASH_CTRL_EMPTY, 16)
            tbl = zeros(Int, 16)
            key = fill((0, 0, Finch.SPARSE_HASH_KEY_FREE), max(maximum(e -> e[3], es; init=0), extent))
            for (p, i, q) in es
                x = Finch.sparse_hash_hash(p, i)
                h = Finch.sparse_hash_vacancy(ctrl, x, 1)
                ctrl[h], tbl[h], key[q] = Finch.sparse_hash_hash_ctrl(x), q, (p, i, Finch.SPARSE_HASH_KEY_RETAINED)
            end
            ptr, perm = Int[], Int[]
            Finch.sparse_hash_freeze!(ptr, perm, key, parents)
            (; ptr, tbl_ctrl=ctrl, tbl, key, perm, tbl_count=[length(es)])
        end
        return SparseHash{Int}(
            child, 10, 1,
            (channels([getfield(s, f) for s in shards]) for f in Finch.SPARSE_HASH_BUFFERS)...,
        )
    end
    entries = [[(1, 5, 3), (2, 7, 1), (2, 9, 2)], [(1, 2, 2), (1, 8, 3), (2, 6, 1)]]
    @testset "child positions differ from traversal order" begin
        for dense_child in (false, true)
            leaf = Element(0, channels([collect(1:(dense_child ? 6 : 3)) for _ in 1:2]))
            child = dense_child ? Dense(leaf, 2) : leaf
            src = hash_shards(child, entries)
            for tid in 1:2, seed in 1:30
                Random.seed!(seed)
                inner, q = Finch.sample(tid, child)
                p, i, _ = only(filter(e -> e[3] == q, entries[tid]))
                Random.seed!(seed)
                @test Finch.sample(tid, src) == ((inner..., i), p)
                Random.seed!(seed)
                @test Finch.sample(tid, Dense(src, 2)) == ((inner..., i, p), 1)
            end
            virtual = Finch.virtualize(Finch.FinchCompiler(), :sample_hash, typeof(src))
            @test Finch.sample_dims(virtual) == (dense_child ? 2 : 1)
            @test !Finch.all_dense(virtual)
        end
    end

    @testset "nested hashes" begin
        leaf = Element(0, channels([collect(1:3) for _ in 1:2]))
        inner_entries = [[(1, 4, 2), (2, 3, 3), (3, 8, 1)] for _ in 1:2]
        inner = hash_shards(leaf, inner_entries; parents=3)
        src = hash_shards(inner, entries)
        for tid in 1:2, seed in 1:30
            Random.seed!(seed)
            _, q = Finch.sample(tid, leaf)
            child_pos, i, _ = only(filter(e -> e[3] == q, inner_entries[tid]))
            parent, j, _ = only(filter(e -> e[3] == child_pos, entries[tid]))
            Random.seed!(seed)
            @test Finch.sample(tid, src) == ((i, j), parent)
        end
    end

    @testset "unused child positions and empty shards" begin
        leaf = Element(0, channels([collect(1:4), collect(1:4)]))
        live = [(1, 5, 3), (2, 7, 1), (2, 9, 4)]
        src = hash_shards(leaf, [live, Tuple{Int,Int,Int}[]]; extents=[4, 4])
        Random.seed!(1)
        samples = [Finch.sample(1, src) for _ in 1:100]
        @test Set(samples) == Set(((i,), p) for (p, i, _) in live)
        @test_throws ArgumentError Finch.sample(2, src)
    end

    @testset "coalesce sampler integration" begin
        leaf = Element(0, channels([collect(1:3) for _ in 1:2]))
        src = Dense(hash_shards(leaf, entries), 2)
        dst = Dense(SparseHash(Element(0), 10), 2)
        lvl = Coalesce(device, src, dst, Finch.FinchStaticSchedule{:dynamic}(), nothing; mode=:fast)
        nnz, _ = Finch.get_total_nnz(lvl, true)
        Random.seed!(1)
        samples = Finch.build_sampler(lvl, 2, nnz, 2)
        @test length(samples) == 400
        @test issorted(samples; by=reverse)
        @test Set(samples) == Set((i, p) for shard in entries for (p, i, _) in shard)
    end
end

@testitem "coalesce_reuse" begin
    # Reusing an output must discard old entries; unused backing values may remain.
    @testset "$(summary(fmt())) on $P workers" for fmt in (
        () -> Dense(SparseByteMap(Element(0))),
        () -> SparseByteMap(SparseByteMap(Element(0))),
    ), P in (1, 2, 3)
        device = cpu(:k, P)
        output = Tensor(Coalesce(device, fmt()))
        first = [1 0 2; 0 3 0; 4 0 0]
        second = [0 5 0; 6 0 0; 0 0 7]
        for x in (first, second, second, zero(first), first)
            input = Tensor(Dense(SparseList(Element(0))), x)
            @finch begin
                output .= 0
                for j in parallel(_, device), i in _
                    output[i, j] += input[i, j]
                end
            end
            @test Array(output) == x
        end
    end
end
