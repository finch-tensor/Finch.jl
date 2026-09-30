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
                () -> SparseHash(SparseHash{Int32,false}(Element(0), 7, 16), 3, 16),
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
            @test plan.off == [0, 2]
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


@testitem "coalesce_hash_setup" begin
    function hash_shards(entries; B=nextpow(2, length(entries)))
        P = length(entries)
        mem = Finch.MultiChannelMemory(cpu(:setup, P), P)
        channels(data) = Finch.MultiChannelBuffer(mem, data)
        counts = [zeros(Int, B) for _ in entries]
        for (tid, es) in enumerate(entries), (p, i, _) in es
            b = Finch.sparse_hash_hash_subtable(Finch.sparse_hash_hash(p, i), B)
            counts[tid][b] += 1
        end
        controls = [fill(Finch.SPARSE_HASH_CTRL_EMPTY,
                         B * Finch.sparse_hash_table_capacity(maximum(c))) for c in counts]
        tables = [fill((0, 0, 0), length(ctrl)) for ctrl in controls]
        perms = Vector{Int}[]
        ptrs = Vector{Int}[]
        vals = Vector{Int}[]
        for tid in 1:P
            es = sort(entries[tid]; by=e -> (e[1], e[2]))
            for (p, i, q) in es
                Finch.sparse_hash_table_insert_noresize!(controls[tid], tables[tid], p, i, q, B)
            end
            push!(perms, [Finch.sparse_hash_table_lookup_slot(
                controls[tid], tables[tid], p, i, B
            ) for (p, i, _) in es])
            parents = maximum(e -> e[1], es; init=0)
            push!(ptrs, [1 + count(e -> e[1] < p, es) for p in 1:(parents + 1)])
            push!(vals, zeros(Int, maximum(e -> e[3], es; init=0)))
        end
        return SparseHash{Int,false}(
            Element(0, channels(vals)), 1000, B, channels(ptrs), channels(controls),
            channels(tables), channels([Int[] for _ in 1:P]), channels(perms),
            channels(counts),
            channels([[length(v)] for v in vals]),
        )
    end

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
        dst = SparseHash{Int,false}(Element(0), 1000, 8)
        shift = [0, 0, 1, 1, 2]
        plan = Finch.setup_coalesce!(src, 4, dst, 5, shift, true)
        initialize!(plan)
        @test plan.shift == shift
        @test eltype(plan.shared) == Int
        @test plan.shared == [0, 0, 3, 2, 0]
        @test plan.shared_dst == [0, 0, 1, 1, 0]
        @test plan.off == [0, 2, 2, 2, 3]
        @test plan.prev == [0, 2, 2, 2, 2]
        @test plan.nnz == 4
        @test plan.child_shift == [0, 4, 4, 7, 11]
        @test plan.max_child_pos == 12
        @test length(dst.ptr) == 5
        @test length(dst.perm) == 4
        @test isempty(dst.pool)
        @test length(dst.tbl) == length(dst.tbl_ctrl)
        @test all(==(Finch.SPARSE_HASH_CTRL_EMPTY), dst.tbl_ctrl)
        @test sum(plan.bucket_counts) == plan.nnz
        @test dst.tbl_count == plan.bucket_counts
        @test dst.qos_stop == [plan.max_child_pos]
        @test plan.bucket_shift == [Int((Finch.SPARSE_HASH_POS_MULTIPLIER * (s % UInt)) & UInt(7)) for s in shift]
        @test src.tbl.data[1][src.perm.data[1][1]] == (1, 2, 4)

        # The allocated tables must accept every owned entry without resizing.
        for tid in 1:5, r in eachindex(src.perm.data[tid])
            p, i, q = src.tbl.data[tid][src.perm.data[tid][r]]
            q == plan.shared[tid] && continue
            Finch.sparse_hash_table_insert_noresize!(
                dst.tbl_ctrl, dst.tbl, p + shift[tid], i, q + plan.child_shift[tid], dst.subtables
            )
        end
        for (p, i, q) in [(1, 2, 4), (2, 5, 1), (2, 7, 11), (3, 7, 12)]
            @test Finch.sparse_hash_table_lookup(dst.tbl_ctrl, dst.tbl, p, i, 8) == q
        end
        for tid in (3, 4)
            p, i, _ = src.tbl.data[tid][first(src.perm.data[tid])]
            @test Finch.sparse_hash_table_lookup(
                dst.tbl_ctrl, dst.tbl, p + shift[tid], i, 8
            ) == plan.shared_dst[tid]
        end

        # Repeated setup schedules control-byte initialization and clears the pool.
        push!(dst.pool, 99)
        plan = Finch.setup_coalesce!(src, 4, dst, 5, shift, true)
        @test (dst.tbl_ctrl, 1, Finch.SPARSE_HASH_CTRL_EMPTY) in plan.init
        initialize!(plan)
        @test all(==(Finch.SPARSE_HASH_CTRL_EMPTY), dst.tbl_ctrl)
        @test isempty(dst.pool)
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
            Finch.sparse_hash_table_insert_noresize!(dst.tbl_ctrl, dst.tbl, 1, i, q, 8)
        end
        @test all(enumerate(indices)) do (q, i)
            Finch.sparse_hash_table_lookup(dst.tbl_ctrl, dst.tbl, 1, i, 8) == q
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
        @test isempty(dst.pool)
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
        tables = [SetupReadGuard(v, Ref(0)) for v in src.tbl.data]
        perms = [SetupReadGuard(v, Ref(0)) for v in src.perm.data]
        guarded = SparseHash{Int,false}(
            src.lvl, src.shape, src.subtables, src.ptr, src.tbl_ctrl,
            Finch.MultiChannelBuffer(src.tbl.device, tables), src.pool,
            Finch.MultiChannelBuffer(src.perm.device, perms), src.tbl_count, src.qos_stop,
        )
        dst = SparseHash(Element(0), 1000, 4)
        plan = Finch.setup_coalesce!(guarded, 3, dst, 3, [0, 0, 0], false)
        @test all(v -> v.reads[] == 2, tables)
        @test all(v -> v.reads[] == 2, perms)
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
    function hash_shards(child, entries; parents=2, pools=[Int[], Int[]])
        controls = [fill(Finch.SPARSE_HASH_CTRL_EMPTY, 16) for _ in entries]
        tables = [fill((0, 0, 0), 16) for _ in entries]
        perms = Vector{Int}[]
        ptrs = Vector{Int}[]
        for tid in eachindex(entries)
            for (p, i, q) in entries[tid]
                Finch.sparse_hash_table_insert_noresize!(controls[tid], tables[tid], p, i, q)
            end
            ordered = sort(entries[tid]; by=e -> (e[1], e[2]))
            push!(perms, [Finch.sparse_hash_table_lookup_slot(
                controls[tid], tables[tid], p, i
            ) for (p, i, _) in ordered])
            push!(ptrs, [1 + count(e -> e[1] < p, ordered) for p in 1:(parents + 1)])
        end
        return SparseHash{Int,false}(
            child, 10, 1, channels(ptrs), channels(controls), channels(tables),
            channels(pools), channels(perms),
            channels([[length(es)] for es in entries]),
            channels([[max(maximum(e -> e[3], es; init=0), maximum(pools[t]; init=0))]
                      for (t, es) in enumerate(entries)]),
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
        src = hash_shards(leaf, [live, Tuple{Int,Int,Int}[]]; pools=[[2], [1, 2, 3, 4]])
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
        @test length(samples) == 2000
        @test issorted(samples; by=reverse)
        @test Set(samples) == Set((i, p) for shard in entries for (p, i, _) in shard)
    end
end
