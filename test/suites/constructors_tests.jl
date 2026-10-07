@testitem "sparse_hash" begin
    function check_counts(lvl)
        B = lvl.subtables
        len = length(lvl.tbl_ctrl) ÷ B
        @test lvl.tbl_count[1:B] == [count(!=(Finch.SPARSE_HASH_CTRL_EMPTY),
            view(lvl.tbl_ctrl, ((b - 1) * len + 1):(b * len))) for b in 1:B]
        @test sum(lvl.tbl_count[1:B]) == length(lvl.perm)
        # Block j counts each bucket's entries of rank at least (j - 1) * B + 1.
        m = cld(length(lvl.perm), B) + 1
        @test length(lvl.tbl_count) == m * B
        for j in 1:m
            expected = zeros(Int, B)
            for q in lvl.perm[((j - 1) * B + 1):end]
                p, i = lvl.key[q]
                expected[Finch.sparse_hash_hash_subtable(Finch.sparse_hash_hash(p, i), B)] += 1
            end
            @test lvl.tbl_count[((j - 1) * B + 1):(j * B)] == expected
        end
        @test length(lvl.key) == maximum(lvl.perm; init=0)
        # Every entry's key looks up its own child position.
        for q in lvl.perm
            p, i = lvl.key[q]
            @test Finch.sparse_hash_lookup(lvl.tbl_ctrl, lvl.tbl, lvl.key, p, i, B) == q
        end
    end
    @testset "bucket rotation and slot hashing" begin
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
        # Buckets use the low bits of the linear hash; slots mix it first.
        x = UInt(0b101)
        @test Finch.sparse_hash_hash_subtable(x, 8) == 6
        @test Finch.sparse_hash_hash_slot_parts(x, 512, 8) == (321, Int((hash(x) >>> 7) & 63), 63)
        @test Finch.sparse_hash_hash_slot_parts(typemax(UInt), 8, 8) == (8, 0, 0)
        @test Finch.sparse_hash_hash_ctrl(x) == 0x80 | (hash(x) % UInt8 & 0x7f)
    end

    @testset "collisions, wraparound, and resizing" for subtables in (1, 4)
        cap = 64 * subtables
        ctrl = fill(Finch.SPARSE_HASH_CTRL_EMPTY, cap)
        tbl = zeros(Int, cap)
        # Choose colliding keys at the last slot of the first subtable, so
        # insertion must wrap without crossing into the next subtable.
        parents = filter(1:20000) do p
            hsh = Finch.sparse_hash_hash(p, 7)
            Finch.sparse_hash_hash_slot_parts(hsh, cap, subtables) == (1, 63, 63)
        end
        key = [(parents[q], 7, Finch.SPARSE_HASH_KEY_RETAINED) for q in 1:12]
        find(p, i) = Finch.sparse_hash_find(ctrl, tbl, key, p, i, Finch.sparse_hash_hash(p, i), subtables)
        for (q, (p, i)) in enumerate(key)
            h = find(p, i)
            ctrl[h] = Finch.sparse_hash_hash_ctrl(Finch.sparse_hash_hash(p, i))
            tbl[h] = q
        end
        @test ctrl[1] != Finch.SPARSE_HASH_CTRL_EMPTY
        for newcap in (cap, 2cap, 4cap)
            if newcap != cap
                Finch.sparse_hash_resize!(
                    ctrl, tbl, key, newcap, subtables, length(key)
                )
            end
            for (q, (p, i)) in enumerate(key)
                @test Finch.sparse_hash_lookup(ctrl, tbl, key, p, i, subtables) == q
            end
            @test Finch.sparse_hash_lookup(ctrl, tbl, key, parents[13], 7, subtables) == 0
        end
        # A present key probes to its own slot rather than a vacancy.
        @test tbl[find(parents[1], 7)] == 1
        @test count(!=(Finch.SPARSE_HASH_CTRL_EMPTY), ctrl) == length(key)
    end

    @testset "dense rehash ignores unused key capacity" for Tp in (Int, Int32), B in (1, 8)
        key = Vector{Tuple{Tp,Int,UInt8}}(undef, 32)
        key[1:12] .= [(Tp(mod(q, 3) + 1), 7q, Finch.SPARSE_HASH_KEY_RETAINED) for q in 1:12]
        ctrl, tbl = UInt8[], Tp[]
        for cap in (64B, 128B)
            Finch.sparse_hash_resize!(ctrl, tbl, key, cap, B, 12)
            @test count(!iszero, ctrl) == 12
            for q in 1:12
                p, i = key[q]
                @test Finch.sparse_hash_lookup(ctrl, tbl, key, p, i, B) == q
                h = Finch.sparse_hash_find(ctrl, tbl, key, p, i, Finch.sparse_hash_hash(p, i), B)
                @test ctrl[h] == Finch.sparse_hash_hash_ctrl(Finch.sparse_hash_hash(p, i))
            end
        end
    end

    @testset "sparse clearing crosses cleared slots and wraps" for Tp in (Int8, Int32, Int), B in (1, 8)
        cap = 4096B
        indices = collect(Iterators.take(Iterators.filter(Iterators.countfrom(1)) do i
            Finch.sparse_hash_hash_slot_parts(Finch.sparse_hash_hash(1, i), cap, B) ==
                (1, 4095, 4095)
        end, 4))
        # Interior child holes and a sorted permutation that differs from
        # insertion order. Clearing its first entry breaks the probe chain.
        perm = Tp[1, 4, 3, 6]
        key = fill((Tp(0), 0, Finch.SPARSE_HASH_KEY_FREE), 6)
        for (q, i) in zip(perm, indices)
            key[q] = (Tp(1), i, Finch.SPARSE_HASH_KEY_RETAINED)
        end
        ctrl, tbl = UInt8[], Tp[]
        Finch.sparse_hash_resize!(ctrl, tbl, key, cap, B)
        @test ctrl[1] != Finch.SPARSE_HASH_CTRL_EMPTY
        ctrl_ptr, tbl_ptr = pointer(ctrl), pointer(tbl)
        Finch.sparse_hash_clear!(ctrl, tbl, key, perm, B)
        @test length(ctrl) == length(tbl) == cap
        @test pointer(ctrl) == ctrl_ptr
        @test pointer(tbl) == tbl_ptr
        @test all(iszero, ctrl)
        # Empty clearing also keeps the allocation, without visiting old keys.
        Finch.sparse_hash_clear!(ctrl, tbl, empty(key), empty(perm), B)
        @test all(iszero, ctrl)
    end

    @testset "declaration reuses capacity across dense, sparse, and empty outputs" for Tp in (Int32, Int), B in (1, 8)
        tensor = Tensor(SparseHash(Element{0,Int,Tp}(), 8192, B))
        input = Tensor(Dense(Element(0)), zeros(Int, 8192))
        capacity = 0
        for indices in (1:2:8192, 1:2:8192, [5, 13, 47, 91], [2, 4], Int[], Int[], 2:2:8192)
            fill!(input.lvl.lvl.val, 0)
            input.lvl.lvl.val[indices] .= 3
            @finch begin
                tensor .= 0
                for i in _
                    if input[i] != 0
                        tensor[i] += input[i]
                    end
                end
            end
            @test Array(tensor) == Array(input)
            @test length(tensor.lvl.tbl_ctrl) >= capacity
            @test length(tensor.lvl.tbl) == length(tensor.lvl.tbl_ctrl)
            capacity = length(tensor.lvl.tbl_ctrl)
            check_counts(tensor.lvl)
        end
    end

    @testset "thawed growth recovers child holes" begin
        # Coalesce can leave free records without linking them together.
        ctrl, tbl = zeros(UInt8, 8), zeros(Int, 8)
        key = fill((0, 0, Finch.SPARSE_HASH_KEY_FREE), 6)
        values = zeros(Int, 6)
        data = zeros(Int, 64)
        for (i, q) in ((1, 1), (4, 3), (6, 6))
            key[q] = (1, i, Finch.SPARSE_HASH_KEY_RETAINED)
            x = Finch.sparse_hash_hash(1, i)
            h = Finch.sparse_hash_vacancy(ctrl, x, 1)
            ctrl[h], tbl[h] = Finch.sparse_hash_hash_ctrl(x), q
            values[q] = data[i] = 10i
        end
        tensor = Tensor(SparseHash{Int}(
            Element(0, values), 64, 1, [1, 4], ctrl, tbl, key, [1, 3, 6], [3]
        ))
        input = Tensor(Dense(Element(0)), ones(Int, 64))
        @finch for i in _
            tensor[i] += input[i]
        end
        @test Array(tensor) == data .+ 1
        @test length(tensor.lvl.perm) == 64
        @test length(tensor.lvl.key) == 64
        for (i, q) in ((1, 1), (4, 3), (6, 6))
            @test Finch.sparse_hash_lookup(ctrl, tbl, key, 1, i, 1) == q
        end
        check_counts(tensor.lvl)
    end

    @testset "tensor assembly and updates" for Ti in (Int, Int32), B in (1, 8)
        data = [mod(i + 3j, 5) == 0 ? i + j : 0 for i in 1:17, j in 1:9]
        input = Tensor(Dense(Dense(Element(0))), data)
        tensor = Tensor(Dense(SparseHash{Ti}(Element(0), 17, B)), data)
        @test Array(tensor) == data
        check_counts(tensor.lvl.lvl)
        counts = tensor.lvl.lvl.tbl_count
        @finch begin
            for j in _, i in _
                tensor[i, j] += input[i, j]
            end
        end
        @test Array(tensor) == 2data
        @test tensor.lvl.lvl.tbl_count === counts
        check_counts(tensor.lvl.lvl)
    end

    @testset "skewed assembly and thawed growth" for Ti in (Int, Int32)
        B = 8
        indices = filter(i -> Finch.sparse_hash_hash_subtable(Finch.sparse_hash_hash(1, i), B) == 1, 1:1000)[1:40]
        data = zeros(Int, 1000)
        data[indices[1:10]] .= 1
        tensor = Tensor(SparseHash{Ti}(Element(0), 1000, B), data)
        @test Array(tensor) == data
        input = Tensor(Dense(Element(0)), zeros(Int, 1000))
        input.lvl.lvl.val[indices] .= 2
        @finch for i in _
            if input[i] != 0
                tensor[i] += input[i]
            end
        end
        @test Array(tensor) == data + Array(input)
        @test length(tensor.lvl.tbl) == B * 128
        @test length(tensor.lvl.perm) == 40
        check_counts(tensor.lvl)
        @test Finch.sparse_hash_lookup(tensor.lvl.tbl_ctrl, tensor.lvl.tbl, tensor.lvl.key, 1, 1001, B) == 0
    end

    @testset "pending writers share keys across bucket growth" begin
        indices = filter(i -> Finch.sparse_hash_hash_subtable(Finch.sparse_hash_hash(1, i), 8) == 1, 1:1000)[1:40]
        coords = Tensor(Dense(Element(0)), indices)
        tensor = Tensor(SparseHash{Int}(Dense(Element(0), 2), 1000, 8))
        @finch begin
            tensor .= 0
            for k in _
                let i = coords[k]
                    for j in 1:2
                        tensor[j, i] += 1
                        tensor[j, i] += 2
                    end
                end
            end
        end
        expected = zeros(Int, 2, 1000)
        expected[:, indices] .= 3
        @test Array(tensor) == expected
        @test length(tensor.lvl.perm) == 40
        @test all(k -> k[3] == Finch.SPARSE_HASH_KEY_RETAINED, tensor.lvl.key)
        check_counts(tensor.lvl)
    end

    @testset "pending keys invalidate cached slots" begin
        indices = filter(1:10000) do i
            hsh = Finch.sparse_hash_hash(1, i)
            Finch.sparse_hash_hash_slot_parts(hsh, 32, 8) == (1, 3, 3)
        end[1:40]
        left = Tensor(Dense(Element(0)), indices[1:2:end])
        right = Tensor(Dense(Element(0)), indices[2:2:end])
        tensor = Tensor(SparseHash{Int}(Dense(Element(0), 2), 10000, 8))
        @finch begin
            tensor .= 0
            for k in _
                let i = left[k], ii = right[k]
                    for j in 1:2
                        tensor[j, i] += 1
                        tensor[j, ii] += 2
                    end
                end
            end
        end
        expected = zeros(Int, 2, 10000)
        expected[:, indices[1:2:end]] .= 1
        expected[:, indices[2:2:end]] .= 2
        @test Array(tensor) == expected
        @test length(tensor.lvl.perm) == 40
        check_counts(tensor.lvl)
    end
end

@testitem "sparse_hash_pending" begin
    using Random

    function table(Tp, B; capacity=4B)
        (; ctrl=zeros(UInt8, capacity), tbl=zeros(Tp, capacity), key=Tuple{Tp,Int,UInt8}[],
            counts=zeros(Int, B), free_head=Ref(zero(Tp)), B)
    end
    function acquire!(t, k)
        Tp = eltype(t.tbl)
        x = Finch.sparse_hash_hash(k...)
        b = Finch.sparse_hash_hash_subtable(x, t.B)
        if 2t.B * (t.counts[b] + 1) > length(t.ctrl)
            Finch.sparse_hash_resize!(t.ctrl, t.tbl, t.key, 2length(t.ctrl), t.B)
        end
        h = Finch.sparse_hash_find(t.ctrl, t.tbl, t.key, k..., x, t.B)
        pending = true
        if t.ctrl[h] == 0
            if t.free_head[] == 0
                push!(t.key, (k..., 0x01))
                q = Tp(length(t.key))
            else
                q = t.free_head[]
                t.free_head[] = t.key[q][1]
                t.key[q] = (k..., 0x01)
            end
            t.ctrl[h], t.tbl[h] = Finch.sparse_hash_hash_ctrl(x), q
            t.counts[b] += 1
        else
            q = t.tbl[h]
            pending = t.key[q][3] != Finch.SPARSE_HASH_KEY_RETAINED
            if pending
                Finch.sparse_hash_share!(t.key, q)
            end
        end
        return (; q, h, x, pending)
    end
    function release!(t, handle, dirty)
        if handle.pending && Finch.sparse_hash_release!(
            t.ctrl, t.tbl, t.key, handle.q, handle.x, t.B, dirty, t.free_head[]
        )
            t.counts[Finch.sparse_hash_hash_subtable(handle.x, t.B)] -= 1
            t.free_head[] = handle.q
        end
    end
    slot(t, k) = Finch.sparse_hash_find(t.ctrl, t.tbl, t.key, k...,
        Finch.sparse_hash_hash(k...), t.B)
    state(t, k) = t.key[t.tbl[slot(t, k)]][3]

    function free_positions(t)
        positions = eltype(t.tbl)[]
        q = t.free_head[]
        while q != 0
            @test 1 <= q <= length(t.key)
            @test t.key[q][3] == Finch.SPARSE_HASH_KEY_FREE
            q in positions && error("Cycle in free child positions")
            push!(positions, q)
            q = t.key[q][1]
        end
        return positions
    end

    @testset "127-writer limit, promotion, and growth" for Tp in (Int, Int32),
        dirty in (false, true)
        t = table(Tp, 4)
        handles = [acquire!(t, (1, 1)) for _ in 1:127]
        q = first(handles).q
        @test state(t, (1, 1)) == 127
        @test_throws "SparseHash supports at most 127 pending writers per entry" acquire!(
            t, (1, 1)
        )
        @test state(t, (1, 1)) == 127
        @test all(h.q == q for h in handles)
        # The limit is per entry. Growth preserves both maximum counts.
        other = [acquire!(t, (1, 2)) for _ in 1:127]
        for i in 3:200
            release!(t, acquire!(t, (1, i)), true)
        end
        @test state(t, (1, 1)) == 127
        @test state(t, (1, 2)) == 127
        @test_throws "SparseHash supports at most 127 pending writers per entry" acquire!(
            t, (1, 1)
        )
        if dirty
            release!(t, first(handles), true)
            @test state(t, (1, 1)) >= Finch.SPARSE_HASH_CTRL_FULL
            # Retained entries no longer need counts or a writer limit.
            retained = [acquire!(t, (1, 1)) for _ in 1:128]
            @test all(!h.pending && h.q == q for h in retained)
            foreach(h -> release!(t, h, false), retained)
            foreach(h -> release!(t, h, false), handles[2:end])
            @test Finch.sparse_hash_lookup(t.ctrl, t.tbl, t.key, 1, 1, t.B) == q
        else
            release!(t, first(handles), false)
            @test state(t, (1, 1)) == 126
            # A failed acquisition did not change the count; releasing a
            # writer permits another acquisition up to the limit again.
            replacement = acquire!(t, (1, 1))
            @test state(t, (1, 1)) == 127
            release!(t, replacement, false)
            foreach(h -> release!(t, h, false), handles[2:end])
            @test Finch.sparse_hash_lookup(t.ctrl, t.tbl, t.key, 1, 1, t.B) == 0
            @test q in free_positions(t)
        end
        @test state(t, (1, 2)) == 127
        foreach(h -> release!(t, h, false), other)
        @test Finch.sparse_hash_lookup(t.ctrl, t.tbl, t.key, 1, 2, t.B) == 0
        @test sum(t.counts) == (dirty ? 199 : 198)
        h = acquire!(t, (2, 201))
        @test h.q == first(other).q
        @test state(t, (2, 201)) == 0x01
        release!(t, h, true)
    end

    @testset "freeze trims free tails and preserves reusable holes" begin
        t = table(Int, 4)
        handles = [acquire!(t, (1, i)) for i in 1:4]
        for (i, h) in enumerate(handles)
            release!(t, h, isodd(i))
        end
        ptr, perm = Int[], Int[]
        extent = Finch.sparse_hash_freeze!(ptr, perm, t.key, 1)
        t.free_head[] = Finch.sparse_hash_free_head!(t.key, perm)
        @test extent == length(t.key) == 3
        @test free_positions(t) == [2]
        @test perm == [1, 3]
        h = acquire!(t, (1, 5))
        @test h.q == 2
        release!(t, h, true)

        t = table(Int, 4)
        handles = [acquire!(t, (1, i)) for i in 1:4]
        foreach(h -> release!(t, h, false), handles)
        extent = Finch.sparse_hash_freeze!(ptr, perm, t.key, 1)
        t.free_head[] = Finch.sparse_hash_free_head!(t.key, perm)
        @test extent == 0
        @test isempty(t.key) && t.free_head[] == 0 && isempty(perm)
        h = acquire!(t, (1, 5))
        @test h.q == 1
        release!(t, h, true)
    end

    @testset "deletion crosses home slots and wraps within a bucket" for B in (1, 4),
        home in (0, 63)
        t = table(Int, B; capacity=64B)
        function candidates(off)
            filter(1:50000) do i
                Finch.sparse_hash_hash_slot_parts(Finch.sparse_hash_hash(1, i), 64B, B) ==
                (1, off, 63)
            end
        end
        a, c = candidates(home)[1:2]
        b = first(candidates((home + 1) & 63))
        ha = acquire!(t, (1, a))
        hb = acquire!(t, (1, b)) # This home entry must not stop the repair scan.
        hc = acquire!(t, (1, c))
        shared = acquire!(t, (1, c))
        release!(t, ha, false)
        @test slot(t, (1, c)) != hc.h
        @test state(t, (1, c)) == 2
        release!(t, hc, false) # The writer count stays at the same child position.
        release!(t, hb, true)
        release!(t, shared, true)
        @test Finch.sparse_hash_lookup(t.ctrl, t.tbl, t.key, 1, a, B) == 0
        @test Finch.sparse_hash_lookup(t.ctrl, t.tbl, t.key, 1, b, B) == hb.q
        @test Finch.sparse_hash_lookup(t.ctrl, t.tbl, t.key, 1, c, B) == hc.q
        @test sum(t.counts) == 2
    end

    @testset "random overlapping lifetimes against a reference" for Tp in (Int, Int32),
        B in (1, 8)
        t = table(Tp, B)
        expected = Dict{Tuple{Int,Int},Tuple{Int,Bool}}()
        active = []
        rng = Xoshiro(17)
        function check()
            @test count(!iszero, t.ctrl) == length(expected) == sum(t.counts)
            @test length(expected) + length(free_positions(t)) == length(t.key)
            for (k, (n, retained)) in expected
                h = slot(t, k)
                entry = t.key[t.tbl[h]]
                @test t.ctrl[h] != 0 && (entry[1], entry[2]) == k
                @test t.ctrl[h] == Finch.sparse_hash_hash_ctrl(Finch.sparse_hash_hash(k...))
                @test entry[3] == (retained ? Finch.SPARSE_HASH_KEY_RETAINED : n)
            end
            width = length(t.ctrl) ÷ B
            @test t.counts == [
                count(!iszero, view(t.ctrl, ((b - 1) * width + 1):(b * width))) for b in 1:B
            ]
        end
        function finish(index, dirty)
            k, handle = active[index]
            deleteat!(active, index)
            release!(t, handle, dirty)
            n, retained = expected[k]
            if n == 1 && !retained && !dirty
                delete!(expected, k)
            else
                expected[k] = (n - 1, retained || dirty)
            end
        end
        for j in 1:5000
            if isempty(active) || (length(active) < 100 && rand(rng) < 0.55)
                k = (rand(rng, 1:4), rand(rng, 1:64))
                push!(active, (k, acquire!(t, k)))
                n, retained = get(expected, k, (0, false))
                expected[k] = (n + 1, retained)
            else
                finish(rand(rng, eachindex(active)), rand(rng) < 0.25)
            end
            j % 100 == 0 && check()
        end
        while !isempty(active)
            finish(lastindex(active), false)
        end
        check()
        @test all(c == 0 || c >= Finch.SPARSE_HASH_CTRL_FULL for c in t.ctrl)
    end

    @testset "generated writers discard, retain, and reuse children across freeze" begin
        indices = reshape(repeat(1:40; inner=2), 2, :)
        coords = Tensor(Dense(Dense(Element(0))), indices)
        weights = Tensor(
            Dense(Dense(Element(0))), [mod(j + k, 5) == 0 ? k : 0 for j in 1:2, k in 1:40]
        )
        tensor = Tensor(SparseHash{Int}(Dense(Element(0), 2), 80, 8))
        @finch begin
            tensor .= 0
            for k in _
                let i = coords[1, k], ii = coords[2, k]
                    for j in 1:2
                        if weights[j, k] != 0
                            tensor[j, i] += weights[j, k]
                        end
                        if weights[j, k] == 0
                            tensor[j, ii] += 0
                        end
                    end
                end
            end
        end
        expected = zeros(Int, 2, 80)
        expected[:, 1:40] .= Array(weights)
        @test Array(tensor) == expected
        @test all(c == 0 || c >= Finch.SPARSE_HASH_CTRL_FULL for c in tensor.lvl.tbl_ctrl)
        input = Tensor(Dense(Dense(Element(0))), ones(Int, 2, 80))
        @finch for i in _, j in _
            tensor[j, i] += input[j, i]
        end
        @test Array(tensor) == expected .+ 1
        @test all(k -> k[3] == Finch.SPARSE_HASH_KEY_RETAINED, tensor.lvl.key)
        @test length(tensor.lvl.perm) == 80
    end

    @testset "overlapping writers in coalesce task shards" begin
        device = cpu(:k, 3)
        coords = Tensor(Dense(Dense(Element(0))), reshape(repeat(1:40; inner=2), 2, :))
        tensor = Tensor(Coalesce(device, SparseHash{Int}(Dense(Element(0), 2), 40, 4)))
        @finch begin
            tensor .= 0
            for k in parallel(_, device)
                let i = coords[1, k], ii = coords[2, k]
                    for j in 1:2
                        tensor[j, i] += 1
                        tensor[j, ii] += 2
                    end
                end
            end
        end
        @test Array(tensor) == fill(3, 2, 40)
    end
end

@testitem "sparse_bytemap_redeclare" begin
    @testset "declaring clears element storage" begin
        tensor = Tensor(SparseByteMap(SparseByteMap(Element(0))), [1 0 2; 0 3 0; 4 0 0])
        @finch tensor .= 0
        @test iszero(Array(tensor))
        @test iszero(tensor.lvl.lvl.lvl.val)
    end

    # Reusing a byte map must not expose entries from earlier writes.
    @testset "$(summary(fmt())), $n columns" for fmt in (
        () -> SparseByteMap(SparseByteMap(Element(0))),
        () -> SparseByteMap(Dense(Element(0))),
        () -> SparseByteMap(SparseList(Element(0))),
    ), n in (3, 64)
        # Exercise both bulk clearing and clearing just the dirty positions.
        first = hcat([1 0 2; 0 3 0; 4 0 0], zeros(Int, 3, n - 3))
        second = hcat([0 5 0; 6 0 0; 0 0 7], zeros(Int, 3, n - 3))
        tensor = Tensor(fmt(), zeros(Int, 3, n))
        for x in (first, second, second, zero(first), first)
            input = Tensor(SparseList(SparseList(Element(0))), x)
            @finch begin
                tensor .= 0
                for j in _, i in _
                    tensor[i, j] += input[i, j]
                end
            end
            @test Array(tensor) == x
        end
    end

    @testset "$(summary(fmt()))" for fmt in (
        () -> Dense(SparseByteMap(SparseByteMap(Element(0)))),
        () -> SparseByteMap(Dense(SparseByteMap(Element(0)))),
        () -> SparseByteMap(Dense(SparseList(Element(0)))),
        () -> SparseByteMap(SparseByteMap(SparseList(Element(0)))),
    )
        tensor = Tensor(fmt(), zeros(Int, 2, 2, 2))
        # The second write reuses parents of the first, under different indices.
        for x in (cat([1 0; 0 0], [0 2; 0 0]; dims=3), cat([0 0; 3 0], [0 0; 0 4]; dims=3))
            input = Tensor(SparseList(SparseList(SparseList(Element(0)))), x)
            @finch begin
                tensor .= 0
                for k in _, j in _, i in _
                    tensor[i, j, k] += input[i, j, k]
                end
            end
            @test Array(tensor) == x
        end
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
