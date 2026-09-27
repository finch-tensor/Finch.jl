@testitem "coalesce_merges" begin
    mem = Finch.MultiChannelMemory(cpu(:t, 2), 2)
    channels(data) = Finch.MultiChannelBuffer(mem, data)

    function merge_fixture(src, dst)
        Finch.setup_coalesce!(
            src, 1, dst, 2, Finch.MergeNormalization(); pos_map=Finch.init_posmap(2)
        )
        offsets = Finch.init_pos_offsets(2)
        shared_flags = Finch.init_shared_flags(2)
        Threads.@threads for tid in 1:2
            Finch.coalesce_fast!(tid, offsets, shared_flags, 2, src, dst, false)
        end
        return offsets
    end

    # Both shards contain column 2, but own disjoint entries within that column.
    function outer_sparse(inner)
        SparseList{Int}(
            inner, 3, channels([[1, 3], [1, 3]]), channels([[1, 2], [2, 3]])
        )
    end

    @testset "shared parent, distinct child boundaries" begin
        for with_bytemap in (false, true)
            leaf = Element(0, channels([[10, 11, 20], [21, 30, 31]]))
            dst_leaf = Element(0)
            if with_bytemap
                # This bytemap must see the child's unshared boundaries, rather
                # than retaining the outer sparse level's shared boundary.
                leaf = SparseByteMap{Int}(
                    leaf, 1, channels([[1, 2, 3, 4], [1, 2, 3, 4]]),
                    channels([fill(true, 3), fill(true, 3)]),
                    channels([[1, 2, 3], [1, 2, 3]]),
                )
                dst_leaf = SparseByteMap(dst_leaf, 1)
            end
            inner = SparseList{Int}(
                leaf, 2, channels([[1, 3, 4], [1, 2, 4]]),
                channels([[1, 2, 1], [2, 1, 2]]),
            )
            src = outer_sparse(inner)
            dst = SparseList(SparseList(dst_leaf, 2), 3)
            merge_fixture(src, dst)

            @test src.idx.data == [[1, -1], [2, 3]]
            @test dst.ptr == [1, 4]
            @test dst.idx == [1, 2, 3]
            @test dst.lvl.ptr == [1, 3, 5, 7]
            @test dst.lvl.idx == [1, 2, 1, 2, 1, 2]
            if with_bytemap
                @test dst.lvl.lvl.ptr == 1:7
                @test dst.lvl.lvl.srt == 1:6
                @test all(dst.lvl.lvl.tbl)
                @test dst.lvl.lvl.lvl.val == [10, 11, 20, 21, 30, 31]
            else
                @test dst.lvl.lvl.val == [10, 11, 20, 21, 30, 31]
            end
        end
    end

    @testset "bytemap and dense preserve ancestor boundaries" begin
        for with_dense in (false, true)
            vals = [[10, 11, 20, 0], [0, 21, 30, 31]]
            expected = [10, 11, 20, 21, 30, 31]
            if with_dense
                vals = [collect(Iterators.flatten((v, 10v) for v in vs)) for vs in vals]
                expected = collect(Iterators.flatten((v, 10v) for v in expected))
            end
            leaf = Element(0, channels(vals))
            dst_leaf = Element(0)
            if with_dense
                leaf = Dense(leaf, 2)
                dst_leaf = Dense(dst_leaf, 2)
            end
            inner = SparseByteMap{Int}(
                leaf, 2, channels([[1, 3, 4], [1, 2, 4]]),
                channels([[true, true, true, false], [false, true, true, true]]),
                channels([[1, 2, 3], [2, 3, 4]]),
            )
            src = outer_sparse(inner)
            dst = SparseList(SparseByteMap(dst_leaf, 2), 3)
            merge_fixture(src, dst)

            @test src.idx.data == [[1, -1], [2, 3]]
            # The bytemap's own indices carry no shared-boundary marker.
            @test src.lvl.srt.data == [[1, 2, 3], [2, 3, 4]]
            @test dst.ptr == [1, 4]
            @test dst.idx == [1, 2, 3]
            @test dst.lvl.ptr == [1, 3, 5, 7]
            @test dst.lvl.srt == 1:6
            @test all(dst.lvl.tbl)
            result_leaf = with_dense ? dst.lvl.lvl.lvl : dst.lvl.lvl
            @test result_leaf.val == expected
        end
    end

    @testset "fast merge without shared boundaries" begin
        src = SparseList{Int}(
            Element(0, channels([[10, 20], [30, 40]])), 4,
            channels([[1, 3], [1, 3]]), channels([[1, 2], [3, 4]]),
        )
        dst = SparseList(Element(0), 4)
        @test Finch.setup_coalesce!(src, 1, dst, 2, Finch.MergeFast())
        offsets = Finch.init_pos_offsets(2)
        shared_flags = Finch.init_shared_flags(2)
        Threads.@threads for tid in 1:2
            Finch.coalesce_fast!(tid, offsets, shared_flags, 2, src, dst, false)
        end
        @test dst.ptr == [1, 5]
        @test dst.idx == [1, 2, 3, 4]
        @test dst.lvl.val == [10, 20, 30, 40]
    end

    @testset "explicit merge ranges" begin
        offsets = Finch.init_pos_offsets(2)
        flags = Finch.init_shared_flags(2)
        src = Element(0, channels([[10, 20, 30], [40, 50, 60]]))
        dst = Element(0, fill(-1, 6))
        Finch.coalesce_fast!(1, offsets, flags, 2, src, dst, false, 2, 4)
        @test dst.val == [-1, 20, 30, 40, -1, -1]
        Finch.coalesce_fast!(1, offsets, flags, 2, src, dst, false, 5, 4)
        @test dst.val == [-1, 20, 30, 40, -1, -1]

        # A range may start or stop partway through a shared dense block.
        offsets[1] = [1, 2, 3]
        flags[1] = [true, false]
        src = Dense(Element(0, channels([[10, 11, 20, 0], [0, 21, 30, 31]])), 2)
        dst = Dense(Element(0, fill(-1, 6)), 2)
        Finch.coalesce_fast!(1, offsets, flags, 2, src, dst, true, 2, 2)
        @test dst.lvl.val == [-1, -1, 20, 21, -1, -1]
        Finch.coalesce_fast!(1, offsets, flags, 2, src.lvl, dst.lvl, true, 4, 5)
        @test dst.lvl.val == [-1, -1, 20, 21, 30, -1]
    end

    @testset "bytemap skips unoccupied child storage" begin
        for child_format in (:element, :dense, :sparse, :bytemap)
            with_dense = child_format == :dense
            child_sparse = child_format == :sparse
            leaf = Element(0, channels([[10, 30], [50]]))
            dst_leaf = Element(0, fill(-1, 3))
            if child_format == :bytemap
                leaf = SparseByteMap{Int}(
                    Element(0, channels([
                        [10, 999, 999, 999, 999, 30], [999, 999, 999, 50, 999, 999],
                    ])),
                    2, channels([[1, 2, 2, 3], [1, 1, 2, 2]]),
                    channels([
                        [true, false, false, false, false, true],
                        [false, false, false, true, false, false],
                    ]),
                    channels([[1, 6], [4]]),
                )
                dst_leaf = SparseByteMap(Element(0, fill(-1, 12)), 2)
            elseif child_sparse
                leaf = SparseList{Int}(
                    leaf, 2, channels([[1, 2, 2, 3], [1, 1, 2, 2]]),
                    channels([[1, 2], [2]]),
                )
                dst_leaf = SparseList(dst_leaf, 2)
            else
                vals = [[10, 999, 30], [999, 50, 999]]
                if with_dense
                    vals = [collect(Iterators.flatten((v, 10v) for v in vs)) for vs in vals]
                end
                leaf = Element(0, channels(vals))
                dst_leaf = Element(0, fill(-1, with_dense ? 12 : 6))
            end
            if with_dense
                leaf = Dense(leaf, 2)
                dst_leaf = Dense(dst_leaf, 2)
            end
            inner = SparseByteMap{Int}(
                leaf, 3, channels([[1, 3], [1, 2]]),
                channels([[true, false, true], [false, true, false]]),
                channels([[1, 3], [2]]),
            )
            src = SparseList{Int}(
                inner, 2, channels([[1, 2], [1, 2]]), channels([[1], [2]])
            )
            dst = SparseList(SparseByteMap(dst_leaf, 3), 2)
            @test Finch.setup_coalesce!(src, 1, dst, 2, Finch.MergeFast())
            offsets = Finch.init_pos_offsets(2)
            flags = Finch.init_shared_flags(2)
            Threads.@threads for tid in 1:2
                Finch.coalesce_fast!(tid, offsets, flags, 2, src, dst, false)
            end
            @test dst.lvl.srt == [1, 3, 5]
            if child_format == :bytemap
                @test dst.lvl.lvl.srt == [1, 6, 10]
                @test dst.lvl.lvl.lvl.val ==
                    [10, -1, -1, -1, -1, 30, -1, -1, -1, 50, -1, -1]
                @test [dst.lvl.lvl.ptr[q:(q + 1)] for q in (1, 3, 5)] ==
                    [[1, 2], [2, 3], [3, 4]]
            elseif child_sparse
                @test dst.lvl.lvl.idx == [1, 2, 2]
                @test dst.lvl.lvl.lvl.val == [10, 30, 50]
                @test [dst.lvl.lvl.ptr[q:(q + 1)] for q in (1, 3, 5)] ==
                    [[1, 2], [2, 3], [3, 4]]
            else
                expected = [10, -1, 30, -1, 50, -1]
                if with_dense
                    expected = collect(Iterators.flatten(
                        v == -1 ? (-1, -1) : (v, 10v) for v in expected
                    ))
                end
                result = with_dense ? dst.lvl.lvl.lvl : dst.lvl.lvl
                @test result.val == expected
            end
        end
    end

    @testset "dense bytemap merge follows the source shard" begin
        src = SparseByteMap{Int}(
            Element(0, channels([[10, 0, 0, 0, 0, 0], [0, 0, 30, 40, 50, 0]])),
            6, channels([[1, 2], [1, 4]]),
            channels([
                [true, false, false, false, false, false],
                [false, false, true, true, true, false],
            ]),
            channels([[1], [3, 4, 5]]),
        )
        dst = SparseByteMap(Element(0, fill(-1, 6)), 6)
        Finch.setup_coalesce!(src, 1, dst, 2, Finch.MergeFast())
        offsets = Finch.init_pos_offsets(2)
        flags = Finch.init_shared_flags(2)
        Threads.@threads for tid in 1:2
            Finch.coalesce_dense!(tid, offsets, flags, 2, src, dst)
        end
        @test dst.srt == [1, 3, 4, 5]
        @test dst.lvl.val == [10, -1, 30, 40, 50, -1]
    end

    @testset "batched merge ranges" begin
        offsets = Finch.init_pos_offsets(2)
        flags = Finch.init_shared_flags(2)
        offsets[1] = [1, 2, 3]
        flags[1] = [true, false]
        src = Element(0, channels([[10, 11, 20, 0], [0, 21, 30, 31]]))
        for ranges in (
            (1:1, 3:4, 6:6), [1:1, 3:4, 6:6],
            Finch.CoalesceRanges([1, 3, 4, 6], 1, 4),
        )
            dst = Element(0, fill(-1, 6))
            Finch.coalesce_fast!(1, offsets, flags, 2, src, dst, true, ranges)
            @test dst.val == [10, -1, 20, 21, -1, 31]
            @test offsets[1] == [1, 2, 3]
            @test flags[1] == [true, false]
        end
        dst = Dense(Element(0, fill(-1, 6)), 2)
        Finch.coalesce_fast!(
            1, offsets, flags, 2, Dense(src, 2), dst, true, (1:1, 2:1, 3:3)
        )
        @test dst.lvl.val == [10, 11, -1, -1, 30, 31]
        Finch.coalesce_fast!(1, offsets, flags, 2, src, dst.lvl, true, UnitRange{Int}[])
        @test dst.lvl.val == [10, 11, -1, -1, 30, 31]

        # The cursor must skip empty shards and positions between requested ranges.
        mem4 = Finch.MultiChannelMemory(cpu(:t, 4), 4)
        src4 = Element(0, Finch.MultiChannelBuffer(mem4, [Int[], [10, 20], Int[], [30, 40]]))
        dst4 = Element(0, fill(-1, 4))
        Finch.coalesce_fast!(
            1, Finch.init_pos_offsets(4), Finch.init_shared_flags(4), 4,
            src4, dst4, false, (1:1, 3:4),
        )
        @test dst4.val == [10, -1, 30, 40]
        @test collect(Finch.CoalesceRanges([1, 2, 5, 8, 9], 2, 4)) == [2:2, 5:5, 8:8]
        @test isempty(collect(Finch.CoalesceRanges(Int[], 1, 0)))

        # Exercise bulk copies across a shared block with a nonzero fill value.
        src = Element(-7, channels([
            vcat(1:32, 101:116, fill(-7, 16)),
            vcat(fill(-7, 16), 117:132, 201:232),
        ]))
        dst = Element(-7, fill(-999, 96))
        expected = copy(dst.val)
        values = vcat(1:32, 101:132, 201:232)
        ranges = (3:40, 45:90)
        for range in ranges
            expected[range] = values[range]
        end
        Finch.coalesce_fast!(1, offsets, flags, 2, src, dst, true, ranges)
        @test dst.val == expected
    end
end
