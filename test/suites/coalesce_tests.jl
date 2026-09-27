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
end
