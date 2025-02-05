@testitem "scheduler" begin
    using Finch: JuliaContext
    using Finch.FinchNotation: finch_unparse_program, @finch_program_instance

    # test `propagate_map_queries`
    let
        plan = Finch.plan(
            Finch.query(
                Finch.alias(:A),
                Finch.aggregate(
                    Finch.immediate(:B),
                    Finch.immediate(:C),
                    Finch.immediate(:D),
                ),
            ),
            Finch.query(Finch.alias(:E), Finch.alias(:A)),
            Finch.produces(Finch.alias(:E)),
        )
        expected = Finch.plan(
            Finch.query(
                Finch.alias(:E),
                Finch.mapjoin(
                    Finch.immediate(:B),
                    Finch.immediate(:C),
                    Finch.immediate(:D),
                ),
            ),
            Finch.produces(Finch.alias(:E)),
        )

        result = Finch.propagate_map_queries(plan)
        @test result == expected
    end

    @testset "finch_unparse" begin
        prgm = @finch_program quote
            A .= 0
            for i in _
                A[i] += 1
            end
        end
        @test prgm.val == @finch_program $(finch_unparse_program(JuliaContext(), prgm))
    end

    @testset "concordize" begin
        using Finch.FinchLogic
        A = alias(:A)
        B = alias(:B)
        C = alias(:C)
        i = field(:i)
        j = field(:j)
        k = field(:k)
        prgm_in = plan(
            query(A, table(0, i, j)),
            query(B, table(0, i, j)),
            query(
                C,
                aggregate(
                    +,
                    0,
                    mapjoin(*,
                        reorder(relabel(A, i, k), i, k, j),
                        reorder(relabel(B, j, k), i, k, j),
                    ),
                ),
            ),
            produces(C))
        B_2 = alias(:B_2)
        prgm_out = plan(
            query(A, table(0, i, j)),
            query(B, table(0, i, j)),
            query(B_2, reorder(relabel(B, i, j), j, i)),
            query(
                C,
                aggregate(
                    +,
                    0,
                    mapjoin(*,
                        reorder(relabel(A, i, k), i, k, j),
                        reorder(relabel(B_2, k, j), i, k, j),
                    ),
                ),
            ),
            produces(C))
        @test Finch.concordize(prgm_in) == prgm_out

        prgm_in = plan(produces())
        prgm_out = plan(produces())
        @test Finch.concordize(prgm_in) == prgm_out

        prgm_in = plan(
            query(A, table(0, i, j)),
            query(B, table(0, i, j)),
            query(
                C,
                mapjoin(+,
                    reorder(relabel(A, i, j), j, i),
                    reorder(relabel(B, j, i), i, j),
                ),
            ),
            produces(C),
        )
        A_2 = alias(:A_2)
        prgm_out = plan(
            query(A, table(0, i, j)),
            query(A_2, reorder(relabel(A, i, j), j, i)),
            query(B, table(0, i, j)),
            query(B_2, reorder(relabel(B, i, j), j, i)),
            query(
                C,
                mapjoin(+,
                    reorder(relabel(A_2, j, i), j, i),
                    reorder(relabel(B_2, i, j), i, j),
                ),
            ),
            produces(C),
        )
        @test Finch.concordize(prgm_in) == prgm_out

        prgm_in = plan(
            query(A, table(0, i, j)),
            query(B, reorder(relabel(A, i, j), i, j)),
            produces(B),
        )
        prgm_out = plan(
            query(A, table(0, i, j)),
            query(B, reorder(relabel(A, i, j), i, j)),
            produces(B),
        )
        @test Finch.concordize(prgm_in) == prgm_out

        D = alias(:D)
        prgm_in = plan(
            query(A, table(0, i, j)),
            query(B, table(0, i, j)),
            query(C, reorder(relabel(A, i, j), j, i)),
            query(D, reorder(relabel(B, j, i), i, j)),
            produces(C, D),
        )
        prgm_out = plan(
            query(A, table(0, i, j)),
            query(A_2, reorder(relabel(A, i, j), j, i)),
            query(B, table(0, i, j)),
            query(B_2, reorder(relabel(B, i, j), j, i)),
            query(C, reorder(relabel(A_2, j, i), j, i)),
            query(D, reorder(relabel(B_2, i, j), i, j)),
            produces(C, D),
        )
        @test Finch.concordize(prgm_in) == prgm_out

        prgm_in = plan(
            query(A, table(0, i, j)),
            query(B, table(0, i, j)),
            query(
                C,
                mapjoin(+,
                    reorder(relabel(A, i, k), k, i),
                    reorder(relabel(B, k, j), j, k),
                ),
            ),
            produces(C),
        )
        C_2 = alias(:C_2)
        prgm_out = plan(
            query(A, table(0, i, j)),
            query(A_2, reorder(relabel(A, i, j), j, i)),
            query(B, table(0, i, j)),
            query(B_2, reorder(relabel(B, i, j), j, i)),
            query(
                C,
                mapjoin(+,
                    reorder(relabel(A_2, k, i), k, i),
                    reorder(relabel(B_2, j, k), j, k),
                ),
            ),
            produces(C),
        )
        @test Finch.concordize(prgm_in) == prgm_out

        prgm_in = plan(
            query(A, table(0)),
            query(B, reorder(relabel(A))),
            produces(B),
        )
        prgm_out = plan(
            query(A, table(0)),
            query(B, reorder(relabel(A))),
            produces(B),
        )
        @test Finch.concordize(prgm_in) == prgm_out

        prgm_in = plan(
            query(A, table(0, i, j, k)),
            query(B, reorder(relabel(A, i, j, k), k, j, i)),
            query(C, reorder(relabel(A, i, j, k), j, k, i)),
            query(
                D,
                mapjoin(*,
                    reorder(relabel(B, k, j, i), i, j, k),
                    reorder(relabel(C, j, k, i), i, j, k),
                ),
            ),
            produces(D),
        )
        A_3 = alias(:A_3)
        C_2 = alias(:C_2)
        prgm_out = plan(
            query(A, table(0, i, j, k)),
            query(A_2, reorder(relabel(A, i, j, k), k, j, i)),
            query(A_3, reorder(relabel(A, i, j, k), j, k, i)),
            query(B, reorder(relabel(A_2, k, j, i), k, j, i)),
            query(B_2, reorder(relabel(B, k, j, i), i, j, k)),
            query(C, reorder(relabel(A_3, j, k, i), j, k, i)),
            query(C_2, reorder(relabel(C, j, k, i), i, j, k)),
            query(
                D,
                mapjoin(*,
                    reorder(relabel(B_2, i, j, k), i, j, k),
                    reorder(relabel(C_2, i, j, k), i, j, k),
                ),
            ),
            produces(D),
        )
        @test Finch.concordize(prgm_in) == prgm_out
    end

    @testset "push_fields" begin
        using Finch.FinchLogic
        A = alias(:A)
        i = field(:i)
        j = field(:j)
        k = field(:k)

        # Test 1: Simple reorder and relabel on a table
        expr_in = reorder(relabel(table(A, i, j, k), k, j, i), i, j, k)
        expr_out = reorder(table(A, k, j, i), i, j, k)  # After push_fields, reorder and relabel should be absorbed
        @test Finch.push_fields(expr_in) == expr_out

        # Test 2: Nested reorders and relabels on a table
        expr_in = reorder(
            relabel(reorder(relabel(table(A, i, j, k), j, i, k), k, j, i), i, k, j), j, i, k
        )
        expr_out = reorder(table(A, k, j, i), j, i, k)
        @test Finch.push_fields(expr_in) == expr_out

        # Test 3: Mapjoin with internal reordering and relabeling
        expr_in = mapjoin(+,
            reorder(relabel(table(A, i, j), j, i), i, j),
            reorder(relabel(table(A, j, i), i, j), j, i))
        expr_out = mapjoin(+,
            reorder(table(A, j, i), i, j),
            reorder(table(A, i, j), j, i))
        @test Finch.push_fields(expr_in) == expr_out

        # Test 4: Immediate values absorbing relabel and reorder
        expr_in = reorder(relabel(immediate(42)), i)
        expr_out = reorder(immediate(42), i)
        @test Finch.push_fields(expr_in) == expr_out

        # Test 5: Complex nested structure with mapjoin and aggregates
        expr_in = mapjoin(+,
            reorder(
                relabel(
                    mapjoin(*,
                        reorder(relabel(table(A, i, j, k), k, j, i), i, j, k),
                        table(A, j, i, k)), i, k, j), j, i, k),
            mapjoin(*,
                reorder(relabel(table(A, i, j, k), j, i, k), k, j, i)))
        expr_out = mapjoin(+,
            mapjoin(*,
                reorder(table(A, j, k, i), j, i, k),
                reorder(table(A, k, i, j), j, i, k)),
            mapjoin(*,
                reorder(table(A, j, i, k), k, j, i)))
        @test Finch.push_fields(expr_in) == expr_out

        #=
        query(A1, table(0, i0, i1))
        query(A2, table(1, i2, i3))
        query(A5,
            aggregate(+, 0.0, relabel(
                mapjoin(*,
                    reorder(relabel(relabel(A2, i2, i3), i7, i8), i7, i8, i9),
                    reorder(relabel(relabel(A0, i0, i1), i8, i9), i7, i8, i9)
                ), i13, i14, i15), i14))
        =#
    end
end
