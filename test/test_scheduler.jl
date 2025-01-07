@testset "scheduler" begin
    @info "Testing Default Scheduler's Internals"

    # test `propagate_map_queries`
    let
        plan = Finch.plan(
            Finch.query(
                Finch.alias(:A),
                Finch.aggregate(
                    Finch.immediate(:B),
                    Finch.immediate(:C),
                    Finch.immediate(:D),
                )
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
                )
            ),
            Finch.produces(Finch.alias(:E)),
        )

        result = Finch.propagate_map_queries(plan)
        @test result == expected
    end

end