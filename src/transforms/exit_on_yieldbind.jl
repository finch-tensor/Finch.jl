"""
    exit_on_yieldbind(prgm)

This pass rewrites the program so that yieldbind expressions
are only present at the end of a block. It also adds a
yieldbind if not present already.
"""
function exit_on_yieldbind(prgm)
    Rewrite(
        Chain([
            Rewrite(
                Fixpoint(
                    Postwalk(
                        Chain([
                            (@rule block(~a_1..., yieldbind(~b...), ~a_2, ~a_3...) =>
                                begin
                                    block(a_1..., yieldbind(b...))
                                end),
                            (@rule block(
                                ~a_1..., block(~b..., yieldbind(~c...)), ~a_2...
                            ) => begin
                                block(a_1..., b..., yieldbind(c...))
                            end),
                        ]),
                    ),
                ),
            ),
        ]),
    )(
        block(prgm, yieldbind())
    )
end
