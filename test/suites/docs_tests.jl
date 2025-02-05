@testitem "docs" skip = (Sys.WORD_SIZE != 64) begin
    using ..Main: parsed_args
    using FileWatching

    if parsed_args["overwrite"]
        mkpidlock(joinpath(@__DIR__, "..", "..", "lock.pid")) do
            include("../../docs/fix.jl")
        end
    else
        include("../../docs/test.jl")
    end
end
