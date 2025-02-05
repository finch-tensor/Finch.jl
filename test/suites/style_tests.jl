@testitem "style" skip = (!Base.Sys.isunix()) begin #https://github.com/domluna/JuliaFormatter.jl/issues/898
    using ..Main: parsed_args
    using JuliaFormatter
    using FileWatching

    if parsed_args["overwrite"]
        mkpidlock(joinpath(@__DIR__, "..", "..", "lock.pid")) do
            for i in 1:10#https://github.com/domluna/JuliaFormatter.jl/issues/897
                if JuliaFormatter.format(Finch)
                    break
                end
            end
            @test JuliaFormatter.format(Finch)
        end
    else
        @test JuliaFormatter.format(Finch, overwrite=false)
    end
end
