module TensorMarketExt

using Finch

isdefined(Base, :get_extension) ? (using TensorMarket) : (using ..TensorMarket)

function Finch.fttread(filename::AbstractString, infoonly=false, retcoord=false)
    infoonly && return ttread(filename, true)
    out = ttread(filename, false, retcoord)
    if out isa Tuple
        (I, V, shape, coords...) = out
        A = fsparse!(I..., V, shape)
        if retcoord
            (A, coords...)
        else
            A
        end
    else
        out
    end
end

function Finch.fttwrite(filename::AbstractString, A)
    IV = ffindnz(A)
    I = IV[1:(end - 1)]
    V = IV[end]
    ttwrite(filename, I, V, size(A))
end

function Finch.ftnsread(filename::AbstractString)
    (I, V) = tnsread(filename)
    IR = (reverse(I)...,)
    if issorted(1:length(V); lt=(i, j) -> Finch.fsparse_lt(IR, i, j))
        swizzle(fsparse!(IR..., V), reverse(1:length(I))...)
    else
        fsparse!(I..., V)
    end
end

function Finch.ftnswrite(filename::AbstractString, A)
    IV = ffindnz(A)
    I = IV[1:(end - 1)]
    V = IV[end]
    tnswrite(filename, I, V)
end

end
