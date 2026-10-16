module TensorMarketExt

using Galley

isdefined(Base, :get_extension) ? (using TensorMarket) : (using ..TensorMarket)

function Galley.fttread(filename::AbstractString, infoonly=false, retcoord=false)
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

function Galley.fttwrite(filename::AbstractString, A)
    IV = ffindnz(A)
    I = IV[1:(end - 1)]
    V = IV[end]
    ttwrite(filename, I, V, size(A))
end

function Galley.ftnsread(filename::AbstractString)
    (I, V) = tnsread(filename)
    IR = (reverse(I)...,)
    if issorted(1:length(V); lt=(i, j) -> Galley.fsparse_lt(IR, i, j))
        swizzle(fsparse!(IR..., V), reverse(1:length(I))...)
    else
        fsparse!(I..., V)
    end
end

function Galley.ftnswrite(filename::AbstractString, A)
    IV = ffindnz(A)
    I = IV[1:(end - 1)]
    V = IV[end]
    tnswrite(filename, I, V)
end

end
