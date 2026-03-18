module MatrixMarketExt

using Finch
using SparseArrays

isdefined(Base, :get_extension) ? (using MatrixMarket) : (using ..MatrixMarket)

function Finch.fmmread(filename::AbstractString)
    Tensor(sparse(mmread(filename)))
end

function Finch.fmmwrite(filename::AbstractString, A)
    mmwrite(filename, SparseMatrixCSC(A))
end

end
