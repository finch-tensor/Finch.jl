module MatrixMarketExt

using Galley
using SparseArrays

isdefined(Base, :get_extension) ? (using MatrixMarket) : (using ..MatrixMarket)

# Exchange coordinates without relying on Galley's SparseArrays conversion methods.
function Galley.fmmread(filename::AbstractString)
    A = sparse(mmread(filename))
    fsparse(findnz(A)..., size(A))
end

function Galley.fmmwrite(filename::AbstractString, A)
    ndims(A) == 2 || throw(ArgumentError("Matrix Market only supports matrices"))
    fill_value(A) === zero(eltype(A)) || throw(
        ArgumentError("Matrix Market only supports zero fill values")
    )
    I, J, V = ffindnz(A)
    matrix = sparse(I, J, V, size(A)...)
    # Pattern files encode every stored entry as true, including stored false values.
    eltype(matrix) <: Bool && dropzeros!(matrix)
    mmwrite(filename, matrix)
end

end
