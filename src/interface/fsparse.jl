"""
    fsparse(I::Tuple, V,[ M::Tuple, combine]; fill_value=zero(eltype(V)))

Create a sparse COO tensor `S` such that `size(S) == M` and `S[(i[q] for i =
I)...] = V[q]`. The combine function is used to combine duplicates. If `M` is
not specified, it is set to `map(maximum, I)`. If the combine function is not
supplied, combine defaults to `+` unless the elements of V are Booleans in which
case combine defaults to `|`. All elements of I must satisfy 1 <= I[n][q] <=
M[n].  Numerical zeros are retained as structural nonzeros; to drop numerical
zeros, use dropzeros!.

See also: [`sparse`](https://docs.julialang.org/en/v1/stdlib/SparseArrays/#SparseArrays.sparse)

# Examples

julia> I = (
    [1, 2, 3],
    [1, 2, 3],
    [1, 2, 3]);

julia> V = [1.0; 2.0; 3.0];

julia> fsparse(I, V)
SparseCOO (0.0) [1:3×1:3×1:3]
│ │ │
└─└─└─[1, 1, 1] [2, 2, 2] [3, 3, 3]
      1.0       2.0       3.0
"""
fsparse(iV::AbstractVector, args...; kwargs...) = fsparse_parse((), iV, args...; kwargs...)
fsparse_parse(I, i::AbstractVector, args...; kwargs...) = fsparse_parse((I..., i), args...; kwargs...)
fsparse_parse(I, V::AbstractVector; kwargs...) = fsparse_impl(I, V; kwargs...)
fsparse_parse(I, V::AbstractVector, m::Tuple; kwargs...) = fsparse_impl(I, V, m; kwargs...)
fsparse_parse(I, V::AbstractVector, m::Tuple, combine; kwargs...) = fsparse_impl(I, V, m, combine; kwargs...)
function fsparse_impl(I::Tuple, V::Vector, shape = map(maximum, I), combine = eltype(V) isa Bool ? (|) : (+); fill_value = zero(eltype(V)))
    C = map(tuple, reverse(I)...)
    updater = false
    if !issorted(C)
        P = sortperm(C)
        C = C[P]
        V = V[P]
        updater = true
    end
    if !allunique(C)
        P = unique(p -> C[p], 1:length(C))
        C = C[P]
        push!(P, length(I[1]) + 1)
        V = map((start, stop) -> foldl(combine, @view V[start:stop - 1]), P[1:end - 1], P[2:end])
        updater = true
    end
    if updater
        I = map(i -> similar(i, length(C)), I)
        foreach(((p, c),) -> ntuple(n->I[n][p] = c[n], length(I)), enumerate(C))
        I = reverse(I)
    else
        I = map(copy, I)
    end
    return fsparse!(I..., V, shape; fill_value=fill_value)
end

"""
    fsparse!(I..., V,[ M::Tuple])

Like [`fsparse`](@ref), but the coordinates must be sorted and unique, and memory
is reused.
"""
fsparse!(args...; kwargs...) = fsparse!_parse((), args...; kwargs...)
fsparse!_parse(I, i::AbstractVector, args...; kwargs...) = fsparse!_parse((I..., i), args...; kwargs...)
fsparse!_parse(I, V::AbstractVector; kwargs...) = fsparse!_impl(I, V; kwargs...)
fsparse!_parse(I, V::AbstractVector, M::Tuple; kwargs...) = fsparse!_impl(I, V, M; kwargs...)
function fsparse!_impl(I::Tuple, V, shape = map(maximum, I); fill_value = zero(eltype(V)))
    return Tensor(SparseCOO{length(I), Tuple{map(eltype, I)...}}(Element{fill_value, eltype(V), Int}(V), shape, [1, length(V) + 1], I))
end

"""
    fsprand([rng],[type], M..., p, [rfn])

Create a random sparse tensor of size `m` in COO format. There are two cases:
    - If `p` is floating point, the probability of any element being nonzero is
    independently given by `p` (and hence the expected density of nonzeros is
    also `p`).
    - If `p` is an integer, exactly `p` nonzeros are distributed uniformly at
    random throughout the tensor (and hence the density of nonzeros is exactly
    `p / prod(M)`).
Nonzero values are sampled from the distribution specified by `rfn` and have the
type `type`. The uniform distribution is used in case `rfn` is not specified.
The optional `rng` argument specifies a random number generator.

See also: (`sprand`)(https://docs.julialang.org/en/v1/stdlib/SparseArrays/#SparseArrays.sprand)

# Examples
```julia
julia> fsprand(Bool, 3, 3, 0.5)
SparseCOO (false) [1:3,1:3]
├─├─[1, 1]: true
├─├─[3, 1]: true
├─├─[2, 2]: true
├─├─[3, 2]: true
├─├─[3, 3]: true

julia> fsprand(Float64, 2, 2, 2, 0.5)
SparseCOO (0.0) [1:2,1:2,1:2]
├─├─├─[2, 2, 1]: 0.6478553157718558
├─├─├─[1, 1, 2]: 0.996665291437684
├─├─├─[2, 1, 2]: 0.7491940599574348
```
"""
fsprand(args...) = fsprand_parse_rng(args...)

fsprand_parse_rng(r::AbstractRNG, args...) = fsprand_parse_type(r, args...)
fsprand_parse_rng(args...) = fsprand_parse_type(default_rng(), args...)

fsprand_parse_type(r, T::Type, args...) = fsprand_parse_shape(r, (T,), (), args...)
fsprand_parse_type(r, args...) = fsprand_parse_shape(r, (), (), args...)

fsprand_parse_shape(r, T, M, m, args...) = fsprand_parse_shape(r, T, (M..., m), args...)
fsprand_parse_shape(r, T, M, p::AbstractFloat) = fsprand_parse_shape(r, T, M, p, rand)
fsprand_parse_shape(r, T, M, p::AbstractFloat, rfn::Function) = fsprand_erdos_renyi_gilbert(r, T, M, p, rfn)
fsprand_parse_shape(r, T, M, nnz::Integer) = fsprand_parse_shape(r, T, M, nnz, rand)
fsprand_parse_shape(r, T, M, nnz::Integer, rfn::Function) = fsprand_erdos_renyi(r, T, M, nnz, rfn)
#fsprand_parse_shape(r, T, M) = throw(ArgumentError("No float p given to fsprand"))

#https://github.com/JuliaStats/StatsBase.jl/blob/60fb5cd400c31d75efd5cdb7e4edd5088d4b1229/src/sampling.jl#L137-L182
function fsprand_erdos_renyi_sample_knuth(r::AbstractRNG, M::Tuple, nnz::Int)
    N = length(M)

    I = ntuple(n -> Vector{typeof(M[n])}(undef, nnz), N)

    k = 1
    function sample(n, i...)
        if n == 0
            if k <= nnz
                for m = 1:N
                    I[m][k] = i[m]
                end
            elseif rand(r) * k < nnz
                l = rand(r, 1:nnz)
                for m = 1:N
                    I[m][l] = i[m]
                end
            end
            k += 1
        else
            m = M[n]
            for i_n = 1:m
                sample(n - 1, i_n, i...)
            end
        end
    end
    sample(N)

    return I
end

#https://github.com/JuliaStats/StatsBase.jl/blob/60fb5cd400c31d75efd5cdb7e4edd5088d4b1229/src/sampling.jl#L234-L278
function fsprand_erdos_renyi_sample_self_avoid(r::AbstractRNG, M::Tuple, nnz::Int)
    N = length(M)

    I = ntuple(n -> Vector{typeof(M[n])}(undef, nnz), length(M))
    S = Set{typeof(M)}()

    k = 0
    while length(S) < nnz
        i = ntuple(n -> rand(r, 1:M[n]), N)
        push!(S, i)
        if length(S) > k
            k += 1
            for m = 1:N
                I[m][k] = i[m]
            end
        end
    end

    return I
end

function fsprand_erdos_renyi(r::AbstractRNG, T, M::Tuple, nnz::Int, rfn)
    if nnz / prod(M, init=1.0) < 0.15
        I = fsprand_erdos_renyi_sample_self_avoid(r, M, nnz)
    else
        I = fsprand_erdos_renyi_sample_knuth(r, M, nnz)
    end
    p = sortperm(map(tuple, reverse(I)...))
    for n = 1:length(I)
        permute!(I[n], p)
    end
    V = rfn(r, T..., nnz)
    return fsparse!(I..., V, M)
end

function fsprand_erdos_renyi_gilbert(r::AbstractRNG, T, M::Tuple, p::AbstractFloat, rfn)
    n = prod(M, init=1.0)
    q = 1 - p
    #We wish to sample nnz from binomial(n, p).
    if n <= typemax(Int)*(1 - eps())
        #Ideally, n is representable as an Int
        _n = Int(prod(M))
        nnz = rand(r, Binomial(_n, p))
    else
        #Otherwise we approximate
        if n * p < 10
            #When n * p < 10, we use a poisson
            #https://math.oxford.emory.edu/site/math117/connectingPoissonAndBinomial/
            nnz = rand(r, Poisson(n * p))
        else
            nnz = -1
            while nnz < 0
                #Otherwise, we use a normal distribution
                #https://stats.libretexts.org/Courses/Las_Positas_College/Math_40%3A_Statistics_and_Probability/06%3A_Continuous_Random_Variables_and_the_Normal_Distribution/6.04%3A_Normal_Approximation_to_the_Binomial_Distribution
                _nnz = rand(r, Normal(n * p, sqrt(n * p * q)))
                @assert _nnz <= typemax(Int) "integer overflow; tried to generate too many nonzeros"
                nnz = round(Int, _nnz)
            end
        end
        # Note that we do not consider n * q < 10, since this would mean we
        # would probably overflow the int buffer anyway. However, subtracting
        # poisson would work in that case
    end
    #now we generate exactly nnz nonzeros:
    return fsprand_erdos_renyi(r, T, M, nnz, rfn)
end

fsprandn(args...) = fsprand(args..., randn)

"""
    fspzeros([type], M...)

Create a random zero tensor of size `M`, with elements of type `type`. The
tensor is in COO format.

See also: (`spzeros`)(https://docs.julialang.org/en/v1/stdlib/SparseArrays/#SparseArrays.spzeros)

# Examples
```jldoctest
julia> A = fspzeros(Bool, 3, 3)
3×3 Tensor{SparseCOOLevel{2, Tuple{Int64, Int64}, Vector{Int64}, Tuple{Vector{Int64}, Vector{Int64}}, ElementLevel{false, Bool, Int64, Vector{Bool}}}}:
 0  0  0
 0  0  0
 0  0  0

julia> countstored(A)
0

julia> B = fspzeros(Float64, 2, 2, 2)
2×2×2 Tensor{SparseCOOLevel{3, Tuple{Int64, Int64, Int64}, Vector{Int64}, Tuple{Vector{Int64}, Vector{Int64}, Vector{Int64}}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}:
[:, :, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2] =
 0.0  0.0
 0.0  0.0

julia> countstored(B)
0

```
"""
fspzeros(M...) = fspzeros(Float64, M...)
function fspzeros(::Type{T}, M...) where {T}
    return fsparse!((Int[] for _ in M)..., T[], M)
end

"""
    ffindnz(arr)

Return the nonzero elements of `arr`, as Finch understands `arr`. Returns `(I...,
V)`, where `I` are the coordinate vectors, one for each mode of `arr`, and
`V` is a vector of corresponding nonzero values, which can be passed to
[`fsparse`](@ref).

See also: (`findnz`)(https://docs.julialang.org/en/v1/stdlib/SparseArrays/#SparseArrays.findnz)
"""
function ffindnz(src)
    tmp = Tensor(
        SparseCOOLevel{ndims(src)}(
            ElementLevel{zero(eltype(src)), eltype(src)}()))
    tmp = copyto!(tmp, src)
    nnz = tmp.lvl.ptr[2] - 1
    tbl = tmp.lvl.tbl
    val = tmp.lvl.lvl.val
    (ntuple(n->tbl[n][1:nnz], ndims(src))..., val[1:nnz])
end
