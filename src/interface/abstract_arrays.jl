@kwdef mutable struct VirtualAbstractArray <: AbstractVirtualTensor
    tag
    data
    eltype
    ndims
    shape
end

function virtual_size(ctx::AbstractCompiler, arr::VirtualAbstractArray)
    return arr.shape
end

function lower(ctx::AbstractCompiler, arr::VirtualAbstractArray, ::DefaultStyle)
    return arr.data
end

function virtualize(ctx, ex, ::Type{<:AbstractArray{T,N}}, tag=:tns) where {T,N}
    tag = freshen(ctx, tag)
    dims = map(i -> Symbol(tag, :_mode, i, :_stop), 1:N)
    data = freshen(ctx, tag, :_data)
    push_preamble!(
        ctx,
        quote
            $data = $ex
            ($(dims...),) = size($ex)
        end,
    )
    VirtualAbstractArray(
        tag, data, T, N, map(i -> Extent(literal(1), value(dims[i], Int)), 1:N)
    )
end

function Finch.reroot_set!(ctx::AbstractCompiler, arr::VirtualAbstractArray, diff)
    diff[arr.tag] = arr
end

function Finch.reroot_get(ctx::AbstractCompiler, arr::VirtualAbstractArray, diff)
    get(diff, arr.tag, arr)
end

function declare!(ctx::AbstractCompiler, arr::VirtualAbstractArray, init)
    push_preamble!(
        ctx,
        quote
            fill!($(arr.data), $(ctx(init)))
        end,
    )
    arr
end

freeze!(ctx::AbstractCompiler, arr::VirtualAbstractArray) = arr
thaw!(ctx::AbstractCompiler, arr::VirtualAbstractArray) = arr

@kwdef struct VirtualAbstractArraySlice
    arr::VirtualAbstractArray
    idx
end

Finch.reroot_set!(ctx::AbstractCompiler, arr::VirtualAbstractArraySlice, diff) = 
    Finch.reroot_set!(ctx, arr.mtx, diff)

Finch.reroot_get(ctx::AbstractCompiler, arr::VirtualAbstractArraySlice, diff) =
    VirtualAbstractArraySlice(Finch.reroot_get(ctx, arr.mtx, diff), arr.idx)

FinchNotation.finch_leaf(x::VirtualAbstractArraySlice) = virtual(x)

function unfurl(ctx, tns::VirtualAbstractArraySlice, ext, mode, proto)
    arr = tns.arr
    idx = tns.idx
    Lookup(;
        body=(ctx, i) -> begin
            idx_2 = (i, idx...)
            if length(idx_2) == arr.ndims
                val = freshen(ctx, :val)
                if mode.kind === reader
                    Thunk(;
                        preamble=quote
                            $val = $(arr.data)[$(map(ctx, idx_2)...)]
                        end,
                        body=(ctx) -> instantiate(
                            ctx,
                            VirtualScalar(
                                nothing, nothing, arr.eltype, nothing, gensym(), val
                            ),
                            mode,
                        ), #=We don't know what init is, but it won't be used here =#
                    )
                else
                    Thunk(;
                        body=(ctx,) -> instantiate(
                            ctx,
                            VirtualScalar(
                                nothing,
                                nothing,
                                arr.eltype,
                                nothing,
                                gensym(),
                                :($(arr.data)[$(map(ctx, idx_2)...)]),
                            ),
                            mode,
                        ), #=We don't know what init is, but it won't be used here=#
                    )
                end
            else
                Thunk(;
                    body=(ctx,) ->
                        instantiate(ctx, VirtualAbstractArraySlice(arr, idx_2), mode),
                )
            end
        end,
    )
end

#is_injective(ctx, tns::VirtualAbstractArraySlice) = is_injective(ctx, tns.body)
#is_atomic(ctx, tns::VirtualAbstractArraySlice) = is_atomic(ctx, tns.body)
#is_concurrent(ctx, tns::VirtualAbstractArraySlice) = is_concurrent(ctx, tns.body)

function instantiate(ctx::AbstractCompiler, arr::VirtualAbstractArray, mode)
    if arr.ndims == 0
        val = freshen(ctx, :val)
        if mode.kind === reader
            Thunk(;
                preamble=quote
                    $val = $(arr.data)[]
                end,
                body=(ctx) -> instantiate(
                    ctx,
                    VirtualScalar(nothing, nothing, arr.eltype, nothing, gensym(), val),
                    mode,
                ), #=We don't know what init is, but it won't be used here =#
            )
        else
            Thunk(;
                body=(ctx,) -> instantiate(
                    ctx,
                    VirtualScalar(
                        nothing, nothing, arr.eltype, nothing, gensym(),
                        :($(arr.data)[]),
                    ),
                    mode,
                ), #=We don't know what init is, but it won't be used here=#
            )
        end
    else
        Unfurled(;
            arr=arr,
            body=VirtualAbstractArraySlice(arr, ()),
        )
    end
end

FinchNotation.finch_leaf(x::VirtualAbstractArray) = virtual(x)

virtual_fill_value(ctx, ::VirtualAbstractArray) = 0
virtual_eltype(ctx, tns::VirtualAbstractArray) = tns.eltype

function virtual_transfer(ctx, arr::VirtualAbstractArray, device, style)
    data = freshen(ctx, arr.data)
    push_preamble!(
        ctx,
        quote
            $data = $transfer($(arr.data), $(ctx(device)), $style)
        end,
    )
    VirtualAbstractArray(arr.tag, data, arr.eltype, arr.ndims, arr.shape)
end

fill_value(a::AbstractArray) = fill_value(typeof(a))
fill_value(T::Type{<:AbstractArray}) = zero(eltype(T))

"""
    Array(arr::Union{Tensor, SwizzleArray})

Construct an array from a tensor or swizzle. May reuse memory, will usually densify the tensor.
"""
function Base.Array(fbr::Union{Tensor,SwizzleArray})
    arr = Array{eltype(fbr)}(undef, size(fbr)...)
    return copyto!(arr, fbr)
end

struct AsArray{T,N,Fbr} <: AbstractArray{T,N}
    fbr::Fbr
    function AsArray{T,N,Fbr}(fbr::Fbr) where {T,N,Fbr}
        @assert T == eltype(fbr)
        @assert N == ndims(fbr)
        new{T,N,Fbr}(fbr)
    end
end

AsArray(fbr::Fbr) where {Fbr} = AsArray{eltype(Fbr),ndims(Fbr),Fbr}(fbr)

function Base.summary(io::IO, arr::AsArray)
    join(io, size(arr), "×")
    print(io, " ", typeof(arr.fbr))
    #print(io, " ")
    #summary(io, arr.fbr)
end

Base.size(arr::AsArray)                                            = size(arr.fbr)
Base.getindex(arr::AsArray{T,N}, i::Vararg{Int,N}) where {T,N}     = arr.fbr[i...]
Base.getindex(arr::AsArray{T,N}, i::Vararg{Any,N}) where {T,N}     = arr.fbr[i...]
Base.setindex!(arr::AsArray{T,N}, v, i::Vararg{Int,N}) where {T,N} = arr.fbr[i...] = v
Base.setindex!(arr::AsArray{T,N}, v, i::Vararg{Any,N}) where {T,N} = arr.fbr[i...] = v

is_injective(ctx, tns::VirtualAbstractArray) = [true for _ in tns.ndims]
is_atomic(ctx, tns::VirtualAbstractArray) = [false, [false for _ in tns.ndims]...]
# is_atomic(ctx, tns::VirtualAbstractArray) = true
