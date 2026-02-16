"""
    DenseLevel{[Ti=Int]}(lvl, [dim])

A subfiber of a dense level is an array which stores every slice `A[:, ..., :,
i]` as a distinct subfiber in `lvl`. Optionally, `dim` is the size of the last
dimension. `Ti` is the type of the indices used to index the level.

```jldoctest
julia> ndims(Tensor(Dense(Element(0.0))))
1

julia> ndims(Tensor(Dense(Dense(Element(0.0)))))
2

julia> tensor_tree(Tensor(Dense(Dense(Element(0.0))), [1 2; 3 4]))
2×2-Tensor
└─ Dense [:,1:2]
   ├─ [:, 1]: Dense [1:2]
   │  ├─ [1]: 1.0
   │  └─ [2]: 3.0
   └─ [:, 2]: Dense [1:2]
      ├─ [1]: 2.0
      └─ [2]: 4.0
```
"""
struct DenseLevel{Ti,Lvl} <: AbstractLevel
    lvl::Lvl
    shape::Ti
end
DenseLevel(lvl) = DenseLevel{Int}(lvl)
#DenseLevel(lvl, shape::Ti) where {Ti} = DenseLevel{Ti}(lvl, shape)
DenseLevel{Ti}(lvl) where {Ti} = DenseLevel{Ti}(lvl, zero(Ti))
DenseLevel{Ti}(lvl::Lvl, shape) where {Ti,Lvl} = DenseLevel{Ti,Lvl}(lvl, shape)

const Dense = DenseLevel

Base.summary(lvl::Dense) = "Dense($(summary(lvl.lvl)))"

function similar_level(lvl::DenseLevel, fill_value, eltype::Type, dims...)
    Dense(similar_level(lvl.lvl, fill_value, eltype, dims[1:(end - 1)]...), dims[end])
end

function postype(::Type{DenseLevel{Ti,Lvl}}) where {Ti,Lvl}
    return postype(Lvl)
end

function transfer(device, lvl::DenseLevel{Ti}) where {Ti}
    return DenseLevel{Ti}(transfer(device, lvl.lvl), lvl.shape)
end

function pattern!(lvl::DenseLevel{Ti,Lvl}) where {Ti,Lvl}
    DenseLevel{Ti}(pattern!(lvl.lvl), lvl.shape)
end

function set_fill_value!(lvl::DenseLevel{Ti}, init) where {Ti}
    DenseLevel{Ti}(set_fill_value!(lvl.lvl, init), lvl.shape)
end

function Base.resize!(lvl::DenseLevel{Ti}, dims...) where {Ti}
    DenseLevel{Ti}(resize!(lvl.lvl, dims[1:(end - 1)]...), dims[end])
end

@inline level_ndims(::Type{<:DenseLevel{Ti,Lvl}}) where {Ti,Lvl} = 1 + level_ndims(Lvl)
@inline level_size(lvl::DenseLevel) = (level_size(lvl.lvl)..., lvl.shape)
@inline level_axes(lvl::DenseLevel) = (level_axes(lvl.lvl)..., Base.OneTo(lvl.shape))
@inline level_eltype(::Type{<:DenseLevel{Ti,Lvl}}) where {Ti,Lvl} = level_eltype(Lvl)
@inline level_fill_value(::Type{<:DenseLevel{Ti,Lvl}}) where {Ti,Lvl} = level_fill_value(
    Lvl
)
data_rep_level(::Type{<:DenseLevel{Ti,Lvl}}) where {Ti,Lvl} = DenseData(data_rep_level(Lvl))

function isstructequal(a::T, b::T) where {T<:Dense}
    a.shape == b.shape &&
        isstructequal(a.lvl, b.lvl)
end

(fbr::AbstractFiber{<:DenseLevel})() = fbr
function (fbr::SubFiber{<:DenseLevel{Ti}})(idxs...) where {Ti}
    isempty(idxs) && return fbr
    lvl = fbr.lvl
    p = fbr.pos
    q = (p - 1) * lvl.shape + idxs[end]
    fbr_2 = SubFiber(lvl.lvl, q)
    fbr_2(idxs[1:(end - 1)]...)
end

function countstored_level(lvl::DenseLevel, pos)
    countstored_level(lvl.lvl, pos * lvl.shape)
end

function Base.show(io::IO, lvl::DenseLevel{Ti}) where {Ti}
    if get(io, :compact, false)
        print(io, "Dense(")
    else
        print(io, "Dense{$Ti}(")
    end
    show(io, lvl.lvl)
    print(io, ", ")
    show(io, lvl.shape)
    print(io, ")")
end

function labelled_show(io::IO, fbr::SubFiber{<:DenseLevel})
    print(io, "Dense [", ":,"^(ndims(fbr) - 1), "1:", size(fbr)[end], "]")
end

function labelled_children(fbr::SubFiber{<:DenseLevel})
    lvl = fbr.lvl
    pos = fbr.pos
    map(1:(lvl.shape)) do idx
        LabelledTree(
            cartesian_label([range_label() for _ in 1:(ndims(fbr) - 1)]..., idx),
            SubFiber(lvl.lvl, (pos - 1) * lvl.shape + idx),
        )
    end
end

mutable struct VirtualDenseLevel <: AbstractVirtualLevel
    tag
    lvl
    Ti
    shape
end

function is_level_injective(ctx, lvl::VirtualDenseLevel)
    [is_level_injective(ctx, lvl.lvl)..., true]
end
function is_level_atomic(ctx, lvl::VirtualDenseLevel)
    (data, atomic) = is_level_atomic(ctx, lvl.lvl)
    return ([data; atomic], atomic)
end
function is_level_concurrent(ctx, lvl::VirtualDenseLevel)
    (data, concurrent) = is_level_concurrent(ctx, lvl.lvl)
    return ([data; concurrent], concurrent)
end

function virtualize(ctx, ex, ::Type{DenseLevel{Ti,Lvl}}, tag=:lvl) where {Ti,Lvl}
    tag = freshen(ctx, tag)
    stop = freshen(ctx, tag, :_stop)
    push_preamble!(
        ctx,
        quote
            $tag = $ex
            $stop = $tag.shape
        end,
    )
    shape = value(stop, Ti)
    lvl_2 = virtualize(ctx, :($tag.lvl), Lvl, tag)
    VirtualDenseLevel(tag, lvl_2, Ti, shape)
end
function lower(ctx::AbstractCompiler, lvl::VirtualDenseLevel, ::DefaultStyle)
    quote
        $DenseLevel{$(lvl.Ti)}(
            $(ctx(lvl.lvl)),
            $(ctx(lvl.shape)),
        )
    end
end

function distribute_level(ctx::AbstractCompiler, lvl::VirtualDenseLevel, arch, diff, style)
    lvl_2 = distribute_level(ctx, lvl.lvl, arch, diff, style)
    diff[lvl.tag] = VirtualDenseLevel(lvl.tag, lvl_2, lvl.Ti, lvl.shape)
end

function redistribute(ctx::AbstractCompiler, lvl::VirtualDenseLevel, diff)
    get(
        diff,
        lvl.tag,
        VirtualDenseLevel(lvl.tag, redistribute(ctx, lvl.lvl, diff), lvl.Ti, lvl.shape),
    )
end

Base.summary(lvl::VirtualDenseLevel) = "Dense($(summary(lvl.lvl)))"

function virtual_level_size(ctx, lvl::VirtualDenseLevel)
    ext = VirtualExtent(literal(lvl.Ti(1)), lvl.shape)
    (virtual_level_size(ctx, lvl.lvl)..., ext)
end

function virtual_level_resize!(ctx, lvl::VirtualDenseLevel, dims...)
    lvl.shape = getstop(dims[end])
    lvl.lvl = virtual_level_resize!(ctx, lvl.lvl, dims[1:(end - 1)]...)
    lvl
end

virtual_level_eltype(lvl::VirtualDenseLevel) = virtual_level_eltype(lvl.lvl)
virtual_level_fill_value(lvl::VirtualDenseLevel) = virtual_level_fill_value(lvl.lvl)

postype(lvl::VirtualDenseLevel) = postype(lvl.lvl)

function declare_level!(ctx::AbstractCompiler, lvl::VirtualDenseLevel, pos, init)
    lvl.lvl = declare_level!(ctx, lvl.lvl, call(*, pos, lvl.shape), init)
    return lvl
end

function assemble_level!(ctx, lvl::VirtualDenseLevel, pos_start, pos_stop)
    qos_start = call(+, call(*, call(-, pos_start, lvl.Ti(1)), lvl.shape), 1)
    qos_stop = call(*, pos_stop, lvl.shape)
    assemble_level!(ctx, lvl.lvl, qos_start, qos_stop)
end

supports_reassembly(::VirtualDenseLevel) = true
function reassemble_level!(ctx, lvl::VirtualDenseLevel, pos_start, pos_stop)
    qos_start = call(+, call(*, call(-, pos_start, lvl.Ti(1)), lvl.shape), 1)
    qos_stop = call(*, pos_stop, lvl.shape)
    reassemble_level!(ctx, lvl.lvl, qos_start, qos_stop)
    lvl
end

function thaw_level!(ctx::AbstractCompiler, lvl::VirtualDenseLevel, pos)
    lvl.lvl = thaw_level!(ctx, lvl.lvl, call(*, pos, lvl.shape))
    return lvl
end

function freeze_level!(ctx::AbstractCompiler, lvl::VirtualDenseLevel, pos)
    lvl.lvl = freeze_level!(ctx, lvl.lvl, call(*, pos, lvl.shape))
    return lvl
end

struct DenseTraversal
    fbr
    subfiber_ctr
end

function unfurl(ctx, fbr::VirtualSubFiber{VirtualDenseLevel}, ext, mode, proto)
    unfurl(ctx, DenseTraversal(fbr, VirtualSubFiber), ext, mode, proto)
end
function unfurl(ctx, fbr::VirtualHollowSubFiber{VirtualDenseLevel}, ext, mode, proto)
    unfurl(
        ctx,
        DenseTraversal(fbr, (lvl, pos) -> VirtualHollowSubFiber(lvl, pos, fbr.dirty)),
        ext,
        mode,
        proto,
    )
end

function unfurl(
    ctx,
    trv::DenseTraversal,
    ext,
    mode,
    ::Union{
        typeof(defaultread),
        typeof(follow),
        typeof(defaultupdate),
        typeof(laminate),
        typeof(extrude),
    },
)
    (lvl, pos) = (trv.fbr.lvl, trv.fbr.pos)
    tag = lvl.tag
    Ti = lvl.Ti

    q = freshen(ctx, tag, :_q)

    Lookup(;
        body=(ctx, i) -> Thunk(;
            preamble=quote
                $q = ($(ctx(pos)) - $(Ti(1))) * $(ctx(lvl.shape)) + $(ctx(i))
            end,
            body=(ctx) ->
                instantiate(ctx, trv.subfiber_ctr(lvl.lvl, value(q, lvl.Ti)), mode),
        ),
    )
end

function coalesce_level!(
    lvl::DenseLevel, global_fbr_map, local_fbr_map, task_map, factor, P, coalescent
)
    shape = lvl.shape
    global_fbr_map, local_fbr_map, task_map, factor = merge_dense(
        global_fbr_map, local_fbr_map, task_map, factor, shape, P
    )

    DenseLevel(
        coalesce_level!(lvl.lvl, global_fbr_map, local_fbr_map, task_map,
            factor, P, coalescent.lvl), lvl.shape)
end
