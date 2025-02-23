"""
    SeparateLevel{Lvl, [Val]}()

A subfiber of a Separate level is a separate tensor of type `Lvl`, in it's
own memory space.

Each sublevel is stored in a vector of type `Val` with `eltype(Val) = Lvl`.

```jldoctest
julia> tensor_tree(Tensor(Dense(Separate(Element(0.0))), [1, 2, 3]))
3-Tensor
└─ Dense [1:3]
   ├─ [1]: Pointer ->
   │  └─ 1.0
   ├─ [2]: Pointer ->
   │  └─ 2.0
   └─ [3]: Pointer ->
      └─ 3.0
```
"""
struct SeparateLevel{Lvl,Val} <: AbstractLevel
    lvl::Lvl
    val::Val
end
const Separate = SeparateLevel

#similar_level(lvl, level_fill_value(typeof(lvl)), level_eltype(typeof(lvl)), level_size(lvl)...)
SeparateLevel(lvl::Lvl) where {Lvl} = SeparateLevel(lvl, Lvl[])
Base.summary(::Separate{Lvl,Val}) where {Lvl,Val} = "Separate($(Lvl))"

function similar_level(
    lvl::Separate{Lvl,Val}, fill_value, eltype::Type, dims...
) where {Lvl,Val}
    SeparateLevel(similar_level(lvl.lvl, fill_value, eltype, dims...))
end

postype(::Type{<:Separate{Lvl,Val}}) where {Lvl,Val} = postype(Lvl)

function transfer(lvl::SeparateLevel, device, style)
    lvl_2 = transfer(lvl.lvl, device, style)
    val_2 = transfer(lvl.val, device, style)
    return SeparateLevel(lvl_2, val_2)
end

pattern!(lvl::SeparateLevel) = SeparateLevel(pattern!(lvl.lvl), map(pattern!, lvl.val))
function set_fill_value!(lvl::SeparateLevel, init)
    SeparateLevel(
        set_fill_value!(lvl.lvl, init), map(lvl_2 -> set_fill_value!(lvl_2, init), lvl.val)
    )
end
function Base.resize!(lvl::SeparateLevel, dims...)
    SeparateLevel(resize!(lvl.lvl, dims...), map(lvl_2 -> resize!(lvl_2, dims...), lvl.val))
end

function isstructequal(a::T, b::T) where {T<:Separate}
    all(isstructequal(x, y) for (x, y) in zip(a.val, b.val)) && isstructequal(a.lvl, b.lvl)
end

function Base.show(io::IO, lvl::SeparateLevel{Lvl,Val}) where {Lvl,Val}
    print(io, "Separate(")
    if get(io, :compact, false)
        print(io, "…")
    else
        show(io, lvl.lvl)
        print(io, ", ")
        show(io, lvl.val)
    end
    print(io, ")")
end

labelled_show(io::IO, ::SubFiber{<:SeparateLevel}) =
    print(io, "Pointer -> ")

function labelled_children(fbr::SubFiber{<:SeparateLevel})
    lvl = fbr.lvl
    pos = fbr.pos
    pos > length(lvl.val) && return []
    [LabelledTree(SubFiber(lvl.val[pos], 1))]
end

@inline level_ndims(::Type{<:SeparateLevel{Lvl,Val}}) where {Lvl,Val} = level_ndims(Lvl)
@inline level_size(lvl::SeparateLevel{Lvl,Val}) where {Lvl,Val} = level_size(lvl.lvl)
@inline level_axes(lvl::SeparateLevel{Lvl,Val}) where {Lvl,Val} = level_axes(lvl.lvl)
@inline level_eltype(::Type{SeparateLevel{Lvl,Val}}) where {Lvl,Val} = level_eltype(Lvl)
@inline level_fill_value(::Type{<:SeparateLevel{Lvl,Val}}) where {Lvl,Val} =
    level_fill_value(Lvl)
data_rep_level(::Type{<:SeparateLevel{Lvl,Val}}) where {Lvl,Val} = data_rep_level(Lvl)

function (fbr::SubFiber{<:SeparateLevel})(idxs...)
    q = fbr.pos
    return SubFiber(fbr.lvl.val[q], 1)(idxs...)
end

countstored_level(lvl::SeparateLevel, pos) = pos

mutable struct VirtualSeparateLevel <: AbstractVirtualLevel
    lvl  # stand in for the sublevel for virutal resize, etc.
    ex
    val
    Tv
    Lvl
    Val
end

postype(lvl::VirtualSeparateLevel) = postype(lvl.lvl)

function is_level_injective(ctx, lvl::VirtualSeparateLevel)
    [is_level_injective(ctx, lvl.lvl)..., true]
end
function is_level_atomic(ctx, lvl::VirtualSeparateLevel)
    (below, atomic) = is_level_atomic(ctx, lvl.lvl)
    return ([below; [atomic]], atomic)
end
function is_level_concurrent(ctx, lvl::VirtualSeparateLevel)
    (data, _) = is_level_concurrent(ctx, lvl.lvl)
    return (data, true)
end

function lower(ctx::AbstractCompiler, lvl::VirtualSeparateLevel, ::DefaultStyle)
    quote
        $SeparateLevel{$(lvl.Lvl),$(lvl.Val)}($(ctx(lvl.lvl)), $(lvl.val))
    end
end

function virtualize(ctx, ex, ::Type{SeparateLevel{Lvl,Val}}, tag=:lvl) where {Lvl,Val}
    sym = freshen(ctx, tag)
    val = freshen(ctx, tag, :_val)

    push_preamble!(
        ctx,
        quote
            $sym = $ex
            $val = $ex.val
        end,
    )
    lvl_2 = virtualize(ctx, :($ex.lvl), Lvl, sym)
    VirtualSeparateLevel(lvl_2, sym, val, typeof(level_fill_value(Lvl)), Lvl, Val)
end

Base.summary(lvl::VirtualSeparateLevel) = "Separate($(lvl.Lvl))"

function virtual_level_resize!(ctx, lvl::VirtualSeparateLevel, dims...)
    (lvl.lvl = virtual_level_resize!(ctx, lvl.lvl, dims...); lvl)
end
virtual_level_size(ctx, lvl::VirtualSeparateLevel) = virtual_level_size(ctx, lvl.lvl)
virtual_level_eltype(lvl::VirtualSeparateLevel) = virtual_level_eltype(lvl.lvl)
virtual_level_fill_value(lvl::VirtualSeparateLevel) = virtual_level_fill_value(lvl.lvl)

function virtual_transfer_level(ctx, lvl::VirtualSeparateLevel, arch, style)
    # Need to move each pointer...
    val_2 = freshen(ctx, lvl.val)
    push_preamble!(
        ctx,
        quote
            $val_2 = $transfer($(lvl.val), $(ctx(arch)), $style)
        end,
    )
    lvl_2 = virtual_transfer_level(ctx, lvl.lvl, arch, style)
    VirtualSeparateLevel(lvl_2, lvl.ex, val_2, lvl.Tv, lvl.Lvl, lvl.Val)
end

function declare_level!(ctx, lvl::VirtualSeparateLevel, pos, init)
    #declare_level!(lvl.lvl, ctx_2, literal(1), init)
    return lvl
end

function assemble_level!(ctx, lvl::VirtualSeparateLevel, pos_start, pos_stop)
    pos_start = cache!(ctx, :pos_start, simplify(ctx, pos_start))
    pos_stop = cache!(ctx, :pos_stop, simplify(ctx, pos_stop))
    pos = freshen(ctx, :pos)
    sym = freshen(ctx, :pointer_to_lvl)
    push_preamble!(
        ctx,
        quote
            Finch.resize_if_smaller!($(lvl.val), $(ctx(pos_stop)))
            for $pos in ($(ctx(pos_start))):($(ctx(pos_stop)))
                $sym = Finch.similar_level(
                    $(lvl.ex).lvl,
                    Finch.level_fill_value(typeof($(lvl.ex).lvl)),
                    Finch.level_eltype(typeof($(lvl.ex).lvl)),
                    $(map(ctx, map(getstop, virtual_level_size(ctx, lvl)))...),
                )
                $(
                    contain(ctx) do ctx_2
                        lvl_2 = virtualize(ctx_2.code, sym, lvl.Lvl, sym)
                        lvl_2 = declare_level!(
                            ctx_2,
                            lvl_2,
                            literal(0),
                            literal(virtual_level_fill_value(lvl_2)),
                        )
                        lvl_2 = virtual_level_resize!(
                            ctx_2, lvl_2, virtual_level_size(ctx_2, lvl.lvl)...
                        )
                        push_preamble!(
                            ctx_2, assemble_level!(ctx_2, lvl_2, literal(1), literal(1))
                        )
                        contain(ctx_2) do ctx_3
                            lvl_2 = freeze_level!(ctx_3, lvl_2, literal(1))
                            :($(lvl.val)[$(ctx_3(pos))] = $(ctx_3(lvl_2)))
                        end
                    end
                )
            end
        end,
    )
    lvl
end

supports_reassembly(::VirtualSeparateLevel) = true
function reassemble_level!(ctx, lvl::VirtualSeparateLevel, pos_start, pos_stop)
    pos_start = cache!(ctx, :pos_start, simplify(ctx, pos_start))
    pos_stop = cache!(ctx, :pos_stop, simplify(ctx, pos_stop))
    pos = freshen(ctx, :pos)
    push_preamble!(
        ctx,
        quote
            for $idx in ($(ctx(pos_start))):($(ctx(pos_stop)))
                $(
                    contain(ctx) do ctx_2
                        lvl_2 = virtualize(ctx_2.code, :($(lvl.val)[$idx]), lvl.Lvl, sym)
                        push_preamble!(
                            ctx_2, assemble_level!(ctx_2, lvl_2, literal(1), literal(1))
                        )
                        lvl_2 = declare_level!(ctx_2, lvl_2, literal(1), init)
                        contain(ctx_2) do ctx_3
                            lvl_2 = freeze_level!(ctx_3, lvl_2, literal(1))
                            :($(lvl.val)[$(ctx_3(pos))] = $(ctx_3(lvl_2)))
                        end
                    end
                )
            end
        end,
    )
    lvl
end

function freeze_level!(ctx, lvl::VirtualSeparateLevel, pos)
    return lvl
end

function thaw_level!(ctx::AbstractCompiler, lvl::VirtualSeparateLevel, pos)
    return lvl
end

function instantiate(ctx, fbr::VirtualSubFiber{VirtualSeparateLevel}, mode)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    sym = freshen(ctx, :pointer_to_lvl)
    if mode.kind === reader
        isnulltest = freshen(ctx, tag, :_nulltest)
        Vf = level_fill_value(lvl.Lvl)
        val = freshen(ctx, lvl.ex, :_val)
        return Thunk(;
            body=(ctx) -> begin
                lvl_2 = virtualize(ctx.code, :($(lvl.val)[$(ctx(pos))]), lvl.Lvl, sym)
                Provenance(;
                    path=SubLevelOf(Parent()),
                    body=instantiate(ctx, VirtualSubFiber(lvl_2, literal(1)), mode),
                )
            end,
        )
    else
        return Thunk(;
            body=(ctx) -> begin
                lvl_2 = virtualize(ctx.code, :($(lvl.val)[$(ctx(pos))]), lvl.Lvl, sym)
                lvl_2 = thaw_level!(ctx, lvl_2, literal(1))
                push_preamble!(ctx, assemble_level!(ctx, lvl_2, literal(1), literal(1)))
                res = Provenance(;
                    path=SubLevelOf(Parent()),
                    body=instantiate(ctx, VirtualSubFiber(lvl_2, literal(1)), mode),
                )
                push_epilogue!(ctx,
                    contain(ctx) do ctx_2
                        lvl_2 = freeze_level!(ctx_2, lvl_2, literal(1))
                        :($(lvl.val)[$(ctx_2(pos))] = $(ctx_2(lvl_2)))
                    end,
                )
                res
            end,
        )
    end
end

function instantiate(ctx, fbr::VirtualHollowSubFiber{VirtualSeparateLevel}, mode)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    sym = freshen(ctx, :pointer_to_lvl)
    @assert mode.kind === updater

    return Thunk(;
        body=(ctx) -> begin
            lvl_2 = virtualize(ctx.code, :($(lvl.val)[$(ctx(pos))]), lvl.Lvl, sym)
            lvl_2 = thaw_level!(ctx, lvl_2, literal(1))
            push_preamble!(ctx, assemble_level!(ctx, lvl_2, literal(1), literal(1)))
            res = Provenance(;
                path=SubLevelOf(Parent()),
                body=instantiate(
                    ctx, VirtualHollowSubFiber(lvl_2, literal(1), fbr.dirty), mode
                ),
            )
            push_epilogue!(ctx,
                contain(ctx) do ctx_2
                    lvl_2 = freeze_level!(ctx_2, lvl_2, literal(1))
                    :($(lvl.val)[$(ctx_2(pos))] = $(ctx_2(lvl_2)))
                end,
            )
            res
        end,
    )
end
