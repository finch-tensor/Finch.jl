"""
    AtomicElementLevel{Vf, [Tv=typeof(Vf)], [Tp=Int], [Val]}()

Like an [`ElementLevel`](@ref), but updates to the level are performed atomically.

```jldoctest
julia> tensor_tree(Tensor(Dense(AtomicElement(0.0)), [1, 2, 3]))
3-Tensor
└─ Dense [1:3]
   ├─ [1]: 1.0
   ├─ [2]: 2.0
   └─ [3]: 3.0
```
"""
struct AtomicElementLevel{Vf,Tv,Tp,Val} <: AbstractLevel
    val::Val
end
const AtomicElement = AtomicElementLevel

function AtomicElementLevel(d, args...)
    isbits(d) || throw(ArgumentError("Finch currently only supports isbits defaults"))
    AtomicElementLevel{d}(args...)
end
AtomicElementLevel{Vf}() where {Vf} = AtomicElementLevel{Vf,typeof(Vf)}()
AtomicElementLevel{Vf}(val::Val) where {Vf,Val} = AtomicElementLevel{Vf,eltype(Val)}(val)
AtomicElementLevel{Vf,Tv}(args...) where {Vf,Tv} = AtomicElementLevel{Vf,Tv,Int}(args...)
AtomicElementLevel{Vf,Tv,Tp}() where {Vf,Tv,Tp} = AtomicElementLevel{Vf,Tv,Tp}(Tv[])

function AtomicElementLevel{Vf,Tv,Tp}(val::Val) where {Vf,Tv,Tp,Val}
    AtomicElementLevel{Vf,Tv,Tp,Val}(val)
end

Base.summary(::AtomicElement{Vf}) where {Vf} = "AtomicElement($(Vf))"

function similar_level(
    ::AtomicElementLevel{Vf,Tv,Tp}, fill_value, eltype::Type, ::Vararg
) where {Vf,Tv,Tp}
    AtomicElementLevel{fill_value,eltype,Tp}()
end

postype(::Type{<:AtomicElementLevel{Vf,Tv,Tp}}) where {Vf,Tv,Tp} = Tp

function transfer(lvl::AtomicElementLevel{Vf,Tv,Tp}, device, style) where {Vf,Tv,Tp}
    return AtomicElementLevel{Vf,Tv,Tp}(transfer(device, lvl.val))
end

pattern!(lvl::AtomicElementLevel{Vf,Tv,Tp}) where {Vf,Tv,Tp} =
    Pattern{Tp}()
function set_fill_value!(lvl::AtomicElementLevel{Vf,Tv,Tp}, init) where {Vf,Tv,Tp}
    AtomicElementLevel{init,Tv,Tp}(lvl.val)
end
Base.resize!(lvl::AtomicElementLevel) = lvl

function Base.show(io::IO, lvl::AtomicElementLevel{Vf,Tv,Tp,Val}) where {Vf,Tv,Tp,Val}
    print(io, "AtomicElement{")
    show(io, Vf)
    print(io, ", $Tv, $Tp}(")
    if get(io, :compact, false)
        print(io, "…")
    else
        show(io, lvl.val)
    end
    print(io, ")")
end

labelled_show(io::IO, fbr::SubFiber{<:AtomicElementLevel}) =
    print(io, fbr.lvl.val[fbr.pos])

@inline level_ndims(::Type{<:AtomicElementLevel}) = 0
@inline level_size(::AtomicElementLevel) = ()
@inline level_axes(::AtomicElementLevel) = ()
@inline level_eltype(::Type{<:AtomicElementLevel{Vf,Tv}}) where {Vf,Tv} = Tv
@inline level_fill_value(::Type{<:AtomicElementLevel{Vf}}) where {Vf} = Vf
data_rep_level(::Type{<:AtomicElementLevel{Vf,Tv}}) where {Vf,Tv} = ElementData(Vf, Tv)

(fbr::Tensor{<:AtomicElementLevel})() = SubFiber(fbr.lvl, 1)()
function (fbr::SubFiber{<:AtomicElementLevel})()
    q = fbr.pos
    return fbr.lvl.val[q]
end

countstored_level(lvl::AtomicElementLevel, pos) = pos

mutable struct VirtualAtomicElementLevel <: AbstractVirtualLevel
    tag
    Vf
    Tv
    Tp
    val
end

function reroot_set!(ctx::AbstractCompiler, lvl::VirtualAtomicElementLevel, diff)
    diff[lvl.tag] = lvl
end

function reroot_get(ctx::AbstractCompiler, lvl::VirtualAtomicElementLevel, diff)
    get(diff, lvl.tag, lvl)
end

is_level_injective(ctx, ::VirtualAtomicElementLevel) = []
is_level_atomic(ctx, lvl::VirtualAtomicElementLevel) = ([], false)
function is_level_concurrent(ctx, lvl::VirtualAtomicElementLevel)
    return ([], true)
end

function lower(ctx::AbstractCompiler, lvl::VirtualAtomicElementLevel, ::DefaultStyle)
    :(AtomicElementLevel{$(lvl.Vf),$(lvl.Tv),$(lvl.Tp)}($(lvl.val)))
end

function virtualize(
    ctx, ex, ::Type{AtomicElementLevel{Vf,Tv,Tp,Val}}, tag=:lvl
) where {Vf,Tv,Tp,Val}
    tag = freshen(ctx, tag)
    val = freshen(ctx, tag, :_val)
    push_preamble!(
        ctx,
        quote
            $tag = $ex
            $val = $tag.val
        end,
    )
    VirtualAtomicElementLevel(tag, Vf, Tv, Tp, val)
end

Base.summary(lvl::VirtualAtomicElementLevel) = "AtomicElement($(lvl.Vf))"

virtual_level_resize!(ctx, lvl::VirtualAtomicElementLevel) = lvl
virtual_level_size(ctx, ::VirtualAtomicElementLevel) = ()
virtual_level_ndims(ctx, lvl::VirtualAtomicElementLevel) = 0
virtual_level_eltype(lvl::VirtualAtomicElementLevel) = lvl.Tv
virtual_level_fill_value(lvl::VirtualAtomicElementLevel) = lvl.Vf

postype(lvl::VirtualAtomicElementLevel) = lvl.Tp

function declare_level!(ctx, lvl::VirtualAtomicElementLevel, pos, init)
    init == literal(lvl.Vf) || throw(
        FinchProtocolError(
            "Cannot initialize AtomicElement Levels to non-fill values (have $init expected $(lvl.Vf))"
        ),
    )
    lvl
end

function freeze_level!(ctx::AbstractCompiler, lvl::VirtualAtomicElementLevel, pos)
    push_preamble!(
        ctx,
        quote
            resize!($(lvl.val), $(ctx(pos)))
        end,
    )
    return lvl
end

thaw_level!(ctx::AbstractCompiler, lvl::VirtualAtomicElementLevel, pos) = lvl

function assemble_level!(ctx, lvl::VirtualAtomicElementLevel, pos_start, pos_stop)
    pos_start = cache!(ctx, :pos_start, simplify(ctx, pos_start))
    pos_stop = cache!(ctx, :pos_stop, simplify(ctx, pos_stop))
    quote
        Finch.resize_if_smaller!($(lvl.val), $(ctx(pos_stop)))
        Finch.fill_range!($(lvl.val), $(lvl.Vf), $(ctx(pos_start)), $(ctx(pos_stop)))
    end
end

supports_reassembly(::VirtualAtomicElementLevel) = true
function reassemble_level!(ctx, lvl::VirtualAtomicElementLevel, pos_start, pos_stop)
    pos_start = cache!(ctx, :pos_start, simplify(ctx, pos_start))
    pos_stop = cache!(ctx, :pos_stop, simplify(ctx, pos_stop))
    push_preamble!(
        ctx,
        quote
            Finch.fill_range!($(lvl.val), $(lvl.Vf), $(ctx(pos_start)), $(ctx(pos_stop)))
        end,
    )
    lvl
end

function distribute_level(
    ctx::AbstractCompiler, lvl::VirtualAtomicElementLevel, arch, style
)
    VirtualAtomicElementLevel(
        lvl.tag,
        lvl.Vf,
        lvl.Tv,
        lvl.Tp,
        distribute_buffer(ctx, lvl.Tv, lvl.val, style),
    )
end

function instantiate(ctx, fbr::VirtualSubFiber{VirtualAtomicElementLevel}, mode)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    if mode.kind === reader
        val = freshen(ctx.code, lvl.tag, :_val)
        return Thunk(;
            preamble=quote
                $val = $(lvl.val)[$(ctx(pos))]
            end,
            body=(ctx) -> VirtualScalar(nothing, nothing, lvl.Tv, lvl.Vf, gensym(), val),
        )
    else
        return fbr
    end
end

function instantiate(ctx, fbr::VirtualHollowSubFiber{VirtualAtomicElementLevel}, mode)
    fbr
end

function lower_assign(ctx, fbr::VirtualSubFiber{VirtualAtomicElementLevel}, mode, op, rhs)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    op = ctx(op)
    rhs = ctx(rhs)
    device = ctx(get_device(get_task(ctx)))
    :(Finch.atomic_modify!($device, $(lvl.val), $(ctx(pos)), $op, $rhs))
end

function lower_assign(
    ctx, fbr::VirtualHollowSubFiber{VirtualAtomicElementLevel}, mode, op, rhs
)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    push_preamble!(
        ctx,
        quote
            $(fbr.dirty) = true
        end,
    )
    op = ctx(op)
    rhs = ctx(rhs)
    device = ctx(get_device(get_task(ctx)))
    :(Finch.atomic_modify!($device, $(lvl.val), $(ctx(pos)), $op, $rhs))
end
