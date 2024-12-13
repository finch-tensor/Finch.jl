"""
    ShardedLevel{Lvl, [Val]}()

Each subfiber of a Sharded level is stored in a thread-local tensor of type
`Lvl`, in a thread-local memory space.

Each sublevel is stored in a vector of type `Val` with `eltype(Val) = Lvl`.

```jldoctest
julia> tensor_tree(Tensor(Dense(Sharded(Element(0.0))), [1, 2, 3]))
3-Tensor
└─ Dense [1:3]
   ├─ [1]: Shard ->
   │  └─ 1.0
   ├─ [2]: Shard ->
   │  └─ 2.0
   └─ [3]: Shard ->
      └─ 3.0
```
"""
struct ShardedLevel{Device, Lvl, Ptr, Task, Val} <: AbstractLevel
    device::Device
    lvl::Lvl
    ptr::Ptr
    task::Task
    val::Val
end
const Sharded = ShardedLevel

ShardedLevel(device::Device, lvl::Lvl) where {Device, Lvl} =
    ShardedLevel{Device}(device, lvl, postype(lvl)[], postype(lvl)[], typeof(lvl)[])

ShardedLevel(device::Device, lvl::Lvl, ptr::Ptr, task::Task, val::Val) where {Device, Lvl, Ptr, Task, Val} =
    ShardedLevel{Device, Lvl, Ptr, Task, Val}(device, lvl, ptr, task, val)

Base.summary(::Sharded{Device, Lvl, Ptr, Task, Val}) where {Device, Lvl, Ptr, Task, Val} = "Sharded($(Lvl))"

similar_level(lvl::Sharded{Device, Lvl, Ptr, Task, Val}, fill_value, eltype::Type, dims...) where {Device, Lvl, Ptr, Task, Val} =
    ShardedLevel(lvl, similar_level(lvl.lvl, fill_value, eltype, dims...))

postype(::Type{<:Sharded{Device, Lvl, Ptr, Task, Val}}) where {Device, Lvl, Ptr, Task, Val} = postype(Lvl)

function moveto(lvl::ShardedLevel, device)
    lvl_2 = moveto(lvl.lvl, device)
    ptr_2 = moveto(lvl.ptr, device)
    task_2 = moveto(lvl.task, device)
    val_2 = moveto(lvl.val, device)
    return ShardedLevel(lvl_2, ptr_2, task_2, val_2)
end

pattern!(lvl::ShardedLevel) = ShardedLevel(pattern!(lvl.lvl), lvl.ptr, lvl.task, map(pattern!, lvl.val))
set_fill_value!(lvl::ShardedLevel, init) = ShardedLevel(set_fill_value!(lvl.lvl, init), lvl.ptr, lvl.task, map(lvl_2 -> set_fill_value!(lvl_2, init), lvl.val))
Base.resize!(lvl::ShardedLevel, dims...) = ShardedLevel(resize!(lvl.lvl, dims...), lvl.ptr, lvl.task, map(lvl_2 -> resize!(lvl_2, dims...), lvl.val))

function Base.show(io::IO, lvl::ShardedLevel{Device, Lvl, Ptr, Task, Val}) where {Device, Lvl, Ptr, Task, Val}
    print(io, "Sharded(")
    if get(io, :compact, false)
        print(io, "…")
    else
        show(io, lvl.lvl)
        print(io, ", ")
        show(io, lvl.ptr)
        print(io, ", ")
        show(io, lvl.task)
        print(io, ", ")
        show(io, lvl.val)
    end
    print(io, ")")
end

function labelled_show(io::IO, fbr::SubFiber{<:ShardedLevel})
    (lvl, pos) = (fbr.lvl, fbr.pos)
    print(io, "shard($(lvl.task[pos])) -> ")
end

function labelled_children(fbr::SubFiber{<:ShardedLevel})
    lvl = fbr.lvl
    pos = fbr.pos
    pos > length(lvl.val) && return []
    [LabelledTree(SubFiber(lvl.val[lvl.task[pos]], lvl.ptr[pos]))]
end

@inline level_ndims(::Type{<:ShardedLevel{Device, Lvl, Ptr, Task, Val}}) where {Device, Lvl, Ptr, Task, Val} = level_ndims(Lvl)
@inline level_size(lvl::ShardedLevel{Device, Lvl, Ptr, Task, Val}) where {Device, Lvl, Ptr, Task, Val} = level_size(lvl.lvl)
@inline level_axes(lvl::ShardedLevel{Device, Lvl, Ptr, Task, Val}) where {Device, Lvl, Ptr, Task, Val} = level_axes(lvl.lvl)
@inline level_eltype(::Type{ShardedLevel{Device, Lvl, Ptr, Task, Val}}) where {Device, Lvl, Ptr, Task, Val} = level_eltype(Lvl)
@inline level_fill_value(::Type{<:ShardedLevel{Device, Lvl, Ptr, Task, Val}}) where {Device, Lvl, Ptr, Task, Val} = level_fill_value(Lvl)

function (fbr::SubFiber{<:ShardedLevel})(idxs...)
    q = fbr.pos
    return SubFiber(fbr.lvl.val[q], 1)(idxs...)
end

countstored_level(lvl::ShardedLevel, pos) = pos

mutable struct VirtualShardedLevel <: AbstractVirtualLevel
    lvl  # stand-in for the sublevel for virtual resize, etc.
    ex
    val
    Tv
    Lvl
    Ptr
    Task
    Val
end

postype(lvl::VirtualShardedLevel) = postype(lvl.lvl)

is_level_injective(ctx, lvl::VirtualShardedLevel) = [is_level_injective(ctx, lvl.lvl)..., true]
function is_level_atomic(ctx, lvl::VirtualShardedLevel)
    (below, atomic) = is_level_atomic(ctx, lvl.lvl)
    return ([below; [atomic]], atomic)
end
function is_level_concurrent(ctx, lvl::VirtualShardedLevel)
    (data, _) = is_level_concurrent(ctx, lvl.lvl)
    return (data, true)
end

function lower(ctx::AbstractCompiler, lvl::VirtualShardedLevel, ::DefaultStyle)
    quote
        $ShardedLevel{$(lvl.Lvl), $(lvl.Ptr), $(lvl.Task), $(lvl.Val)}($(ctx(lvl.lvl)), $(lvl.val))
    end
end

function virtualize(ctx, ex, ::Type{ShardedLevel{Device, Lvl, Ptr, Task, Val}}, tag=:lvl) where {Device, Lvl, Ptr, Task, Val}
    sym = freshen(ctx, tag)
    ptr = freshen(ctx, tag, :_ptr)
    task = freshen(ctx, tag, :_task)
    val = freshen(ctx, tag, :_val)

    push_preamble!(ctx, quote
              $sym = $ex
              $ptr = $ex.ptr
              $task = $ex.task
              $val = $ex.val
    end)
    lvl_2 = virtualize(ctx, :($ex.lvl), Lvl, sym)
    VirtualShardedLevel(lvl_2, sym, val, typeof(level_fill_value(Lvl)), Lvl, Ptr, Task, Val)
end

Base.summary(lvl::VirtualShardedLevel) = "Sharded($(lvl.Lvl))"

virtual_level_resize!(ctx, lvl::VirtualShardedLevel, dims...) = (lvl.lvl = virtual_level_resize!(ctx, lvl.lvl, dims...); lvl)
virtual_level_size(ctx, lvl::VirtualShardedLevel) = virtual_level_size(ctx, lvl.lvl)
virtual_level_eltype(lvl::VirtualShardedLevel) = virtual_level_eltype(lvl.lvl)
virtual_level_fill_value(lvl::VirtualShardedLevel) = virtual_level_fill_value(lvl.lvl)

function virtual_moveto_level(ctx, lvl::VirtualShardedLevel, arch)
    val_2 = freshen(ctx, lvl.val)
    push_preamble!(ctx, quote
            $val_2 = $(lvl.val)
            $(lvl.val) = $moveto($(lvl.val), $(ctx(arch)))
        end)
    push_epilogue!(ctx, quote
            $(lvl.val) = $val_2
        end)
    virtual_moveto_level(ctx, lvl.lvl, arch)
end

function declare_level!(ctx, lvl::VirtualShardedLevel, pos, init)
    return lvl
end

function assemble_level!(ctx, lvl::VirtualShardedLevel, pos_start, pos_stop)
    pos_start = cache!(ctx, :pos_start, simplify(ctx, pos_start))
    pos_stop = cache!(ctx, :pos_stop, simplify(ctx, pos_stop))
    pos = freshen(ctx, :pos)
    sym = freshen(ctx, :pointer_to_lvl)
    push_preamble!(ctx, quote
        Finch.resize_if_smaller!($(lvl.val), $(ctx(pos_stop)))
        for $pos in $(ctx(pos_start)):$(ctx(pos_stop))
            $sym = Finch.similar_level(
                $(lvl.ex).lvl,
                Finch.level_fill_value(typeof($(lvl.ex).lvl)),
                Finch.level_eltype(typeof($(lvl.ex).lvl)),
                $(map(ctx, map(getstop, virtual_level_size(ctx, lvl)))...)
            )
            $(contain(ctx) do ctx_2
                lvl_2 = virtualize(ctx_2.code, sym, lvl.Lvl, sym)
                lvl_2 = declare_level!(ctx_2, lvl_2, literal(0), literal(virtual_level_fill_value(lvl_2)))
                lvl_2 = virtual_level_resize!(ctx_2, lvl_2, virtual_level_size(ctx_2, lvl.lvl)...)
                push_preamble!(ctx_2, assemble_level!(ctx_2, lvl_2, literal(1), literal(1)))
                contain(ctx_2) do ctx_3
                    lvl_2 = freeze_level!(ctx_3, lvl_2, literal(1))
                    :($(lvl.val)[$(ctx_3(pos))] = $(ctx_3(lvl_2)))
                end
            end)
        end
    end)
    lvl
end

supports_reassembly(::VirtualShardedLevel) = true

function reassemble_level!(ctx, lvl::VirtualShardedLevel, pos_start, pos_stop)
    pos_start = cache!(ctx, :pos_start, simplify(ctx, pos_start))
    pos_stop = cache!(ctx, :pos_stop, simplify(ctx, pos_stop))
    pos = freshen(ctx, :pos)
    push_preamble!(ctx, quote
        for $idx in $(ctx(pos_start)):$(ctx(pos_stop))
            $(contain(ctx) do ctx_2
                lvl_2 = virtualize(ctx_2.code, :($(lvl.val)[$idx]), lvl.Lvl, sym)
                push_preamble!(ctx_2, assemble_level!(ctx_2, lvl_2, literal(1), literal(1)))
                lvl_2 = declare_level!(ctx_2, lvl_2, literal(1), init)
                contain(ctx_2) do ctx_3
                    lvl_2 = freeze_level!(ctx_3, lvl_2, literal(1))
                    :($(lvl.val)[$(ctx_3(pos))] = $(ctx_3(lvl_2)))
                end
            end)
        end
    end)
    lvl
end

function freeze_level!(ctx, lvl::VirtualShardedLevel, pos)
    return lvl
end

function thaw_level!(ctx::AbstractCompiler, lvl::VirtualShardedLevel, pos)
    return lvl
end

function instantiate(ctx, fbr::VirtualSubFiber{VirtualShardedLevel}, mode)
    if mode.kind === reader
        (lvl, pos) = (fbr.lvl, fbr.pos)
        tag = lvl.ex
        isnulltest = freshen(ctx, tag, :_nulltest)
        Vf = level_fill_value(lvl.Lvl)
        sym = freshen(ctx, :pointer_to_lvl)
        val = freshen(ctx, lvl.ex, :_val)
        return Thunk(
            body = (ctx) -> begin
                lvl_2 = virtualize(ctx.code, :($(lvl.val)[$(ctx(pos))]), lvl.Lvl, sym)
                instantiate(ctx, VirtualSubFiber(lvl_2, literal(1)), mode)
            end,
        )
    else
        (lvl, pos) = (fbr.lvl, fbr.pos)
        tag = lvl.ex
        sym = freshen(ctx, :pointer_to_lvl)

        return Thunk(
            body = (ctx) -> begin
                lvl_2 = virtualize(ctx.code, :($(lvl.val)[$(ctx(pos))]), lvl.Lvl, sym)
                lvl_2 = thaw_level!(ctx, lvl_2, literal(1))
                push_preamble!(ctx, assemble_level!(ctx, lvl_2, literal(1), literal(1)))
                res = instantiate(ctx, VirtualSubFiber(lvl_2, literal(1)), mode)
                push_epilogue!(ctx,
                    contain(ctx) do ctx_2
                        lvl_2 = freeze_level!(ctx_2, lvl_2, literal(1))
                        :($(lvl.val)[$(ctx_2(pos))] = $(ctx_2(lvl_2)))
                    end
                )
                res
            end
        )
    end
end

function instantiate(ctx, fbr::VirtualHollowSubFiber{VirtualShardedLevel}, mode)
    @assert mode.kind === updater
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    sym = freshen(ctx, :pointer_to_lvl)

    return Thunk(
        body = (ctx) -> begin
            lvl_2 = virtualize(ctx.code, :($(lvl.val)[$(ctx(pos))]), lvl.Lvl, sym)
            lvl_2 = thaw_level!(ctx, lvl_2, literal(1))
            push_preamble!(ctx, assemble_level!(ctx, lvl_2, literal(1), literal(1)))
            res = instantiate(ctx, VirtualHollowSubFiber(lvl_2, literal(1), fbr.dirty), mode)
            push_epilogue!(ctx,
                contain(ctx) do ctx_2
                    lvl_2 = freeze_level!(ctx_2, lvl_2, literal(1))
                    :($(lvl.val)[$(ctx_2(pos))] = $(ctx_2(lvl_2)))
                end
            )
            res
            end
        )
    end
end