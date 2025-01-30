"""
    ShardLevel{Lvl, [Val]}()

Each subfiber of a Shard level is stored in a thread-local tensor of type
`Lvl`, in a thread-local memory space.

Each sublevel is stored in a vector of type `Val` with `eltype(Val) = Lvl`.

```jldoctest
julia> tensor_tree(Tensor(Dense(Shard(Element(0.0))), [1, 2, 3]))
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
struct ShardLevel{Device, Lvl, Ptr, Task, Val} <: AbstractLevel
    device::Device
    lvl::Lvl
    ptr::Ptr
    task::Task
    val::Val
end
const Shard = ShardLevel

ShardLevel(device::Device, lvl::Lvl) where {Device, Lvl} =
    ShardLevel{Device}(device, lvl, postype(lvl)[], postype(lvl)[], moveto(lvl, device)) #TODO scatterto?

ShardLevel{Device}(device, lvl::Lvl, ptr::Ptr, task::Task, val::Val) where {Device, Lvl, Ptr, Task, Val} =
    ShardLevel{Device, Lvl, Ptr, Task, Val}(device, lvl, ptr, task, val)

Base.summary(::Shard{Device, Lvl, Ptr, Task, Val}) where {Device, Lvl, Ptr, Task, Val} = "Shard($(Lvl))"

similar_level(lvl::Shard{Device, Lvl, Ptr, Task, Val}, fill_value, eltype::Type, dims...) where {Device, Lvl, Ptr, Task, Val} =
    ShardLevel(lvl.device, similar_level(lvl.lvl, fill_value, eltype, dims...))

postype(::Type{<:Shard{Device, Lvl, Ptr, Task, Val}}) where {Device, Lvl, Ptr, Task, Val} = postype(Lvl)

function moveto(lvl::ShardLevel, device)
    lvl_2 = moveto(lvl.lvl, device)
    ptr_2 = moveto(lvl.ptr, device)
    task_2 = moveto(lvl.task, device)
    return ShardLevel(lvl_2, ptr_2, task_2, val_2)
end

pattern!(lvl::ShardLevel) = ShardLevel(pattern!(lvl.lvl), lvl.ptr, lvl.task, map(pattern!, lvl.val))
set_fill_value!(lvl::ShardLevel, init) = ShardLevel(set_fill_value!(lvl.lvl, init), lvl.ptr, lvl.task, map(lvl_2 -> set_fill_value!(lvl_2, init), lvl.val))
Base.resize!(lvl::ShardLevel, dims...) = ShardLevel(resize!(lvl.lvl, dims...), lvl.ptr, lvl.task, map(lvl_2 -> resize!(lvl_2, dims...), lvl.val))

function Base.show(io::IO, lvl::ShardLevel{Device, Lvl, Ptr, Task, Val}) where {Device, Lvl, Ptr, Task, Val}
    print(io, "Shard(")
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

function labelled_show(io::IO, fbr::SubFiber{<:ShardLevel})
    (lvl, pos) = (fbr.lvl, fbr.pos)
    print(io, "shard($(lvl.task[pos])) -> ")
end

function labelled_children(fbr::SubFiber{<:ShardLevel})
    lvl = fbr.lvl
    pos = fbr.pos
    pos > length(lvl.val) && return []
    [LabelledTree(SubFiber(lvl.val[lvl.task[pos]], lvl.ptr[pos]))]
end

@inline level_ndims(::Type{<:ShardLevel{Device, Lvl, Ptr, Task, Val}}) where {Device, Lvl, Ptr, Task, Val} = level_ndims(Lvl)
@inline level_size(lvl::ShardLevel{Device, Lvl, Ptr, Task, Val}) where {Device, Lvl, Ptr, Task, Val} = level_size(lvl.lvl)
@inline level_axes(lvl::ShardLevel{Device, Lvl, Ptr, Task, Val}) where {Device, Lvl, Ptr, Task, Val} = level_axes(lvl.lvl)
@inline level_eltype(::Type{ShardLevel{Device, Lvl, Ptr, Task, Val}}) where {Device, Lvl, Ptr, Task, Val} = level_eltype(Lvl)
@inline level_fill_value(::Type{<:ShardLevel{Device, Lvl, Ptr, Task, Val}}) where {Device, Lvl, Ptr, Task, Val} = level_fill_value(Lvl)

function (fbr::SubFiber{<:ShardLevel})(idxs...)
    q = fbr.pos
    return SubFiber(fbr.lvl.val[q], 1)(idxs...)
end

countstored_level(lvl::ShardLevel, pos) = pos

mutable struct VirtualShardLevel <: AbstractVirtualLevel
    device
    lvl  # stand-in for the sublevel for virtual resize, etc.
    ex
    ptr
    task
    val
    Tv
    Device
    Lvl
    Ptr
    Task
    Val
end

postype(lvl::VirtualShardLevel) = postype(lvl.lvl)

is_level_injective(ctx, lvl::VirtualShardLevel) = [is_level_injective(ctx, lvl.lvl)..., true]
function is_level_atomic(ctx, lvl::VirtualShardLevel)
    (below, atomic) = is_level_atomic(ctx, lvl.lvl)
    return ([below; [atomic]], atomic)
end
function is_level_concurrent(ctx, lvl::VirtualShardLevel)
    (data, _) = is_level_concurrent(ctx, lvl.lvl)
    return (data, true)
end

function lower(ctx::AbstractCompiler, lvl::VirtualShardLevel, ::DefaultStyle)
    quote
        $ShardLevel{$(lvl.Lvl), $(lvl.Ptr), $(lvl.Task), $(lvl.Val)}($(ctx(lvl.lvl)), $(lvl.val))
    end
end

function virtualize(ctx, ex, ::Type{ShardLevel{Device, Lvl, Ptr, Task, Val}}, tag=:lvl) where {Device, Lvl, Ptr, Task, Val}
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
    device_2 = virtualize(ctx, :($ex.device), Device, sym)
    lvl_2 = virtualize(ctx, :($ex.lvl), Lvl, sym)
    VirtualShardLevel(device_2, lvl_2, sym, ptr, task, val, typeof(level_fill_value(Lvl)), Device, Lvl, Ptr, Task, Val)
end

Base.summary(lvl::VirtualShardLevel) = "Shard($(lvl.Lvl))"

virtual_level_resize!(ctx, lvl::VirtualShardLevel, dims...) = (lvl.lvl = virtual_level_resize!(ctx, lvl.lvl, dims...); lvl)
virtual_level_size(ctx, lvl::VirtualShardLevel) = virtual_level_size(ctx, lvl.lvl)
virtual_level_eltype(lvl::VirtualShardLevel) = virtual_level_eltype(lvl.lvl)
virtual_level_fill_value(lvl::VirtualShardLevel) = virtual_level_fill_value(lvl.lvl)

function virtual_moveto_level(ctx, lvl::VirtualShardLevel, arch)
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

function declare_level!(ctx, lvl::VirtualShardLevel, pos, init)
    push_preamble!(ctx, 
        virtual_parallel_region(ctx, lvl.device) do ctx_2
            lvl_2 = virtualize(ctx_2, :($(lvl.ex).val[$(ctx_2(get_task_num(get_task(ctx_2))))]), lvl.Lvl) #TODO should this virtualize the eltype of Val?
            declare_level!(ctx_2, lvl_2, literal(1), init)
        end
    )
    lvl
end

"""
assemble:
    mapping is pos -> task, ptr. task says which task has it, ptr says which position in that task has it.

read:
    read from pos to task, ptr. simple.

write:
    allocate something for this task on that position, assemble on the task itself on demand. Complain if the task is wrong.

The outer level needs to be concurrent, like denselevel.
"""
function assemble_level!(ctx, lvl::VirtualShardLevel, pos_start, pos_stop)
    pos_start = cache!(ctx, :pos_start, simplify(ctx, pos_start))
    pos_stop = cache!(ctx, :pos_stop, simplify(ctx, pos_stop))
    pos = freshen(ctx, :pos)
    sym = freshen(ctx, :pointer_to_lvl)
    push_preamble!(ctx, quote
        Finch.resize_if_smaller!($(lvl.task), $(ctx(pos_stop)))
        Finch.resize_if_smaller!($(lvl.ptr), $(ctx(pos_stop)))
        Finch.fill_range!($(lvl.task), $(ctx(pos_start)), $(ctx(pos_stop)), 0)
    end)
    lvl
end

supports_reassembly(::VirtualShardLevel) = false

"""
these two are no-ops, we insteaed do these on instantiate
"""
function freeze_level!(ctx, lvl::VirtualShardLevel, pos)
    return lvl
end

function thaw_level!(ctx::AbstractCompiler, lvl::VirtualShardLevel, pos)
    return lvl
end

function instantiate(ctx, fbr::VirtualSubFiber{VirtualShardLevel}, mode)
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

#we need some sort of localization step at the start of a parallel region whereby we can thaw the shart level

"""
assemble:
    mapping is pos -> task, ptr. task says which task has it, ptr says which position in that task has it.

read:
    read from pos to task, ptr. simple.

write:
    allocate something for this task on that position, assemble on the task itself on demand. Complain if the task is wrong.

The outer level needs to be concurrent, like denselevel.
"""
function instantiate(ctx, fbr::VirtualHollowSubFiber{VirtualShardLevel}, mode)
    @assert mode.kind === updater
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.ex
    sym = freshen(ctx, :pointer_to_lvl)

    task = freshen(ctx, tag, :_task)

    return Thunk(
        preamble = quote
            $task = $(lvl.task)[$(ctx(pos))]
            if task == 0
                $(lvl.task)[$(ctx(pos))] = $(gettasknum(ctx))
                qos = local_qos_fill
                if $(lvl.local_qos_fill) > $(lvl.local_qos_stop)
                    $local_qos_stop = max($local_qos_stop << 1, 1)
                    $(contain(ctx_2->assemble_level!(ctx_2, lvl.lvl, value(qos_fill, Tp), value(qos_stop, Tp)), ctx))
                end
            else
                qos = $(lvl.ptr)[$(ctx(pos))]
                qos_stop = $(lvl.local_qos_stop)
                #only in safe mode, we check if task == $(gettasknum(ctx)) and if not error("Task mismatch in ShardLevel")
            end
            dirty = true
        end,
        body = (ctx) -> VirtualHollowSubFiber(lvl.lvl, value(qos), dirty),
        epilogue = quote
            #this task will always own this position forever, even if we don't write to it. Still, we try to be conservative of memory usage of the underlying level.
            if dirty && $(lvl.ptr)[$(ctx(pos))] == 0
                local_qos_fill += 1
                $(lvl.ptr)[$(ctx(pos))] = $(lvl.local_qos_fill) += 1
            end
        end
    )
end