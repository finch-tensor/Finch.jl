struct MultiChannelMemory{Device} <: AbstractDevice
    device
    n::Int
end

get_num_tasks(device::MultiChannelMemory) = device.n
get_device(device::MultiChannelMemory) = device.device

struct VirtualMultiChannelMemory <: AbstractVirtualDevice
    device
    n::Int
end

get_num_tasks(device::VirtualMultiChannelMemory) = device.n
get_device(device::VirtualMultiChannelMemory) = device.device

function MultiChannelMemory(device::Device) where {Device}
    MultiChannelMemory{Device}(device)
end

function virtualize(ctx, ex, ::Type{MultiChannelMemory{Device}}) where {Device}
    device = virtualize(ctx, :($ex.device), Device)
    n = freshen(ctx, :n)
    push_preamble!(quote
        $n = $(ctx(ex.n))
    end)
    VirtualMultiChannelMemory(device, n)
end

function lower(ctx::AbstractCompiler, mem::VirtualMultiChannelMemory, ::DefaultStyle)
    quote
        MultiChannelMemory($(ctx(mem.device)), $(ctx(mem.n)))
    end
end

struct MemoryChannel{Device<:MultiChannelMemory, Parent} <: AbstractTask
    t::Int
    device::Device
    Parent::Parent
end

get_device(device::MemoryChannel) = device.device
get_parent_task(device::MemoryChannel) = device.parent
get_task_num(device::MemoryChannel) = device.t

struct VirtualMemoryChannel <: AbstractVirtualTask
    t
    device
    parent
end

function virtualize(ctx, ex, ::Type{MemoryChannel{Device, Parent}}) where {Device, Parent}
    device = virtualize(ctx, :($ex.device), Device)
    parent = virtualize(ctx, :($ex.parent), Parent)
    t = freshen(ctx, :t)
    push_preamble!(quote
        $t = $(ctx(ex.t))
    end)
    VirtualMemoryChannel(device, t)
end

function lower(ctx::AbstractCompiler, mem::VirtualMemoryChannel, ::DefaultStyle)
    quote
        MultiChannelMemory($(ctx(mem.device)), $(ctx(mem.n)))
    end
end

struct MultiChannelBuffer{A}
    device::MultiChannelMemory
    data::Vector{A}
end

function MultiChannelBuffer{A}(device::MultiChannelMemory) where {A}
    MultiChannelBuffer{A}(device, [A([]) for _ in 1:(device.n)])
end

Base.eltype(::Type{MultiChannelBuffer{A}}) where {A} = eltype(A)
Base.ndims(::Type{MultiChannelBuffer{A}}) where {A} = ndims(A)

function transfer(device::MultiChannelMemory, arr::AbstractArray)
    MultiChannelBuffer{A}(mem.device, [transfer(device.device, copy(arr)) for _ in 1:(mem.device.n)])
end
function transfer(device::MultiChannelMemory, arr::MultiChannelBuffer)
    @assert device.device = arr.device.device
    if arr.device.n > device.n
        return arr
    end
    return MultiChannelBuffer{A}(mem.device, resize!(arr.data, device.n))
end
function transfer(task::MemoryChannel, arr::MultiChannelArray)
    if task.device == arr.device
        temp = arr.data[task.tid]
        return temp
    else
        return arr
    end
end
function transfer(dst::MultiChannelArray, arr::MultiChannelArray)
    return arr
end
function transfer(dst::AbstractDevice, arr::MultiChannelArray)
    if dst == arr.device
        return arr
    else
        return MultiChannelArray(dst, [transfer(dst, arr.data[i]) for i in 1:length(arr.data)])
    end
end

"""
    ShardLevel{Lvl}()

Each subfiber of a Shard level is stored in a thread-specific tensor of type
`Lvl`, managed by MultiChannelMemory.

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
struct ShardLevel{Device,Lvl,Ptr,Task,Used,Alloc} <: AbstractLevel
    device::Device
    lvl::Lvl
    ptr::Ptr
    task::Task
    used::Used
    alloc::Alloc
end
const Shard = ShardLevel

function ShardLevel(device::Device, lvl::Lvl) where {Device,Lvl}
    ShardLevel{Device}(device, lvl, postype(lvl)[], postype(lvl)[], postype(lvl)[], postype(lvl)[], transfer(lvl, device))
end

function ShardLevel{Device}(
    device, lvl::Lvl, ptr::Ptr, task::Task, used::Used, alloc::Alloc
) where {Device,Lvl,Ptr,Task,Used,Alloc}
    ShardLevel{Device,Lvl,Ptr,Task,Used,Alloc}(device, lvl, ptr, task, used, alloc)
end

function Base.summary(::Shard{Device,Lvl,Ptr,Task,Used,Alloc}) where {Device,Lvl,Ptr,Task,Used,Alloc}
    "Shard($(Lvl))"
end

function similar_level(
    lvl::Shard{Device,Lvl,Ptr,Task,Used,Alloc}, fill_value, eltype::Type, dims...
) where {Device,Lvl,Ptr,Task,Used,Alloc}
    ShardLevel(lvl.device, similar_level(lvl.lvl, fill_value, eltype, dims...))
end

function postype(::Type{<:Shard{Device,Lvl,Ptr,Task,Used,Alloc}}) where {Device,Lvl,Ptr,Task,Used,Alloc}
    postype(Lvl)
end

function transfer(device, lvl::ShardLevel)
    lvl_2 = transfer(device, lvl.lvl)
    ptr_2 = transfer(device, lvl.ptr)
    task_2 = transfer(device, lvl.task)
    qos_used_2 = transfer(device, lvl.used)
    qos_alloc_2 = transfer(device, lvl.alloc)
    return ShardLevel(lvl_2, ptr_2, task_2, qos_used_2, qos_alloc_2)
end

function pattern!(lvl::ShardLevel)
    ShardLevel(pattern!(lvl.lvl), lvl.ptr, lvl.task, lvl.used, lvl.alloc)
end

function set_fill_value!(lvl::ShardLevel, init)
    ShardLevel(
        set_fill_value!(lvl.lvl, init),
        lvl.ptr,
        lvl.task,
        lvl.used,
        lvl.alloc,
        map(lvl_2 -> set_fill_value!(lvl_2, init)),
    )
end

function Base.resize!(lvl::ShardLevel, dims...)
    ShardLevel(
        resize!(lvl.lvl, dims...),
        lvl.ptr,
        lvl.task,
        lvl.used,
        lvl.alloc,
        map(lvl_2 -> resize!(lvl_2, dims...)),
    )
end

function Base.show(
    io::IO, lvl::ShardLevel{Device,Lvl,Ptr,Task,Used,Alloc}
) where {Device,Lvl,Ptr,Task,Used,Alloc}
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
        show(io, lvl.used)
        print(io, ", ")
        show(io, lvl.alloc)
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
    pos > length(lvl.ptr) && return []
    lvl_2 = transfer(MemoryChannel(lvl.task[pos]), lvl.lvl)
    [LabelledTree(SubFiber(lvl.val[lvl.task[pos]], lvl.ptr[pos]))]
end

@inline level_ndims(
    ::Type{<:ShardLevel{Device,Lvl,Ptr,Task,Used,Alloc,Val}}
) where {Device,Lvl,Ptr,Task,Used,Alloc,Val} = level_ndims(Lvl)
@inline level_size(
    lvl::ShardLevel{Device,Lvl,Ptr,Task,Used,Alloc,Val}
) where {Device,Lvl,Ptr,Task,Used,Alloc,Val} = level_size(lvl.lvl)
@inline level_axes(
    lvl::ShardLevel{Device,Lvl,Ptr,Task,Used,Alloc,Val}
) where {Device,Lvl,Ptr,Task,Used,Alloc,Val} = level_axes(lvl.lvl)
@inline level_eltype(
    ::Type{ShardLevel{Device,Lvl,Ptr,Task,Used,Alloc,Val}}
) where {Device,Lvl,Ptr,Task,Used,Alloc,Val} = level_eltype(Lvl)
@inline level_fill_value(
    ::Type{<:ShardLevel{Device,Lvl,Ptr,Task,Used,Alloc,Val}}
) where {Device,Lvl,Ptr,Task,Used,Alloc,Val} = level_fill_value(Lvl)

function (fbr::SubFiber{<:ShardLevel})(idxs...)
    q = fbr.pos
    return SubFiber(fbr.lvl.val[q], 1)(idxs...)
end

countstored_level(lvl::ShardLevel, pos) = pos

mutable struct VirtualShardLevel <: AbstractVirtualLevel
    tag
    device
    lvl  # stand-in for the sublevel for virtual resize, etc.
    ptr
    task
    used
    alloc
    val
    qos_used
    qos_alloc
    Tv
    Device
    Lvl
    Ptr
    Task
    Used
    Alloc
    Val
end

postype(lvl::VirtualShardLevel) = postype(lvl.lvl)

function is_level_injective(ctx, lvl::VirtualShardLevel)
    [is_level_injective(ctx, lvl.lvl)..., true]
end
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
        $ShardLevel{$(lvl.Lvl),$(lvl.Ptr),$(lvl.Task),$(lvl.Used),$(lvl.Alloc),$(lvl.Val)}(
            $(ctx(lvl.lvl)), $(lvl.val)
        )
    end
end

function virtualize(
    ctx, ex, ::Type{ShardLevel{Device,Lvl,Ptr,Task,Used,Alloc,Val}}, tag=:lvl
) where {Device,Lvl,Ptr,Task,Used,Alloc,Val}
    tag = freshen(ctx, tag)
    ptr = freshen(ctx, tag, :_ptr)
    task = freshen(ctx, tag, :_task)
    used = freshen(ctx, tag, :_qos_used)
    alloc = freshen(ctx, tag, :_qos_alloc)
    val = freshen(ctx, tag, :_val)

    push_preamble!(
        ctx,
        quote
            $tag = $ex
            $ptr = $tag.ptr
            $task = $tag.task
            $used = $tag.used
            $alloc = $tag.alloc
            $val = $tag.val
        end,
    )
    device_2 = virtualize(ctx, :($tag.device), Device, tag)
    lvl_2 = virtualize(ctx, :($tag.lvl), Lvl, tag)
    VirtualShardLevel(tag, device_2, lvl_2, ptr, task, used, alloc, val, typeof(level_fill_value(Lvl)), nothing, nothing, Device, Lvl, Ptr, Task, Used, Alloc, Val)
end

function distribute_level(ctx, lvl::VirtualShardLevel, arch, diff, style)
    diff[lvl.tag] = VirtualShardLevel(
        lvl.tag,
        lvl.device,
        distribute_level(ctx, lvl.lvl, arch, diff, style),
        distribute_buffer(ctx, lvl.ptr, arch, style),
        distribute_buffer(ctx, lvl.task, arch, style),
        distribute_buffer(ctx, lvl.used, arch, style),
        distribute_buffer(ctx, lvl.alloc, arch, style),
        distribute_buffer(ctx, lvl.val, arch, style),
        lvl.qos_used,
        lvl.qos_alloc,
        lvl.Tv,
        lvl.Device,
        lvl.Lvl,
        lvl.Ptr,
        lvl.Task,
        lvl.Used,
        lvl.Alloc,
        lvl.Val,
    )
end

function distribute_level(ctx, lvl::VirtualShardLevel, arch, diff, style::Union{DeviceShared})
    tag = lvl.tag
    qos_used = freshen(ctx, tag, :qos_used)
    qos_alloc = freshen(ctx, tag, :qos_alloc)
    Tp = postype(lvl)
    
    #lvl_2 = freshen(ctx, :lvl)
    #push_preamble!(
    #    ctx,
    #    quote
    #        val_2 = map($lvl_2 -> $(contain(ctx) do ctx_2
    #                lvl_3 = virtualize(ctx_2, lvl_2, lvl_2.Lvl)
    #                ctx_2(distribute(ctx_2, $val_2[i], arch, diff, style))
    #            end)
    #        end
    #        $(lvl.ptr) = $transfer($(ctx(arch)), $(lvl.ptr), style)
    #        $(lvl.task) = $transfer($(ctx(arch)), $(lvl.task), style)
    #        $(lvl.val) = $transfer($(ctx(arch)), $(lvl.val), style)
    #    end,
    #)
    if true #get_device(arch) == lvl.device
        qos_used = freshen(ctx, tag, :_qos_used)
        qos_alloc = freshen(ctx, tag, :_qos_alloc)
        tid = ctx(get_task_num(arch))
        push_preamble!(ctx, quote
            $qos_used = $(lvl.used)[$tid]
            $qos_alloc = $(lvl.alloc)[$tid]
        end)
        lvl_2 = virtualize(ctx, :($(lvl.val)[$tid]), lvl.Lvl)
        lvl_2 = thaw_level!(ctx, lvl_2, value(qos_alloc, Tp))
        push_epilogue!(ctx, contain(ctx) do ctx_2
            lvl_3 = freeze_level!(ctx_2, lvl_2, qos_alloc)
            quote
                $(lvl.used)[$tid] = $qos_used 
                $(lvl.alloc)[$tid] = $qos_alloc
                $(lvl.val)[$tid] = $(ctx_2(lvl_3))
            end
        end)
        diff[lvl.tag] = VirtualShardLevel(
            lvl.tag,
            lvl.device,
            lvl_2,#distribute_level(ctx, lvl_2, arch, diff, style),
            distribute_buffer(ctx, lvl.ptr, arch, style),
            distribute_buffer(ctx, lvl.task, arch, style),
            distribute_buffer(ctx, lvl.used, arch, style),
            distribute_buffer(ctx, lvl.alloc, arch, style),
            distribute_buffer(ctx, lvl.val, arch, style),
            qos_used,
            qos_alloc,
            lvl.Tv,
            lvl.Device,
            lvl.Lvl,
            lvl.Ptr,
            lvl.Task,
            lvl.Used,
            lvl.Alloc,
            lvl.Val,
        )
    else
        diff[lvl.tag] = VirtualShardLevel(
            lvl.tag,
            lvl.device,
            lvl_2,#distribute_level(ctx, lvl.lvl, arch, diff, style),
            distribute_buffer(ctx, lvl.ptr, arch, style),
            distribute_buffer(ctx, lvl.task, arch, style),
            distribute_buffer(ctx, lvl.used, arch, style),
            distribute_buffer(ctx, lvl.alloc, arch, style),
            distribute_buffer(ctx, lvl.val, arch, style),
            lvl.qos_used,
            lvl.qos_alloc,
            lvl.Tv,
            lvl.Device,
            lvl.Lvl,
            lvl.Ptr,
            lvl.Task,
            lvl.Used,
            lvl.Alloc,
            lvl.Val,
        )
    end
end

function redistribute(ctx::AbstractCompiler, lvl::VirtualShardLevel, diff)
    get(
        diff,
        lvl.tag,
        VirtualShardLevel(
            lvl.tag,
            lvl.device,
            redistribute(ctx, lvl.lvl, diff),
            lvl.ptr,
            lvl.task,
            lvl.used,
            lvl.alloc,
            lvl.val,
            lvl.qos_used,
            lvl.qos_alloc,
            lvl.Tv,
            lvl.Device,
            lvl.Lvl,
            lvl.Ptr,
            lvl.Task,
            lvl.Used,
            lvl.Alloc,
            lvl.Val,
        ),
    )
end

Base.summary(lvl::VirtualShardLevel) = "Shard($(lvl.Lvl))"

function virtual_level_resize!(ctx, lvl::VirtualShardLevel, dims...)
    (lvl.lvl = virtual_level_resize!(ctx, lvl.lvl, dims...); lvl)
end
virtual_level_size(ctx, lvl::VirtualShardLevel) = virtual_level_size(ctx, lvl.lvl)
virtual_level_eltype(lvl::VirtualShardLevel) = virtual_level_eltype(lvl.lvl)
virtual_level_fill_value(lvl::VirtualShardLevel) = virtual_level_fill_value(lvl.lvl)

function declare_level!(ctx, lvl::VirtualShardLevel, pos, init)
    push_preamble!(ctx,
        virtual_parallel_region(ctx, lvl.device) do ctx_2
            lvl_2 = virtualize(
                ctx_2, :($(lvl.val)[$(ctx_2(get_task_num(get_task(ctx_2))))]), lvl.Lvl
            ) #TODO should this virtualize the eltype of Val?
            declare_level!(ctx_2, lvl_2, literal(1), init)
        end,
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
    push_preamble!(
        ctx,
        quote
            Finch.resize_if_smaller!($(lvl.task), $(ctx(pos_stop)))
            Finch.resize_if_smaller!($(lvl.ptr), $(ctx(pos_stop)))
            Finch.fill_range!($(lvl.task), $(ctx(pos_start)), $(ctx(pos_stop)), 0)
        end,
    )
    lvl
end

supports_reassembly(::VirtualShardLevel) = false

"""
these two are no-ops, we instead do these on instantiate
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
        tag = lvl.tag
        isnulltest = freshen(ctx, tag, :_nulltest)
        Vf = level_fill_value(lvl.Lvl)
        sym = freshen(ctx, :pointer_to_lvl)
        val = freshen(ctx, lvl.tag, :_val)
        return Thunk(;
            body=(ctx) -> begin
                lvl_2 = virtualize(ctx.code, :($(lvl.val)[$(ctx(pos))]), lvl.Lvl, sym)
                instantiate(ctx, VirtualSubFiber(lvl_2, literal(1)), mode)
            end,
        )
    else
        instantiate(ctx, VirtualHollowSubFiber(fbr.lvl, fbr.pos, freshen(ctx, :dirty)), mode)
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
    tag = lvl.tag
    sym = freshen(ctx, :pointer_to_lvl)
    Tp = postype(lvl)

    tid = freshen(ctx, tag, :_tid)
    qos = freshen(ctx, :qos)

    return Thunk(;
        preamble = quote
            $qos = $(lvl.ptr)[$(ctx(pos))]
            $tid = $(ctx(get_task_num(ctx)))
            if $qos == 0
                #this task will always own this position forever, even if we don't write to it.
                $qos = $(lvl.qos_used) += 1
                $(lvl.task)[$(ctx(pos))] = $tid
                $(lvl.ptr)[$(ctx(pos))] = $(lvl.qos_used)
                if $(lvl.qos_used) > $(lvl.qos_alloc)
                    $(lvl.qos_alloc) = max($(lvl.qos_alloc) << 1, 1)
                    $(contain(ctx_2 -> assemble_level!(ctx_2, lvl.lvl, value(lvl.qos_used, Tp), value(lvl.qos_alloc, Tp)), ctx))
                end
            else
                if $(get_mode_flag(ctx) === :safe)
                    @assert $(lvl.task)[$(ctx(pos))] == $tid "Task mismatch in ShardLevel"
                end
            end
        end,
        body = (ctx) -> VirtualHollowSubFiber(lvl.lvl, value(qos), fbr.dirty),
    )
end
