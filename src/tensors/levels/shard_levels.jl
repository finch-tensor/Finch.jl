###task[pos] gives the processor that owns shard in position pos. AKA which channel in the multimemory channel to access.
###the subfiber p is contained at position ptr[p] on the sublevel in CHANNEL task[p].
###ptr[p] = 0 means unallocated.


struct MultiChannelMemory{Device} <: AbstractDevice
    device::Device
    n::Int
end

function Base.:(==)(device::MultiChannelMemory, other::MultiChannelMemory)
    device.device == other.device
end

get_num_tasks(device::MultiChannelMemory) = device.n
get_device(device::MultiChannelMemory) = device.device

struct VirtualMultiChannelMemory <: AbstractVirtualDevice
    device
    n
end

function Base.:(==)(device::VirtualMultiChannelMemory, other::VirtualMultiChannelMemory)
    device.device == other.device
end

get_num_tasks(device::VirtualMultiChannelMemory) = device.n
get_device(device::VirtualMultiChannelMemory) = device.device

function virtualize(ctx, ex, ::Type{MultiChannelMemory{Device}}) where {Device}
    device = virtualize(ctx, :($ex.device), Device)
    n = freshen(ctx, :n)
    push_preamble!(
        quote
            $n = $ex.n
        end,
    )
    VirtualMultiChannelMemory(device, n)
end

function lower(ctx::AbstractCompiler, mem::VirtualMultiChannelMemory, ::DefaultStyle)
    quote
        $MultiChannelMemory($(ctx(mem.device)), $(ctx(mem.n)))
    end
end

struct MemoryChannel{Device<:MultiChannelMemory,Parent} <: AbstractTask
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

function virtualize(ctx, ex, ::Type{MemoryChannel{Device,Parent}}) where {Device,Parent}
    device = virtualize(ctx, :($ex.device), Device)
    parent = virtualize(ctx, :($ex.parent), Parent)
    t = freshen(ctx, :t)
    push_preamble!(
        quote
            $t = $(ctx(ex.t))
        end,
    )
    VirtualMemoryChannel(device, t)
end

function lower(ctx::AbstractCompiler, mem::VirtualMemoryChannel, ::DefaultStyle)
    quote
        $MemoryChannel($(ctx(mem.t)), $(ctx(mem.device)), $(ctx(mem.parent)))
    end
end

struct MultiChannelBuffer{A}
    device::MultiChannelMemory
    data::Vector{A}
end

Base.eltype(::Type{MultiChannelBuffer{A}}) where {A} = eltype(A)
Base.ndims(::Type{MultiChannelBuffer{A}}) where {A} = ndims(A)

function transfer(device::MultiChannelMemory, arr::AbstractArray)
    data = [transfer(device.device, copy(arr)) for _ in 1:(device.n)]
    MultiChannelBuffer(device, data)
end

function transfer(device::MultiChannelMemory, arr::AbstractDict)
    data = [transfer(device.device, copy(arr)) for _ in 1:device.n]
    MultiChannelBuffer(device, data)
end

function transfer(device::MultiChannelMemory, arr::MultiChannelBuffer)
    data = arr.data
    if device.device != arr.device
        data = map(buf -> transfer(device.device, buf), data)
    end
    if arr.device.n > device.n
        MultiChannelBuffer(device, data)
    else
        padding = [
            transfer(device, Vector{eltype(data[1])}()) for _ in 1:(device.n - arr.device.n)
        ]
        if length(padding) > 0
            MultiChannelBuffer(
                device, vcat(data, padding)
            )
        else
            MultiChannelBuffer(
                device, data
            )
        end
    end
end

function transfer(task::MemoryChannel, arr::MultiChannelBuffer)
    if task.device == arr.device
        temp = arr.data[task.t]
        return temp
    else
        return arr
    end
end
function transfer(dst::MultiChannelBuffer, arr::MultiChannelBuffer)
    return arr
end
function transfer(dst::AbstractDevice, arr::MultiChannelBuffer)
    if dst == arr.device
        return arr
    else
        data = map(buf -> transfer(device.device, buf), arr.data)
        return MultiChannelBuffer(arr.device, data)
    end
end

"""
    ShardLevel{Lvl}()

Each subfiber of a Shard level is stored in a thread-specific tensor of type
`Lvl`, managed by MultiChannelMemory.

```jldoctest
julia> tensor_tree(Tensor(Dense(Shard(Element(0.0))), [1, 2, 3]))
ERROR: MethodError: no method matching ShardLevel(::ElementLevel{0.0, Float64, Int64, Vector{Float64}})
The type `ShardLevel` exists, but no method is defined for this combination of argument types when trying to construct it.

Closest candidates are:
  ShardLevel(::Device, !Matched::Lvl, !Matched::Ptr, !Matched::Task, !Matched::Used, !Matched::Alloc) where {Device, Lvl, Ptr, Task, Used, Alloc}
   @ Finch ~/Projects/Finch.jl/src/tensors/levels/shard_levels.jl:145
  ShardLevel(::Device, !Matched::Lvl) where {Device, Lvl}
   @ Finch ~/Projects/Finch.jl/src/tensors/levels/shard_levels.jl:154

Stacktrace:
 [1] top-level scope
   @ none:1
```
"""
struct ShardLevel{Device,Lvl,Ptr,Task,Used,Alloc,Schedule} <: AbstractLevel
    device::Device
    lvl::Lvl
    ptr::Ptr
    task::Task
    used::Used
    alloc::Alloc
    schedule::Schedule
end
const Shard = ShardLevel

function ShardLevel(device::Device, lvl::Lvl) where {Device,Lvl}
    Tp = postype(lvl)
    ptr = transfer(shared_memory(device), Tp[])
    task = transfer(shared_memory(device), Tp[])
    used = transfer(shared_memory(device), zeros(Tp, get_num_tasks(device)))
    alloc = transfer(shared_memory(device), zeros(Tp, get_num_tasks(device)))
    lvl = transfer(MultiChannelMemory(device, get_num_tasks(device)), lvl)
    schedule = FinchStaticSchedule{:dynamic}()
    ShardLevel{Device}(
        device,
        transfer(MultiChannelMemory(device, get_num_tasks(device)), lvl),
        ptr,
        task,
        used,
        alloc,
        schedule
    )
end

function ShardLevel{Device}(
    device, lvl::Lvl, ptr::Ptr, task::Task, used::Used, alloc::Alloc, schedule::Schedule
) where {Device,Lvl,Ptr,Task,Used,Alloc,Schedule}
    ShardLevel{Device,Lvl,Ptr,Task,Used,Alloc,Schedule}(device, lvl, ptr, task, used, alloc, schedule)
end

function Base.summary(
    ::Shard{Device,Lvl,Ptr,Task,Used,Alloc}
) where {Device,Lvl,Ptr,Task,Used,Alloc}
    "Shard($(Lvl))"
end

function similar_level(
    lvl::Shard{Device,Lvl,Ptr,Task,Used,Alloc}, fill_value, eltype::Type, dims...
) where {Device,Lvl,Ptr,Task,Used,Alloc}
    lvl_2 = similar(lvl.lvl, fill_value, eltype, dims...)
    ShardLevel(
        lvl.device,
        transfer(MultiChannelMemory(lvl.device, get_num_tasks(lvl.device)), lvl_2),
    )
end

function postype(
    ::Type{<:Shard{Device,Lvl,Ptr,Task,Used,Alloc}}
) where {Device,Lvl,Ptr,Task,Used,Alloc}
    postype(Lvl)
end

function transfer(device, lvl::ShardLevel)
    #lvl_2 = transfer(MultiChannelMemory(lvl.device, get_num_tasks(lvl.device)), lvl.lvl)
    lvl_2 = transfer(device, lvl.lvl) #TODO unclear
    ptr_2 = transfer(device, lvl.ptr)
    task_2 = transfer(device, lvl.task)
    qos_fill_2 = transfer(device, lvl.used)
    qos_stop_2 = transfer(device, lvl.alloc)
    return ShardLevel(lvl_2, ptr_2, task_2, qos_fill_2, qos_stop_2)
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
    )
end

function Base.resize!(lvl::ShardLevel, dims...)
    ShardLevel(
        lvl.device,
        resize!(lvl.lvl, dims...),
        lvl.ptr,
        lvl.task,
        lvl.used,
        lvl.alloc,
        lvl.schedule
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
    if lvl.ptr[pos] < 1
        print(io, "shard(?) -> ?")
    else
        print(io, "shard($(lvl.task[pos])) -> ")
    end
end

function labelled_children(fbr::SubFiber{<:ShardLevel})
    lvl = fbr.lvl
    pos = fbr.pos
    lvl.ptr[pos] < 1 && return []
    pos > length(lvl.ptr) && return []
    lvl_2 = transfer(
        MemoryChannel(
            lvl.task[pos],
            MultiChannelMemory(lvl.device, get_num_tasks(lvl.device)),
            SerialTask(),
        ),
        lvl.lvl,
    )
    [LabelledTree(SubFiber(lvl_2, lvl.ptr[pos]))]
end

@inline level_ndims(
    ::Type{<:ShardLevel{Device,Lvl,Ptr,Task,Used,Alloc}}
) where {Device,Lvl,Ptr,Task,Used,Alloc} = level_ndims(Lvl)
@inline level_size(
    lvl::ShardLevel{Device,Lvl,Ptr,Task,Used,Alloc}
) where {Device,Lvl,Ptr,Task,Used,Alloc} = level_size(lvl.lvl)
@inline level_axes(
    lvl::ShardLevel{Device,Lvl,Ptr,Task,Used,Alloc}
) where {Device,Lvl,Ptr,Task,Used,Alloc} = level_axes(lvl.lvl)
@inline level_eltype(
    ::Type{ShardLevel{Device,Lvl,Ptr,Task,Used,Alloc,Schedule}}
) where {Device,Lvl,Ptr,Task,Used,Alloc,Schedule} = level_eltype(Lvl)
@inline level_fill_value(
    ::Type{<:ShardLevel{Device,Lvl,Ptr,Task,Used,Alloc}}
) where {Device,Lvl,Ptr,Task,Used,Alloc} = level_fill_value(Lvl)

function (fbr::SubFiber{<:ShardLevel})(idxs...)
    lvl = fbr.lvl
    pos = fbr.pos
    pos > length(lvl.ptr) && return []
    lvl_2 = transfer(
        MemoryChannel(
            lvl.task[pos],
            MultiChannelMemory(lvl.device, get_num_tasks(lvl.device)),
            SerialTask(),
        ),
        lvl.lvl,
    )
    SubFiber(lvl_2, lvl.ptr[pos])(idxs...)
end

function countstored_level(lvl::ShardLevel, pos)
    sum(1:pos) do qos
        lvl_2 = transfer(
            MemoryChannel(
                lvl.task[qos],
                MultiChannelMemory(lvl.device, get_num_tasks(lvl.device)),
                SerialTask(),
            ),
            lvl.lvl,
        )
        countstored_level(lvl_2, lvl.used[qos])
    end
end

mutable struct VirtualShardLevel <: AbstractVirtualLevel
    tag
    device
    lvl
    ptr
    task
    used
    alloc
    schedule
    qos_fill
    qos_stop
    Tv
    Device
    Lvl
    Ptr
    Task
    Used
    Alloc
    Schedule
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
        $ShardLevel(
            $(ctx(lvl.device)),
            $(ctx(lvl.lvl)),
            $(ctx(lvl.ptr)),
            $(ctx(lvl.task)),
            $(ctx(lvl.used)),
            $(ctx(lvl.alloc)),
            $(lvl.tag).schedule
        )
    end
end

function virtualize(
    ctx, ex, ::Type{ShardLevel{Device,Lvl,Ptr,Task,Used,Alloc,Schedule}}, tag=:lvl
) where {Device,Lvl,Ptr,Task,Used,Alloc, Schedule}
    tag = freshen(ctx, tag)
    ptr = freshen(ctx, tag, :_ptr)
    task = freshen(ctx, tag, :_task)
    used = freshen(ctx, tag, :_qos_fill)
    alloc = freshen(ctx, tag, :_qos_stop)
    schedule = freshen(ctx, tag, :_schedule)

    push_preamble!(
        ctx,
        quote
            $tag = $ex
            $ptr = $tag.ptr
            $task = $tag.task
            $used = $tag.used
            $alloc = $tag.alloc
            $schedule = $tag.schedule
        end,
    )
    device_2 = virtualize(ctx, :($tag.device), Device, tag)
    lvl_2 = virtualize(ctx, :($tag.lvl), Lvl, tag)
    schedule_2 = virtualize(ctx, :($tag.schedule), Schedule, tag)
    VirtualShardLevel(
        tag,
        device_2,
        lvl_2,
        ptr,
        task,
        used,
        alloc,
        schedule_2,
        nothing,
        nothing,
        typeof(level_fill_value(Lvl)),
        Device,
        Lvl,
        Ptr,
        Task,
        Used,
        Alloc,
        Schedule
    )
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
        lvl.schedule,
        lvl.qos_fill,
        lvl.qos_stop,
        lvl.Tv,
        lvl.Device,
        lvl.Lvl,
        lvl.Ptr,
        lvl.Task,
        lvl.Used,
        lvl.Alloc,
        lvl.Schedule
    )
end

function distribute_level(
    ctx, lvl::VirtualShardLevel, arch, diff, style::Union{DeviceShared}
)
    Tp = postype(lvl)
    tag = lvl.tag
    if true #get_device(arch) == lvl.device
        qos_fill = freshen(ctx, tag, :_qos_fill)
        qos_stop = freshen(ctx, tag, :_qos_stop)
        tid = ctx(get_task_num(arch))
        push_preamble!(
            ctx,
            quote
                $qos_fill = $(lvl.used)[$tid]
                $qos_stop = $(lvl.alloc)[$tid]
            end,
        )
        dev = get_device(arch)
        multi_channel_dev = VirtualMultiChannelMemory(dev, get_num_tasks(dev))
        channel_task = VirtualMemoryChannel(get_task_num(arch), multi_channel_dev, arch)
        lvl_2 = distribute_level(ctx, lvl.lvl, channel_task, diff, style)
        lvl_2 = thaw_level!(ctx, lvl_2, value(qos_stop, Tp))
        push_epilogue!(
            ctx,
            contain(ctx) do ctx_2
                quote
                    $(lvl.used)[$tid] = $qos_fill
                    $(lvl.alloc)[$tid] = $qos_stop
                end
                freeze_level!(ctx_2, lvl_2, value(qos_stop))
            end,
        )
        diff[lvl.tag] = VirtualShardLevel(
            lvl.tag,
            lvl.device,
            lvl_2,
            distribute_buffer(ctx, lvl.ptr, arch, style),
            distribute_buffer(ctx, lvl.task, arch, style),
            distribute_buffer(ctx, lvl.used, arch, style),
            distribute_buffer(ctx, lvl.alloc, arch, style),
            lvl.schedule,
            qos_fill,
            qos_stop,
            lvl.Tv,
            lvl.Device,
            lvl.Lvl,
            lvl.Ptr,
            lvl.Task,
            lvl.Used,
            lvl.Alloc,
            lvl.Schedule
        )
    else
        diff[lvl.tag] = VirtualShardLevel(
            lvl.tag,
            lvl.device,
            distribute_level(ctx, lvl.lvl, arch, diff, style),
            distribute_buffer(ctx, lvl.ptr, arch, style),
            distribute_buffer(ctx, lvl.task, arch, style),
            distribute_buffer(ctx, lvl.used, arch, style),
            distribute_buffer(ctx, lvl.alloc, arch, style),
            lvl.schedule,
            lvl.qos_fill,
            lvl.qos_stop,
            lvl.Tv,
            lvl.Device,
            lvl.Lvl,
            lvl.Ptr,
            lvl.Task,
            lvl.Used,
            lvl.Alloc,
            lvl.Schedule
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
            lvl.schedule,
            lvl.qos_fill,
            lvl.qos_stop,
            lvl.Tv,
            lvl.Device,
            lvl.Lvl,
            lvl.Ptr,
            lvl.Task,
            lvl.Used,
            lvl.Alloc,
            lvl.Schedule
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
    @assert !is_on_device(ctx, lvl.device)
    push_preamble!(
        ctx,
        contain(ctx) do ctx_2
            diff = Dict()
            lvl_2 = distribute_level(ctx_2, lvl.lvl, lvl.device, diff, HostShared())
            used = distribute_buffer(ctx_2, lvl.used, lvl.device, HostShared())
            alloc = distribute_buffer(ctx_2, lvl.alloc, lvl.device, HostShared())

            ext = VirtualExtent(literal(1), pos)
            parallel_dim = VirtualParallelDimension(ext, lvl.device, lvl.schedule)

            virtual_parallel_region(ctx_2, parallel_dim, lvl.device, lvl.schedule) do f, ctx_3, i_lo, i_hi
                task = get_task(ctx_3)
                tid = ctx_3(get_task_num(ctx_3))

                alloced_pos = freshen(ctx_3, :alloced_pos)
                push_preamble!(ctx_3, 
                quote
                    $alloced_pos = $(ctx_3(alloc))[$tid]
                end)
                
                multi_channel_dev = VirtualMultiChannelMemory(
                    lvl.device, get_num_tasks(lvl.device)
                )
                channel_task = VirtualMemoryChannel(
                    get_task_num(task), multi_channel_dev, task
                )
                lvl_3 = distribute_level(ctx_3, lvl.lvl, channel_task, diff, DeviceShared())
                used = distribute_buffer(ctx_3, lvl.used, task, DeviceShared())
                alloc = distribute_buffer(ctx_3, lvl.alloc, task, DeviceShared())
                lvl_4 = declare_level!(ctx_3, lvl_3, value(alloced_pos), init)
                freeze_level!(ctx_3, lvl_4, value(alloced_pos))
                quote
                    $(ctx_3(used))[$tid] = 0
                    $(ctx_3(alloc))[$tid] = $alloced_pos
                end
            end
        end,
    )
    lvl
end

function assemble_level!(ctx, lvl::VirtualShardLevel, pos_start, pos_stop)
    @assert !is_on_device(ctx, lvl.device)
    pos_start = cache!(ctx, :pos_start, simplify(ctx, pos_start))
    pos_stop = cache!(ctx, :pos_stop, simplify(ctx, pos_stop))
    pos = freshen(ctx, :pos)
    sym = freshen(ctx, :pointer_to_lvl)
    push_preamble!(
        ctx,
        quote
            Finch.resize_if_smaller!($(lvl.task), $(ctx(pos_stop)))
            Finch.resize_if_smaller!($(lvl.ptr), $(ctx(pos_stop)))
            Finch.fill_range!($(lvl.ptr), 0, $(ctx(pos_start)), $(ctx(pos_stop)))
        end,
    )
    lvl
end

supports_reassembly(::VirtualShardLevel) = false

"""
these two are no-ops, we instead do these on distribute
"""
function freeze_level!(ctx, lvl::VirtualShardLevel, pos)
    @assert !is_on_device(ctx, lvl.device)
    return lvl
end

function thaw_level!(ctx::AbstractCompiler, lvl::VirtualShardLevel, pos)
    @assert !is_on_device(ctx, lvl.device)
    return lvl
end

function instantiate(ctx, fbr::VirtualSubFiber{VirtualShardLevel}, mode)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    Tp = postype(lvl)
    if mode.kind === reader
        tag = lvl.tag
        isnulltest = freshen(ctx, tag, :_nulltest)
        Vf = level_fill_value(lvl.Lvl)
        sym = freshen(ctx, :pointer_to_lvl)
        val = freshen(ctx, lvl.tag, :_val)
        t = freshen(ctx, tag, :_t)
        qos = freshen(ctx, tag, :_q)
        push_preamble!(
            ctx,
            quote
                $t = $(lvl.task)[$(ctx(pos))]
                $qos = $(lvl.ptr)[$(ctx(pos))]
            end,
        )
        Switch([
            value(:($qos != 0)) => Thunk(; body=(ctx_2) -> begin
                task = get_task(ctx_2)
                multi_channel_dev = VirtualMultiChannelMemory(lvl.device, get_num_tasks(lvl.device))
                channel_task = VirtualMemoryChannel(value(t, Tp), multi_channel_dev, task)
                lvl_2 = distribute_level(ctx_2, lvl.lvl, channel_task, Dict(), DeviceGlobal())
                instantiate(ctx_2, VirtualSubFiber(lvl_2, value(qos, Tp)), mode)
            end),
            literal(true) => FillLeaf(virtual_level_fill_value(lvl))
        ])
    else
        @assert is_on_device(ctx, lvl.device)
        instantiate(ctx, VirtualHollowSubFiber(lvl, pos, freshen(ctx, :dirty)), mode)
    end
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
function instantiate(ctx, fbr::VirtualHollowSubFiber{VirtualShardLevel}, mode)
    @assert mode.kind === updater
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.tag
    sym = freshen(ctx, :pointer_to_lvl)
    Tp = postype(lvl)

    tid = freshen(ctx, tag, :_tid)
    qos = freshen(ctx, :qos)

    @assert is_on_device(ctx, lvl.device)

    return Thunk(;
        preamble=quote
            $qos = $(lvl.ptr)[$(ctx(pos))]
            $tid = $(ctx(get_task_num(ctx)))
            if $qos == 0
                #this task will always own this position forever, even if we don't write to it.
                $qos = $(lvl.qos_fill) += 1
                $(lvl.task)[$(ctx(pos))] = $tid
                $(lvl.ptr)[$(ctx(pos))] = $(lvl.qos_fill)
                if $(lvl.qos_fill) > $(lvl.qos_stop)
                    $(lvl.qos_stop) = max($(lvl.qos_stop) << 1, 1)
                    $(contain(
                        ctx_2 -> assemble_level!(
                            ctx_2,
                            lvl.lvl,
                            value(lvl.qos_fill, Tp),
                            value(lvl.qos_stop, Tp),
                        ),
                        ctx,
                    ))
                end
            else
                if $(get_mode_flag(ctx) === :safe)
                    @assert $(lvl.task)[$(ctx(pos))] == $tid "Task mismatch in ShardLevel"
                end
            end
        end,
        body=(ctx) -> VirtualHollowSubFiber(lvl.lvl, value(qos), fbr.dirty),
    )
end
