###task[pos] gives the processor that owns Coalesce in position pos. AKA which channel in the multimemory channel to access.
###the subfiber p is contained at position ptr[p] on the sublevel in CHANNEL task[p].
###ptr[p] = 0 means unallocated.

"""
    CoalesceLevel{device, Lvl}()

CoalesceLevel uses an internal Coalesced representation, but unified the result into a single Tensor when
entering read-only mode.

```jldoctest
julia> tensor_tree(Tensor(Dense(Coalesce(cpu(1,2),Element(0.0))), 4))
4-Tensor
└─ Dense [1:4]
   ├─ [1]: Coalesce(?) -> ?
   ├─ [2]: Coalesce(?) -> ?
   ├─ [3]: Coalesce(?) -> ?
   └─ [4]: Coalesce(?) -> ?
```
"""
struct CoalesceLevel{Device,Lvl,Task,Schedule} <: AbstractLevel
    device::Device
    lvl::Lvl
    task::Task
    schedule::Schedule
end
const Coalesce = CoalesceLevel

function CoalesceLevel(device::Device, lvl::Lvl) where {Device,Lvl}
    Tp = postype(lvl)
    lvl = transfer(MultiChannelMemory(device, get_num_tasks(device)), lvl)
    task = transfer(shared_memory(device), Tp[])
    schedule = FinchStaticSchedule{:dynamic}()
    CoalesceLevel{Device}(
        device,
        transfer(MultiChannelMemory(device, get_num_tasks(device)), lvl),
        task,
        schedule,
    )
end

function CoalesceLevel{Device}(
    device, lvl::Lvl, task::Task, schedule::Schedule
) where {Device,Lvl,Task,Schedule}
    CoalesceLevel{Device,Lvl,Task,Schedule}(
        device, lvl, task, schedule
    )
end

function Base.summary(
    ::Coalesce{Device,Lvl,Task,Schedule}
) where {Device,Lvl,Task,Schedule}
    "Coalesce($(Lvl))"
end

function similar_level(
    lvl::Coalesce{Device,Lvl,Task,Schedule}, fill_value, eltype::Type, dims...
) where {Device,Lvl,Task,Schedule}
    lvl_2 = similar(lvl.lvl, fill_value, eltype, dims...)
    CoalesceLevel(
        lvl.device,
        transfer(MultiChannelMemory(lvl.device, get_num_tasks(lvl.device)), lvl_2),
        lvl.task,
        lvl.schedule
    )
end

function postype(
    ::Type{<:Coalesce{Device,Lvl,Task,Schedule}}
) where {Device,Lvl,Task,Schedule}
    postype(Lvl)
end

function transfer(device, lvl::CoalesceLevel)
    #lvl_2 = transfer(MultiChannelMemory(lvl.device, get_num_tasks(lvl.device)), lvl.lvl)
    lvl_2 = transfer(device, lvl.lvl) #TODO unclear
    task_2 = transfer(device, lvl.task)
    return CoalesceLevel(lvl.device, lvl_2, task_2, lvl.schedule)
end

function pattern!(lvl::CoalesceLevel)
    CoalesceLevel(lvl.device, pattern!(lvl.lvl), lvl.task, lvl.schedule)
end

function set_fill_value!(lvl::CoalesceLevel, init)
    CoalesceLevel(
        lvl.device,
        set_fill_value!(lvl.lvl, init),
        lvl.task,
        lvl.schedule,
    )
end

function Base.resize!(lvl::CoalesceLevel, dims...)
    CoalesceLevel(
        lvl.device,
        resize!(lvl.lvl, dims...),
        lvl.task,
        lvl.schedule,
    )
end

function Base.show(
    io::IO, lvl::CoalesceLevel{Device,Lvl,Task,Schedule}
) where {Device,Lvl,Task,Schedule}
    print(io, "Coalesce(")
    if get(io, :compact, false)
        print(io, "…")
    else
        show(io, lvl.lvl)
        print(io, ", ")
        show(io, lvl.task)
        print(io, ", ")
        show(io, lvl.schedule)
    end
    print(io, ")")
end

function labelled_show(io::IO, fbr::SubFiber{<:CoalesceLevel})
    (lvl, pos) = (fbr.lvl, fbr.pos)
    print(io, "Coalesce($(pos)) -> ")
end

function labelled_children(fbr::SubFiber{<:CoalesceLevel})
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
    ::Type{<:CoalesceLevel{Device,Lvl,Task,Schedule}}
) where {Device,Lvl,Task,Schedule} = level_ndims(Lvl)
@inline level_size(
    lvl::CoalesceLevel{Device,Lvl,Task,Schedule}
) where {Device,Lvl,Task,Schedule} = level_size(lvl.lvl)
@inline level_axes(
    lvl::CoalesceLevel{Device,Lvl,Task,Schedule}
) where {Device,Lvl,Task,Schedule} = level_axes(lvl.lvl)
@inline level_eltype(
    ::Type{CoalesceLevel{Device,Lvl,Task,Schedule}}
) where {Device,Lvl,Task,Schedule} = level_eltype(Lvl)
@inline level_fill_value(
    ::Type{<:CoalesceLevel{Device,Lvl,Task,Schedule}}
) where {Device,Lvl,Task,Schedule} = level_fill_value(Lvl)

function (fbr::SubFiber{<:CoalesceLevel})(idxs...)
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

function countstored_level(lvl::CoalesceLevel, pos)
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

mutable struct VirtualCoalesceLevel <: AbstractVirtualLevel
    tag
    device
    lvl
    task
    schedule
    Tv
    Device
    Lvl
    Task
    Schedule
end

postype(lvl::VirtualCoalesceLevel) = postype(lvl.lvl)

function is_level_injective(ctx, lvl::VirtualCoalesceLevel)
    [is_level_injective(ctx, lvl.lvl)..., true]
end
function is_level_atomic(ctx, lvl::VirtualCoalesceLevel)
    (below, atomic) = is_level_atomic(ctx, lvl.lvl)
    return ([below; [atomic]], atomic)
end
function is_level_concurrent(ctx, lvl::VirtualCoalesceLevel)
    (data, _) = is_level_concurrent(ctx, lvl.lvl)
    return (data, true)
end

function lower(ctx::AbstractCompiler, lvl::VirtualCoalesceLevel, ::DefaultStyle)
    quote
        $CoalesceLevel(
            $(ctx(lvl.device)),
            $(ctx(lvl.lvl)),
            $(ctx(lvl.task)),
            $(lvl.tag).schedule,
        )
    end
end

function virtualize(
    ctx, ex, ::Type{CoalesceLevel{Device,Lvl,Task,Schedule}}, tag=:lvl
) where {Device,Lvl,Task,Schedule}
    tag = freshen(ctx, tag)
    task = freshen(ctx, tag, :_task)
    schedule = freshen(ctx, tag, :_schedule)

    push_preamble!(
        ctx,
        quote
            $tag = $ex
            $task = $tag.task
            $schedule = $tag.schedule
        end,
    )
    device_2 = virtualize(ctx, :($tag.device), Device, tag)
    lvl_2 = virtualize(ctx, :($tag.lvl), Lvl, tag)
    schedule_2 = virtualize(ctx, :($tag.schedule), Schedule, tag)
    VirtualCoalesceLevel(
        tag,
        device_2,
        lvl_2,
        task,
        schedule_2,
        typeof(level_fill_value(Lvl)),
        Device,
        Lvl,
        Task,
        Schedule,
    )
end

function distribute_level(ctx, lvl::VirtualCoalesceLevel, arch, diff, style)
    diff[lvl.tag] = VirtualCoalesceLevel(
        lvl.tag,
        lvl.device,
        distribute_level(ctx, lvl.lvl, arch, diff, style),
        distribute_buffer(ctx, lvl.task, arch, style),
        lvl.schedule,
        lvl.Tv,
        lvl.Device,
        lvl.Lvl,
        lvl.Task,
        lvl.Schedule,
    )
end

function distribute_level(
    ctx, lvl::VirtualCoalesceLevel, arch, diff, style::Union{DeviceShared}
)
    Tp = postype(lvl)
    tag = lvl.tag
    if true #get_device(arch) == lvl.device
        qos_stop = freshen(ctx, tag, :_qos_stop)
        tid = ctx(get_task_num(arch))
        push_preamble!(
            ctx,
            quote
                $qos_stop = length($(lvl.task)) #all positions in task are always used?
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
                freeze_level!(ctx_2, lvl_2, value(qos_stop))
            end,
        )
        diff[lvl.tag] = VirtualCoalesceLevel(
            lvl.tag,
            lvl.device,
            distribute_level(ctx, lvl.lvl, arch, diff, style),
            distribute_buffer(ctx, lvl.task, arch, style),
            lvl.schedule,
            lvl.Tv,
            lvl.Device,
            lvl.Lvl,
            lvl.Task,
            lvl.Schedule,
        )
    else
        diff[lvl.tag] = VirtualCoalesceLevel(
            lvl.tag,
            lvl.device,
            distribute_level(ctx, lvl.lvl, arch, diff, style),
            distribute_buffer(ctx, lvl.task, arch, style),
            lvl.schedule,
            lvl.Tv,
            lvl.Device,
            lvl.Lvl,
            lvl.Task,
            lvl.Schedule,
        )
    end
end

function redistribute(ctx::AbstractCompiler, lvl::VirtualCoalesceLevel, diff)
    get(
        diff,
        lvl.tag,
        VirtualCoalesceLevel(
            lvl.tag,
            lvl.device,
            distribute_level(ctx, lvl.lvl, arch, diff, style),
            distribute_buffer(ctx, lvl.task, arch, style),
            lvl.schedule,
            lvl.Tv,
            lvl.Device,
            lvl.Lvl,
            lvl.Task,
            lvl.Schedule,
        ),
    )
end

Base.summary(lvl::VirtualCoalesceLevel) = "Coalesce($(lvl.Lvl))"

function virtual_level_resize!(ctx, lvl::VirtualCoalesceLevel, dims...)
    (lvl.lvl = virtual_level_resize!(ctx, lvl.lvl, dims...); lvl)
end
virtual_level_size(ctx, lvl::VirtualCoalesceLevel) = virtual_level_size(ctx, lvl.lvl)
virtual_level_eltype(lvl::VirtualCoalesceLevel) = virtual_level_eltype(lvl.lvl)
virtual_level_fill_value(lvl::VirtualCoalesceLevel) = virtual_level_fill_value(lvl.lvl)

function declare_level!(ctx, lvl::VirtualCoalesceLevel, pos, init)
    @assert !is_on_device(ctx, lvl.device)
    push_preamble!(
        ctx,
        contain(ctx) do ctx_2
            diff = Dict()
            lvl_2 = distribute_level(ctx_2, lvl.lvl, lvl.device, diff, HostShared())

            ext = VirtualExtent(literal(1), pos)
            parallel_dim = VirtualParallelDimension(ext, lvl.device, lvl.schedule)

            virtual_parallel_region(
                ctx_2, parallel_dim, lvl.device, lvl.schedule
            ) do f, ctx_3, i_lo, i_hi
                task = get_task(ctx_3)
                
                multi_channel_dev = VirtualMultiChannelMemory(
                    lvl.device, get_num_tasks(lvl.device)
                )
                channel_task = VirtualMemoryChannel(
                    get_task_num(task), multi_channel_dev, task
                )
                lvl_3 = distribute_level(ctx_3, lvl.lvl, channel_task, diff, DeviceShared())
                lvl_4 = declare_level!(ctx_3, lvl_3, pos, init)
                freeze_level!(ctx_3, lvl_4, pos)
            end
        end,
    )
    lvl
end

function assemble_level!(ctx, lvl::VirtualCoalesceLevel, pos_start, pos_stop)
    @assert !is_on_device(ctx, lvl.device)
    pos_start = cache!(ctx, :pos_start, simplify(ctx, pos_start))
    pos_stop = cache!(ctx, :pos_stop, simplify(ctx, pos_stop))
    pos = freshen(ctx, :pos)
    sym = freshen(ctx, :pointer_to_lvl)
    push_preamble!(
        ctx,
        quote
            Finch.resize_if_smaller!($(lvl.task), $(ctx(pos_stop)))
            Finch.fill_range!($(lvl.task), -1, $(ctx(pos_start)), $(ctx(pos_stop)))
        end,
    )
    lvl
end

supports_reassembly(::VirtualCoalesceLevel) = false

"""
these two are no-ops, we instead do these on distribute
"""
function freeze_level!(ctx, lvl::VirtualCoalesceLevel, pos)
    @assert !is_on_device(ctx, lvl.device)
    return lvl
end

function thaw_level!(ctx::AbstractCompiler, lvl::VirtualCoalesceLevel, pos)
    @assert !is_on_device(ctx, lvl.device)
    return lvl
end

function instantiate(ctx, fbr::VirtualSubFiber{VirtualCoalesceLevel}, mode)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    Tp = postype(lvl)
    if mode.kind === reader
        tag = lvl.tag
        isnulltest = freshen(ctx, tag, :_nulltest)
        Vf = level_fill_value(lvl.Lvl)
        sym = freshen(ctx, :pointer_to_lvl)
        val = freshen(ctx, lvl.tag, :_val)
        t = freshen(ctx, tag, :_t)
        push_preamble!(
            ctx,
            quote
                $t = $(lvl.task)[$(ctx(pos))]
            end,
        )
        #How to generalize this switch? No more pointer array to check if alloced. Do we just assume unalloced?
        Switch([
            value(:($t > 0)) => Thunk(;
                body=(ctx_2) -> begin
                    task = get_task(ctx_2)
                    multi_channel_dev = VirtualMultiChannelMemory(
                        lvl.device, get_num_tasks(lvl.device)
                    )
                    channel_task = VirtualMemoryChannel(
                        value(t, Tp), multi_channel_dev, task
                    )
                    lvl_2 = distribute_level(
                        ctx_2, lvl.lvl, channel_task, Dict(), DeviceGlobal()
                    )
                    instantiate(ctx_2, VirtualSubFiber(lvl_2, pos), mode)
                end,
            ),
            literal(true) => FillLeaf(virtual_level_fill_value(lvl)),
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
function instantiate(ctx, fbr::VirtualHollowSubFiber{VirtualCoalesceLevel}, mode)
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
                    @assert $(lvl.task)[$(ctx(pos))] == $tid "Task mismatch in CoalesceLevel"
                end
            end
        end,
        body=(ctx) -> VirtualHollowSubFiber(lvl.lvl, value(qos), fbr.dirty),
    )
end
