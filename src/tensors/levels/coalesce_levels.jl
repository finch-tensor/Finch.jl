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
struct CoalesceLevel{Device,Lvl,Coalescent,Schedule} <: AbstractLevel
    device::Device
    lvl::Lvl
    coalescent::Coalescent
    schedule::Schedule
end
const Coalesce = CoalesceLevel

function CoalesceLevel(device::Device, lvl::Lvl) where {Device,Lvl}
    Tp = postype(lvl)
    coalescent = similar_level(lvl, level_fill_value(Lvl), level_eltype(Lvl), level_size(lvl)...)
    lvl = transfer(MultiChannelMemory(device, get_num_tasks(device)), lvl)
    schedule = FinchStaticSchedule{:dynamic}()
    CoalesceLevel{Device}(
        device,
        transfer(MultiChannelMemory(device, get_num_tasks(device)), lvl),
        coalescent,
        schedule,
    )
end

function CoalesceLevel{Device}(
    device, lvl::Lvl, coalescent::Coalescent, schedule::Schedule
) where {Device,Lvl,Coalescent,Schedule}
    CoalesceLevel{Device,Lvl,Coalescent,Schedule}(
        device, lvl, coalescent, schedule
    )
end

function Base.summary(
    ::Coalesce{Device,Lvl,Coalescent,Schedule}
) where {Device,Lvl,Coalescent,Schedule}
    "Coalesce($(Lvl))"
end

function similar_level(
    lvl::Coalesce{Device,Lvl,Coalescent,Schedule}, fill_value, eltype::Type, dims...
) where {Device,Lvl,Coalescent,Schedule}
    lvl_2 = similar(lvl.lvl, fill_value, eltype, dims...)
    CoalesceLevel(
        lvl.device,
        transfer(MultiChannelMemory(lvl.device, get_num_tasks(lvl.device)), lvl_2),
        lvl.schedule
    )
end

function postype(
    ::Type{<:Coalesce{Device,Lvl,Coalescent,Schedule}}
) where {Device,Lvl,Coalescent,Schedule}
    postype(Lvl)
end

function transfer(device, lvl::CoalesceLevel)
    #lvl_2 = transfer(MultiChannelMemory(lvl.device, get_num_tasks(lvl.device)), lvl.lvl)
    lvl_2 = transfer(device, lvl.lvl) #TODO unclear
    return CoalesceLevel(lvl.device, lvl_2, lvl.coalescent, lvl.schedule)
end

function pattern!(lvl::CoalesceLevel)
    CoalesceLevel(lvl.device, pattern!(lvl.lvl), lvl.coalescent, lvl.schedule)
end

function set_fill_value!(lvl::CoalesceLevel, init)
    CoalesceLevel(
        lvl.device,
        set_fill_value!(lvl.lvl, init),
        set_fill_value!(lvl.coalescent, init),
        lvl.schedule,
    )
end

function Base.resize!(lvl::CoalesceLevel, dims...)
    CoalesceLevel(
        lvl.device,
        resize!(lvl.lvl, dims...),
        resize!(lvl.coalescent, dims...),
        lvl.schedule,
    )
end

function Base.show(
    io::IO, lvl::CoalesceLevel{Device,Lvl,Coalescent,Schedule}
) where {Device,Lvl,Coalescent,Schedule}
    print(io, "Coalesce(")
    if get(io, :compact, false)
        print(io, "…")
    else
        show(io, lvl.lvl)
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
    # n_threads = get_num_tasks(lvl.device)
    # children = []

    # for tid in 1:n_threads
    #     lvl_2 = transfer(
    #         MemoryChannel(
    #             tid,
    #             MultiChannelMemory(lvl.device, get_num_tasks(lvl.device)),
    #             SerialTask(),
    #         ),
    #         lvl.lvl,
    #     )
    #     push!(children, LabelledTree(SubFiber(lvl_2, pos)))
    # end
    labelled_children(SubFiber(lvl.coalescent, pos))
end

@inline level_ndims(
    ::Type{<:CoalesceLevel{Device,Lvl,Coalescent,Schedule}}
) where {Device,Lvl,Coalescent,Schedule} = level_ndims(Lvl)
@inline level_size(
    lvl::CoalesceLevel{Device,Lvl,Coalescent,Schedule}
) where {Device,Lvl,Coalescent,Schedule} = level_size(lvl.lvl)
@inline level_axes(
    lvl::CoalesceLevel{Device,Lvl,Coalescent,Schedule}
) where {Device,Lvl,Coalescent,Schedule} = level_axes(lvl.lvl)
@inline level_eltype(
    ::Type{CoalesceLevel{Device,Lvl,Coalescent,Schedule}}
) where {Device,Lvl,Coalescent,Schedule} = level_eltype(Lvl)
@inline level_fill_value(
    ::Type{<:CoalesceLevel{Device,Lvl,Coalescent,Schedule}}
) where {Device,Lvl,Coalescent,Schedule} = level_fill_value(Lvl)

function (fbr::SubFiber{<:CoalesceLevel})(idxs...)
    lvl = fbr.lvl
    pos = fbr.pos
    # pos > length(lvl.ptr) && return []
    # lvl_2 = transfer(
    #     MemoryChannel(
    #         lvl.task[pos],
    #         MultiChannelMemory(lvl.device, get_num_tasks(lvl.device)),
    #         SerialTask(),
    #     ),
    #     lvl.lvl,
    # )
    SubFiber(lvl.coalescent, pos)(idxs...)
end

function countstored_level(lvl::CoalesceLevel, pos)
    countstored_level(lvl.coalescent, pos)
end

function coalesce_nnz(lvl::CoalesceLevel, pos)
    n_tasks = get_num_tasks(lvl.device)
    sum(1:n_tasks) do tid
        total = 0
        lvl_2 = transfer(
            MemoryChannel(
                tid,
                MultiChannelMemory(lvl.device, get_num_tasks(lvl.device)),
                SerialTask(),
            ),
            lvl.lvl,
        )
        for qos in 1:pos
            total += countstored_level(lvl_2, qos)
        end
        total
    end
end

mutable struct VirtualCoalesceLevel <: AbstractVirtualLevel
    tag
    device
    lvl
    coalescent
    schedule
    Tv
    Device
    Lvl
    Coalescent
    Schedule
    qos_stop
    coal_ref
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
            $(ctx(lvl.coal_ref)),
            $(lvl.tag).schedule,
        )
    end
end

function virtualize(
    ctx, ex, ::Type{CoalesceLevel{Device,Lvl,Coalescent,Schedule}}, tag=:lvl
) where {Device,Lvl,Coalescent,Schedule}
    tag = freshen(ctx, tag)
    schedule = freshen(ctx, tag, :_schedule)
    coal_ref = freshen(ctx, tag, :_coalref)

    push_preamble!(
        ctx,
        quote
            $tag = $ex
            $schedule = $tag.schedule
            $coal_ref = $tag.coalescent
        end,
    )
    device_2 = virtualize(ctx, :($tag.device), Device, tag)
    lvl_2 = virtualize(ctx, :($tag.lvl), Lvl, tag)
    coalescent_2 = virtualize(ctx, :($tag.coalescent), Coalescent, tag)
    schedule_2 = virtualize(ctx, :($tag.schedule), Schedule, tag)
    qos_stop = freshen(ctx, tag, :_qos_stop)
    VirtualCoalesceLevel(
        tag,
        device_2,
        lvl_2,
        coalescent_2,
        schedule_2,
        typeof(level_fill_value(Lvl)),
        Device,
        Lvl,
        Coalescent,
        Schedule,
        qos_stop,
        coal_ref
    )
end

function distribute_level(ctx, lvl::VirtualCoalesceLevel, arch, diff, style::Union{HostShared})
    diff[lvl.tag] = VirtualCoalesceLevel(
        lvl.tag,
        lvl.device,
        distribute_level(ctx, lvl.lvl, arch, diff, style),
        lvl.coalescent,
        lvl.schedule,
        lvl.Tv,
        lvl.Device,
        lvl.Lvl,
        lvl.Coalescent,
        lvl.Schedule,
        lvl.qos_stop,
        lvl.coal_ref
    )
end

function distribute_level(ctx, lvl::VirtualCoalesceLevel, arch, diff, style::Union{DeviceGlobal, HostGlobal})
    diff[lvl.tag] = VirtualCoalesceLevel(
        lvl.tag,
        lvl.device,
        lvl.lvl,
        distribute_level(ctx, lvl.coalescent, arch, diff, style),
        lvl.schedule,
        lvl.Tv,
        lvl.Device,
        lvl.Lvl,
        lvl.Coalescent,
        lvl.Schedule,
        lvl.qos_stop,
        lvl.coal_ref
    )
end

function distribute_level(ctx, lvl::VirtualCoalesceLevel, arch, diff, style::Union{DeviceLocal, HostLocal})
    diff[lvl.tag] = VirtualCoalesceLevel(
        lvl.tag,
        lvl.device,
        distribute_level(ctx, lvl.lvl, arch, diff, style),
        distribute_level(ctx, lvl.coalescent, arch, diff, style),
        lvl.schedule,
        lvl.Tv,
        lvl.Device,
        lvl.Lvl,
        lvl.Coalescent,
        lvl.Schedule,
        lvl.qos_stop,
        lvl.coal_ref
    )
end

function distribute_level(
    ctx, lvl::VirtualCoalesceLevel, arch, diff, style::Union{DeviceShared}
)
    Tp = postype(lvl)
    tag = lvl.tag
    if true #get_device(arch) == lvl.device
        dev = get_device(arch)
        multi_channel_dev = VirtualMultiChannelMemory(dev, get_num_tasks(dev))
        channel_task = VirtualMemoryChannel(get_task_num(arch), multi_channel_dev, arch)
        lvl_2 = distribute_level(ctx, lvl.lvl, channel_task, diff, style)
        lvl_2 = thaw_level!(ctx, lvl_2, value(lvl.qos_stop, Tp))
        push_epilogue!(
            ctx,
            contain(ctx) do ctx_2
                freeze_level!(ctx_2, lvl_2, value(lvl.qos_stop))
            end,
        )
        diff[lvl.tag] = VirtualCoalesceLevel(
            lvl.tag,
            lvl.device,
            distribute_level(ctx, lvl.lvl, arch, diff, style),
            lvl.coalescent,
            lvl.schedule,
            lvl.Tv,
            lvl.Device,
            lvl.Lvl,
            lvl.Coalescent,
            lvl.Schedule,
            lvl.qos_stop,
            lvl.coal_ref
        )
    else
        diff[lvl.tag] = VirtualCoalesceLevel(
            lvl.tag,
            lvl.device,
            distribute_level(ctx, lvl.lvl, arch, diff, style),
            lvl.coalescent,
            lvl.schedule,
            lvl.Tv,
            lvl.Device,
            lvl.Lvl,
            lvl.Coalescent,
            lvl.Schedule,
            lvl.qos_stop,
            lvl.coal_ref
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
            redistribute(ctx, lvl.lvl, diff),
            lvl.coalescent,
            lvl.schedule,
            lvl.Tv,
            lvl.Device,
            lvl.Lvl,
            lvl.Coalescent,
            lvl.Schedule,
            lvl.qos_stop,
            lvl.coal_ref
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

            push_preamble!(ctx_2,
            quote
                $(lvl.qos_stop) = $(ctx_2(pos))
            end)

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
    coalescent_2 = declare_level!(ctx, lvl.coalescent, pos, init)
    freeze_level!(ctx, coalescent_2, pos)
    lvl.coalescent = coalescent_2

    lvl
end

function assemble_level!(ctx, lvl::VirtualCoalesceLevel, pos_start, pos_stop)
    @assert !is_on_device(ctx, lvl.device)
    pos_start = cache!(ctx, :pos_start, simplify(ctx, pos_start))
    pos_stop = cache!(ctx, :pos_stop, simplify(ctx, pos_stop))
    pos = freshen(ctx, :pos)
    sym = freshen(ctx, :pointer_to_lvl)
    push_preamble!(ctx,
        contain(ctx) do ctx_2
            diff = Dict()
            lvl_2 = distribute_level(ctx_2, lvl.lvl, lvl.device, diff, HostShared())

            ext = VirtualExtent(pos_start, pos_stop)
            parallel_dim = VirtualParallelDimension(ext, lvl.device, lvl.schedule)

            push_preamble!(ctx_2, virtual_parallel_region(
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
                push_preamble!(ctx_3,
                    contain(ctx_3) do ctx_4
                        lvl_3 = declare_level!(ctx_4, lvl_3, pos_start, literal(0))
                        assemble_level!(ctx_4, lvl_3, pos_start, pos_stop)
                    end,
                )
                lvl_3 = freeze_level!(ctx_3, lvl_3, pos_stop)
            end)
            
            push_preamble!(ctx_2,
                contain(ctx_2) do ctx_3
                    declare_level!(ctx_3, lvl.coalescent, pos_start, literal(0))
                    assemble_level!(ctx_3, lvl.coalescent, pos_start, pos_stop)
                end)
            freeze_level!(ctx_2, lvl.coalescent, pos_stop)
        end)
    lvl
end

supports_reassembly(::VirtualCoalesceLevel) = false

# function `freeze`_level!(ctx, lvl::VirtualCoalesceLevel, pos)
#     @assert !is_on_device(ctx, lvl.device)
#     return lvl
# end

function freeze_level!(ctx, lvl::VirtualCoalesceLevel, pos)
    @assert !is_on_device(ctx, lvl.device)
    P = ctx(get_num_tasks(lvl.device))
    lvl_e = ctx(lvl)
    lvl_ce = ctx(lvl.coalescent)
    factor = ctx(pos)

    task_map = freshen(ctx, :tm)
    global_fbr_map = freshen(ctx, :gfm)
    local_fbr_map = freshen(ctx, :lfm)
    # factor = freshen(ctx, :fac)

    push_preamble!(
        ctx,
        quote
            $task_map = collect(1:($P))
            $global_fbr_map = ones(Int, $P)
            $local_fbr_map = ones(Int, $P)
            
            $(lvl.coal_ref) = coalesce_level!(
                $(lvl_e), $global_fbr_map, $local_fbr_map, $task_map, $factor, $P, $(lvl_ce)
            )
        end,
    )

    return lvl
end

function thaw_level!(ctx::AbstractCompiler, lvl::VirtualCoalesceLevel, pos)
    @assert !is_on_device(ctx, lvl.device)

    push_preamble!(
        ctx,
        contain(ctx) do ctx_2
            diff = Dict()
            lvl_2 = distribute_level(ctx_2, lvl.lvl, lvl.device, diff, HostShared())

            ext = VirtualExtent(literal(1), pos)
            parallel_dim = VirtualParallelDimension(ext, lvl.device, lvl.schedule)

            push_preamble!(ctx_2,
            quote
                $(lvl.qos_stop) = $(ctx_2(pos))
            end)

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
                lvl_4 = declare_level!(ctx_3, lvl_3, pos, literal(0))
                freeze_level!(ctx_3, lvl_4, pos)
            end
        end,
    )
    return lvl
end

function instantiate(ctx, fbr::VirtualSubFiber{VirtualCoalesceLevel}, mode)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    if mode.kind === reader
        Thunk(;
            body=(ctx_2) -> begin
                instantiate(ctx_2, VirtualSubFiber(lvl.coalescent, pos), mode)
            end,
        )
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

    @assert is_on_device(ctx, lvl.device)

    return Thunk(;
        body=(ctx) -> VirtualHollowSubFiber(lvl.lvl, pos, fbr.dirty),
    )
end

function coalesce_level!(lvl::CoalesceLevel, global_fbr_map, local_fbr_map, task_map, factor, P, coalescent)
    coalesce_level!(lvl.lvl, global_fbr_map, local_fbr_map, task_map, factor, P, coalescent)
end