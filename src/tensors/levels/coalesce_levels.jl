###task[pos] gives the processor that owns Coalesce in position pos. AKA which channel in the multimemory channel to access.
###the subfiber p is contained at position ptr[p] on the sublevel in CHANNEL task[p].
###ptr[p] = 0 means unallocated.

##### NOTE: In order to get coalesce levels to work, I need to recursively construct a new coalescent FROM THE ORIGINAL COALESCENT

"""
    CoalesceLevel{device, Lvl}()

CoalesceLevel uses an internal Coalesced representation, but unified the result into a single Tensor when
entering read-only mode.

```jldoctest
julia> tensor_tree(Tensor(Dense(Coalesce(cpu(:t, 2), Element(0.0))), 4))
4-Tensor
└─ Dense [1:4]
   ├─ [1]: Coalesce(1) -> 
   ├─ [2]: Coalesce(2) -> 
   ├─ [3]: Coalesce(3) -> 
   └─ [4]: Coalesce(4) -> 
```
"""
struct CoalesceLevel{mode,Device,Lvl,Coalescent,Schedule,Accumulator} <: AbstractLevel
    device::Device
    lvl::Lvl
    coalescent::Coalescent
    schedule::Schedule
    accumulator::Accumulator
end
const Coalesce = CoalesceLevel

getmode(lvl::CoalesceLevel{mode,Device,Lvl,Coalescent,Schedule,Accumulator}) where {mode,Device,Lvl,Coalescent,Schedule,Accumulator} = mode

gen_accumulator(lvl::AbstractLevel, fill_value, eltype::Type, dims...) =
    similar_level(lvl, fill_value, eltype, dims...)

function gen_accumulator(lvl::DenseLevel, fill_value, eltype::Type, dims...)
    Dense(gen_accumulator(lvl.lvl, fill_value, eltype, dims[1:(end - 1)]...), dims[end])
end

function gen_accumulator(
    lvl::SparseListLevel{Ti}, fill_value, eltype::Type, dim, tail...
) where {Ti}
    SparseHashLevel{Ti,true}(gen_accumulator(lvl.lvl, fill_value, eltype, tail...), dim)
end

function gen_accumulator(lvl::SparseByteMapLevel, fill_value, eltype::Type, dims...)
    SparseByteMap(
        gen_accumulator(lvl.lvl, fill_value, eltype, dims[1:(end - 1)]...), dims[end]
    )
end

function CoalesceLevel(device::Device, lvl::Lvl; mode=:normalize) where {Device,Lvl}
    Tp = postype(lvl)
    coal_lvl = lvl
    while typeof(coal_lvl) <: CoalesceLevel
        coal_lvl = coal_lvl.lvl
    end
    P = get_num_tasks(device)
    coalescent = similar_level(
        coal_lvl, level_fill_value(Lvl), level_eltype(Lvl), level_size(coal_lvl)...
    )
    if mode == :fast
        accum = nothing
    else
        accum = gen_accumulator(
            coal_lvl, level_fill_value(Lvl), level_eltype(Lvl), level_size(coal_lvl)...
        )
    end
    schedule = FinchStaticSchedule{:dynamic}()
    CoalesceLevel{Device}(
        device,
        transfer(MultiChannelMemory(device, P), lvl),
        coalescent,
        schedule,
        transfer(MultiChannelMemory(device, P), accum),;
        mode
    )
end

function CoalesceLevel(device, lvl, coalescent, schedule, accumulator; mode=:normalize)
    CoalesceLevel{typeof(device)}(device, lvl, coalescent, schedule, accumulator; mode)
end

function CoalesceLevel{Device}(
    device, lvl::Lvl, coalescent::Coalescent, schedule::Schedule, accumulator::Accumulator; mode=:normalize
) where {Device,Lvl,Coalescent,Schedule,Accumulator}
    CoalesceLevel{mode,Device,Lvl,Coalescent,Schedule,Accumulator}(
        device, lvl, coalescent, schedule, accumulator
    )
end

function Base.summary(
    ::Coalesce{mode,Device,Lvl,Coalescent,Schedule,Accumulator}
) where {mode,Device,Lvl,Coalescent,Schedule,Accumulator}
    "Coalesce($(Lvl))"
end

function similar_level(
    lvl::Coalesce{mode,Device,Lvl,Coalescent,Schedule,Accumulator}, fill_value, eltype::Type, dims...
) where {mode,Device,Lvl,Coalescent,Schedule,Accumulator}
    lvl_2 = similar_level(lvl.lvl, fill_value, eltype, dims...)
    coal_2 = similar_level(lvl.coalescent, fill_value, eltype, dims...)
    CoalesceLevel(
        lvl.device,
        lvl_2,
        coal_2,
        lvl.schedule,
        lvl.accumulator;
        mode=getmode(lvl)
    )
end

function postype(
    ::Type{<:Coalesce{mode,Device,Lvl,Coalescent,Schedule,Accumulator}}
) where {mode,Device,Lvl,Coalescent,Schedule,Accumulator}
    postype(Lvl)
end

function transfer(device, lvl::CoalesceLevel)
    lvl_2 = transfer(device, lvl.lvl)
    coal_2 = transfer(device, lvl.coalescent)
    return CoalesceLevel(lvl.device, lvl_2, coal_2, lvl.schedule, lvl.accumulator; mode=getmode(lvl))
end

function pattern!(lvl::CoalesceLevel)
    CoalesceLevel(lvl.device, pattern!(lvl.lvl), lvl.coalescent, lvl.schedule, lvl.accumulator; mode=getmode(lvl))
end

function set_fill_value!(lvl::CoalesceLevel, init)
    CoalesceLevel(
        lvl.device,
        set_fill_value!(lvl.lvl, init),
        set_fill_value!(lvl.coalescent, init),
        lvl.schedule,
        lvl.accumulator;
        mode=getmode(lvl)
    )
end

function Base.resize!(lvl::CoalesceLevel, dims...)
    CoalesceLevel(
        lvl.device,
        resize!(lvl.lvl, dims...),
        resize!(lvl.coalescent, dims...),
        lvl.schedule,
        resize!(lvl.accumulator, dims...);
        mode=getmode(lvl)
    )
end

function Base.show(
    io::IO, lvl::CoalesceLevel{mode,Device,Lvl,Coalescent,Schedule,Accumulator}
) where {mode,Device,Lvl,Coalescent,Schedule,Accumulator}
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
    ::Type{<:CoalesceLevel{mode,Device,Lvl,Coalescent,Schedule,Accumulator}}
) where {mode,Device,Lvl,Coalescent,Schedule,Accumulator} = level_ndims(Lvl)
@inline level_size(
    lvl::CoalesceLevel{mode,Device,Lvl,Coalescent,Schedule,Accumulator}
) where {mode,Device,Lvl,Coalescent,Schedule,Accumulator} = level_size(lvl.lvl)
@inline level_axes(
    lvl::CoalesceLevel{mode,Device,Lvl,Coalescent,Schedule,Accumulator}
) where {mode,Device,Lvl,Coalescent,Schedule,Accumulator} = level_axes(lvl.lvl)
@inline level_eltype(
    ::Type{CoalesceLevel{mode,Device,Lvl,Coalescent,Schedule,Accumulator}}
) where {mode,Device,Lvl,Coalescent,Schedule,Accumulator} = level_eltype(Lvl)
@inline level_fill_value(
    ::Type{<:CoalesceLevel{mode,Device,Lvl,Coalescent,Schedule,Accumulator}}
) where {mode,Device,Lvl,Coalescent,Schedule,Accumulator} = level_fill_value(Lvl)

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
    accumulator
    Tv
    Device
    Lvl
    Coalescent
    Schedule
    qos_stop
    mode
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
            $(ctx(lvl.coalescent)),
            $(lvl.tag).schedule,
            $(ctx(lvl.accumulator));
            mode=($(QuoteNode(lvl.mode))),
        )
    end
end

function virtualize(
    ctx, ex, ::Type{CoalesceLevel{mode,Device,Lvl,Coalescent,Schedule,Accumulator}}, tag=:lvl
) where {mode,Device,Lvl,Coalescent,Schedule,Accumulator}
    tag = freshen(ctx, tag)
    schedule = freshen(ctx, tag, :_schedule)

    push_preamble!(
        ctx,
        quote
            $tag = $ex
            $schedule = $tag.schedule
        end,
    )
    device_2 = virtualize(ctx, :($tag.device), Device, tag)
    lvl_2 = virtualize(ctx, :($tag.lvl), Lvl, tag)
    coalescent_2 = virtualize(ctx, :($tag.coalescent), Coalescent, tag)
    schedule_2 = virtualize(ctx, :($tag.schedule), Schedule, tag)
    accumulator_2 = virtualize(ctx, :($tag.accumulator), Accumulator, tag)
    qos_stop = freshen(ctx, tag, :_qos_stop)
    VirtualCoalesceLevel(
        tag,
        device_2,
        lvl_2,
        coalescent_2,
        schedule_2,
        accumulator_2,
        typeof(level_fill_value(Lvl)),
        Device,
        Lvl,
        Coalescent,
        Schedule,
        qos_stop,
        mode,
    )
end

function distribute_level(
    ctx, lvl::VirtualCoalesceLevel, arch, diff, style::Union{HostShared}
)
    diff[lvl.tag] = VirtualCoalesceLevel(
        lvl.tag,
        lvl.device,
        distribute_level(ctx, lvl.lvl, arch, diff, style),
        lvl.coalescent,
        lvl.schedule,
        lvl.accumulator,
        lvl.Tv,
        lvl.Device,
        lvl.Lvl,
        lvl.Coalescent,
        lvl.Schedule,
        lvl.qos_stop,
        lvl.mode,
    )
end

function distribute_level(
    ctx, lvl::VirtualCoalesceLevel, arch, diff, style::Union{DeviceGlobal,HostGlobal}
)
    diff[lvl.tag] = VirtualCoalesceLevel(
        lvl.tag,
        lvl.device,
        lvl.lvl,
        distribute_level(ctx, lvl.coalescent, arch, diff, style),
        lvl.schedule,
        lvl.accumulator,
        lvl.Tv,
        lvl.Device,
        lvl.Lvl,
        lvl.Coalescent,
        lvl.Schedule,
        lvl.qos_stop,
        lvl.mode,
    )
end

function distribute_level(
    ctx, lvl::VirtualCoalesceLevel, arch, diff, style::Union{DeviceLocal,HostLocal}
)
    diff[lvl.tag] = VirtualCoalesceLevel(
        lvl.tag,
        lvl.device,
        distribute_level(ctx, lvl.lvl, arch, diff, style),
        distribute_level(ctx, lvl.coalescent, arch, diff, style),
        lvl.schedule,
        lvl.accumulator,
        lvl.Tv,
        lvl.Device,
        lvl.Lvl,
        lvl.Coalescent,
        lvl.Schedule,
        lvl.qos_stop,
        lvl.mode,
    )
end

function distribute_level(
    ctx, lvl::VirtualCoalesceLevel, arch, diff, style::Union{DeviceShared}
)
    Tp = postype(lvl)
    tag = lvl.tag
    if lvl.device == get_device(arch)
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
            lvl_2,
            lvl.coalescent,
            lvl.schedule,
            lvl.accumulator,
            lvl.Tv,
            lvl.Device,
            lvl.Lvl,
            lvl.Coalescent,
            lvl.Schedule,
            lvl.qos_stop,
            lvl.mode,
        )
    else
        dev = get_device(get_device(arch))
        distribute_level(ctx, lvl.coalescent, dev, diff, HostShared())
        diff[lvl.tag] = VirtualCoalesceLevel(
            lvl.tag,
            lvl.device,
            distribute_level(ctx, lvl.lvl, arch, diff, style),
            distribute_level(ctx, lvl.coalescent, arch, diff, style),
            lvl.schedule,
            lvl.accumulator,
            lvl.Tv,
            lvl.Device,
            lvl.Lvl,
            lvl.Coalescent,
            lvl.Schedule,
            lvl.qos_stop,
            lvl.mode,
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
            lvl.accumulator,
            lvl.Tv,
            lvl.Device,
            lvl.Lvl,
            lvl.Coalescent,
            lvl.Schedule,
            lvl.qos_stop,
            lvl.mode,
        ),
    )
end

Base.summary(lvl::VirtualCoalesceLevel) = "Coalesce($(lvl.Lvl))"

function virtual_level_resize!(ctx, lvl::VirtualCoalesceLevel, dims...)
    lvl.lvl = virtual_level_resize!(ctx, lvl.lvl, dims...)
    lvl.coalescent = virtual_level_resize!(ctx, lvl.coalescent, dims...)
    return lvl
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
                lvl_3 = distribute_level(
                    ctx_3, lvl.lvl, channel_task, diff, DeviceShared()
                )
                lvl_4 = declare_level!(ctx_3, lvl_3, literal(0), init)
                freeze_level!(ctx_3, lvl_4, literal(0))
                nothing
            end
        end,
    )
    coalescent_2 = declare_level!(ctx, lvl.coalescent, literal(0), init)
    freeze_level!(ctx, coalescent_2, literal(0))
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

            push_preamble!(ctx_2,
                quote
                    $(lvl.qos_stop) = $(ctx_2(pos_stop))
                end)

            push_preamble!(
                ctx_2,
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
                    lvl_3 = distribute_level(
                        ctx_3, lvl.lvl, channel_task, diff, DeviceShared()
                    )
                    push_preamble!(ctx_3,
                        contain(ctx_3) do ctx_4
                            lvl_3 = thaw_level!(
                                ctx_4, lvl_3, call(-, pos_start, literal(1))
                            )
                            assemble_level!(ctx_4, lvl_3, pos_start, pos_stop)
                        end,
                    )
                    lvl_4 = freeze_level!(ctx_3, lvl_3, pos_stop)
                    nothing
                end,
            )

            push_preamble!(ctx_2,
                contain(ctx_2) do ctx_3
                    thaw_level!(ctx_3, lvl.coalescent, call(-, pos_start, literal(1)))
                    assemble_level!(ctx_3, lvl.coalescent, pos_start, pos_stop)
                end)
            freeze_level!(ctx_2, lvl.coalescent, pos_stop)
        end)
    lvl
end

supports_reassembly(::VirtualCoalesceLevel) = false
init_gfm(P) = [[1] for _ in 1:P]
init_fast_meta(P) = [ones(Int, P) for _ in 1:P]

function freeze_level!(ctx, lvl::VirtualCoalesceLevel, pos)
    @assert !is_on_device(ctx, lvl.device)
    P = ctx(get_num_tasks(lvl.device))
    lvl_e = ctx(lvl)
    lvl_c = ctx(lvl.coalescent)
    
    ##On init, factor is both a dimensional maximum and communicates unwrapping for Dense(Coalesce(Sparse)) data
    factor = ctx(pos)
    max_pos = factor
    mode = lvl.mode

    global_fbr_map = freshen(ctx, :gfm)
    lastpos = freshen(ctx, :lastpos)
    tid = freshen(ctx, :tid)
    dec = freshen(ctx, :declared)
    if mode == :fast
        push_preamble!(
            ctx,
            quote
                $dec = Finch.setup_coalesce!($(lvl_e), $max_pos, $(lvl_c))
                $lastpos = Finch.init_fast_meta($P)
                if $dec
                    Threads.@threads for $tid in 1:($P)
                        Finch.coalesce_fast!($tid, $lastpos, $P, $(lvl_e), $(lvl_c), false)
                    end
                end
            end,
        )
    else
        lb = freshen(ctx, :lb)
        ub = freshen(ctx, :ub)
        mask = freshen(ctx, :mask)
        nnz = freshen(ctx, :nnz)
        sid = freshen(ctx, :sid)
        push_preamble!(ctx,
            quote
                $nnz = Finch.get_total_nnz($(lvl_e))
                if $nnz > 0
                    Threads.@threads for $tid in 1:($P)
                        $lb, $ub = Finch.balance(
                            $(lvl_e).lvl, $tid, $P, $nnz, Finch.MergeNormalization()
                        )
                        $mask = Finch.tuplemask($lb, $ub)

                        $(contain(ctx) do ctx_2
                            diff = Dict()
                            channel_dev = VirtualMultiChannelMemory(
                                lvl.device, get_num_tasks(lvl.device)
                            )
                            channel_task = VirtualMemoryChannel(
                                value(tid, Int), channel_dev, get_task(ctx_2)
                            )
                            accum_2 = distribute_level(
                                ctx_2, lvl.accumulator, channel_task, diff, DeviceShared()
                            )
                            accum_2 = declare_level!(ctx_2, accum_2, literal(0), literal(0))
                            push_preamble!(
                                ctx_2,
                                assemble_level!(ctx_2, accum_2, literal(1), literal(1)),
                            )

                            N = level_ndims(lvl.Lvl)
                            Tp = postype(lvl)

                            accumulator_var = variable(freshen(ctx_2, :accumulator))
                            set_binding!(
                                ctx_2, accumulator_var,
                                virtual(VirtualSubFiber(accum_2, literal(1)))
                            )

                            push_preamble!(ctx_2,
                                quote
                                    for $sid in 1:($P)
                                        $(contain(ctx_2) do ctx_3
                                            channel_dev_2 = VirtualMultiChannelMemory(
                                                lvl.device, get_num_tasks(lvl.device)
                                            )
                                            channel_task_2 = VirtualMemoryChannel(
                                                value(sid, Int),
                                                channel_dev_2,
                                                get_task(ctx_3),
                                            )
                                            shard_2 = distribute_level(
                                                ctx_3,
                                                lvl.lvl,
                                                channel_task_2,
                                                diff,
                                                DeviceShared(),
                                            )

                                            shard_var = variable(freshen(ctx_3, :shard))
                                            set_binding!(
                                                ctx_3,
                                                shard_var,
                                                virtual(
                                                    VirtualSubFiber(shard_2, literal(1))
                                                ),
                                            )

                                            mask_var = variable(freshen(ctx_3, :mask))
                                            set_binding!(
                                                ctx_3,
                                                mask_var,
                                                virtual(
                                                    virtualize(ctx_3, mask, TupleMask{N,Tp})
                                                ),
                                            )

                                            exts = virtual_level_size(ctx_3, shard_2)
                                            inds = [
                                                index(freshen(ctx_3, :i, n)) for
                                                n in 1:length(exts)
                                            ]

                                            op = literal(+)
                                            prgm = assign(
                                                access(
                                                    accumulator_var, updater(op), inds...
                                                ),
                                                op,
                                                access(shard_var, reader(), inds...),
                                            )
                                            prgm = sieve(
                                                access(mask_var, reader(), inds...), prgm
                                            )
                                            for (ind, ext) in zip(inds, exts)
                                                prgm = loop(ind, ext, prgm)
                                            end
                                            prgm = instantiate!(ctx_3, prgm)
                                            ctx_3(prgm)
                                        end)
                                    end
                                end)
                            accum_2 = freeze_level!(ctx_2, accum_2, literal(1))
                            nothing
                        end)
                    end

                    $dec = Finch.setup_coalesce!($(lvl_e).accumulator, $max_pos, $(lvl_c))
                    if $dec
                        Threads.@threads for $tid in 1:($P)
                            Finch.coalesce_fast!(
                                $tid, nothing, $P, $(lvl_e), $(lvl_c), false
                            )
                        end
                    else
                        Threads.@threads for $tid in 1:($P)
                            $(contain(ctx) do ctx_2
                                diff = Dict()
                                channel_dev = VirtualMultiChannelMemory(
                                    lvl.device, get_num_tasks(lvl.device)
                                )
                                channel_task = VirtualMemoryChannel(
                                    value(tid, Int), channel_dev, get_task(ctx_2)
                                )
                                accum_2 = distribute_level(
                                    ctx_2, lvl.accumulator, channel_task, diff,
                                    DeviceShared(),
                                )

                                own_2 = distribute_level(
                                    ctx_2, lvl.lvl, channel_task, diff, DeviceShared()
                                )
                                own_2 = declare_level!(ctx_2, own_2, literal(0), literal(0))
                                push_preamble!(
                                    ctx_2,
                                    assemble_level!(ctx_2, own_2, literal(1), literal(1)),
                                )

                                own_var = variable(freshen(ctx_2, :own))
                                set_binding!(
                                    ctx_2, own_var,
                                    virtual(VirtualSubFiber(own_2, literal(1))),
                                )

                                accum_r_var = variable(freshen(ctx_2, :accum_r))
                                set_binding!(
                                    ctx_2, accum_r_var,
                                    virtual(VirtualSubFiber(accum_2, literal(1))),
                                )

                                exts_cp = virtual_level_size(ctx_2, own_2)
                                inds_cp = [
                                    index(freshen(ctx_2, :k, n)) for n in 1:length(exts_cp)
                                ]

                                op_cp = literal(initwrite(virtual_level_fill_value(own_2)))
                                prgm_cp = assign(
                                    access(own_var, updater(op_cp), inds_cp...),
                                    op_cp,
                                    access(accum_r_var, reader(), inds_cp...),
                                )
                                for (ind, ext) in zip(inds_cp, exts_cp)
                                    prgm_cp = loop(ind, ext, prgm_cp)
                                end
                                prgm_cp = instantiate!(ctx_2, prgm_cp)
                                push_preamble!(ctx_2, ctx_2(prgm_cp))

                                own_2 = freeze_level!(ctx_2, own_2, literal(1))
                                nothing
                            end)
                        end
                        
                        $lastpos = Finch.init_fast_meta($P)
                        Threads.@threads for $tid in 1:($P)
                            Finch.coalesce_fast!(
                                $tid, $lastpos, $P, $(lvl_e).lvl, $(lvl_c), false
                            )
                        end
                    end
                end
            end,
        )
    end
    return lvl
end

function thaw_level!(ctx::AbstractCompiler, lvl::VirtualCoalesceLevel, pos)
    @assert !is_on_device(ctx, lvl.device)

    push_preamble!(ctx,
        quote
            $(lvl.qos_stop) = $(ctx(pos))
        end)

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

    return Thunk(;
        body=(ctx) -> VirtualHollowSubFiber(lvl.lvl, pos, fbr.dirty)
    )
end

function setup_coalesce!(lvl::CoalesceLevel, max_pos, coalescent)
    return setup_coalesce!(lvl.lvl, max_pos, coalescent)
end

function coalesce_level!(
    lvl::CoalesceLevel, global_fbr_map, factor, max_dim, P, coalescent, mode
)
    if max_dim < 1
        return nothing
    end

    coalesce_level!(lvl.lvl, global_fbr_map, factor, max_dim, P, coalescent, mode)
end

function coalesce_fast!(tid, meta, P, lvl::CoalesceLevel, coalescent, was_dense)
    coalesce_fast!(tid, meta, P, lvl.lvl, coalescent, was_dense)
end

###Load balancer stuff

struct MergeNormalization end

@inbounds function decrement_idxs(idxs, shapes)
    idxs = copy(idxs)
    pos = 1
    while pos <= length(idxs)
        if idxs[pos] > 1
            idxs[pos] -= 1
            return idxs
        else
            idxs[pos] = shapes[pos]
            pos += 1
        end
    end
    error("nnz too small to load balance across P processors")
end

@inbounds function balance(lvl, tid, P, nnz, style::Union{MergeNormalization})
    shapes = collect(Finch.level_size(lvl))
    max_dim = length(shapes)

    base = div(nnz, P)
    remainder = nnz % P
    lower_work = (tid - 1) * base + min(tid - 1, remainder)
    upper_work = tid * base + min(tid, remainder)

    if tid == 1
        lb_idxs = ntuple(_ -> 1, max_dim)
    else
        lb_idxs = find_normalizer_split(lvl, lower_work, P, copy(shapes), max_dim)
    end
    lb = Tuple(lb_idxs)

    if tid == P
        ub_idxs = shapes
    else
        next_lb_idxs = find_normalizer_split(lvl, upper_work, P, copy(shapes), max_dim)
        ub_idxs = decrement_idxs(next_lb_idxs, shapes)
    end
    ub = Tuple(ub_idxs)

    return (lb, ub)
end

@inbounds function find_normalizer_split(lvl, target_work, P, idxs, max_dim)
    dim = 0
    while dim < max_dim
        lo = 1
        hi = idxs[max_dim - dim]
        truth = -1
        while lo <= hi
            candidate = div(lo + hi, 2)
            idxs[max_dim - dim] = candidate
            stored = 0
            for p in 1:P
                stored += Finch.countstored_level(lvl, 1, idxs, 0, p, true)
            end
            if stored >= target_work
                truth = candidate
                hi = candidate - 1
            else
                lo = candidate + 1
            end
        end
        idxs[max_dim - dim] = truth
        dim += 1
    end
    return idxs
end

function get_total_nnz(lvl::AbstractLevel)
    while !(lvl isa ElementLevel)
        lvl = lvl.lvl
    end
    sum(length, lvl.val.data)
end