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

function getmode(
    lvl::CoalesceLevel{mode,Device,Lvl,Coalescent,Schedule,Accumulator}
) where {mode,Device,Lvl,Coalescent,Schedule,Accumulator}
    mode
end

function gen_accumulator(lvl::AbstractLevel, fill_value, eltype::Type, dims...)
    similar_level(lvl, fill_value, eltype, dims...)
end

function gen_accumulator(lvl::DenseLevel, fill_value, eltype::Type, dims...)
    Dense(gen_accumulator(lvl.lvl, fill_value, eltype, dims[1:(end - 1)]...), dims[end])
end

function gen_accumulator(lvl::SparseHashLevel{Ti}, fill_value, eltype::Type, dims...) where {Ti}
    SparseHashLevel{Ti}(
        gen_accumulator(lvl.lvl, fill_value, eltype, dims[1:(end - 1)]...),
        dims[end], lvl.subtables,
    )
end

function gen_accumulator(
    lvl::SparseListLevel{Ti}, fill_value, eltype::Type, dim, tail...
) where {Ti}
    SparseHashLevel{Ti}(gen_accumulator(lvl.lvl, fill_value, eltype, tail...), dim)
end

function gen_accumulator(lvl::SparseByteMapLevel, fill_value, eltype::Type, dims...)
    SparseByteMap(
        gen_accumulator(lvl.lvl, fill_value, eltype, dims[1:(end - 1)]...), dims[end]
    )
end

function coalesce_similar_level(lvl::DenseLevel{Ti}, P) where {Ti}
    DenseLevel{Ti}(coalesce_similar_level(lvl.lvl, P), lvl.shape)
end

function coalesce_similar_level(lvl::SparseListLevel{Ti}, P) where {Ti}
    SparseListLevel{Ti}(coalesce_similar_level(lvl.lvl, P), lvl.shape)
end

function coalesce_similar_level(lvl::SparseByteMapLevel{Ti}, P) where {Ti}
    SparseByteMapLevel{Ti}(coalesce_similar_level(lvl.lvl, P), lvl.shape)
end

function CoalesceLevel(device::Device, lvl::Lvl; mode=:normalize) where {Device,Lvl}
    Tp = postype(lvl)
    coal_lvl = lvl
    while typeof(coal_lvl) <: CoalesceLevel
        coal_lvl = coal_lvl.lvl
    end
    P = get_num_tasks(device)
    # All participating hashes use the same bucket layout.
    coalescent = coalesce_similar_level(coal_lvl, P)
    if mode == :fast
        accum = nothing
    else
        accum = gen_accumulator(
            coal_lvl, level_fill_value(Lvl), level_eltype(Lvl), level_size(coal_lvl)...
        )
        accum = coalesce_similar_level(accum, P)
    end
    schedule = FinchStaticSchedule{:dynamic}()
    CoalesceLevel{Device}(
        device,
        transfer(MultiChannelMemory(device, P), coalesce_similar_level(lvl, P)),
        coalescent,
        schedule,
        transfer(MultiChannelMemory(device, P), accum), ;
        mode,
    )
end

function CoalesceLevel(device, lvl, coalescent, schedule, accumulator; mode=:normalize)
    CoalesceLevel{typeof(device)}(device, lvl, coalescent, schedule, accumulator; mode)
end

function CoalesceLevel{Device}(
    device, lvl::Lvl, coalescent::Coalescent, schedule::Schedule, accumulator::Accumulator;
    mode=:normalize
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
    lvl::Coalesce{mode,Device,Lvl,Coalescent,Schedule,Accumulator}, fill_value,
    eltype::Type, dims...
) where {mode,Device,Lvl,Coalescent,Schedule,Accumulator}
    lvl_2 = similar_level(lvl.lvl, fill_value, eltype, dims...)
    coal_2 = similar_level(lvl.coalescent, fill_value, eltype, dims...)
    CoalesceLevel(
        lvl.device,
        lvl_2,
        coal_2,
        lvl.schedule,
        lvl.accumulator;
        mode=getmode(lvl),
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
    return CoalesceLevel(
        lvl.device, lvl_2, coal_2, lvl.schedule, lvl.accumulator; mode=getmode(lvl)
    )
end

function pattern!(lvl::CoalesceLevel)
    CoalesceLevel(
        lvl.device,
        pattern!(lvl.lvl),
        lvl.coalescent,
        lvl.schedule,
        lvl.accumulator;
        mode=getmode(lvl),
    )
end

function set_fill_value!(lvl::CoalesceLevel, init)
    CoalesceLevel(
        lvl.device,
        set_fill_value!(lvl.lvl, init),
        set_fill_value!(lvl.coalescent, init),
        lvl.schedule,
        lvl.accumulator;
        mode=getmode(lvl),
    )
end

function Base.resize!(lvl::CoalesceLevel, dims...)
    CoalesceLevel(
        lvl.device,
        resize!(lvl.lvl, dims...),
        resize!(lvl.coalescent, dims...),
        lvl.schedule,
        resize!(lvl.accumulator, dims...);
        mode=getmode(lvl),
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
    sampler
    declared
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
    ctx, ex, ::Type{CoalesceLevel{mode,Device,Lvl,Coalescent,Schedule,Accumulator}},
    tag=:lvl
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
    if mode == :fast
        sampler = nothing
    else
        sampler = freshen(ctx, tag, :sampler)
    end
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
        sampler,
        false,
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
        lvl.sampler,
        lvl.declared,
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
        lvl.sampler,
        lvl.declared,
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
        # Each outer task needs its own accumulator, or nested tasks share channels.
        lvl.mode == :fast ? lvl.accumulator :
        distribute_level(ctx, lvl.accumulator, arch, diff, style),
        lvl.Tv,
        lvl.Device,
        lvl.Lvl,
        lvl.Coalescent,
        lvl.Schedule,
        lvl.qos_stop,
        lvl.mode,
        lvl.sampler,
        lvl.declared,
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
            lvl.sampler,
            lvl.declared,
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
            lvl.sampler,
            lvl.declared,
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
            lvl.sampler,
            lvl.declared,
        ),
    )
end

Base.summary(lvl::VirtualCoalesceLevel) = "Coalesce($(lvl.Lvl))"

function virtual_level_resize!(ctx, lvl::VirtualCoalesceLevel, dims...)
    lvl.lvl = virtual_level_resize!(ctx, lvl.lvl, dims...)
    lvl.coalescent = virtual_level_resize!(ctx, lvl.coalescent, dims...)
    if lvl.mode != :fast
        lvl.accumulator = virtual_level_resize!(ctx, lvl.accumulator, dims...)
    end
    return lvl
end
virtual_level_size(ctx, lvl::VirtualCoalesceLevel) = virtual_level_size(ctx, lvl.lvl)
virtual_level_eltype(lvl::VirtualCoalesceLevel) = virtual_level_eltype(lvl.lvl)
virtual_level_fill_value(lvl::VirtualCoalesceLevel) = virtual_level_fill_value(lvl.lvl)
@inline sample_dims(lvl::VirtualCoalesceLevel) = sample_dims(lvl.lvl)
@inline all_dense(lvl::VirtualCoalesceLevel) = true & all_dense(lvl.lvl)

function declare_level!(ctx, lvl::VirtualCoalesceLevel, pos, init)
    @assert !is_on_device(ctx, lvl.device)
    dcl_pos = supports_reassembly(lvl.lvl) ? pos : literal(0)

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
                lvl_4 = declare_level!(ctx_3, lvl_3, dcl_pos, init)
                freeze_level!(ctx_3, lvl_4, dcl_pos)
                nothing
            end
        end,
    )
    coalescent_2 = declare_level!(ctx, lvl.coalescent, dcl_pos, init)
    freeze_level!(ctx, coalescent_2, dcl_pos)
    lvl.declared = true
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

# Named functions so the generated code has no comprehension for the parser to trip on.
get_bands(P, shapes, style::MergeDense) = [balance(t, P, shapes, style) for t in 1:(P)]
get_bands(sampler, P, shapes, style::MergeRandom) =
    [balance(sampler, t, P, shapes, style) for t in 1:(P)]
do_band_ranges(shapes, bands) = [band_range(b..., shapes) for b in bands]

function freeze_level!(ctx, lvl::VirtualCoalesceLevel, pos)
    lvl.declared || return lvl
    @assert !is_on_device(ctx, lvl.device)
    P = ctx(get_num_tasks(lvl.device))
    lvl_e = ctx(lvl)
    lvl_c = ctx(lvl.coalescent)

    ##On init, factor is both a dimensional maximum and communicates unwrapping for Dense(Coalesce(Sparse)) data
    factor = ctx(pos)
    max_pos = factor
    mode = lvl.mode

    tid = freshen(ctx, :tid)
    if mode == :fast
        # Tasks wrote their shards directly, without accumulating.
        push_preamble!(
            ctx,
            quote
                Finch.coalesce_shards!($(lvl_e).lvl, $(lvl_c), $P, $max_pos, nothing)
            end,
        )
    else
        lb = freshen(ctx, :lb)
        ub = freshen(ctx, :ub)
        bands = freshen(ctx, :bands)
        ranges = freshen(ctx, :ranges)
        mask = freshen(ctx, :mask)
        nnz = freshen(ctx, :nnz)
        sid = freshen(ctx, :sid)
        unordered = freshen(ctx, :unordered)
        shapes = freshen(ctx, :shapes)
        tsize = sample_dims(lvl)
        dense = all_dense(lvl)
        band = if dense
            :(Finch.get_bands($P, $shapes, Finch.MergeDense()))
        else
            :(Finch.get_bands($(lvl.sampler), $P, $shapes, Finch.MergeRandom()))
        end

        push_preamble!(ctx,
            quote
                $nnz, $unordered = Finch.get_total_nnz($(lvl_e), true)
                $shapes = Finch.level_size($(lvl_e))
                if $nnz > 0
                    if !$dense
                        $(lvl.sampler) = Finch.build_sampler($(lvl_e), $P, $nnz, $tsize)
                    end
                    # Task tid sums every shard's entries in its band into its accumulator.
                    $bands = $band
                    $ranges = Finch.do_band_ranges($shapes, $bands)
                    Finch.@barrier Threads.@threads for $tid in 1:($P)
                        $lb, $ub = $bands[$tid]
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
                            accum_2 = declare_level!(ctx_2, accum_2, pos, literal(0))
                            push_preamble!(
                                ctx_2,
                                assemble_level!(ctx_2, accum_2, literal(1), literal(1)),
                            )

                            N = level_ndims(lvl.Lvl)
                            Tp = postype(lvl)

                            accumulator_var = variable(freshen(ctx_2, :accumulator))
                            set_binding!(
                                ctx_2, accumulator_var,
                                virtual(VirtualSubFiber(accum_2, literal(1))),
                            )

                            push_preamble!(ctx_2,
                                quote
                                    Finch.isempty_band($lb, $ub) || for $sid in 1:($P)
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

                    if typeof($(lvl_e).accumulator) === typeof($(lvl_e).lvl)
                        Finch.coalesce_shards!(
                            $(lvl_e).accumulator, $(lvl_c), $P, $max_pos, $ranges
                        )
                    else
                        # SparseList levels, including children of hashes,
                        # accumulate into hashes. Restore the output format
                        # whenever any level differs before merging the bands.
                        Finch.@barrier Threads.@threads for $tid in 1:($P)

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
                        Finch.coalesce_shards!($(lvl_e).lvl, $(lvl_c), $P, $max_pos, $ranges)
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

function coalesce_level!(
    lvl::CoalesceLevel, global_fbr_map, factor, max_dim, P, coalescent, mode
)
    if max_dim < 1
        return nothing
    end

    coalesce_level!(lvl.lvl, global_fbr_map, factor, max_dim, P, coalescent, mode)
end

###Load balancer stuff

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

@inbounds function balance(
    sampler::Vector{NTuple{m,Int}}, tid, P, shapes, style::MergeRandom
) where {m}
    neg = ntuple(_ -> -1, m)
    start = searchsortedlast(sampler, neg; by=reverse) + 1
    n = length(sampler) - start + 1
    base = div(n, P)
    remainder = n % P

    lb_at(t) =
        t == 1 ? ntuple(_ -> 1, m) : sampler[start + (t - 1) * base + min(t - 1, remainder)]

    lb = Tuple(lb_at(tid))

    if tid == P
        ub = Tuple(shapes)
    else
        next = lb_at(tid + 1)
        # Repeated sample points leave a band empty.
        all(isone, next) && return empty_band(shapes)
        ub = Tuple(decrement_idxs(collect(next), shapes))
    end

    return isempty_band(lb, ub) ? empty_band(shapes) : (lb, ub)
end

@inbounds function idxs_at_flat(flat, shapes)
    idxs = Vector{Int}(undef, length(shapes))
    remaining = flat - 1
    for pos in 1:length(shapes)
        idxs[pos] = remaining % shapes[pos] + 1
        remaining = remaining ÷ shapes[pos]
    end
    idxs
end

@inbounds function balance(tid, P, shapes, style::MergeDense)
    total = prod(shapes)
    base, rem = divrem(total, P)
    lower = (tid - 1) * base + min(tid - 1, rem) + 1
    upper = tid * base + min(tid, rem)
    lower > upper && return empty_band(shapes)

    lb = Tuple(idxs_at_flat(lower, shapes))
    ub = Tuple(idxs_at_flat(upper, shapes))

    return (lb, ub)
end

# Stored entries per shard. A byte map over an element stores a dense `val`, so
# count its `srt` instead.
function shard_counts(lvl::AbstractLevel)
    while !(lvl isa ElementLevel)
        lvl isa SparseByteMapLevel && lvl.lvl isa ElementLevel && return length.(lvl.srt.data)
        lvl = lvl.lvl
    end
    return length.(lvl.val.data)
end

function get_total_nnz(lvl::AbstractLevel, unordered)
    counts = shard_counts(lvl)
    while !(lvl isa ElementLevel)
        lvl = lvl.lvl
        unordered = unordered & !isa(lvl, SparseListLevel)
    end
    return sum(counts), unordered
end

function sample(tid, lvl::CoalesceLevel)
    tup, idx = sample(tid, lvl.lvl)
    return tup
end

function build_sampler(lvl::AbstractLevel, P, nnz, tsize)
    counts = shard_counts(lvl)
    sampler = Vector{NTuple{tsize,Int}}(undef, 0)
    for p in 1:P
        active = round(Int, 200 * P * counts[p] / nnz)
        for _ in 1:active
            push!(sampler, sample(p, lvl))
        end
    end
    sort!(sampler; by=reverse)
    sampler
end
