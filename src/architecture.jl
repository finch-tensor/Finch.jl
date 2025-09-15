"""
    AbstractDevice

A datatype representing a device on which tasks can be executed.
"""

abstract type AbstractDevice end
abstract type AbstractVirtualDevice end

"""
    local_memory(dev::AbstractDevice)

Return the default local memory space of `dev`.
"""
function local_memory end

"""
    shared_memory(dev::AbstractDevice)

Return the default shared memory space of `dev`.
"""
function shared_memory end

"""
    global_memory(dev::AbstractDevice)

Return the default global memory space of `dev`.
"""
function global_memory end

"""
    AbstractTask

An individual processing unit on a device, responsible for running code.
"""
abstract type AbstractTask end
abstract type AbstractVirtualTask end

"""
    get_num_tasks(dev::AbstractDevice)

Return the number of tasks on the device dev.
"""
function get_num_tasks end

"""
    get_task_num(task::AbstractTask)

Return the task number of `task`.
"""
function get_task_num end

"""
    get_device(task::AbstractTask)

Return the device that `task` is running on.
"""
function get_device end

"""
    get_parent_task(task::AbstractTask)

Return the task which spawned `task`.
"""
function get_parent_task end

get_num_tasks(ctx::AbstractCompiler) = get_num_tasks(get_task(ctx))
get_num_tasks(task::AbstractTask) = get_num_tasks(get_device(task))
get_task_num(ctx::AbstractCompiler) = get_task_num(get_task(ctx))
get_device(ctx::AbstractCompiler) = get_device(get_task(ctx))
get_parent_task(ctx::AbstractCompiler) = get_parent_task(get_task(ctx))

function is_on_device(ctx::AbstractCompiler, dev)
    res = false
    task = get_task(ctx)
    while task != nothing
        if get_device(task) == dev
            res = true
            break
        end
        task = get_parent_task(task)
    end
    return res
end

"""
    aquire_lock!(dev::AbstractDevice, val)

Lock the lock, val, on the device dev, waiting until it can acquire lock.
"""
aquire_lock!(dev::AbstractDevice, val) = nothing

"""
    release_lock!(dev::AbstractDevice, val)

Release the lock, val, on the device dev.
"""
release_lock!(dev::AbstractDevice, val) = nothing

"""
    get_lock(dev::AbstractDevice, arr, idx, ty)

Given a device, an array of elements of type ty, and an index to the array, idx, gets a lock of type ty associated to arr[idx] on dev.
"""
get_lock(dev::AbstractDevice, arr, idx, ty) = nothing

"""
    make_lock(ty)

Makes a lock of type ty.
"""
function make_lock end

"""
    Serial()

A Task that represents a serial CPU execution.
"""
struct Serial <: AbstractDevice end
serial() = Serial()
get_num_tasks(::Serial) = 1
struct VirtualSerial <: AbstractVirtualTask end
virtualize(ctx, ex, ::Type{Serial}) = VirtualSerial()
lower(ctx::AbstractCompiler, task::VirtualSerial, ::DefaultStyle) = :(Finch.Serial())
virtual_call_def(ctx, alg, ::typeof(serial), Any) = VirtualSerial()
FinchNotation.finch_leaf(device::VirtualSerial) = virtual(device)
get_num_tasks(::VirtualSerial) = literal(1)
Base.:(==)(::Serial, ::Serial) = true
Base.:(==)(::VirtualSerial, ::VirtualSerial) = true

"""
    SerialTask()

A Task that represents a serial CPU execution.
"""
struct SerialTask <: AbstractDevice end
get_device(::SerialTask) = Serial()
get_parent_task(::SerialTask) = nothing
get_task_num(::SerialTask) = 1
struct VirtualSerialTask <: AbstractVirtualTask end
virtualize(ctx, ex, ::Type{SerialTask}) = VirtualSerialTask()
lower(ctx::AbstractCompiler, task::VirtualSerialTask, ::DefaultStyle) = :(SerialTask())
FinchNotation.finch_leaf(device::VirtualSerialTask) = virtual(device)
get_device(::VirtualSerialTask) = VirtualSerial()
get_parent_task(::VirtualSerialTask) = nothing
get_task_num(::VirtualSerialTask) = literal(1)

struct SerialMemory end
struct VirtualSerialMemory end
FinchNotation.finch_leaf(mem::SerialMemory) = virtual(mem)
virtualize(ctx, ex, ::Type{SerialMemory}) = VirtualSerialMemory()
local_memory(::Serial) = SerialMemory()
shared_memory(::Serial) = SerialMemory()
global_memory(::Serial) = SerialMemory()
local_memory(::VirtualSerial) = VirtualSerialMemory()
shared_memory(::VirtualSerial) = VirtualSerialMemory()
global_memory(::VirtualSerial) = VirtualSerialMemory()

transfer(device::Union{Serial,SerialMemory}, arr) = arr

"""
    CPU(n)

A device that represents a CPU with n threads.
"""
struct CPU{id} <: AbstractDevice
    n::Int
end
cpu(id, n=Threads.nthreads()) = CPU{id}(n)
get_num_tasks(dev::CPU{tag}) where {tag} = dev.n
@kwdef struct VirtualCPU <: AbstractVirtualDevice
    n
    id
end
function virtualize(ctx, ex, ::Type{CPU{id}}) where {id}
    n = freshen(ctx, :n)
    push_preamble!(
        ctx,
        quote
            $n = ($ex.n)
        end,
    )
    VirtualCPU(value(n, Int), literal(id))
end
function virtual_call_def(
    ctx, alg, ::typeof(cpu), ::Any, n=value(:($(Threads.nthreads)()), Int)
)
    n_2 = freshen(ctx, :n)
    push_preamble!(
        ctx,
        quote
            $n_2 = $(ctx(n))
        end,
    )
    VirtualCPU(value(n_2, Int))
end
function lower(ctx::AbstractCompiler, device::VirtualCPU, ::DefaultStyle)
    :(Finch.CPU($(ctx(device.id)), $(ctx(device.n))))
end
get_num_tasks(device::VirtualCPU) = device.n
Base.:(==)(::CPU{id1}, ::CPU{id2}) where {id1, id2} = id1 == id2
Base.:(==)(a::VirtualCPU, b::VirtualCPU) = a.id == b.id

FinchNotation.finch_leaf(device::VirtualCPU) = virtual(device)

struct CPULocalMemory
    device::CPU{id} where {id}
end
struct VirtualCPULocalMemory
    device::VirtualCPU
end
FinchNotation.finch_leaf(mem::VirtualCPULocalMemory) = virtual(mem)
function virtualize(ctx, ex, ::Type{CPULocalMemory})
    VirtualCPULocalMemory(virtualize(ctx, :($ex.device), CPU))
end
function lower(ctx::AbstractCompiler, mem::VirtualCPULocalMemory, ::DefaultStyle)
    :(Finch.CPULocalMemory($(ctx(mem.device))))
end

struct CPUSharedMemory
    device::CPU{id} where {id}
end
struct VirtualCPUSharedMemory
    device::VirtualCPU
end
FinchNotation.finch_leaf(mem::VirtualCPUSharedMemory) = virtual(mem)
function virtualize(ctx, ex, ::Type{CPUSharedMemory})
    VirtualCPULocalMemory(virtualize(ctx, :($ex.device), CPU{ex.device.id}))
end
function lower(ctx::AbstractCompiler, mem::VirtualCPUSharedMemory, ::DefaultStyle)
    :(Finch.CPUSharedMemory($(ctx(mem.device{mem.device.id}))))
end

local_memory(device::CPU) = CPULocalMemory(device)
shared_memory(device::CPU) = CPUSharedMemory(device)
global_memory(device::CPU) = CPUSharedMemory(device)
local_memory(device::VirtualCPU) = VirtualCPULocalMemory(device)
shared_memory(device::VirtualCPU) = VirtualCPUSharedMemory(device)
global_memory(device::VirtualCPU) = VirtualCPUSharedMemory(device)

struct CPUThread{Parent} <: AbstractTask
    tid::Int
    dev::CPU{id} where {id}
    parent::Parent
end
get_device(task::CPUThread) = task.device
get_parent_task(task::CPUThread) = task.parent
get_task_num(task::CPUThread) = task.tid

struct CPULocalArray{A}
    device::CPU{id} where {id}
    data::Vector{A}
end

function CPULocalArray{A}(device::CPU{id}) where {A, id}
    CPULocalArray{A}(device, [A([]) for _ in 1:(device.n)])
end

Base.eltype(::Type{CPULocalArray{A}}) where {A} = eltype(A)
Base.ndims(::Type{CPULocalArray{A}}) where {A} = ndims(A)

transfer(device::Union{CPUThread,CPUSharedMemory}, arr::AbstractArray) = arr
function transfer(mem::CPULocalMemory, arr::AbstractArray)
    CPULocalArray{typeof(arr)}(mem.device, [copy(arr) for _ in 1:(mem.device.n)])
end
function transfer(task::CPUThread, arr::CPULocalArray)
    if get_device(task) == arr.device
        temp = arr.data[task.tid]
        return temp
    else
        return arr
    end
end
function transfer(dst::AbstractArray, arr::AbstractArray)
    return arr
end

"""
    transfer(device, arr)

If the array is not on the given device, it creates a new version of this array
on that device and copies the data in to it, according to the `device` trait. If
the device is simply a data buffer, we copy the array into the buffer.
"""
transfer(device, arr) = arr

"""
    distribute(ctx, arr, device, diff, style)

If the virtual array is not on the given device, copy the array to that device. This
function may modify underlying data arrays, but cannot change the virtual itself. This
function is used to move data to the device before a kernel is launched. Since this
function may modify the root node, iterators in-progress may need to be updated.
We can store new root objects in the `diff` dictionary.
"""
distribute(ctx, arr, device, diff, style) = arr

"""
redistribute(ctx, node, diff)

    When the root node is distributed, several iterators may need to be updated.
The `redistribute` function traverses `tns` and updates it based on the updated
objects in the `diff` dictionary.
"""
redistribute(ctx, node, diff) = node

function redistribute(ctx::AbstractCompiler, node::FinchNode, diff)
    if node.kind === virtual
        virtual(redistribute(ctx, node.val, diff))
    elseif istree(node)
        similarterm(
            node, operation(node), map(x -> redistribute(ctx, x, diff), arguments(node))
        )
    else
        node
    end
end

"""
    HostLocal()

From the host, distribute the tensor to device local memory.
"""
struct HostLocal end
const host_local = HostLocal()
"""
    DeviceLocal()

From the device, load the local version of the tensor.
"""
struct DeviceLocal end
const device_local = DeviceLocal()
"""
    HostShared()

From the host, distribute the tensor to device shared memory.
"""
struct HostShared end
const host_shared = HostShared()
"""
    DeviceShared()

From the device, load the shared view of the tensor.
"""
struct DeviceShared end
const device_shared = DeviceShared()
"""
    HostGlobal()

From the host, distribute the tensor to device global memory.
"""
struct HostGlobal end
const host_global = HostGlobal()
"""
    DeviceGlobal()

From the device, load the global view of the tensor.
"""
struct DeviceGlobal end
const device_global = DeviceGlobal()

function distribute_buffer(ctx, buf, device, ::HostLocal)
    buf_2 = freshen(ctx, buf)
    push_preamble!(
        ctx,
        quote
            $buf_2 = $transfer($(ctx(local_memory(device))), $buf)
        end,
    )
    return buf_2
end

function distribute_buffer(ctx, buf, device, ::HostGlobal)
    buf_2 = freshen(ctx, buf)
    push_preamble!(
        ctx,
        quote
            $buf_2 = $transfer($(ctx(global_memory(device))), $buf)
        end,
    )
    return buf_2
end

function distribute_buffer(ctx, buf, device, ::HostShared)
    buf_2 = freshen(ctx, buf)
    push_preamble!(
        ctx,
        quote
            $buf_2 = $transfer($(ctx(shared_memory(device))), $buf)
        end,
    )
    push_epilogue!(
        ctx,
        quote
            $buf = $transfer($buf, $buf_2)
        end,
    )
    return buf_2
end

###*******
##Check for errors
###*******
function distribute_buffer(
    ctx, buf, task, style::Union{DeviceLocal,DeviceShared,DeviceGlobal}
)
    buf_2 = freshen(ctx, buf)
    push_preamble!(
        ctx,
        quote
            $buf_2 = $transfer($(ctx(task)), $buf)
        end,
    )
    return buf_2
end

@inline function make_lock(::Type{Threads.Atomic{T}}) where {T}
    return Threads.Atomic{T}(zero(T))
end

@inline function make_lock(::Type{Base.Threads.SpinLock})
    return Threads.SpinLock()
end

@inline function aquire_lock!(dev::CPU, val::Threads.Atomic{T}) where {T}
    # Keep trying to catch x === false so we can set it to true.
    while (Threads.atomic_cas!(x, zero(T), one(T)) === one(T))
    end
    # when it is true because we did it, we leave, but let's make sure it is true in debug mode.
    @assert x === one(T)
end

@inline function aquire_lock!(dev::CPU, val::Threads.SpinLock)
    lock(val)
    @assert islocked(val)
end

@inline function release_lock!(dev::CPU, val::Threads.Atomic{T}) where {T}
    # set the atomic to false so someone else can grab it.
    Threads.atomic_cas!(x, one(T), zero(T))
end

@inline function release_lock!(dev::CPU, val::Base.Threads.SpinLock)
    @assert islocked(val)
    unlock(val)
end

function get_lock(dev::CPU, arr, idx, ::Type{Threads.Atomic{T}}) where {T}
    return arr[idx]
end

function get_lock(dev::CPU, arr, idx, ::Type{Base.Threads.SpinLock})
    return arr[idx]
end

struct VirtualCPUThread <: AbstractVirtualTask
    tid
    dev::VirtualCPU
    parent
end

function virtualize(ctx, ex, ::Type{CPUThread{Parent}}) where {Parent}
    VirtualCPUThread(
        value(sym.tid, Int),
        virtualize(ctx, :($sym.dev), CPU),
        virtualize(ctx, :($sym.parent), Parent),
    )
end
function lower(ctx::AbstractCompiler, task::VirtualCPUThread, ::DefaultStyle)
    :(Finch.CPUThread($(ctx(task.tid)), $(ctx(task.dev)), $(ctx(task.parent))))
end
FinchNotation.finch_leaf(device::VirtualCPUThread) = virtual(device)
get_device(task::VirtualCPUThread) = task.dev
get_parent_task(task::VirtualCPUThread) = task.parent
get_task_num(task::VirtualCPUThread) = task.tid

struct Converter{f,T} end

(::Converter{f,T})(x) where {f,T} = T(f(x))

@propagate_inbounds function atomic_modify!(::Serial, vec, idx, op, x)
    @inbounds begin
        vec[idx] = op(vec[idx], x)
    end
end

@propagate_inbounds function atomic_modify!(::CPU, vec, idx, op, x)
    Base.unsafe_modify!(pointer(vec, idx), op, x, :sequentially_consistent)
end

@propagate_inbounds function atomic_modify!(::CPU, vec, idx, op::Chooser{Vf}, x) where {Vf}
    Base.unsafe_replace!(pointer(vec, idx), Vf, x, :sequentially_consistent)
end

@propagate_inbounds function atomic_modify!(::CPU, vec, idx, op::typeof(overwrite), x)
    Base.unsafe_store!(pointer(vec, idx), x, :sequentially_consistent)
end

@propagate_inbounds function atomic_modify!(
    ::CPU, vec, idx, op::InitWriter{Vf}, x
) where {Vf}
    Base.unsafe_store!(pointer(vec, idx), x, :sequentially_consistent)
end

for T in [
    Bool,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
    Int128,
    UInt128,
    Float16,
    Float32,
    Float64,
]
    if T <: AbstractFloat
        ops = [+, -]
    else
        ops = [+, -, *, /, %, &, |, ⊻, ⊼, max, min]
    end
    for op in ops
        @eval @propagate_inbounds function atomic_modify!(
            ::CPU, vec::Vector{$T}, idx, ::typeof($op), x::$T
        )
            UnsafeAtomics.modify!(pointer(vec, idx), $op, x, UnsafeAtomics.seq_cst)
        end
    end

    @eval @propagate_inbounds function atomic_modify!(
        ::CPU, vec::Vector{$T}, idx, op::Chooser{Vf}, x::$T
    ) where {Vf}
        UnsafeAtomics.cas!(
            pointer(vec, idx), $T(Vf), x, UnsafeAtomics.seq_cst, UnsafeAtomics.seq_cst
        )
    end
end

abstract type AbstractSchedule end

abstract type AbstractVirtualSchedule end

struct FinchStaticSchedule{schedule} <: AbstractSchedule end

struct VirtualFinchStaticSchedule <: AbstractVirtualSchedule
    schedule
end

FinchNotation.finch_leaf(x::VirtualFinchStaticSchedule) = virtual(x)

function virtualize(ctx, ex, ::Type{FinchStaticSchedule{schedule}}) where {schedule}
    VirtualFinchStaticSchedule(schedule)
end

function lower(ctx, ex::VirtualFinchStaticSchedule)
    :(FinchStaticSchedule{$(QuoteNode(ex.schedule))}())
end

static_schedule(schedule::Symbol=:dynamic) = FinchStaticSchedule{schedule}()

function virtual_call_def(
    ctx, alg, ::typeof(static_schedule), ::Any, schedule=value(:dynamic, Symbol)
)
    VirtualFinchStaticSchedule(schedule.val)
end

struct FinchGreedySchedule{schedule} <: AbstractSchedule
    chk::Int
end

struct VirtualFinchGreedySchedule <: AbstractVirtualSchedule
    chk
    schedule
end

FinchNotation.finch_leaf(x::VirtualFinchGreedySchedule) = virtual(x)

function virtualize(ctx, ex, ::Type{FinchGreedySchedule{schedule}}) where {schedule}
    chk = freshen(ctx, :chk)
    push_preamble!(
        ctx,
        quote
            $chk = ($ex.chk)
        end,
    )
    VirtualFinchGreedySchedule(value(chk, Int), schedule)
end

function lower(ctx, ex::VirtualFinchGreedySchedule)
    :(FinchGreedySchedule{$(QuoteNode(ex.schedule))}($(ctx(ex.chk))))
end

greedy_schedule(chk::Int=1, schedule::Symbol=:static) = FinchGreedySchedule{schedule}(chk)

function virtual_call_def(
    ctx,
    alg,
    ::typeof(greedy_schedule),
    ::Any,
    chk=value(:(1), Int),
    schedule=value(:static, Symbol),
)
    chk_2 = freshen(ctx, :chk)
    push_preamble!(
        ctx,
        quote
            $chk_2 = $(ctx(chk))
        end,
    )
    VirtualFinchGreedySchedule(value(chk_2, Int), schedule.val)
end

struct FinchJuliaSchedule{schedule} <: AbstractSchedule
    chk::Int
end

struct VirtualFinchJuliaSchedule <: AbstractVirtualSchedule
    chk
    schedule
end

FinchNotation.finch_leaf(x::VirtualFinchJuliaSchedule) = virtual(x)

function virtualize(ctx, ex, ::Type{FinchJuliaSchedule{schedule}}) where {schedule}
    chk = freshen(ctx, :chk)
    push_preamble!(
        ctx,
        quote
            $chk = ($ex.chk)
        end,
    )
    VirtualFinchJuliaSchedule(value(chk, Int), schedule)
end

function lower(ctx, ex::VirtualFinchJuliaSchedule)
    :(FinchJuliaSchedule{$(QuoteNode(ex.schedule))}($(ctx(ex.chk))))
end

function julia_schedule(chk::Int=1, schedule::Union{Symbol,Nothing}=nothing)
    if isnothing(schedule)
        schedule = if VERSION >= v"1.11.0"
            :greedy
        else
            :dynamic
        end
    end
    FinchJuliaSchedule{schedule}(chk)
end

function virtual_call_def(
    ctx,
    alg,
    ::typeof(julia_schedule),
    ::Any,
    chk=value(:(1), Int),
    schedule=value(nothing, Union{Symbol,Nothing}),
)
    chk_2 = freshen(ctx, :chk)
    push_preamble!(
        ctx,
        quote
            $chk_2 = $(ctx(chk))
        end,
    )
    if isnothing(schedule.val)
        schedule.val = if VERSION >= v"1.11.0"
            :greedy
        else
            :dynamic
        end
    end
    VirtualFinchJuliaSchedule(value(chk_2, Int), schedule.val)
end

function virtual_parallel_region(f, ctx, ::Serial)
    contain(f, ctx)
end

function virtual_parallel_region(
    f, ctx, ext::VirtualParallelDimension, device::VirtualCPU,
    schedule::VirtualFinchStaticSchedule,
)
    tid = freshen(ctx, :tid)
    i_lo = call(
        +,
        call(fld, call(*, measure(ext.ext), call(-, value(tid, Int), 1)), device.n),
        1,
    )
    i_hi = call(fld, call(*, measure(ext.ext), value(tid, Int)), device.n)

    code = contain(ctx) do ctx_2
        subtask = VirtualCPUThread(value(tid, Int), device, ctx_2.code.task)
        contain(ctx_2; task=subtask) do ctx_3
            f(ctx_3, i_lo, i_hi) do inner
                inner
            end
        end
    end
    return quote
        Threads.@threads $(QuoteNode(schedule.schedule)) for $tid in 1:($(ctx(device.n)))
            Finch.@barrier begin
                @inbounds @fastmath begin
                    $code
                end
                nothing
            end
        end
    end
end

function virtual_parallel_region(
    f, ctx, ext::VirtualParallelDimension, device::VirtualCPU,
    schedule::VirtualFinchGreedySchedule,
)
    tid = freshen(ctx, :tid)
    chk_id = freshen(ctx, :chk_id)
    chk_ctr = freshen(ctx, :chk_ctr)
    num_chks = freshen(ctx, :num_chks)
    i_lo = call(+, call(*, schedule.chk, call(-, value(chk_id, Int), 1)), 1)
    i_hi = call(min, call(*, schedule.chk, value(chk_id, Int)), measure(ext.ext))

    push_preamble!(
        ctx,
        quote
            $chk_ctr = Threads.Atomic{Int}(0)
            $num_chks = cld($(ctx(measure(ext.ext))), $(ctx(schedule.chk)))
        end,
    )

    code = contain(ctx) do ctx_2
        subtask = VirtualCPUThread(value(tid, Int), device, ctx_2.code.task)
        contain(ctx_2; task=subtask) do ctx_3
            f(ctx_3, i_lo, i_hi) do inner
                quote
                    while true
                        $chk_id = Threads.atomic_add!($chk_ctr, 1)
                        if $chk_id > $num_chks
                            break
                        end
                        $inner
                    end
                end
            end
        end
    end

    return quote
        Threads.@threads $(QuoteNode(schedule.schedule)) for $tid in 1:($(ctx(device.n)))
            Finch.@barrier begin
                @inbounds @fastmath begin
                    $code
                end
                nothing
            end
        end
    end
end

function virtual_parallel_region(
    f, ctx, ext::VirtualParallelDimension, device::VirtualCPU,
    schedule::VirtualFinchJuliaSchedule,
)
    tid_tmp = freshen(ctx, :tid)
    tid_ch = freshen(ctx, :tid_ch)
    tid = freshen(ctx, :tid)
    chk_id = freshen(ctx, :chk_id)
    num_chks = freshen(ctx, :num_chks)
    i_lo = call(+, call(*, schedule.chk, call(-, value(chk_id, Int), 1)), 1)
    i_hi = call(min, call(*, schedule.chk, value(chk_id, Int)), measure(ext.ext))

    push_preamble!(
        ctx,
        quote
            $num_chks = cld($(ctx(measure(ext.ext))), $(ctx(schedule.chk)))
        end,
    )

    code = contain(ctx) do ctx_2
        push_preamble!(
            ctx_2,
            quote
                $tid = take!($tid_ch)
            end,
        )
        push_epilogue!(
            ctx_2,
            quote
                put!($tid_ch, $tid)
            end,
        )
        subtask = VirtualCPUThread(value(tid, Int), device, ctx_2.code.task)
        contain(ctx_2; task=subtask) do ctx_3
            f(ctx_3, i_lo, i_hi) do inner
                inner
            end
        end
    end

    return quote
        $tid_ch = Channel{Int}($(ctx(device.n)))
        for $tid_tmp in 1:($(ctx(device.n)))
            put!($tid_ch, $tid_tmp)
        end

        Threads.@threads $(QuoteNode(schedule.schedule)) for $chk_id in 1:($num_chks)
            Finch.@barrier begin
                @inbounds @fastmath begin
                    $code
                end
                nothing
            end
        end
    end
end
