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
const serial = Serial()
get_num_tasks(::Serial) = 1
struct VirtualSerial <: AbstractVirtualTask end
virtualize(ctx, ex, ::Type{Serial}) = VirtualSerial()
lower(ctx::AbstractCompiler, task::VirtualSerial, ::DefaultStyle) = :(Serial())
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
struct CPU <: AbstractDevice
    n::Int
end
CPU() = CPU(Threads.nthreads())
get_num_tasks(dev::CPU) = dev.n
@kwdef struct VirtualCPU <: AbstractVirtualDevice
    ex
    n
end
function virtualize(ctx, ex, ::Type{CPU})
    sym = freshen(ctx, :cpu)
    push_preamble!(
        ctx,
        quote
            $sym = $ex
        end,
    )
    VirtualCPU(sym, virtualize(ctx, :($sym.n), Int))
end
function lower(ctx::AbstractCompiler, device::VirtualCPU, ::DefaultStyle)
    something(device.ex, :(CPU($(ctx(device.n)))))
end
get_num_tasks(::VirtualCPU) = literal(1)
Base.:(==)(::CPU, ::CPU) = true
Base.:(==)(::VirtualCPU, ::VirtualCPU) = true #This is not strictly true. A better approach would name devices, and give them parents so that we can be sure to parallelize through the processor hierarchy.

FinchNotation.finch_leaf(device::VirtualCPU) = virtual(device)

struct CPULocalMemory
    device::CPU
end
struct VirtualCPULocalMemory
    device::VirtualCPU
end
FinchNotation.finch_leaf(mem::VirtualCPULocalMemory) = virtual(mem)
function virtualize(ctx, ex, ::Type{CPULocalMemory})
    VirtualCPULocalMemory(virtualize(ctx, :($ex.device), CPU))
end
function lower(ctx::AbstractCompiler, mem::VirtualCPULocalMemory, ::DefaultStyle)
    :(CPULocalMemory($(ctx(mem.device))))
end

struct CPUSharedMemory
    device::CPU
end
struct VirtualCPUSharedMemory
    device::VirtualCPU
end
FinchNotation.finch_leaf(mem::VirtualCPUSharedMemory) = virtual(mem)
function virtualize(ctx, ex, ::Type{CPUSharedMemory})
    VirtualCPULocalMemory(virtualize(ctx, :($ex.device), CPU))
end
function lower(ctx::AbstractCompiler, mem::VirtualCPUSharedMemory, ::DefaultStyle)
    :(CPUSharedMemory($(ctx(mem.device))))
end

local_memory(device::CPU) = CPULocalMemory(device)
shared_memory(device::CPU) = CPUSharedMemory(device)
global_memory(device::CPU) = CPUSharedMemory(device)
local_memory(device::VirtualCPU) = VirtualCPULocalMemory(device)
shared_memory(device::VirtualCPU) = VirtualCPUSharedMemory(device)
global_memory(device::VirtualCPU) = VirtualCPUSharedMemory(device)

struct CPUThread{Parent} <: AbstractTask
    tid::Int
    dev::CPU
    parent::Parent
end
get_device(task::CPUThread) = task.device
get_parent_task(task::CPUThread) = task.parent
get_task_num(task::CPUThread) = task.tid

struct CPULocalArray{A}
    device::CPU
    data::Vector{A}
end

function CPULocalArray{A}(device::CPU) where {A}
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
        virtualize(ctx, :($sym.tid), Int),
        virtualize(ctx, :($sym.dev), CPU),
        virtualize(ctx, :($sym.parent), Parent),
    )
end
function lower(ctx::AbstractCompiler, task::VirtualCPUThread, ::DefaultStyle)
    :(CPUThread($(ctx(task.tid)), $(ctx(task.dev)), $(ctx(task.parent))))
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

function virtual_parallel_region(f, ctx, ::VirtualSerial)
    contain(f, ctx; task=VirtualSerialTask())
end

function virtual_parallel_region(f, ctx, device::VirtualCPU)
    tid = freshen(ctx, :tid)

    code = contain(ctx) do ctx_2
        subtask = VirtualCPUThread(value(tid, Int), device, ctx_2.code.task)
        contain(f, ctx_2; task=subtask)
    end

    return quote
        Threads.@threads for $tid in 1:($(ctx(device.n)))
            Finch.@barrier begin
                @inbounds @fastmath begin
                    $code
                end
                nothing
            end
        end
    end
end
