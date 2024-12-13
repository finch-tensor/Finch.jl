abstract type AbstractDevice end
abstract type AbstractVirtualDevice end
abstract type AbstractTask end
abstract type AbstractVirtualTask end


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
    CPU(n)

A device that represents a CPU with n threads.
"""
struct CPU <: AbstractDevice
    n::Int
end
CPU() = CPU(Threads.nthreads())
@kwdef struct VirtualCPU <: AbstractVirtualDevice
    ex
    n
end
function virtualize(ctx, ex, ::Type{CPU})
    sym = freshen(ctx, :cpu)
    push_preamble!(ctx, quote
        $sym = $ex
    end)
    VirtualCPU(sym, virtualize(ctx, :($sym.n), Int))
end
lower(ctx::AbstractCompiler, device::VirtualCPU, ::DefaultStyle) =
    something(device.ex, :(CPU($(ctx(device.n)))))

FinchNotation.finch_leaf(device::VirtualCPU) = virtual(device)


"""
    Serial()

A device that represents a serial CPU execution.
"""
struct Serial <: AbstractTask end
const serial = Serial()
get_device(::Serial) = CPU(1)
get_task(::Serial) = nothing
struct VirtualSerial <: AbstractVirtualTask end
virtualize(ctx, ex, ::Type{Serial}) = VirtualSerial()
lower(ctx::AbstractCompiler, task::VirtualSerial, ::DefaultStyle) = :(Serial())
FinchNotation.finch_leaf(device::VirtualSerial) = virtual(device)
virtual_get_device(::VirtualSerial) = VirtualCPU(nothing, 1)
virtual_get_task(::VirtualSerial) = nothing


struct CPUThread{Parent} <: AbstractTask
    tid::Int
    dev::CPU
    parent::Parent
end
get_device(task::CPUThread) = task.device
get_task(task::CPUThread) = task.parent

@inline function make_lock(::Type{Threads.Atomic{T}}) where {T}
    return Threads.Atomic{T}(zero(T))
end

@inline function make_lock(::Type{Base.Threads.SpinLock})
    return Threads.SpinLock()
end

@inline function aquire_lock!(dev:: CPU, val::Threads.Atomic{T}) where {T}
    # Keep trying to catch x === false so we can set it to true.
    while (Threads.atomic_cas!(x, zero(T), one(T)) === one(T))

    end
    # when it is true because we did it, we leave, but let's make sure it is true in debug mode.
    @assert x === one(T)
end

@inline function aquire_lock!(dev:: CPU, val::Threads.SpinLock)
    lock(val)
    @assert islocked(val)
end

@inline function release_lock!(dev:: CPU, val::Threads.Atomic{T}) where {T}
    # set the atomic to false so someone else can grab it.
    Threads.atomic_cas!(x, one(T), zero(T))
end

@inline function release_lock!(dev:: CPU, val::Base.Threads.SpinLock)
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
        virtualize(ctx, :($sym.parent), Parent)
    )
end
lower(ctx::AbstractCompiler, task::VirtualCPUThread, ::DefaultStyle) = :(CPUThread($(ctx(task.tid)), $(ctx(task.dev)), $(ctx(task.parent))))
FinchNotation.finch_leaf(device::VirtualCPUThread) = virtual(device)
virtual_get_device(task::VirtualCPUThread) = task.dev
virtual_get_task(task::VirtualCPUThread) = task.parent

struct CPULocalMemory
    device::CPU
end
function moveto(vec::V, mem::CPULocalMemory) where {V <: Vector}
    CPULocalVector{V}(mem.device, [copy(vec) for _ in 1:mem.device.n])
end

struct CPULocalVector{V}
    device::CPU
    data::Vector{V}
end

CPULocalVector{V}(device::CPU) where {V} =
    CPULocalVector{V}(device, [V([]) for _ in 1:device.n])

Base.eltype(::Type{CPULocalVector{V}}) where {V} = eltype(V)
Base.ndims(::Type{CPULocalVector{V}}) where {V} = ndims(V)

function moveto(vec::Vector, device::CPU)
    return vec
end

function moveto(vec::Vector, task::CPUThread)
    return copy(vec)
end

function moveto(vec::CPULocalVector, task::CPUThread)
    temp = vec.data[task.tid]
    return temp
end

"""
    local_memory(device)

Returns the local memory type for a given device.
"""
function local_memory(device::CPU)
    return CPULocalMemory(device)
end

function local_memory(device::Serial)
    return device
end

struct Converter{f, T} end

(::Converter{f, T})(x) where {f, T} = T(f(x))

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

@propagate_inbounds function atomic_modify!(::CPU, vec, idx, op::InitWriter{Vf}, x) where {Vf}
    Base.unsafe_store!(pointer(vec, idx), x, :sequentially_consistent)
end

for T = [Bool, Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Int128, UInt128, Float16, Float32, Float64]
    if T <: AbstractFloat
        ops = [+, -]
    else
        ops = [+, -, *, /, %, &, |, ⊻, ⊼, max, min]
    end
    for op in ops
        @eval @propagate_inbounds function atomic_modify!(::CPU, vec::Vector{$T}, idx, ::typeof($op), x::$T)
            UnsafeAtomics.modify!(pointer(vec, idx), $op, x, UnsafeAtomics.seq_cst)
        end
    end

    @eval @propagate_inbounds function atomic_modify!(::CPU, vec::Vector{$T}, idx, op::Chooser{Vf}, x::$T) where {Vf}
        UnsafeAtomics.cas!(pointer(vec, idx), $T(Vf), x, UnsafeAtomics.seq_cst, UnsafeAtomics.seq_cst)
    end

end