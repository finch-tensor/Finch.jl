"""
    MutexLevel{Val, Lvl}()

Mutex Level Protects the level directly below it with atomics

Each position in the level below the Mutex level is protected by a lock.
```jldoctest
julia> tensor_tree(Tensor(Dense(Mutex(Element(0.0))), [1, 2, 3]))
3-Tensor
└─ Dense [1:3]
   ├─ [1]: Mutex ->
   │  └─ 1.0
   ├─ [2]: Mutex ->
   │  └─ 2.0
   └─ [3]: Mutex ->
      └─ 3.0
```
"""
struct MutexLevel{AVal,Lvl} <: AbstractLevel
    lvl::Lvl
    locks::AVal
end
const Mutex = MutexLevel

MutexLevel(lvl) = MutexLevel(lvl, Base.Threads.SpinLock[])
#MutexLevel(lvl::Lvl, locks::AVal) where {Lvl, AVal} =
#    MutexLevel{AVal, Lvl}(lvl, locks)
Base.summary(::MutexLevel{AVal,Lvl}) where {Lvl,AVal} = "MutexLevel($(AVal), $(Lvl))"

function similar_level(
    lvl::Mutex{AVal,Lvl}, fill_value, eltype::Type, dims...
) where {Lvl,AVal}
    MutexLevel(similar_level(lvl.lvl, fill_value, eltype, dims...))
end

postype(::Type{<:MutexLevel{AVal,Lvl}}) where {Lvl,AVal} = postype(Lvl)

function transfer(lvl::MutexLevel, device)
    lvl_2 = transfer(lvl.lvl, device)
    locks_2 = transfer(lvl.locks, device)
    return MutexLevel(lvl_2, locks_2)
end

pattern!(lvl::MutexLevel) = MutexLevel(pattern!(lvl.lvl), lvl.locks)
function set_fill_value!(lvl::MutexLevel, init)
    MutexLevel(set_fill_value!(lvl.lvl, init), lvl.locks)
end
# TODO: FIXME: Need toa dopt the number of dims
Base.resize!(lvl::MutexLevel, dims...) = MutexLevel(resize!(lvl.lvl, dims...), lvl.locks)

function Base.show(io::IO, lvl::MutexLevel{AVal,Lvl}) where {AVal,Lvl}
    print(io, "Mutex(")
    if get(io, :compact, false)
        print(io, "…")
    else
        show(IOContext(io), lvl.lvl)
        print(io, ", ")
        show(IOContext(io, :typeinfo => AVal), lvl.locks)
    end
    print(io, ")")
end

labelled_show(io::IO, ::SubFiber{<:MutexLevel}) =
    print(io, "Mutex -> ")

function labelled_children(fbr::SubFiber{<:MutexLevel})
    lvl = fbr.lvl
    pos = fbr.pos
    [LabelledTree(SubFiber(lvl.lvl, pos))]
end

@inline level_ndims(::Type{<:MutexLevel{AVal,Lvl}}) where {AVal,Lvl} = level_ndims(Lvl)
@inline level_size(lvl::MutexLevel{AVal,Lvl}) where {AVal,Lvl} = level_size(lvl.lvl)
@inline level_axes(lvl::MutexLevel{AVal,Lvl}) where {AVal,Lvl} = level_axes(lvl.lvl)
@inline level_eltype(::Type{MutexLevel{AVal,Lvl}}) where {AVal,Lvl} = level_eltype(Lvl)
@inline level_fill_value(::Type{<:MutexLevel{AVal,Lvl}}) where {AVal,Lvl} =
    level_fill_value(Lvl)
data_rep_level(::Type{<:MutexLevel{AVal,Lvl}}) where {AVal,Lvl} = data_rep_level(Lvl)

function isstructequal(a::T, b::T) where {T<:Mutex}
    typeof(a.locks) == typeof(b.locks) &&
        isstructequal(a.lvl, b.lvl)
end
# Temporary hack to deal with SpinLock allocate undefined references.

# FIXME: These.
(fbr::Tensor{<:MutexLevel})() = SubFiber(fbr.lvl, 1)()
(fbr::SubFiber{<:MutexLevel})() = fbr #TODO this is not consistent somehow
function (fbr::SubFiber{<:MutexLevel})(idxs...)
    return Tensor(fbr.lvl.lvl)(idxs...)
end

countstored_level(lvl::MutexLevel, pos) = countstored_level(lvl.lvl, pos)

mutable struct VirtualMutexLevel <: AbstractVirtualLevel
    lvl # the level below us.
    ex
    locks
    Tv
    Val
    AVal
    Lvl
end
postype(lvl::MutexLevel) = postype(lvl.lvl)

postype(lvl::VirtualMutexLevel) = postype(lvl.lvl)

is_level_injective(ctx, lvl::VirtualMutexLevel) = [is_level_injective(ctx, lvl.lvl)...]

function is_level_concurrent(ctx, lvl::VirtualMutexLevel)
    (below, c) = is_level_concurrent(ctx, lvl.lvl)
    return (below, c)
end

function is_level_atomic(ctx, lvl::VirtualMutexLevel)
    (below, _) = is_level_atomic(ctx, lvl.lvl)
    return (below, true)
end

function lower(ctx::AbstractCompiler, lvl::VirtualMutexLevel, ::DefaultStyle)
    quote
        $MutexLevel{$(lvl.AVal),$(lvl.Lvl)}($(ctx(lvl.lvl)), $(lvl.locks))
    end
end

function virtualize(ctx, ex, ::Type{MutexLevel{AVal,Lvl}}, tag=:lvl) where {AVal,Lvl}
    sym = freshen(ctx, tag)
    atomics = freshen(ctx, tag, :_locks)
    push_preamble!(
        ctx,
        quote
            $sym = $ex
            $atomics = $ex.locks
        end,
    )
    lvl_2 = virtualize(ctx, :($sym.lvl), Lvl, sym)
    temp = VirtualMutexLevel(
        lvl_2, sym, atomics, typeof(level_fill_value(Lvl)), Val, AVal, Lvl
    )
    temp
end

Base.summary(lvl::VirtualMutexLevel) = "Mutex($(lvl.Lvl))"
function virtual_level_resize!(ctx, lvl::VirtualMutexLevel, dims...)
    (lvl.lvl = virtual_level_resize!(ctx, lvl.lvl, dims...); lvl)
end
virtual_level_size(ctx, lvl::VirtualMutexLevel) = virtual_level_size(ctx, lvl.lvl)
virtual_level_ndims(ctx, lvl::VirtualMutexLevel) = length(virtual_level_size(ctx, lvl.lvl))
virtual_level_eltype(lvl::VirtualMutexLevel) = virtual_level_eltype(lvl.lvl)
virtual_level_fill_value(lvl::VirtualMutexLevel) = virtual_level_fill_value(lvl.lvl)

function declare_level!(ctx, lvl::VirtualMutexLevel, pos, init)
    lvl.lvl = declare_level!(ctx, lvl.lvl, pos, init)
    return lvl
end

function assemble_level!(ctx, lvl::VirtualMutexLevel, pos_start, pos_stop)
    pos_start = cache!(ctx, :pos_start, simplify(ctx, pos_start))
    pos_stop = cache!(ctx, :pos_stop, simplify(ctx, pos_stop))
    idx = freshen(ctx, :idx)
    lockVal = freshen(ctx, :lock)
    push_preamble!(
        ctx,
        quote
            Finch.resize_if_smaller!($(lvl.locks), $(ctx(pos_stop)))
            @inbounds for $idx in ($(ctx(pos_start))):($(ctx(pos_stop)))
                $(lvl.locks)[$idx] = Finch.make_lock(eltype($(lvl.AVal)))
            end
        end,
    )
    assemble_level!(ctx, lvl.lvl, pos_start, pos_stop)
end

supports_reassembly(lvl::VirtualMutexLevel) = supports_reassembly(lvl.lvl)
function reassemble_level!(ctx, lvl::VirtualMutexLevel, pos_start, pos_stop)
    pos_start = cache!(ctx, :pos_start, simplify(ctx, pos_start))
    pos_stop = cache!(ctx, :pos_stop, simplify(ctx, pos_stop))
    idx = freshen(ctx, :idx)
    lockVal = freshen(ctx, :lock)
    push_preamble!(
        ctx,
        quote
            Finch.resize_if_smaller!($lvl.locks, $(ctx(pos_stop)))
            @inbounds for $idx in ($(ctx(pos_start))):($(ctx(pos_stop)))
                $lvl.locks[$idx] = Finch.make_lock(eltype($(lvl.AVal)))
            end
        end,
    )
    reassemble_level!(ctx, lvl.lvl, pos_start, pos_stop)
    lvl
end

function freeze_level!(ctx, lvl::VirtualMutexLevel, pos)
    idx = freshen(ctx, :idx)
    push_preamble!(
        ctx,
        quote
            resize!($(lvl.locks), $(ctx(pos)))
        end,
    )
    lvl.lvl = freeze_level!(ctx, lvl.lvl, pos)
    return lvl
end

function thaw_level!(ctx::AbstractCompiler, lvl::VirtualMutexLevel, pos)
    lvl.lvl = thaw_level!(ctx, lvl.lvl, pos)
    return lvl
end

function virtual_transfer_level(ctx::AbstractCompiler, lvl::VirtualMutexLevel, arch)
    #Add for seperation level too.
    atomics = freshen(ctx, :locksArray)

    push_preamble!(
        ctx,
        quote
            $atomics = $(lvl.locks)
            $(lvl.locks) = $transfer($(lvl.locks), $(ctx(arch)))
        end,
    )
    push_epilogue!(
        ctx,
        quote
            $(lvl.locks) = $atomics
        end,
    )
    virtual_transfer_level(ctx, lvl.lvl, arch)
end

function instantiate(ctx, fbr::VirtualSubFiber{VirtualMutexLevel}, mode)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    if mode.kind === reader
        instantiate(ctx, VirtualSubFiber(lvl.lvl, pos), mode)
    else
        fbr
    end
end

function unfurl(ctx, fbr::VirtualSubFiber{VirtualMutexLevel}, ext, mode, proto)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    if mode.kind === reader
        return unfurl(ctx, VirtualSubFiber(lvl.lvl, pos), ext, mode, proto)
    else
        sym = freshen(ctx, lvl.ex, :after_atomic_lvl)
        atomicData = freshen(ctx, lvl.ex, :atomicArraysAcc)
        lockVal = freshen(ctx, lvl.ex, :lockVal)
        dev = lower(ctx, get_device(ctx.code.task), DefaultStyle())
        push_preamble!(
            ctx,
            quote
                $atomicData = Finch.get_lock(
                    $dev, $(lvl.locks), $(ctx(pos)), eltype($(lvl.AVal))
                )
                $lockVal = Finch.aquire_lock!($dev, $atomicData)
            end,
        )
        res = unfurl(ctx, VirtualSubFiber(lvl.lvl, pos), ext, mode, proto)
        push_epilogue!(
            ctx,
            quote
                Finch.release_lock!($dev, $atomicData)
            end,
        )
        return res
    end
end

function unfurl(ctx, fbr::VirtualHollowSubFiber{VirtualMutexLevel}, ext, mode, proto)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    @assert mode.kind === updater
    sym = freshen(ctx, lvl.ex, :after_atomic_lvl)
    atomicData = freshen(ctx, lvl.ex, :atomicArraysAcc)
    lockVal = freshen(ctx, lvl.ex, :lockVal)
    dev = lower(ctx, get_device(ctx.code.task), DefaultStyle())
    push_preamble!(
        ctx,
        quote
            $atomicData = Finch.get_lock(
                $dev, $(lvl.locks), $(ctx(pos)), eltype($(lvl.AVal))
            )
            $lockVal = Finch.aquire_lock!($dev, $atomicData)
        end,
    )
    res = unfurl(ctx, VirtualHollowSubFiber(lvl.lvl, pos, fbr.dirty), ext, mode, proto)
    push_epilogue!(
        ctx,
        quote
            Finch.release_lock!($dev, $atomicData)
        end,
    )
    return res
end

function lower_assign(ctx, fbr::VirtualSubFiber{VirtualMutexLevel}, mode, op, rhs)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    sym = freshen(ctx, lvl.ex, :after_atomic_lvl)
    atomicData = freshen(ctx, lvl.ex, :atomicArraysAcc)
    lockVal = freshen(ctx, lvl.ex, :lockVal)
    dev = lower(ctx, get_device(ctx.code.task), DefaultStyle())
    push_preamble!(
        ctx,
        quote
            $atomicData = Finch.get_lock(
                $dev, $(lvl.locks), $(ctx(pos)), eltype($(lvl.AVal))
            )
            $lockVal = Finch.aquire_lock!($dev, $atomicData)
        end,
    )
    res = lower_assign(ctx, VirtualSubFiber(lvl.lvl, pos), mode, op, rhs)
    push_epilogue!(
        ctx,
        quote
            Finch.release_lock!($dev, $atomicData)
        end,
    )
    return res
end

function lower_assign(ctx, fbr::VirtualHollowSubFiber{VirtualMutexLevel}, mode, op, rhs)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    sym = freshen(ctx, lvl.ex, :after_atomic_lvl)
    atomicData = freshen(ctx, lvl.ex, :atomicArraysAcc)
    lockVal = freshen(ctx, lvl.ex, :lockVal)
    dev = lower(ctx, get_device(ctx.code.task), DefaultStyle())
    push_preamble!(
        ctx,
        quote
            $atomicData = Finch.get_lock(
                $dev, $(lvl.locks), $(ctx(pos)), eltype($(lvl.AVal))
            )
            $lockVal = Finch.aquire_lock!($dev, $atomicData)
        end,
    )
    res = lower_assign(ctx, VirtualHollowSubFiber(lvl.lvl, pos, fbr.dirty), mode, op, rhs)
    push_epilogue!(
        ctx,
        quote
            Finch.release_lock!($dev, $atomicData)
        end,
    )
    return res
end
