"""
    ParallelSparseDictLevel{[Ti=Int], [Tp=Int], [Ptr, Idx, Val, Tbl, Pool=Dict]}(lvl, [dim])

A subfiber of a sparse level does not need to represent slices `A[:, ..., :, i]`
which are entirely [`fill_value`](@ref). Instead, only potentially non-fill
slices are stored as subfibers in `lvl`.  A datastructure specified by Tbl is used to record which
slices are stored. Optionally, `dim` is the size of the last dimension.

`Ti` is the type of the last fiber index, and `Tp` is the type used for
positions in the level. The types `Ptr` and `Idx` are the types of the
arrays used to store positions and indicies.

```jldoctest
julia> tensor_tree(Tensor(Dense(SparseDict(Element(0.0))), [10 0 20; 30 0 0; 0 0 40]))
3×3-Tensor
└─ Dense [:,1:3]
   ├─ [:, 1]: SparseDict (0.0) [1:3]
   │  ├─ [1]: 10.0
   │  └─ [2]: 30.0
   ├─ [:, 2]: SparseDict (0.0) [1:3]
   └─ [:, 3]: SparseDict (0.0) [1:3]
      ├─ [1]: 20.0
      └─ [3]: 40.0

julia> tensor_tree(Tensor(SparseDict(SparseDict(Element(0.0))), [10 0 20; 30 0 0; 0 0 40]))
3×3-Tensor
└─ SparseDict (0.0) [:,1:3]
   ├─ [:, 1]: SparseDict (0.0) [1:3]
   │  ├─ [1]: 10.0
   │  └─ [2]: 30.0
   └─ [:, 3]: SparseDict (0.0) [1:3]
      ├─ [1]: 20.0
      └─ [3]: 40.0

```
"""

struct ParallelSparseDictLevel{Ti,Ptr,Idx,Val,Tbl,Pool,Lvl} <: AbstractLevel
    lvl::Lvl
    shape::Ti
    ptr::Ptr
    idx::Idx
    val::Val
    tbl::Tbl
    pool::Pool
end

const ParallelSparseDict = ParallelSparseDictLevel

ParallelSparseDictLevel(lvl) = ParallelSparseDictLevel{Int}(lvl)
ParallelSparseDictLevel(lvl, shape::Ti) where {Ti} = ParallelSparseDictLevel{Ti}(lvl, shape)
ParallelSparseDictLevel{Ti}(lvl) where {Ti} = ParallelSparseDictLevel{Ti}(lvl, zero(Ti))
function ParallelSparseDictLevel{Ti}(lvl, shape) where {Ti}
    ParallelSparseDictLevel{Ti}(
        lvl,
        shape,
        postype(lvl)[1],
        Ti[],
        postype(lvl)[],
        Dict{Tuple{postype(lvl),Ti},postype(lvl)}(),
        postype(lvl)[],
    )
end

function ParallelSparseDictLevel{Ti}(
    lvl::Lvl, shape, ptr::Ptr, idx::Idx, val::Val, tbl::Tbl, pool::Pool
) where {Ti,Ptr,Idx,Val,Tbl,Pool,Lvl}
    ParallelSparseDictLevel{Ti,Ptr,Idx,Val,Tbl,Pool,Lvl}(lvl, shape, ptr, idx, val, tbl, pool)
end

Base.summary(lvl::ParallelSparseDictLevel) = "SparseDict($(summary(lvl.lvl)))"
function similar_level(lvl::ParallelSparseDictLevel, fill_value, eltype::Type, dim, tail...)
    ParallelSparseDict(similar_level(lvl.lvl, fill_value, eltype, tail...), dim)
end

function postype(
    ::Type{ParallelSparseDictLevel{Ti,Ptr,Idx,Val,Tbl,Pool,Lvl}}
) where {Ti,Ptr,Idx,Val,Tbl,Pool,Lvl}
    return postype(Lvl)
end

function Base.resize!(lvl::ParallelSparseDictLevel{Ti}, dims...) where {Ti}
    ParallelSparseDictLevel{Ti}(
        resize!(lvl.lvl, dims[1:(end - 1)]...),
        dims[end],
        lvl.ptr,
        lvl.idx,
        lvl.val,
        lvl.tbl,
        lvl.pool,
    )
end

function transfer(
    Tm, lvl::ParallelSparseDictLevel{Ti,Ptr,Idx,Val,Tbl,Pool,Lvl}
) where {Ti,Ptr,Idx,Val,Tbl,Pool,Lvl}
    lvl_2 = transfer(Tm, lvl.lvl)
    ptr_2 = transfer(Tm, lvl.ptr)
    idx_2 = transfer(Tm, lvl.idx)
    val_2 = transfer(Tm, lvl.val)
    tbl_2 = transfer(Tm, lvl.tbl)
    pool_2 = transfer(Tm, lvl.pool)
    return ParallelSparseDictLevel{Ti}(lvl_2, lvl.shape, ptr_2, idx_2, val_2, tbl_2, pool_2)
end

function countstored_level(lvl::ParallelSparseDictLevel, pos)
    pos == 0 && return countstored_level(lvl.lvl, pos)
    countstored_level(lvl.lvl, lvl.ptr[pos + 1] - 1)
end

function pattern!(lvl::ParallelSparseDictLevel{Ti}) where {Ti}
    ParallelSparseDictLevel{Ti}(
        pattern!(lvl.lvl), lvl.shape, lvl.ptr, lvl.idx, lvl.val, lvl.tbl, lvl.pool
    )
end

function set_fill_value!(lvl::ParallelSparseDictLevel{Ti}, init) where {Ti}
    ParallelSparseDictLevel{Ti}(
        set_fill_value!(lvl.lvl, init),
        lvl.shape,
        lvl.ptr,
        lvl.idx,
        lvl.val,
        lvl.tbl,
        lvl.pool,
    )
end

function Base.show(io::IO, lvl::ParallelSparseDictLevel{Ti}) where {Ti}
    if get(io, :compact, false)
        print(io, "SparseDict(")
    else
        print(io, "SparseDict{$Ti}(")
    end
    show(io, lvl.lvl)
    print(io, ", ")
    show(IOContext(io, :typeinfo => Ti), lvl.shape)
    print(io, ", ")
    if get(io, :compact, false)
        print(io, "…")
    else
        show(io, lvl.ptr)
        print(io, ", ")
        show(io, lvl.idx)
        print(io, ", ")
        show(io, lvl.val)
        print(io, ", ")
        show(io, lvl.tbl)
        print(io, ", ")
        show(io, lvl.pool)
    end
    print(io, ")")
end

function labelled_show(io::IO, fbr::SubFiber{<:ParallelSparseDictLevel})
    print(
        io,
        "SparseDict (",
        fill_value(fbr),
        ") [",
        ":,"^(ndims(fbr) - 1),
        "1:",
        size(fbr)[end],
        "]",
    )
end

function labelled_children(fbr::SubFiber{<:ParallelSparseDictLevel})
    lvl = fbr.lvl
    pos = fbr.pos
    pos + 1 > length(lvl.ptr) && return []
    map(lvl.ptr[pos]:(lvl.ptr[pos + 1] - 1)) do qos
        LabelledTree(
            cartesian_label([range_label() for _ in 1:(ndims(fbr) - 1)]..., lvl.idx[qos]),
            SubFiber(lvl.lvl, lvl.val[qos]),
        )
    end
end

@inline level_ndims(
    ::Type{<:ParallelSparseDictLevel{Ti,Ptr,Idx,Val,Tbl,Pool,Lvl}}
) where {Ti,Ptr,Idx,Val,Tbl,Pool,Lvl} = 1 + level_ndims(Lvl)
@inline level_size(lvl::ParallelSparseDictLevel) = (level_size(lvl.lvl)..., lvl.shape)
@inline level_axes(lvl::ParallelSparseDictLevel) = (level_axes(lvl.lvl)..., Base.OneTo(lvl.shape))
@inline level_eltype(
    ::Type{<:ParallelSparseDictLevel{Ti,Ptr,Idx,Val,Tbl,Pool,Lvl}}
) where {Ti,Ptr,Idx,Val,Tbl,Pool,Lvl} = level_eltype(Lvl)
@inline level_fill_value(
    ::Type{<:ParallelSparseDictLevel{Ti,Ptr,Idx,Val,Tbl,Pool,Lvl}}
) where {Ti,Ptr,Idx,Val,Tbl,Pool,Lvl} = level_fill_value(Lvl)
function data_rep_level(
    ::Type{<:ParallelSparseDictLevel{Ti,Ptr,Idx,Val,Tbl,Pool,Lvl}}
) where {Ti,Ptr,Idx,Val,Tbl,Pool,Lvl}
    SparseData(data_rep_level(Lvl))
end

(fbr::AbstractFiber{<:ParallelSparseDictLevel})() = fbr
function (fbr::SubFiber{<:ParallelSparseDictLevel{Ti}})(idxs...) where {Ti}
    isempty(idxs) && return fbr
    lvl = fbr.lvl
    p = fbr.pos
    crds = @view lvl.idx[lvl.ptr[p]:(lvl.ptr[p + 1] - 1)]
    r = searchsorted(crds, idxs[end])
    q = lvl.ptr[p] + first(r) - 1
    length(r) == 0 ? fill_value(fbr) : SubFiber(lvl.lvl, lvl.val[q])(idxs[1:(end - 1)]...)
end

mutable struct VirtualParallelSparseDictLevel <: AbstractVirtualLevel
    tag
    lvl
    Ti
    ptr
    idx
    val
    tbl
    pool
    shape
    qos_stop
end

function is_level_injective(ctx, lvl::VirtualParallelSparseDictLevel)
    [is_level_injective(ctx, lvl.lvl)..., false]
end
function is_level_atomic(ctx, lvl::VirtualParallelSparseDictLevel)
    (below, atomic) = is_level_atomic(ctx, lvl.lvl)
    return ([below; [atomic]], atomic)
end
function is_level_concurrent(ctx, lvl::VirtualParallelSparseDictLevel)
    (data, _) = is_level_concurrent(ctx, lvl.lvl)
    return ([data; [false]], false)
end

function virtualize(
    ctx, ex, ::Type{ParallelSparseDictLevel{Ti,Ptr,Idx,Val,Tbl,Pool,Lvl}}, tag=:lvl
) where {Ti,Ptr,Idx,Val,Tbl,Pool,Lvl}
    tag = freshen(ctx, tag)
    ptr = freshen(ctx, tag, :_ptr)
    idx = freshen(ctx, tag, :_idx)
    val = freshen(ctx, tag, :_val)
    tbl = freshen(ctx, tag, :_tbl)
    pool = freshen(ctx, tag, :_pool)
    stop = freshen(ctx, tag, :_stop)
    push_preamble!(
        ctx,
        quote
            $tag = $ex
            $ptr = $tag.ptr
            $idx = $tag.idx
            $val = $tag.val
            $tbl = $tag.tbl
            $pool = $tag.pool
            $stop = $tag.shape
        end,
    )
    qos_stop = freshen(ctx, tag, :_qos_stop)
    shape = value(stop, Int)
    lvl_2 = virtualize(ctx, :($tag.lvl), Lvl, tag)
    VirtualParallelSparseDictLevel(tag, lvl_2, Ti, ptr, idx, val, tbl, pool, shape, qos_stop)
end
function lower(ctx::AbstractCompiler, lvl::VirtualParallelSparseDictLevel, ::DefaultStyle)
    quote
        $ParallelSparseDictLevel{$(lvl.Ti)}(
            $(ctx(lvl.lvl)),
            $(ctx(lvl.shape)),
            $(lvl.ptr),
            $(lvl.idx),
            $(lvl.val),
            $(lvl.tbl),
            $(lvl.pool),
        )
    end
end

function distribute_level(
    ctx::AbstractCompiler, lvl::VirtualParallelSparseDictLevel, arch, diff, style
)
    return diff[lvl.tag] = VirtualParallelSparseDictLevel(
        lvl.tag,
        distribute_level(ctx, lvl.lvl, arch, diff, style),
        lvl.Ti,
        distribute_buffer(ctx, lvl.ptr, arch, style),
        distribute_buffer(ctx, lvl.idx, arch, style),
        distribute_buffer(ctx, lvl.val, arch, style),
        distribute_buffer(ctx, lvl.tbl, arch, style),
        distribute_buffer(ctx, lvl.pool, arch, style),
        lvl.shape,
        lvl.qos_stop,
    )
end

function redistribute(ctx::AbstractCompiler, lvl::VirtualParallelSparseDictLevel, diff)
    get(
        diff,
        lvl.tag,
        VirtualParallelSparseDictLevel(
            lvl.tag,
            redistribute(ctx, lvl.lvl, diff),
            lvl.Ti,
            lvl.ptr,
            lvl.idx,
            lvl.val,
            lvl.tbl,
            lvl.pool,
            lvl.shape,
            lvl.qos_stop,
        ),
    )
end

Base.summary(lvl::VirtualParallelSparseDictLevel) = "SparseDict($(summary(lvl.lvl)))"

function virtual_level_size(ctx, lvl::VirtualParallelSparseDictLevel)
    ext = virtual_call(ctx, extent, literal(lvl.Ti(1)), lvl.shape)
    (virtual_level_size(ctx, lvl.lvl)..., ext)
end

function virtual_level_resize!(ctx, lvl::VirtualParallelSparseDictLevel, dims...)
    lvl.shape = getstop(dims[end])
    lvl.lvl = virtual_level_resize!(ctx, lvl.lvl, dims[1:(end - 1)]...)
    lvl
end

virtual_level_eltype(lvl::VirtualParallelSparseDictLevel) = virtual_level_eltype(lvl.lvl)
virtual_level_fill_value(lvl::VirtualParallelSparseDictLevel) = virtual_level_fill_value(lvl.lvl)

postype(lvl::VirtualParallelSparseDictLevel) = postype(lvl.lvl)

function declare_level!(ctx::AbstractCompiler, lvl::VirtualParallelSparseDictLevel, pos, init)
    #TODO check that init == fill_value
    Ti = lvl.Ti
    Tp = postype(lvl)
    qos = freshen(ctx, tag, :qos)
    push_preamble!(
        ctx,
        quote
            empty!($(lvl.tbl))
            empty!($(lvl.pool))
            $qos = $(Tp(0))
            $(lvl.qos_stop) = 0
        end,
    )
    lvl.lvl = declare_level!(ctx, lvl.lvl, value(qos, Tp), init)
    return lvl
end

function assemble_level!(ctx, lvl::VirtualParallelSparseDictLevel, pos_start, pos_stop)
    pos_start = ctx(cache!(ctx, :p_start, pos_start))
    pos_stop = ctx(cache!(ctx, :p_start, pos_stop))
end

function freeze_level!(ctx::AbstractCompiler, lvl::VirtualParallelSparseDictLevel, pos_stop)
    p = freshen(ctx, :p)
    Tp = postype(lvl)
    Ti = lvl.Ti
    pos_stop = cache!(ctx, :pos_stop, simplify(ctx, pos_stop))
    qos_stop = freshen(ctx, :qos_stop)
    p = freshen(ctx, :p)
    q = freshen(ctx, :q)
    r = freshen(ctx, :r)
    i = freshen(ctx, :i)
    v = freshen(ctx, :v)
    idx_tmp = freshen(ctx, :idx_tmp)
    val_tmp = freshen(ctx, :val_tmp)
    perm = freshen(ctx, :perm)
    pdx_tmp = freshen(ctx, :pdx_tmp)
    entry = freshen(ctx, :entry)
    ptr_2 = freshen(ctx, :ptr_2)
    push_preamble!(
        ctx,
        quote
            resize!($(lvl.ptr), $(ctx(pos_stop)) + 1)
            $(lvl.ptr)[1] = 1
            Finch.fill_range!($(lvl.ptr), 0, 2, $(ctx(pos_stop)) + 1)
            $pdx_tmp = Vector{$Tp}(undef, length($(lvl.tbl)))
            resize!($(lvl.idx), length($(lvl.tbl)))
            resize!($(lvl.val), length($(lvl.tbl)))
            $idx_tmp = Vector{$Ti}(undef, length($(lvl.tbl)))
            $val_tmp = Vector{$Tp}(undef, length($(lvl.tbl)))
            $q = 0
            for $entry in pairs($(lvl.tbl))
                (($p, $i), $v) = $entry
                $q += 1
                $idx_tmp[$q] = $i
                $val_tmp[$q] = $v
                $pdx_tmp[$q] = $p
                $(lvl.ptr)[$p + 1] += 1
            end
            for $p in 2:($(ctx(pos_stop)) + 1)
                $(lvl.ptr)[$p] += $(lvl.ptr)[$p - 1]
            end
            $perm = sortperm($idx_tmp)
            $ptr_2 = copy($(lvl.ptr))
            for $q in $perm
                $p = $pdx_tmp[$q]
                $r = $ptr_2[$p]
                $(lvl.idx)[$r] = $idx_tmp[$q]
                $(lvl.val)[$r] = $val_tmp[$q]
                $ptr_2[$p] += 1
            end
            $qos_stop = $(lvl.ptr)[$(ctx(pos_stop)) + 1] - 1
        end,
    )
    lvl.lvl = freeze_level!(ctx, lvl.lvl, value(qos_stop))
    return lvl
end

function thaw_level!(ctx::AbstractCompiler, lvl::VirtualParallelSparseDictLevel, pos_stop)
    p = freshen(ctx, :p)
    pos_stop = ctx(cache!(ctx, :pos_stop, simplify(ctx, pos_stop)))
    push_preamble!(
        ctx,
        quote
            $(lvl.qos_stop) = $(lvl.ptr)[$(ctx(pos_stop)) + 1] - 1
        end,
    )
    lvl.lvl = thaw_level!(ctx, lvl.lvl, value(lvl.qos_stop))
    return lvl
end

function unfurl(
    ctx,
    fbr::VirtualSubFiber{VirtualParallelSparseDictLevel},
    ext,
    mode,
    ::Union{typeof(defaultread),typeof(walk)},
)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.tag
    Tp = postype(lvl)
    Ti = lvl.Ti
    my_i = freshen(ctx, tag, :_i)
    my_q = freshen(ctx, tag, :_q)
    my_q_stop = freshen(ctx, tag, :_q_stop)
    my_i1 = freshen(ctx, tag, :_i1)
    my_v = freshen(ctx, tag, :_v)

    Thunk(;
        preamble=quote
            $my_q = $(lvl.ptr)[$(ctx(pos))]
            $my_q_stop = $(lvl.ptr)[$(ctx(pos)) + $(Tp(1))]
            if $my_q < $my_q_stop
                $my_i = $(lvl.idx)[$my_q]
                $my_i1 = $(lvl.idx)[$my_q_stop - $(Tp(1))]
            else
                $my_i = $(Ti(1))
                $my_i1 = $(Ti(0))
            end
        end,
        body=(ctx) -> Sequence([
            Phase(;
                stop=(ctx, ext) -> value(my_i1),
                body=(ctx, ext) -> Stepper(;
                    seek=(ctx, ext) -> quote
                        if $(lvl.idx)[$my_q] < $(ctx(getstart(ext)))
                            $my_q = Finch.scansearch(
                                $(lvl.idx),
                                $(ctx(getstart(ext))),
                                $my_q,
                                $my_q_stop - 1,
                            )
                            $my_i = $(lvl.idx)[$my_q]
                        end
                    end,
                    preamble=quote
                        $my_i = $(lvl.idx)[$my_q]
                        $my_v = $(lvl.val)[$my_q]
                    end,
                    stop=(ctx, ext) -> value(my_i),
                    chunk=Spike(;
                        body=FillLeaf(virtual_level_fill_value(lvl)),
                        tail=Simplify(
                            instantiate(
                                ctx, VirtualSubFiber(lvl.lvl, value(my_v, Ti)), mode
                            ),
                        ),
                    ),
                    next=(ctx, ext) -> :($my_q += $(Tp(1))),
                ),
            ),
            Phase(;
                body=(ctx, ext) -> Run(FillLeaf(virtual_level_fill_value(lvl)))
            ),
        ]),
    )
end

function unfurl(
    ctx, fbr::VirtualSubFiber{VirtualParallelSparseDictLevel}, ext, mode, ::typeof(follow)
)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.tag
    Tp = postype(lvl)
    my_q = freshen(ctx, tag, :_q)
    h = freshen(ctx, :_h)

    Lookup(;
        body=(ctx, i) -> Thunk(;
            preamble=quote
                $h = hash(($(ctx(pos)), $(ctx(i)))) % length($(lvl.tbl))
                $my_q = get($(lvl.tbl).data[$h], ($(ctx(pos)), $(ctx(i))), 0)
            end,
            body=(ctx) -> Switch(
                [
                    value(:($my_q != 0)) => instantiate(
                        ctx, VirtualSubFiber(lvl.lvl, value(my_q, Tp)), mode
                    )
                    literal(true) => FillLeaf(virtual_level_fill_value(lvl))
                ],
            ),
        ),
    )
end

function unfurl(
    ctx,
    fbr::VirtualSubFiber{VirtualParallelSparseDictLevel},
    ext,
    mode,
    proto::Union{typeof(defaultupdate),typeof(extrude)},
)
    unfurl(
        ctx, VirtualHollowSubFiber(fbr.lvl, fbr.pos, freshen(ctx, :null)), ext, mode, proto
    )
end
function unfurl(
    ctx,
    fbr::VirtualHollowSubFiber{VirtualParallelSparseDictLevel},
    ext,
    mode,
    ::Union{typeof(defaultupdate),typeof(extrude)},
)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    tag = lvl.tag
    Tp = postype(lvl)
    qos = freshen(ctx, tag, :_qos)
    qos_stop = lvl.qos_stop
    dirty = freshen(ctx, tag, :_dirty)

    Thunk(;
        body=(ctx) -> Lookup(;
            body=(ctx, idx) -> Thunk(;
                preamble=quote
                    $qos = get($(lvl.tbl), ($(ctx(pos)), $(ctx(idx))), 0)
                    if $qos == 0
                        #If the qos is not in the table, we need to add it.
                        #We need to commit it to the table in the event that
                        #another accessor tries to write it in the same loop.
                        if !isempty($(lvl.pool))
                            $qos = pop!($(lvl.pool))
                        else
                            $qos = length($(lvl.tbl)) + 1
                            if $qos > $qos_stop
                                $qos_stop = max($qos_stop << 1, 1)
                                $(contain(
                                    ctx_2 -> assemble_level!(
                                        ctx_2,
                                        lvl.lvl,
                                        value(qos, Tp),
                                        value(qos_stop, Tp),
                                    ),
                                    ctx,
                                ))
                                Finch.resize_if_smaller!($(lvl.val), $qos_stop)
                                Finch.fill_range!($(lvl.val), 0, $qos, $qos_stop)
                            end
                        end
                        $(lvl.tbl)[($(ctx(pos)), $(ctx(idx)))] = $qos
                    end
                    $dirty = false
                end,
                body=(ctx) -> instantiate(
                    ctx,
                    VirtualHollowSubFiber(lvl.lvl, value(qos, Tp), dirty),
                    mode,
                ),
                epilogue=quote
                    if $dirty
                        $(lvl.val)[$qos] = $qos
                        $(fbr.dirty) = true
                    elseif $(lvl.val)[$qos] == 0 #here, val is being used as a dirty bit
                        push!($(lvl.pool), $qos)
                        delete!($(lvl.tbl), ($(ctx(pos)), $(ctx(idx))))
                    end
                end,
            ),
        ),
    )
end

function coalesce_level!(
    lvl::ParallelSparseDictLevel, global_fbr_map, local_fbr_map, task_map, factor, P, coalescent
)
    if factor > 1
        global_fbr_map, local_fbr_map, task_map = unroll_dense_coalesce(
            global_fbr_map, local_fbr_map, task_map, factor, P
        )
        factor = 1
    end

    #lvl.idx and lvl.ptr should be MultiChannelBuffers
    idx = lvl.idx.data
    ptr = lvl.ptr.data
    dev = lvl.tbl.device
    tbl = lvl.tbl.data
    max_level_dim = global_fbr_map[length(global_fbr_map)]
    cutoffs = compute_proc_cutoffs(idx, P)

    #Don't merge zero-ed arrays.
    if cutoffs[P + 1] == 1
        return coalescent
    end

    pos_map, idx_map, lfm, tm = gen_pos_idx_map_hash(
        global_fbr_map, local_fbr_map, task_map, ptr, idx, cutoffs, P, tbl
    )
    global_fbr_map, local_fbr_map, task_map, ptr_2, idx_2, tbl_2 = process_next_lvl_parallel_hash(
        pos_map, idx_map, tm, lfm, P, max_level_dim
    )

    my_tbl = MultiChannelBuffer(dev, tbl_2)

    ParallelSparseDictLevel(
        coalesce_level!(
            lvl.lvl, global_fbr_map, local_fbr_map, task_map, factor, P, coalescent.lvl
        ),
        lvl.shape, ptr_2, idx_2, global_fbr_map, my_tbl, Vector{Int}(undef, 0))
end

Base.@propagate_inbounds function gen_pos_idx_map_hash(
    global_fbr_map, local_fbr_map, task_map, ptr, index, cutoffs, P, tbl
)
    ordering = Base.Order.By(j -> (task_map[j], local_fbr_map[j]))
    sorter = AcceleratedKernels.sortperm(collect(1:length(task_map)); order=ordering)

    nnz = cutoffs[length(cutoffs)] - 1
    merged_positions = Vector{Int}(undef, nnz)
    merged_indices = Vector{Int}(undef, nnz)

    task_map2 = Vector{Int}(undef, nnz)
    local_fbr_map2 = Vector{Int}(undef, nnz)

    chk_size = fld(nnz + P - 1, P)
    Threads.@threads for tid in 1:P
        init = (tid - 1) * chk_size + 1
        proc_id = binary_search(init, cutoffs)
        idx_id = init - cutoffs[proc_id] + 1

        local_fbr = binary_search(idx_id, ptr[proc_id])

        tag = get_permute_idx(proc_id, ptr) + local_fbr

        @assert local_fbr > 0
        @assert tag > 0

        global_fbr = global_fbr_map[sorter[tag]]

        j = 0
        for i in 0:(chk_size - 1)
            offset = init + i
            if offset > nnz
                break
            end

            nz_id = j + idx_id
            idx = index[proc_id][nz_id]
            merged_positions[offset] = global_fbr
            merged_indices[offset] = idx
            task_map2[offset] = proc_id
            local_fbr_map2[offset] = tbl[proc_id][(local_fbr, idx)]

            if nz_id >= length(index[proc_id]) && proc_id < P
                proc_id += 1
                idx_id = 1
                j = 0
                
                local_fbr = binary_search(idx_id, ptr[proc_id])
                tag = get_permute_idx(proc_id, ptr) + local_fbr

                global_fbr = global_fbr_map[sorter[tag]]
            elseif nz_id + 1 >= ptr[proc_id][local_fbr + 1] &&
                local_fbr + 1 < length(ptr[proc_id]) &&
                ptr[proc_id][local_fbr + 1] != ptr[proc_id][local_fbr]
                local_fbr += 1

                tag += 1
                global_fbr = global_fbr_map[sorter[tag]]
                j += 1
            else
                j += 1
            end
        end
    end
    return merged_positions, merged_indices, local_fbr_map2, task_map2
end

Base.@propagate_inbounds function process_next_lvl_parallel_hash(
    merged_positions, merged_indices, task_map, local_fbr_map, P, max_level_dim
)
    ordering = Base.Order.By(j -> (merged_positions[j], merged_indices[j]))
    shuffler = AcceleratedKernels.sortperm(
        collect(1:length(merged_positions)); order=ordering
    )

    nnz = length(local_fbr_map)
    global_fbr_map2 = Vector{Int}(undef, nnz)

    merged_positions_s = p_permute(shuffler, merged_positions)
    merged_indices_s = p_permute(shuffler, merged_indices)
    task_map = p_permute(shuffler, task_map)
    local_fbr_map = p_permute(shuffler, local_fbr_map)

    uq_ptr = zeros(Int, P + 1)
    uq_idx = zeros(Int, P + 1)

    chk_size = fld(nnz + P - 1, P)

    Threads.@threads for tid in 1:P
        init = (tid - 1) * chk_size + 1
        seen = 0
        prev =
            init > 1 ? (merged_positions_s[init - 1], merged_indices_s[init - 1]) : (-1, -1)
        prev_ptr = init > 1 ? merged_positions_s[init - 1] : 1
        seen_ptr = 0

        for i in 0:(chk_size - 1)
            offset = init + i
            if offset > nnz
                break
            end

            tup = (merged_positions_s[offset], merged_indices_s[offset])
            if tup != prev
                prev = tup
                seen += 1
            end

            p = merged_positions_s[offset]
            if prev_ptr != p
                seen_ptr += (p - prev_ptr)
                prev_ptr = p
            end
        end
        uq_idx[tid + 1] = seen
        uq_ptr[tid + 1] = seen_ptr
    end
    uq_ptr_s = s_prefix_sum(uq_ptr)
    uq_idx_s = s_prefix_sum(uq_idx)

    lvl_ptr = zeros(Int, max_level_dim + 1)
    lvl_idx = zeros(Int, uq_idx_s[length(uq_idx_s)])
    tbls = [[Dict{Tuple{Int, Int}, Int}() for _ in 1:P] for _ in 1:P]

    Threads.@threads for tid in 1:P
        init = (tid - 1) * chk_size + 1
        seen_ptr = uq_ptr_s[tid] + 2
        seen_idx = uq_idx_s[tid] + 1
        prev =
            init > 1 ? (merged_positions_s[init - 1], merged_indices_s[init - 1]) : (1, -1)

        for i in 0:(chk_size - 1)
            offset = init + i
            if offset > nnz
                break
            end

            while seen_ptr < merged_positions_s[offset]
                lvl_ptr[seen_ptr] = seen_idx
                seen_ptr += 1
            end

            tup = (merged_positions_s[offset], merged_indices_s[offset])
            if tup != prev
                lvl_idx[seen_idx] = tup[2]

                p = merged_positions_s[offset]
                if prev[1] != p
                    lvl_ptr[seen_ptr] = seen_idx
                    seen_ptr += 1
                end
                prev = tup
                seen_idx += 1
            end
            global_fbr_map2[offset] = seen_idx - 1
            h = hash(tup)
            tbls[(h % P) + 1][tid][tup] = seen_idx - 1
        end
    end

    tbl_2 = [Dict{Tuple{Int, Int}, Int}() for _ in 1:P]

    Threads.@threads for tid in 1:P
        merge!(tbl_2[tid], reduce(merge!, tbls[tid]))
    end

    lvl_ptr[1] = 1
    i = length(lvl_ptr)
    while lvl_ptr[i] == 0
        lvl_ptr[i] = length(lvl_idx) + 1
        i -= 1
    end

    return global_fbr_map2, local_fbr_map, task_map, lvl_ptr, lvl_idx, tbl_2
end