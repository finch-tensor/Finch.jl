"""
    ElementLevel{Vf, [Tv=typeof(Vf)], [Tp=Int], [Val]}()

A subfiber of an element level is a scalar of type `Tv`, initialized to `Vf`. `Vf`
may optionally be given as the first argument.

The data is stored in a vector
of type `Val` with `eltype(Val) = Tv`. The type `Tp` is the index type used to
access Val.

```jldoctest
julia> tensor_tree(Tensor(Dense(Element(0.0)), [1, 2, 3]))
3-Tensor
└─ Dense [1:3]
   ├─ [1]: 1.0
   ├─ [2]: 2.0
   └─ [3]: 3.0
```
"""
struct ElementLevel{Vf,Tv,Tp,Val} <: AbstractLevel
    val::Val
end
const Element = ElementLevel

function ElementLevel(d, args...)
    isbits(d) || throw(ArgumentError("Finch currently only supports isbits defaults"))
    ElementLevel{d}(args...)
end
ElementLevel{Vf}() where {Vf} = ElementLevel{Vf,typeof(Vf)}()
ElementLevel{Vf}(val::Val) where {Vf,Val} = ElementLevel{Vf,eltype(Val)}(val)
ElementLevel{Vf,Tv}(args...) where {Vf,Tv} = ElementLevel{Vf,Tv,Int}(args...)
ElementLevel{Vf,Tv,Tp}() where {Vf,Tv,Tp} = ElementLevel{Vf,Tv,Tp}(Tv[])

ElementLevel{Vf,Tv,Tp}(val::Val) where {Vf,Tv,Tp,Val} = ElementLevel{Vf,Tv,Tp,Val}(val)

Base.summary(::Element{Vf}) where {Vf} = "Element($(Vf))"

function similar_level(
    ::ElementLevel{Vf,Tv,Tp}, fill_value, eltype::Type, ::Vararg
) where {Vf,Tv,Tp}
    ElementLevel{fill_value,eltype,Tp}()
end

postype(::Type{<:ElementLevel{Vf,Tv,Tp}}) where {Vf,Tv,Tp} = Tp

function transfer(device, lvl::ElementLevel{Vf,Tv,Tp}) where {Vf,Tv,Tp}
    return ElementLevel{Vf,Tv,Tp}(transfer(device, lvl.val))
end

pattern!(lvl::ElementLevel{Vf,Tv,Tp}) where {Vf,Tv,Tp} = Pattern{Tp}()
function set_fill_value!(lvl::ElementLevel{Vf,Tv,Tp}, init) where {Vf,Tv,Tp}
    ElementLevel{init,Tv,Tp}(lvl.val)
end
Base.resize!(lvl::ElementLevel) = lvl

isstructequal(a::T, b::T) where {T<:Element} =
    a.val == b.val

function Base.show(io::IO, lvl::ElementLevel{Vf,Tv,Tp,Val}) where {Vf,Tv,Tp,Val}
    print(io, "Element{")
    show(io, Vf)
    print(io, ", $Tv, $Tp}(")
    if get(io, :compact, false)
        print(io, "…")
    else
        show(io, lvl.val)
    end
    print(io, ")")
end

labelled_show(io::IO, fbr::SubFiber{<:ElementLevel}) = print(io, fbr.lvl.val[fbr.pos])

@inline level_ndims(::Type{<:ElementLevel}) = 0
@inline level_size(::ElementLevel) = ()
@inline level_axes(::ElementLevel) = ()
@inline level_eltype(::Type{<:ElementLevel{Vf,Tv}}) where {Vf,Tv} = Tv
@inline level_fill_value(::Type{<:ElementLevel{Vf}}) where {Vf} = Vf
data_rep_level(::Type{<:ElementLevel{Vf,Tv}}) where {Vf,Tv} = ElementData(Vf, Tv)

(fbr::Tensor{<:ElementLevel})() = SubFiber(fbr.lvl, 1)()
function (fbr::SubFiber{<:ElementLevel})()
    q = fbr.pos
    return fbr.lvl.val[q]
end

countstored_level(lvl::ElementLevel, pos) = pos

mutable struct VirtualElementLevel <: AbstractVirtualLevel
    tag
    Vf
    Tv
    Tp
    val
end

is_level_injective(ctx, ::VirtualElementLevel) = []
is_level_atomic(ctx, lvl::VirtualElementLevel) = ([], false)
function is_level_concurrent(ctx, lvl::VirtualElementLevel)
    return ([], true)
end

function lower(ctx::AbstractCompiler, lvl::VirtualElementLevel, ::DefaultStyle)
    :(ElementLevel{$(lvl.Vf),$(lvl.Tv),$(lvl.Tp)}($(lvl.val)))
end

function virtualize(
    ctx, ex, ::Type{ElementLevel{Vf,Tv,Tp,Val}}, tag=:lvl
) where {Vf,Tv,Tp,Val}
    tag = freshen(ctx, tag)
    val = freshen(ctx, tag, :_val)
    push_preamble!(
        ctx,
        quote
            $tag = $ex
            $val = $tag.val
        end,
    )
    VirtualElementLevel(tag, Vf, Tv, Tp, val)
end

function distribute_level(
    ctx::AbstractCompiler, lvl::VirtualElementLevel, arch, diff, style
)
    diff[lvl.tag] = VirtualElementLevel(
        lvl.tag, lvl.Vf, lvl.Tv, lvl.Tp, distribute_buffer(ctx, lvl.val, arch, style)
    )
end

function redistribute(ctx::AbstractCompiler, lvl::VirtualElementLevel, diff)
    get(diff, lvl.tag, lvl)
end

Base.summary(lvl::VirtualElementLevel) = "Element($(lvl.Vf))"

virtual_level_resize!(ctx, lvl::VirtualElementLevel) = lvl
virtual_level_size(ctx, ::VirtualElementLevel) = ()
virtual_level_ndims(ctx, lvl::VirtualElementLevel) = 0
virtual_level_eltype(lvl::VirtualElementLevel) = lvl.Tv
virtual_level_fill_value(lvl::VirtualElementLevel) = lvl.Vf

postype(lvl::VirtualElementLevel) = lvl.Tp

function declare_level!(ctx, lvl::VirtualElementLevel, pos, init)
    init == literal(lvl.Vf) || throw(
        FinchProtocolError(
            "Cannot initialize Element Levels to non-fill values (have $init expected $(lvl.Vf))"
        ),
    )
    lvl
end

function freeze_level!(ctx::AbstractCompiler, lvl::VirtualElementLevel, pos)
    push_preamble!(
        ctx,
        quote
            resize!($(lvl.val), $(ctx(pos)))
        end,
    )
    return lvl
end

thaw_level!(ctx::AbstractCompiler, lvl::VirtualElementLevel, pos) = lvl

function assemble_level!(ctx, lvl::VirtualElementLevel, pos_start, pos_stop)
    pos_start = cache!(ctx, :pos_start, simplify(ctx, pos_start))
    pos_stop = cache!(ctx, :pos_stop, simplify(ctx, pos_stop))
    quote
        Finch.resize_if_smaller!($(lvl.val), $(ctx(pos_stop)))
        Finch.fill_range!($(lvl.val), $(lvl.Vf), $(ctx(pos_start)), $(ctx(pos_stop)))
    end
end

supports_reassembly(::VirtualElementLevel) = true
function reassemble_level!(ctx, lvl::VirtualElementLevel, pos_start, pos_stop)
    pos_start = cache!(ctx, :pos_start, simplify(ctx, pos_start))
    pos_stop = cache!(ctx, :pos_stop, simplify(ctx, pos_stop))
    push_preamble!(
        ctx,
        quote
            Finch.fill_range!($(lvl.val), $(lvl.Vf), $(ctx(pos_start)), $(ctx(pos_stop)))
        end,
    )
    lvl
end

function instantiate(ctx, fbr::VirtualSubFiber{VirtualElementLevel}, mode)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    if mode.kind === reader
        val = freshen(ctx, lvl.tag, :_val)
        return Thunk(;
            preamble=quote
                $val = $(lvl.val)[$(ctx(pos))]
            end,
            body=(ctx) -> VirtualScalar(nothing, nothing, lvl.Tv, lvl.Vf, gensym(), val),
        )
    else
        VirtualScalar(
            nothing, nothing, lvl.Tv, lvl.Vf, gensym(), :($(lvl.val)[$(ctx(pos))])
        )
    end
end

function instantiate(ctx, fbr::VirtualHollowSubFiber{VirtualElementLevel}, mode)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    @assert mode.kind === updater
    VirtualSparseScalar(
        nothing, nothing, lvl.Tv, lvl.Vf, gensym(), :($(lvl.val)[$(ctx(pos))]), fbr.dirty
    )
end

function lower_assign(ctx, fbr::VirtualHollowSubFiber{VirtualElementLevel}, mode, op, rhs)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    lower_assign(
        ctx,
        VirtualSparseScalar(
            nothing, nothing, lvl.Tv, lvl.Vf, gensym(), :($(lvl.val)[$(ctx(pos))]),
            fbr.dirty,
        ),
        mode,
        op,
        rhs,
    )
end

function lower_assign(ctx, fbr::VirtualSubFiber{VirtualElementLevel}, mode, op, rhs)
    (lvl, pos) = (fbr.lvl, fbr.pos)
    lower_assign(
        ctx,
        VirtualScalar(
            nothing, nothing, lvl.Tv, lvl.Vf, gensym(), :($(lvl.val)[$(ctx(pos))])
        ),
        mode,
        op,
        rhs,
    )
end

function coalesce_level!(
    lvl::ElementLevel{Vf,Tv,Tp,Val},
    global_fbr_map,
    factor,
    max_dim,
    P,
    coalescent,
    weak
) where {Vf,Tv,Tp,Val}
    val = lvl.val.data
    lvl_val = coalescent.val

    if length(val) < 1
        return nothing
    end
    if factor > 1
        merge_dense_element(
            factor, val, P, val2
        )
    else
        merge_element(global_fbr_map, val, max_dim, P, lvl_val)
    end
end

function setup_coalesce!(lvl::ElementLevel, max_pos, coalescent)
    resize!(coalescent.val, max_pos)
    return true
end

function coalesce_fast!(tid, meta, P, lvl::ElementLevel, coalescent, was_dense)
    val = lvl.val.data
    lvl_val = coalescent.val

    fastmerge_element(tid, val, P, lvl_val)
end

@inbounds function fastmerge_element(tid, val, P, lvl_val)
    nnz_cutoffs = Vector{Int}(undef, P + 1)
    nnz_cutoffs[1] = 1
    for p in 2:P+1
        nnz_cutoffs[p] = nnz_cutoffs[p - 1] + length(val[p - 1])
    end
    nnz = nnz_cutoffs[end] - 1

    base, rem = divrem(nnz, P)
    offset = (tid - 1) * base + min(tid - 1, rem)
    chunksize = base + (tid <= rem ? 1 : 0)
    work_lb = 1 + offset
    work_ub = work_lb + chunksize - 1

    proc_id_lower = binary_search(work_lb, nnz_cutoffs)
    nz_offset = work_lb - nnz_cutoffs[proc_id_lower] + 1
    proc = proc_id_lower
    write_idx = work_lb
    while write_idx <= work_ub
        lvl_val[write_idx] = val[proc][nz_offset]
        write_idx += 1
        nz_offset += 1
        if nz_offset > length(val[proc])
            proc += 1
            nz_offset = 1
        end
    end
end

Base.@propagate_inbounds function merge_element(gfm, val, max_pos, P, lvl_val)
    resize!(lvl_val, max_pos)
    chk = fld(max_pos + P - 1, P)

    Threads.@threads for tid in 1:P
        pos_start = (tid - 1) * chk + 1
        pos_stop = min(tid * chk, max_pos)
        if pos_start > max_pos
            continue
        end
        
        for p in pos_start:pos_stop
            lvl_val[p] = 0
        end

        for proc in 1:P
            lo, hi = 1, length(gfm[proc])
            lfbr = binary_search_lb(pos_start, gfm[proc], lo, hi)

            ##Can prove the processor doesn't contain the range.
            if lfbr < 1
                continue
            end

            curr = gfm[proc][lfbr]

            while lfbr <= length(val[proc]) && (curr = gfm[proc][lfbr]) <= pos_stop
                @fastmath lvl_val[curr] += val[proc][lfbr]
                lfbr += 1
            end
        end

    end
end

##TODO: Rewrite to new API and double check correctness.
##As it stands this is probably wrong > 1 mode tensors.
Base.@propagate_inbounds function merge_dense_element(factor, val, P, val2)
    Threads.@threads for i in 1:factor
        for proc_id in 1:P
            @fastmath val2[i] += val[proc_id][i]
        end
    end
end
