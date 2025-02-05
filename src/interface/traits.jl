using Base.Broadcast: Broadcasted

"""
    SparseData(lvl)

Represents a tensor `A` where `A[:, ..., :, i]` is sometimes entirely fill_value(lvl)
and is sometimes represented by `lvl`.
"""
struct SparseData
    lvl
end
Finch.finch_leaf(x::SparseData) = virtual(x)

Base.ndims(fbr::SparseData) = 1 + ndims(fbr.lvl)
fill_value(fbr::SparseData) = fill_value(fbr.lvl)
Base.eltype(fbr::SparseData) = eltype(fbr.lvl)
is_concordant_rep(fbr::SparseData) = true

"""
    RepeatData(lvl)

Represents a tensor `A` where `A[:, ..., :, i]` is sometimes entirely fill_value(lvl)
and is sometimes represented by repeated runs of `lvl`.
"""
struct RepeatData
    lvl
end
Finch.finch_leaf(x::RepeatData) = virtual(x)

Base.ndims(fbr::RepeatData) = 1 + ndims(fbr.lvl)
fill_value(fbr::RepeatData) = fill_value(fbr.lvl)
Base.eltype(fbr::RepeatData) = eltype(fbr.lvl)
is_concordant_rep(fbr::RepeatData) = true

"""
    DenseData(lvl)

Represents a tensor `A` where each `A[:, ..., :, i]` is represented by `lvl`.
"""
struct DenseData
    lvl
end
Finch.finch_leaf(x::DenseData) = virtual(x)
fill_value(fbr::DenseData) = fill_value(fbr.lvl)
Base.ndims(fbr::DenseData) = 1 + ndims(fbr.lvl)
Base.eltype(fbr::DenseData) = eltype(fbr.lvl)
is_concordant_rep(fbr::DenseData) = is_concordant_rep(fbr.lvl)

"""
    ExtrudeData(lvl)

Represents a tensor `A` where `A[:, ..., :, 1]` is the only slice, and is represented by `lvl`.
"""
struct ExtrudeData
    lvl
end
Finch.finch_leaf(x::ExtrudeData) = virtual(x)
fill_value(fbr::ExtrudeData) = fill_value(fbr.lvl)
Base.ndims(fbr::ExtrudeData) = 1 + ndims(fbr.lvl)
Base.eltype(fbr::ExtrudeData) = eltype(fbr.lvl)
is_concordant_rep(fbr::ExtrudeData) = is_concordant_rep(fbr.lvl)

"""
    HollowData(lvl)

Represents a tensor which is represented by `lvl` but is sometimes entirely `fill_value(lvl)`.
"""
struct HollowData
    lvl
end
Finch.finch_leaf(x::HollowData) = virtual(x)
fill_value(fbr::HollowData) = fill_value(fbr.lvl)
Base.ndims(fbr::HollowData) = ndims(fbr.lvl)
Base.eltype(fbr::HollowData) = eltype(fbr.lvl)
is_concordant_rep(fbr::HollowData) = true

"""
    ElementData(fill_value, eltype)

Represents a scalar element of type `eltype` and fill_value `fill_value`.
"""
struct ElementData
    fill_value
    eltype
end
Finch.finch_leaf(x::ElementData) = virtual(x)
fill_value(fbr::ElementData) = fbr.fill_value
Base.ndims(fbr::ElementData) = 0
Base.eltype(fbr::ElementData) = fbr.eltype
is_concordant_rep(fbr::ElementData) = false

"""
    data_rep(tns)

Return a trait object representing everything that can be learned about the data
based on the storage format (type) of the tensor
"""
data_rep(tns) = (DenseData^(ndims(tns)))(ElementData(fill_value(tns), eltype(tns)))

data_rep(T::Type{<:Number}) = ElementData(zero(T), T)

"""
    collapse_rep(tns)

Normalize a trait object to collapse subfiber information into the parent tensor.
"""
collapse_rep(fbr) = fbr

collapse_rep(fbr::HollowData) = collapse_rep(fbr, collapse_rep(fbr.lvl))
collapse_rep(::HollowData, lvl::HollowData) = collapse_rep(lvl)
collapse_rep(::HollowData, lvl) = HollowData(collapse_rep(lvl))

collapse_rep(fbr::DenseData) = collapse_rep(fbr, collapse_rep(fbr.lvl))
collapse_rep(::DenseData, lvl::HollowData) = collapse_rep(SparseData(lvl.lvl))
collapse_rep(::DenseData, lvl) = DenseData(collapse_rep(lvl))

collapse_rep(fbr::ExtrudeData) = collapse_rep(fbr, collapse_rep(fbr.lvl))
function collapse_rep(::ExtrudeData, lvl::HollowData)
    HollowData(collapse_rep(ExtrudeData(lvl.lvl)))
end
collapse_rep(::ExtrudeData, lvl) = ExtrudeData(collapse_rep(lvl))

collapse_rep(fbr::SparseData) = collapse_rep(fbr, collapse_rep(fbr.lvl))
collapse_rep(::SparseData, lvl::HollowData) = collapse_rep(SparseData(lvl.lvl))
collapse_rep(::SparseData, lvl) = SparseData(collapse_rep(lvl))

collapse_rep(fbr::RepeatData) = collapse_rep(fbr, collapse_rep(fbr.lvl))
collapse_rep(::RepeatData, lvl::HollowData) = collapse_rep(RepeatData(lvl.lvl))
collapse_rep(::RepeatData, lvl) = RepeatData(collapse_rep(lvl))

"""
    map_rep(f, args...)

Return a storage trait object representing the result of mapping `f` over
storage traits `args`. Assumes representation is collapsed.
"""
function map_rep(f, args...)
    map_rep_def(f, map(arg -> paddims_rep(arg, maximum(ndims, args)), args))
end

paddims_rep(rep, n) = ndims(rep) < n ? paddims_rep(ExtrudeData(rep), n) : rep

"""
    expanddims_rep(tns, dims)
Expand the representation of `tns` by inserting singleton dimensions `dims`.
"""
function expanddims_rep(tns, dims)
    @assert allunique(dims)
    @assert issubset(dims, 1:(ndims(tns) + length(dims)))
    expanddims_rep_def(tns, ndims(tns) + length(dims), dims)
end
function expanddims_rep_def(tns::HollowData, dim, dims)
    HollowData(expanddims_rep_def(tns.lvl, dim, dims))
end
function expanddims_rep_def(tns::ElementData, dim, dims)
    dim in dims ? ExtrudeData(expanddims_rep_def(tns, dim - 1, dims)) : tns
end
function expanddims_rep_def(tns::SparseData, dim, dims)
    if dim in dims
        ExtrudeData(expanddims_rep_def(tns, dim - 1, dims))
    else
        SparseData(expanddims_rep_def(tns.lvl, dim - 1, dims))
    end
end
function expanddims_rep_def(tns::RepeatData, dim, dims)
    if dim in dims
        ExtrudeData(expanddims_rep_def(tns, dim - 1, dims))
    else
        RepeatData(expanddims_rep_def(tns.lvl, dim - 1, dims))
    end
end
function expanddims_rep_def(tns::DenseData, dim, dims)
    if dim in dims
        ExtrudeData(expanddims_rep_def(tns, dim - 1, dims))
    else
        DenseData(expanddims_rep_def(tns.lvl, dim - 1, dims))
    end
end
function expanddims_rep_def(tns::ExtrudeData, dim, dims)
    if dim in dims
        ExtrudeData(expanddims_rep_def(tns, dim - 1, dims))
    else
        ExtrudeData(expanddims_rep_def(tns.lvl, dim - 1, dims))
    end
end

struct MapRepHollowStyle end
struct MapRepExtrudeStyle end
struct MapRepSparseStyle end
struct MapRepDenseStyle end
struct MapRepRepeatStyle end
struct MapRepElementStyle end

combine_style(a::MapRepHollowStyle, b::MapRepHollowStyle) = a
combine_style(a::MapRepHollowStyle, b::MapRepExtrudeStyle) = a
combine_style(a::MapRepHollowStyle, b::MapRepSparseStyle) = a
combine_style(a::MapRepHollowStyle, b::MapRepDenseStyle) = a
combine_style(a::MapRepHollowStyle, b::MapRepRepeatStyle) = a

combine_style(a::MapRepHollowStyle, b::MapRepElementStyle) = a

combine_style(a::MapRepSparseStyle, b::MapRepExtrudeStyle) = a
combine_style(a::MapRepSparseStyle, b::MapRepSparseStyle) = a
combine_style(a::MapRepSparseStyle, b::MapRepDenseStyle) = a
combine_style(a::MapRepSparseStyle, b::MapRepRepeatStyle) = a

combine_style(a::MapRepDenseStyle, b::MapRepExtrudeStyle) = a
combine_style(a::MapRepDenseStyle, b::MapRepDenseStyle) = a
combine_style(a::MapRepDenseStyle, b::MapRepRepeatStyle) = a

combine_style(a::MapRepRepeatStyle, b::MapRepExtrudeStyle) = a
combine_style(a::MapRepRepeatStyle, b::MapRepRepeatStyle) = a

combine_style(a::MapRepExtrudeStyle, b::MapRepExtrudeStyle) = a

combine_style(a::MapRepElementStyle, b::MapRepElementStyle) = a

map_rep_style(r::HollowData) = MapRepHollowStyle()
map_rep_style(r::ExtrudeData) = MapRepExtrudeStyle()
map_rep_style(r::SparseData) = MapRepSparseStyle()
map_rep_style(r::DenseData) = MapRepDenseStyle()
map_rep_style(r::RepeatData) = MapRepRepeatStyle()
map_rep_style(r::ElementData) = MapRepElementStyle()

map_rep_def(f, args) = map_rep_def(mapreduce(map_rep_style, result_style, args), f, args)

map_rep_child(r::ExtrudeData) = r.lvl
map_rep_child(r::SparseData) = r.lvl
map_rep_child(r::DenseData) = r.lvl
map_rep_child(r::RepeatData) = r.lvl

function map_rep_def(::MapRepDenseStyle, f, args)
    DenseData(map_rep_def(f, map(map_rep_child, args)))
end
function map_rep_def(::MapRepExtrudeStyle, f, args)
    ExtrudeData(map_rep_def(f, map(map_rep_child, args)))
end

function map_rep_def(::MapRepHollowStyle, f, args)
    lvl = map_rep_def(f, map(arg -> arg isa HollowData ? arg.lvl : arg, args))
    if all(arg -> isa(arg, HollowData), args)
        return HollowData(lvl)
    end
    for (n, arg) in enumerate(args)
        if arg isa HollowData
            args_2 = map(arg -> value(gensym(), eltype(arg)), collect(args))
            args_2[n] = literal(fill_value(arg))
            if finch_leaf(simplify(FinchCompiler(), call(f, args_2...))) ==
                literal(fill_value(lvl))
                return HollowData(lvl)
            end
        end
    end
    return lvl
end

function map_rep_def(::MapRepSparseStyle, f, args)
    lvl = map_rep_def(f, map(map_rep_child, args))
    if all(arg -> isa(arg, SparseData), args)
        return SparseData(lvl)
    end
    for (n, arg) in enumerate(args)
        if arg isa SparseData
            args_2 = map(arg -> value(gensym(), eltype(arg)), collect(args))
            args_2[n] = literal(fill_value(arg))
            if finch_leaf(simplify(FinchCompiler(), call(f, args_2...))) ==
                literal(fill_value(lvl))
                return SparseData(lvl)
            end
        end
    end
    return DenseData(lvl)
end

function map_rep_def(::MapRepRepeatStyle, f, args)
    lvl = map_rep_def(f, map(map_rep_child, args))
    if all(arg -> isa(arg, RepeatData), args)
        return RepeatData(lvl)
    end
    for (n, arg) in enumerate(args)
        if arg isa RepeatData
            args_2 = map(arg -> value(gensym(), eltype(arg)), collect(args))
            args_2[n] = literal(fill_value(arg))
            if finch_leaf(simplify(FinchCompiler(), call(f, args_2...))) ==
                literal(fill_value(lvl))
                return RepeatData(lvl)
            end
        end
    end
    return DenseData(lvl)
end

function map_rep_def(::MapRepElementStyle, f, args)
    return ElementData(
        f(map(fill_value, args)...), return_type(DefaultAlgebra(), f, map(eltype, args)...)
    )
end

"""
    aggregate_rep(op, init, tns, dims)

Return a trait object representing the result of reducing a tensor represented
by `tns` on `dims` by `op` starting at `init`.
"""
function aggregate_rep(op, init, tns, dims)
    aggregate_rep_def(op, init, tns, reverse(map(n -> n in dims, 1:ndims(tns)))...)
end

#TODO I think HollowData here is wrong
function aggregate_rep_def(op, z, fbr::HollowData, drops...)
    HollowData(aggregate_rep_def(op, z, fbr.lvl, drops...))
end
function aggregate_rep_def(op, z, lvl::HollowData, drop, drops...)
    if op(z, fill_value(lvl)) == z
        HollowData(aggregate_rep_def(op, z, lvl.lvl, drops...))
    else
        HollowData(aggregate_rep_def(op, z, lvl.lvl, drops...))
    end
end

function aggregate_rep_def(op, z, lvl::SparseData, drop, drops...)
    if drop
        aggregate_rep_def(op, z, lvl.lvl, drops...)
    else
        if op(z, fill_value(lvl)) == z
            SparseData(aggregate_rep_def(op, z, lvl.lvl, drops...))
        else
            DenseData(aggregate_rep_def(op, z, lvl.lvl, drops...))
        end
    end
end

function aggregate_rep_def(op, z, lvl::RepeatData, drop, drops...)
    if drop
        aggregate_rep_def(op, z, lvl.lvl, drops...)
    else
        RepeatData(aggregate_rep_def(op, z, lvl.lvl, drops...))
    end
end

function aggregate_rep_def(op, z, lvl::DenseData, drop, drops...)
    if drop
        aggregate_rep_def(op, z, lvl.lvl, drops...)
    else
        DenseData(aggregate_rep_def(op, z, lvl.lvl, drops...))
    end
end

function aggregate_rep_def(op, z, lvl::ExtrudeData, drop, drops...)
    if drop
        aggregate_rep_def(op, z, lvl.lvl, drops...)
    else
        ExtrudeData(aggregate_rep_def(op, z, lvl.lvl, drops...))
    end
end

function aggregate_rep_def(op, z, lvl::ElementData)
    ElementData(z, fixpoint_type(op, z, eltype(lvl)))
end

"""
    permutedims_rep(tns, perm)

Return a trait object representing the result of permuting a tensor represented
by `tns` to the permutation `perm`.
"""
function permutedims_rep(tns, perm)
    if length(perm) == 0
        return tns
    end
    tns_2 = collapse_rep(
        permutedims_rep_select_def(tns, reverse([i == perm[end] for i in 1:ndims(tns)])...)
    )
    leaf = permutedims_rep(tns_2, [p - (p > perm[end]) for p in perm[1:(end - 1)]])
    collapse_rep(
        permutedims_rep_aggregate_def(
            leaf, tns, reverse([i != perm[end] for i in 1:ndims(tns)])...
        ),
    )
end

function permutedims_rep(tns::HollowData, perm)
    return HollowData(permutedims_rep(tns.lvl, perm))
end

function permutedims_rep_aggregate_def(tns, lvl::HollowData, drops...)
    permutedims_rep_aggregate_def(tns, lvl.lvl, drops...)
end
function permutedims_rep_aggregate_def(tns, lvl::SparseData, drop, drops...)
    if drop
        permutedims_rep_aggregate_def(tns, lvl.lvl, drops...)
    else
        SparseData(permutedims_rep_aggregate_def(tns, lvl.lvl, drops...))
    end
end
function permutedims_rep_aggregate_def(tns, lvl::ExtrudeData, drop, drops...)
    if drop
        permutedims_rep_aggregate_def(tns, lvl.lvl, drops...)
    else
        ExtrudeData(permutedims_rep_aggregate_def(tns, lvl.lvl, drops...))
    end
end
function permutedims_rep_aggregate_def(tns, lvl::DenseData, drop, drops...)
    if drop
        permutedims_rep_aggregate_def(tns, lvl.lvl, drops...)
    else
        DenseData(permutedims_rep_aggregate_def(tns, lvl.lvl, drops...))
    end
end
function permutedims_rep_aggregate_def(tns, lvl::RepeatData, drop, drops...)
    if drop
        permutedims_rep_aggregate_def(tns, lvl.lvl, drops...)
    else
        RepeatData(permutedims_rep_aggregate_def(tns, lvl.lvl, drops...))
    end
end
permutedims_rep_aggregate_def(tns, lvl::ElementData) = tns

function permutedims_rep_select_def(lvl::SparseData, drop, drops...)
    if drop
        HollowData(permutedims_rep_select_def(lvl.lvl, drops...))
    else
        SparseData(permutedims_rep_select_def(lvl.lvl, drops...))
    end
end
function permutedims_rep_select_def(lvl::DenseData, drop, drops...)
    if drop
        permutedims_rep_select_def(lvl.lvl, drops...)
    else
        DenseData(permutedims_rep_select_def(lvl.lvl, drops...))
    end
end
function permutedims_rep_select_def(lvl::ExtrudeData, drop, drops...)
    if drop
        permutedims_rep_select_def(lvl.lvl, drops...)
    else
        ExtrudeData(permutedims_rep_select_def(lvl.lvl, drops...))
    end
end
function permutedims_rep_select_def(lvl::RepeatData, drop, drops...)
    if drop
        permutedims_rep_select_def(lvl.lvl, drops...)
    else
        RepeatData(permutedims_rep_select_def(lvl.lvl, drops...))
    end
end
permutedims_rep_select_def(lvl::ElementData) = lvl

"""
    rep_construct(tns, protos...)

Construct a tensor suitable to hold data with a representation described by
`tns`. Assumes representation is collapsed.
"""
function rep_construct end
rep_construct(fbr) = rep_construct(fbr, [nothing for _ in 1:ndims(fbr)])
rep_construct(fbr::HollowData, protos) = rep_construct_hollow(fbr.lvl, protos)
function rep_construct_hollow(fbr::DenseData, protos)
    Tensor(construct_level_rep(SparseData(fbr.lvl), protos...))
end
function rep_construct_hollow(fbr::ExtrudeData, protos)
    Tensor(construct_level_rep(SparseData(fbr.lvl), protos...))
end
rep_construct_hollow(fbr::RepeatData, protos) = Tensor(construct_level_rep(fbr, protos...))
rep_construct_hollow(fbr::SparseData, protos) = Tensor(construct_level_rep(fbr, protos...))
rep_construct(fbr, protos) = Tensor(construct_level_rep(fbr, protos...))

construct_level_rep(fbr::SparseData, proto::Union{Nothing,typeof(walk),typeof(extrude)}, protos...) = SparseDict(construct_level_rep(fbr.lvl, protos...))
construct_level_rep(fbr::SparseData, proto::Union{typeof(laminate)}, protos...)                     = SparseDict(construct_level_rep(fbr.lvl, protos...))
construct_level_rep(fbr::RepeatData, proto::Union{Nothing,typeof(walk),typeof(extrude)}, protos...) = SparseRunList(construct_level_rep(fbr.lvl, protos...))
construct_level_rep(fbr::RepeatData, proto::Union{typeof(laminate)}, protos...)                     = SparseDict(construct_level_rep(fbr.lvl, protos...))
construct_level_rep(fbr::DenseData, proto, protos...)                                               = Dense(construct_level_rep(fbr.lvl, protos...))
construct_level_rep(fbr::ExtrudeData, proto, protos...)                                             = Dense(construct_level_rep(fbr.lvl, protos...), 1)
construct_level_rep(fbr::ElementData)                                                               = Element{fbr.fill_value,fbr.eltype}()

"""
    fiber_ctr(tns, protos...)

Return an expression that would construct a tensor suitable to hold data with a
representation described by `tns`. Assumes representation is collapsed.
"""
function fiber_ctr end
fiber_ctr(fbr) = fiber_ctr(fbr, [nothing for _ in 1:ndims(fbr)])
fiber_ctr(fbr::HollowData, protos) = fiber_ctr_hollow(fbr.lvl, protos)
function fiber_ctr_hollow(fbr::DenseData, protos)
    :(Tensor($(level_ctr(SparseData(fbr.lvl), protos...))))
end
function fiber_ctr_hollow(fbr::ExtrudeData, protos)
    :(Tensor($(level_ctr(SparseData(fbr.lvl), protos...))))
end
fiber_ctr_hollow(fbr::SparseData, protos) = :(Tensor($(level_ctr(fbr, protos...))))
fiber_ctr_hollow(fbr::RepeatData, protos) = :(Tensor($(level_ctr(fbr, protos...))))
fiber_ctr(fbr, protos) = :(Tensor($(level_ctr(fbr, protos...))))

level_ctr(fbr::SparseData, proto::Union{Nothing,typeof(walk),typeof(extrude)}, protos...) = :(SparseDict($(level_ctr(fbr.lvl, protos...))))
level_ctr(fbr::SparseData, proto::Union{typeof(laminate)}, protos...)                     = :(SparseDict($(level_ctr(fbr.lvl, protos...))))
level_ctr(fbr::RepeatData, proto::Union{Nothing,typeof(walk),typeof(extrude)}, protos...) = :(SparseRunList($(level_ctr(fbr.lvl, protos...))))
level_ctr(fbr::RepeatData, proto::Union{typeof(laminate)}, protos...)                     = :(SparseDict($(level_ctr(fbr.lvl, protos...))))
level_ctr(fbr::DenseData, proto, protos...)                                               = :(Dense($(level_ctr(fbr.lvl, protos...))))
level_ctr(fbr::ExtrudeData, proto, protos...)                                             = :(Dense($(level_ctr(fbr.lvl, protos...)), 1))
level_ctr(fbr::RepeatData, proto::Union{Nothing,typeof(walk),typeof(extrude)})            = :(Repeat{$(fbr.fill_value),$(fbr.eltype)}())
level_ctr(fbr::RepeatData, proto::Union{typeof(laminate)})                                = level_ctr(DenseData(ElementData(fbr.fill_value, fbr.eltype)), proto)
level_ctr(fbr::ElementData)                                                               = :(Element{$(fbr.fill_value),$(fbr.eltype)}())
