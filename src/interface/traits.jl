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
get_level_rep(fbr::SparseData) = fbr.lvl
set_level_rep(fbr::SparseData, lvl) = SparseData(lvl)

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
get_level_rep(fbr::RepeatData) = fbr.lvl
set_level_rep(fbr::RepeatData, lvl) = RepeatData(lvl)

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
get_level_rep(fbr::DenseData) = fbr.lvl
set_level_rep(fbr::DenseData, lvl) = DenseData(lvl)

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
get_level_rep(fbr::ExtrudeData) = fbr.lvl
set_level_rep(fbr::ExtrudeData, lvl) = ExtrudeData(lvl)

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
get_level_rep(fbr::HollowData) = fbr.lvl
set_level_rep(fbr::HollowData, lvl) = HollowData(lvl)

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
    rep_construct(tns, args...)

Construct a tensor suitable to hold data with a representation described by
`tns`. Assumes representation is collapsed.
"""
function rep_construct end
rep_construct(fbr::HollowData, args...) = rep_construct_hollow(fbr.lvl, args)
function rep_construct_hollow(fbr::DenseData, args)
    Tensor(construct_level_rep(SparseData(fbr.lvl)), args...)
end
function rep_construct_hollow(fbr::ExtrudeData, args)
    Tensor(construct_level_rep(SparseData(fbr.lvl)), args...)
end
rep_construct_hollow(fbr::RepeatData, args) = Tensor(construct_level_rep(fbr), args...)
rep_construct_hollow(fbr::SparseData, args) = Tensor(construct_level_rep(fbr), args...)
rep_construct(fbr, args...) = Tensor(construct_level_rep(fbr), args...)

function construct_level_rep(fbr::SparseData)
    SparseDict(construct_level_rep(fbr.lvl))
end
function construct_level_rep(fbr::RepeatData)
    SparseDict(construct_level_rep(fbr.lvl))
end
function construct_level_rep(fbr::DenseData)
    Dense(construct_level_rep(fbr.lvl))
end
function construct_level_rep(fbr::ExtrudeData)
    Dense(construct_level_rep(fbr.lvl), 1)
end
construct_level_rep(fbr::ElementData) = Element{fbr.fill_value,fbr.eltype}()

"""
    fiber_ctr(tns, args...)

Return an expression that would construct a tensor suitable to hold data with a
representation described by `tns`. Assumes representation is collapsed.
"""
function fiber_ctr end
fiber_ctr(fbr::HollowData, args...) = fiber_ctr_hollow(fbr.lvl, args)
function fiber_ctr_hollow(fbr::DenseData, args)
    :(Tensor($(level_ctr(SparseData(fbr.lvl)))), $(args...))
end
function fiber_ctr_hollow(fbr::ExtrudeData, args)
    :(Tensor($(level_ctr(SparseData(fbr.lvl)))), $(args...))
end
fiber_ctr_hollow(fbr::SparseData, args) = :(Tensor($(level_ctr(fbr)), $(args...)))
fiber_ctr_hollow(fbr::RepeatData, args) = :(Tensor($(level_ctr(fbr)), $(args...)))
fiber_ctr(fbr, args...) = :(Tensor($(level_ctr(fbr)), $(args...)))

function level_ctr(fbr::SparseData)
    :(SparseDict($(level_ctr(fbr.lvl))))
end
function level_ctr(fbr::RepeatData)
    :(SparseDict($(level_ctr(fbr.lvl))))
end
level_ctr(fbr::DenseData) = :(Dense($(level_ctr(fbr.lvl))))
function level_ctr(fbr::ExtrudeData)
    :(Dense($(level_ctr(fbr.lvl)), 1))
end
level_ctr(fbr::ElementData) = :(Element{$(fbr.fill_value),$(fbr.eltype)}())
