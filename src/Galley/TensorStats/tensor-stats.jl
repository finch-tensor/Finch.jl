# This struct holds the high-level definition of a tensor. This information should be
# agnostic to the statistics used for cardinality estimation. Any information which may be
# `Nothing` is considered a part of the physical definition which may be undefined for logical
# intermediates but is required to be defined for the inputs to an executable query.
@auto_hash_equals mutable struct TensorDef
    index_set::StableSet{IndexExpr}
    dim_sizes::OrderedDict{IndexExpr,Float64}
    fill_val::Any
    level_formats::Union{Nothing,Vector{LevelFormat}}
    index_order::Union{Nothing,Vector{IndexExpr}}
    index_protocols::Union{Nothing,Vector{AccessProtocol}}
end
function TensorDef(x)
    TensorDef(
        StableSet{IndexExpr}(),
        OrderedDict{IndexExpr,Float64}(),
        x,
        IndexExpr[],
        IndexExpr[],
        IndexExpr[],
    )
end

function copy_def(def::TensorDef)
    TensorDef(StableSet{IndexExpr}(x for x in def.index_set),
        OrderedDict{IndexExpr,Float64}(x for x in def.dim_sizes),
        def.fill_val,
        isnothing(def.level_formats) ? nothing : [x for x in def.level_formats],
        isnothing(def.index_order) ? nothing : [x for x in def.index_order],
        isnothing(def.index_protocols) ? nothing : [x for x in def.index_protocols])
end

function level_to_enums(lvl)
    if typeof(lvl) <: SparseListLevel
        return [t_sparse_list]
    elseif typeof(lvl) <: SparseDictLevel
        return [t_hash]
    elseif typeof(lvl) <: SparseCOOLevel
        return [t_coo for _ in lvl.shape]
    elseif typeof(lvl) <: SparseByteMap
        return [t_bytemap]
    elseif typeof(lvl) <: DenseLevel
        return [t_dense]
    else
        throw(Base.error("Level Not Recognized"))
    end
end

function get_tensor_formats(tensor::Tensor)
    level_formats = LevelFormat[]
    current_lvl = tensor.lvl
    while length(level_formats) < length(size(tensor))
        append!(level_formats, level_to_enums(current_lvl))
        current_lvl = current_lvl.lvl
    end
    # Because levels are built outside-in, we need to reverse this.
    level_formats = reverse(level_formats)
end

function TensorDef(tensor::Tensor, indices)
    shape_tuple = size(tensor)
    level_formats = get_tensor_formats(tensor::Tensor)
    dim_size = OrderedDict{IndexExpr,Float64}(
        indices[i] => shape_tuple[i] for i in 1:length(size(tensor))
    )
    fill_val = Finch.fill_value(tensor)
    return TensorDef(
        StableSet{IndexExpr}(indices), dim_size, fill_val, level_formats, indices, nothing
    )
end

function reindex_def(indices, def::TensorDef)
    @assert length(indices) == length(def.index_order)
    rename_dict = OrderedDict{IndexExpr,IndexExpr}()
    for i in eachindex(indices)
        rename_dict[def.index_order[i]] = indices[i]
    end
    new_index_set = StableSet{IndexExpr}()
    for idx in def.index_set
        push!(new_index_set, rename_dict[idx])
    end

    new_dim_sizes = OrderedDict{IndexExpr,Float64}()
    for (idx, size) in def.dim_sizes
        new_dim_sizes[rename_dict[idx]] = size
    end

    return TensorDef(
        new_index_set,
        new_dim_sizes,
        def.fill_val,
        def.level_formats,
        indices,
        def.index_protocols,
    )
end

function set_fill_value!(def::TensorDef, fill_val)
    TensorDef(
        def.index_set,
        def.dim_sizes,
        fill_val,
        def.level_formats,
        def.index_order,
        def.index_protocols,
    )
end

function relabel_index!(def::TensorDef, i::IndexExpr, j::IndexExpr)
    if i == j || i ∉ def.index_set
        return nothing
    end
    delete!(def.index_set, i)
    push!(def.index_set, j)
    def.dim_sizes[j] = def.dim_sizes[i]
    delete!(def.dim_sizes, i)
    if !isnothing(def.index_order)
        for k in eachindex(def.index_order)
            if def.index_order[k] == i
                def.index_order[k] = j
            end
        end
    end
end

function add_dummy_idx!(def::TensorDef, i::IndexExpr; idx_pos=-1)
    def.dim_sizes[i] = 1
    push!(def.index_set, i)
    if idx_pos > -1
        insert!(def.index_order, idx_pos, i)
        insert!(def.level_formats, idx_pos, t_dense)
    end
end

get_dim_sizes(def::TensorDef) = def.dim_sizes
get_dim_size(def::TensorDef, idx::IndexExpr) = def.dim_sizes[idx]
get_index_set(def::TensorDef) = def.index_set
get_index_order(def::TensorDef) = def.index_order
get_fill_value(def::TensorDef) = def.fill_val
function get_index_format(def::TensorDef, idx::IndexExpr)
    def.level_formats[findfirst(x -> x == idx, def.index_order)]
end
get_index_formats(def::TensorDef) = def.level_formats
function get_index_protocol(def::TensorDef, idx::IndexExpr)
    def.index_protocols[findfirst(x -> x == idx, def.index_order)]
end
get_index_protocols(def::TensorDef) = def.index_protocols

function get_dim_space_size(def::TensorDef, indices)
    dim_space_size::Float64 = 1
    for idx in indices
        dim_space_size *= def.dim_sizes[idx]
    end
    if dim_space_size == 0 || dim_space_size > typemax(Int)
        return Float64(typemax(Int))^(sizeof(Int) * 8 - 1)
    end
    return dim_space_size
end

abstract type TensorStats end

get_dim_space_size(stat::TensorStats, indices) = get_dim_space_size(get_def(stat), indices)
get_dim_sizes(stat::TensorStats) = get_dim_sizes(get_def(stat))
get_dim_size(stat::TensorStats, idx::IndexExpr) = get_dim_size(get_def(stat), idx)
get_index_set(stat::TensorStats) = get_index_set(get_def(stat))
get_index_order(stat::TensorStats) = get_index_order(get_def(stat))
get_fill_value(stat::TensorStats) = get_fill_value(get_def(stat))
get_index_format(stat::TensorStats, idx::IndexExpr) = get_index_format(get_def(stat), idx)
get_index_formats(stat::TensorStats) = get_index_formats(get_def(stat))
function get_index_protocol(stat::TensorStats, idx::IndexExpr)
    get_index_protocol(get_def(stat), idx)
end
get_index_protocols(stat::TensorStats) = get_index_protocols(get_def(stat))
copy_stats(stat::Nothing) = nothing
