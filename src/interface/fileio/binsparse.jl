const BINSPARSE_VERSION = 0.1

"""
    bspwrite(::AbstractString, tns)
    bspwrite(::HDF5.File, tns)
    bspwrite(::NPYPath, tns)

Write the Finch tensor to a file using
[Binsparse](https://github.com/GraphBLAS/binsparse-specification) file format.

Supported file extensions are:

- `.bsp.h5`: HDF5 file format ([HDF5](https://github.com/JuliaIO/HDF5.jl) must be loaded)
- `.bspnpy`: NumPy and JSON directory format ([NPZ](https://github.com/fhs/NPZ.jl) must be loaded)

!!! warning
    The Binsparse spec is under development. Additionally, this function may not
    be fully conformant. Please file bug reports if you see anything amiss.
"""
function bspwrite end

"""
bspread(::AbstractString)
bspread(::HDF5.File)
bspread(::NPYPath)

Read the [Binsparse](https://github.com/GraphBLAS/binsparse-specification) file into a Finch tensor.

Supported file extensions are:

- `.bsp.h5`: HDF5 file format ([HDF5](https://github.com/JuliaIO/HDF5.jl) must be loaded)
- `.bspnpy`: NumPy and JSON directory format ([NPZ](https://github.com/fhs/NPZ.jl) must be loaded)

!!! warning
The Binsparse spec is under development. Additionally, this function may not
be fully conformant. Please file bug reports if you see anything amiss.
"""
function bspread end

using Finch: level_ndims, SwizzleArray

bspread_type_lookup = OrderedDict(
    "uint8" => UInt8,
    "uint16" => UInt16,
    "uint32" => UInt32,
    "uint64" => UInt64,
    "int8" => Int8,
    "int16" => Int16,
    "int32" => Int32,
    "int64" => Int64,
    "float32" => Float32,
    "float64" => Float64,
    "bint8" => Bool,
)

function bspread_vector end
function bspwrite_vector end

function bspread_data(f, desc, key)
    t = desc["data_types"][key]
    if (m = match(r"^iso\[([^\[]*)\]$", t)) !== nothing
        throw(ArgumentError("iso values not currently supported"))
    elseif (m = match(r"^complex\[([^\[]*)\]$", t)) !== nothing
        desc["data_types"][key] = m.captures[1]
        data = bspread_data(f, desc, key)
        return reinterpret(Complex{eltype(data)}, data)
    elseif (m = match(r"^[^\]]*$", t)) !== nothing
        haskey(bspread_type_lookup, t) || throw(ArgumentError("unknown binsparse type $t"))
        convert(Vector{bspread_type_lookup[t]}, bspread_vector(f, key))
    else
        throw(ArgumentError("unknown binsparse type wrapper $t"))
    end
end

bspwrite_type_lookup = OrderedDict(v => k for (k, v) in bspread_type_lookup)

function bspwrite_data(f, desc, key, data)
    type_desc = bspwrite_data_helper(f, desc, key, data)
end

function bspwrite_data_helper(f, desc, key, data::AbstractVector{T}) where {T}
    haskey(bspwrite_type_lookup, T) || throw(ArgumentError("Cannot write $T to binsparse"))
    bspwrite_vector(f, data, key)
    desc["data_types"][key] = bspwrite_type_lookup[T]
end

function bspwrite_data_helper(f, desc, key, data::AbstractVector{Complex{T}}) where {T}
    data = reinterpret(T, data)
    bspwrite_data_helper(f, desc, key, data)
    desc["data_types"][key] = "complex[$(desc["data_types"][key])]"
end

bspread_tensor_lookup = OrderedDict(
    "DVEC" => OrderedDict(
        "level" => OrderedDict(
            "level_kind" => "dense",
            "rank" => 1,
            "level" => OrderedDict(
                "level_kind" => "element"
            ),
        ),
    ),
    "DMAT" => OrderedDict(
        "level" => OrderedDict(
            "level_kind" => "dense",
            "rank" => 1,
            "level" => OrderedDict(
                "level_kind" => "dense",
                "rank" => 1,
                "level" => OrderedDict(
                    "level_kind" => "element"
                ),
            ),
        ),
    ),
    "DMATR" => OrderedDict(
        "level" => OrderedDict(
            "level_kind" => "dense",
            "rank" => 1,
            "level" => OrderedDict(
                "level_kind" => "dense",
                "rank" => 1,
                "level" => OrderedDict(
                    "level_kind" => "element"
                ),
            ),
        ),
    ),
    "DMATC" => OrderedDict(
        "transpose" => [1, 0],
        "level" => OrderedDict(
            "level_kind" => "dense",
            "rank" => 1,
            "level" => OrderedDict(
                "level_kind" => "dense",
                "rank" => 1,
                "level" => OrderedDict(
                    "level_kind" => "element"
                ),
            ),
        ),
    ),
    "CVEC" => OrderedDict(
        "level" => OrderedDict(
            "level_kind" => "sparse",
            "rank" => 1,
            "level" => OrderedDict(
                "level_kind" => "element"
            ),
        ),
    ),
    "CSR" => OrderedDict(
        "level" => OrderedDict(
            "level_kind" => "dense",
            "rank" => 1,
            "level" => OrderedDict(
                "level_kind" => "sparse",
                "rank" => 1,
                "level" => OrderedDict(
                    "level_kind" => "element"
                ),
            ),
        ),
    ),
    "CSC" => OrderedDict(
        "transpose" => [1, 0],
        "level" => OrderedDict(
            "level_kind" => "dense",
            "rank" => 1,
            "level" => OrderedDict(
                "level_kind" => "sparse",
                "rank" => 1,
                "level" => OrderedDict(
                    "level_kind" => "element"
                ),
            ),
        ),
    ),
    "DCSR" => OrderedDict(
        "level" => OrderedDict(
            "level_kind" => "sparse",
            "rank" => 1,
            "level" => OrderedDict(
                "level_kind" => "sparse",
                "rank" => 1,
                "level" => OrderedDict(
                    "level_kind" => "element"
                ),
            ),
        ),
    ),
    "DCSC" => OrderedDict(
        "transpose" => [1, 0],
        "level" => OrderedDict(
            "level_kind" => "sparse",
            "rank" => 1,
            "level" => OrderedDict(
                "level_kind" => "sparse",
                "rank" => 1,
                "level" => OrderedDict(
                    "level_kind" => "element"
                ),
            ),
        ),
    ),
    "COO" => OrderedDict(
        "level" => OrderedDict(
            "level_kind" => "sparse",
            "rank" => 2,
            "level" => OrderedDict(
                "level_kind" => "element"
            ),
        ),
    ),
    "COOR" => OrderedDict(
        "level" => OrderedDict(
            "level_kind" => "sparse",
            "rank" => 2,
            "level" => OrderedDict(
                "level_kind" => "element"
            ),
        ),
    ),
    "COOC" => OrderedDict(
        "transpose" => [1, 0],
        "level" => OrderedDict(
            "level_kind" => "sparse",
            "rank" => 2,
            "level" => OrderedDict(
                "level_kind" => "element"
            ),
        ),
    ),
)

bspwrite_format_lookup = OrderedDict(v => k for (k, v) in bspread_tensor_lookup)

#indices_zero_to_one(vec::Vector{Ti}) where {Ti} = PlusOneVector(vec)
indices_zero_to_one(vec::Vector) = vec .+ one(eltype(vec))
indices_one_to_zero(vec::Vector) = vec .- one(eltype(vec))

struct NPYPath
    dirname::String
end

function bspwrite_h5(args...)
    throw(
        FinchExtensionError(
            "HDF5.jl must be loaded to write .bsp.h5 files (hint: `using HDF5`)"
        ),
    )
end
function bspwrite_bspnpy(args...)
    throw(
        FinchExtensionError(
            "NPZ.jl must be loaded to write .bspnpy files (hint: `using NPZ`)"
        ),
    )
end

function bspwrite(fname::AbstractString, arr, attrs=OrderedDict())
    if endswith(fname, ".h5") || endswith(fname, ".hdf5")
        bspwrite_h5(fname, arr, attrs)
    elseif endswith(fname, ".bspnpy")
        bspwrite_bspnpy(fname, arr, attrs)
    else
        error("Unknown file extension for file $fname")
    end
end
bspwrite(fname, arr, attrs=OrderedDict()) = bspwrite_tensor(fname, arr, attrs)

function bspwrite_tensor(io, fbr::Tensor, attrs=OrderedDict())
    bspwrite_tensor(io, swizzle(fbr, 1:ndims(fbr)...), attrs)
end

function bspwrite_tensor(
    io, arr::SwizzleArray{dims,<:Tensor}, attrs=OrderedDict()
) where {dims}
    desc = OrderedDict(
        "tensor" => OrderedDict{Any,Any}(
            "level" => OrderedDict()
        ),
        "fill" => true,
        "shape" => map(Int, size(arr)),
        "data_types" => OrderedDict(),
        "version" => "$BINSPARSE_VERSION",
        "number_of_stored_values" => countstored(arr),
        "attrs" => attrs,
    )
    if !issorted(reverse(collect(dims)))
        desc["tensor"]["transpose"] = reverse(collect(dims)) .- 1
    end
    bspwrite_level(io, desc, desc["tensor"]["level"], arr.body.lvl)
    if haskey(bspwrite_format_lookup, desc["tensor"])
        desc["format"] = bspwrite_format_lookup[desc["tensor"]]
    end
    bspwrite_header(io, json(Dict("binsparse" => desc), 4))
end

function bspwrite_header end

function bspread_h5(args...)
    throw(
        FinchExtensionError(
            "HDF5.jl must be loaded to read .bsp.h5 files (hint: `using HDF5`)"
        ),
    )
end
function bspread_bspnpy(args...)
    throw(
        FinchExtensionError(
            "NPZ.jl must be loaded to read .bspnpy files (hint: `using NPZ`)"
        ),
    )
end

function bspread(fname::AbstractString)
    if endswith(fname, ".h5") || endswith(fname, ".hdf5")
        bspread_h5(fname)
    elseif endswith(fname, ".bspnpy")
        bspread_bspnpy(fname)
    else
        error("Unknown file extension for file $fname")
    end
end

function bspread_header end

function bspread(f)
    desc = bspread_header(f)["binsparse"]
    @assert desc["version"] == "$BINSPARSE_VERSION"
    fmt = OrderedDict{Any,Any}(
        get(() -> bspread_tensor_lookup[desc["format"]], desc, "tensor")
    )
    if !haskey(fmt, "transpose")
        fmt["transpose"] = collect(0:(length(desc["shape"]) - 1))
    end
    if !issorted(reverse(fmt["transpose"]))
        sigma = sortperm(reverse(fmt["transpose"] .+ 1))
        desc["shape"] = desc["shape"][sigma]
    end
    fbr = Tensor(bspread_level(f, desc, fmt["level"]))
    if !issorted(reverse(fmt["transpose"]))
        fbr = swizzle(fbr, reverse(fmt["transpose"] .+ 1)...)
    end
    if haskey(desc, "structure")
        throw(ArgumentError("binsparse structure field currently unsupported"))
    end
    fbr
end

bspread_level(f, desc, fmt) = bspread_level(f, desc, fmt, Val(Symbol(fmt["level_kind"])))

function bspwrite_level(f, desc, fmt, lvl::ElementLevel{Vf}) where {Vf}
    fmt["level_kind"] = "element"
    bspwrite_data(f, desc, "values", lvl.val)
    bspwrite_data(f, desc, "fill_value", [Vf])
end
function bspread_level(f, desc, fmt, ::Val{:element})
    val = convert(Vector, bspread_data(f, desc, "values"))
    if haskey(f, "fill_value")
        Vf = bspread_data(f, desc, "fill_value")[1]
    else
        Vf = zero(eltype(val))
    end
    ElementLevel(Vf, val)
end

function bspwrite_level(f, desc, fmt, lvl::DenseLevel{Vf}) where {Vf}
    fmt["level_kind"] = "dense"
    fmt["rank"] = 1
    fmt["level"] = OrderedDict()
    bspwrite_level(f, desc, fmt["level"], lvl.lvl)
end
function bspread_level(f, desc, fmt, ::Val{:dense})
    lvl = bspread_level(f, desc, fmt["level"])
    R = fmt["rank"]
    for r in 1:R
        n = level_ndims(typeof(lvl))
        shape = Int(desc["shape"][n + 1])
        lvl = DenseLevel(lvl, shape)
    end
    lvl
end

function bspwrite_level(f, desc, fmt, lvl::SparseListLevel)
    fmt["level_kind"] = "sparse"
    fmt["rank"] = 1
    n = level_ndims(typeof(lvl))
    N = length(desc["shape"])
    if N - n > 0
        bspwrite_data(f, desc, "pointers_to_$(N - n)", indices_one_to_zero(lvl.ptr))
    end
    bspwrite_data(f, desc, "indices_$(N - n)", indices_one_to_zero(lvl.idx))
    fmt["level"] = OrderedDict()
    bspwrite_level(f, desc, fmt["level"], lvl.lvl)
end
function bspwrite_level(f, desc, fmt, lvl::SparseCOOLevel{R}) where {R}
    fmt["level_kind"] = "sparse"
    fmt["rank"] = R
    n = level_ndims(typeof(lvl))
    N = length(desc["shape"])
    if N - n > 0
        bspwrite_data(f, desc, "pointers_to_$(N - n)", indices_one_to_zero(lvl.ptr))
    end
    for r in 1:R
        bspwrite_data(f, desc, "indices_$(N - n + R - r)", indices_one_to_zero(lvl.tbl[r]))
    end
    fmt["level"] = OrderedDict()
    bspwrite_level(f, desc, fmt["level"], lvl.lvl)
end
function bspread_level(f, desc, fmt, ::Val{:sparse})
    R = fmt["rank"]
    lvl = bspread_level(f, desc, fmt["level"])
    n = level_ndims(typeof(lvl)) + R
    N = length(desc["shape"])
    tbl = (map(1:R) do r
        indices_zero_to_one(bspread_data(f, desc, "indices_$(N - n + R - r)"))
    end...,)
    if N - n > 0
        ptr = bspread_data(f, desc, "pointers_to_$(N - n)")
    else
        ptr = [0, length(tbl[1])]
    end
    ptr = indices_zero_to_one(ptr)
    shape = ntuple(r -> eltype(tbl[r])(desc["shape"][n - R + r]), R)
    if R == 1
        SparseListLevel(lvl, shape[1], ptr, tbl[1])
    else
        SparseCOOLevel{Int(R),typeof(shape)}(lvl, shape, ptr, tbl)
    end
end
