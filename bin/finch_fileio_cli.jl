#!/usr/bin/env julia

using Finch
using HDF5
using MatrixMarket
using SparseArrays

function usage(io::IO = stderr)
    println(io, "usage:")
    println(io, "  finch_fileio_cli.jl mtx2bsp INPUT.mtx OUTPUT.bsp.h5[:DATASET]")
    println(io, "  finch_fileio_cli.jl bsp2mtx INPUT.bsp.h5[:DATASET] OUTPUT.mtx")
    println(io, "  finch_fileio_cli.jl check_equivalence FILE1 FILE2")
    return 1
end

function split_path_and_dataset(value::AbstractString)
    if !occursin(':', value)
        return String(value), nothing
    end

    path, dataset = rsplit(value, ':'; limit=2)
    if endswith(path, ".h5") || endswith(path, ".hdf5")
        return path, isempty(dataset) ? nothing : dataset
    end

    return String(value), nothing
end

function lookup_group(io::HDF5.File, dataset::AbstractString)
    group = io
    for part in split(dataset, '/')
        isempty(part) && continue
        group = group[part]
    end
    return group
end

function ensure_group(io::HDF5.File, dataset::AbstractString)
    group = io
    for part in split(dataset, '/')
        isempty(part) && continue
        group = haskey(group, part) ? group[part] : create_group(group, part)
    end
    return group
end

function read_tensor(path_with_dataset::AbstractString)
    path, dataset = split_path_and_dataset(path_with_dataset)
    if dataset === nothing
        return Finch.fread(path)
    end

    if !(endswith(path, ".bsp.h5") || endswith(path, ".bsp.hdf5"))
        throw(ArgumentError("dataset addressing is only supported for binsparse HDF5 files"))
    end

    h5open(path, "r") do io
        Finch.bspread(lookup_group(io, dataset))
    end
end

function write_tensor(path_with_dataset::AbstractString, tensor)
    path, dataset = split_path_and_dataset(path_with_dataset)
    if dataset === nothing
        Finch.fwrite(path, tensor)
        return path
    end

    if !(endswith(path, ".bsp.h5") || endswith(path, ".bsp.hdf5"))
        throw(ArgumentError("dataset addressing is only supported for binsparse HDF5 files"))
    end

    h5open(path, "w") do io
        Finch.bspwrite(ensure_group(io, dataset), tensor)
    end
    return path
end

function tensors_equivalent(left, right)
    size(left) == size(right) || return false
    if ndims(left) == 2 && ndims(right) == 2
        return SparseMatrixCSC(left) == SparseMatrixCSC(right)
    end
    return left == right
end

function main(args)
    length(args) == 3 || return usage()
    command, arg1, arg2 = args

    if command == "mtx2bsp"
        write_tensor(arg2, Finch.fread(arg1))
        return 0
    elseif command == "bsp2mtx"
        Finch.fwrite(arg2, read_tensor(arg1))
        return 0
    elseif command == "check_equivalence"
        left = read_tensor(arg1)
        right = read_tensor(arg2)
        if tensors_equivalent(left, right)
            println("OK")
            return 0
        end
        println(stderr, "files are not equivalent")
        return 1
    else
        return usage()
    end
end

exit(main(ARGS))
