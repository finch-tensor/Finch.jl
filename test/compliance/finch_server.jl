#!/usr/bin/env julia

# SPDX-FileCopyrightText: 2026 Finch Developers
# SPDX-License-Identifier: MIT
#
# Persistent Julia server for the Binsparse compliance test suite.
#
# Julia startup + package loading takes ~30 s.  The binsparse-tests harness
# invokes the converter executables hundreds of times via subprocess.run().
# Starting a fresh Julia process each time would be prohibitively slow.
#
# Instead we start ONE Julia process that loads Finch / HDF5 / NPZ once, then
# sits in a loop reading JSON-line requests from a named pipe (FIFO).  The
# three thin bash wrappers (binsparse_to_npy, npy_to_binsparse,
# binsparse_to_binsparse) each write a single request line and wait for a
# response file.
#
# Protocol
# --------
# Request  (one JSON line written to the FIFO):
#   {"cmd": "<command>", "args": ["arg1", ...], "response": "/tmp/resp_$$"}
#
# Response (JSON written to the response-file path):
#   {"exit_code": 0}                       on success
#   {"exit_code": 1, "error": "message"}   on failure
#
# The special command {"cmd": "shutdown"} causes a clean exit.

using Finch
using HDF5
using NPZ
using JSON

# ─── binsparse_to_npy ────────────────────────────────────────────────
# Read a Binsparse HDF5 file with Finch, then write its dense values,
# explicit-storage pattern, and fill value as separate .npy files.

function cmd_binsparse_to_npy(args)
    length(args) == 4 || error(
        "binsparse_to_npy requires 4 args: tensor_in tensor_out pattern_out fill_value_out"
    )
    tensor_in, tensor_out, pattern_out, fill_value_out = args
    tns = h5open(Finch.bspread, tensor_in, "r")
    npzwrite(tensor_out, dense_array(tns))
    npzwrite(pattern_out, dense_array(pattern!(tns)))
    npzwrite(fill_value_out, fill(Finch.fill_value(tns)))
    return nothing
end

# Scalar access avoids compiling a new copy kernel for every generated
# fill value and layout in the compliance suite.
dense_array(tns::Tensor) = [tns(Tuple(i)...) for i in CartesianIndices(size(tns))]
function dense_array(tns::Finch.SwizzleArray{dims}) where {dims}
    permutedims(dense_array(tns.body), dims)
end

# ─── npy_to_binsparse ────────────────────────────────────────────────
# Read dense .npy, pattern .npy, fill-value .npy, and a partial JSON
# header, then write a Binsparse HDF5 file using Finch.

function cmd_npy_to_binsparse(args)
    length(args) == 5 || error(
        "npy_to_binsparse requires 5 args: tensor_in pattern_in fill_value_in header_in tensor_out"
    )
    tensor_in, pattern_in, fill_value_in, header_in, tensor_out = args
    dense = read_array(tensor_in)
    pat = read_array(pattern_in)
    fill_value = only(npzread(fill_value_in))
    header = JSON.parsefile(header_in)
    tns = finch_tensor(dense, pat, fill_value, header)
    h5open(tensor_out, "w") do io
        Finch.bspwrite(io, tns; alias=header["format"] != "custom")
        match_header!(io, header)
    end
    return nothing
end

# NPZ.jl reads 0-d arrays as scalars.
read_array(path) = (x = npzread(path); x isa AbstractArray ? x : fill(x))

# Build the Finch tensor with the header's layout, storing exactly the entries
# marked in `pat` (including those equal to the fill value).
function finch_tensor(dense, pat, fill_value, header)
    fmt = binsparse_format(header)
    # Stored dimension i is logical dimension transpose[i], outermost first.
    transpose = Vector{Int}(get(fmt, "transpose", 0:(ndims(dense) - 1))) .+ 1
    stored = permutedims(dense, transpose)
    coords = sort!(map(Tuple, findall(permutedims(pat, transpose))))
    fill_value = convert(eltype(dense), fill_value)
    lvl = finch_level(fmt["level"], stored, coords, [()], 0, fill_value)
    # Finch lists dimensions innermost first.
    return swizzle(Tensor(lvl), invperm(reverse(transpose))...)
end

function binsparse_format(header)
    if header["format"] == "custom"
        header["custom"]
    else
        Finch.bspread_tensor_lookup[header["format"]]
    end
end

# Build the level for stored dimensions `depth + 1` onwards. `parents` lists the
# coordinate prefixes stored by the enclosing levels, in storage order.
function finch_level(fmt, A, coords, parents, depth, fill_value)
    if fmt["level_desc"] == "element"
        return Element(fill_value, eltype(A)[A[p...] for p in parents])
    end
    rank = fmt["rank"]
    shape = size(A)[(depth + 1):(depth + rank)]
    children = Tuple[]
    ptr = [1]
    for p in parents
        if fmt["level_desc"] == "dense"
            block = [(p..., Tuple(i)...) for i in CartesianIndices(shape)]
            append!(children, sort!(vec(block)))
        else
            append!(
                children, unique(c[1:(depth + rank)] for c in coords if c[1:depth] == p)
            )
        end
        push!(ptr, length(children) + 1)
    end
    lvl = finch_level(fmt["level"], A, coords, children, depth + rank, fill_value)
    # Finch levels list dimensions innermost first.
    if fmt["level_desc"] == "dense"
        for n in reverse(shape)
            lvl = Dense(lvl, n)
        end
        return lvl
    elseif rank == 1
        return SparseList{Int}(lvl, only(shape), ptr, Int[c[end] for c in children])
    else
        tbl = ntuple(r -> Int[c[depth + rank + 1 - r] for c in children], rank)
        return SparseCOO{rank}(lvl, reverse(shape), ptr, tbl)
    end
end

# ─── binsparse_to_binsparse ──────────────────────────────────────────
# Read the file into Finch, then write it back out with the input's header.
# Like the reference, write DMAT and COO under their canonical names.

function cmd_binsparse_to_binsparse(args)
    length(args) == 2 ||
        error("binsparse_to_binsparse requires 2 args: tensor_in tensor_out")
    tensor_in, tensor_out = args
    header, tns = h5open(tensor_in, "r") do io
        Finch.bspread_header(io)["binsparse"], Finch.bspread(io)
    end
    h5open(tensor_out, "w") do io
        Finch.bspwrite(io, tns; alias=header["format"] != "custom")
        match_header!(io, header; rename_aliases=false)
    end
    return nothing
end

# ─── header matching ─────────────────────────────────────────────────
# The harness expects the requested header verbatim, but some spellings have no
# Finch equivalent: DMAT and COO are written as DMATR and COOR, dense levels of
# rank r are written as r levels of rank 1, identity transposes are omitted,
# and ISO values are expanded. Adopt the requested spelling wherever it
# describes what Finch wrote. Anything else is left for the harness to report.

function match_header!(io, requested; rename_aliases=true)
    desc = Finch.bspread_header(io)
    actual = desc["binsparse"]
    custom = requested["format"] == "custom"
    if (actual["format"] == "custom") == custom && (custom || rename_aliases) &&
        layout(actual) == layout(requested)
        actual["format"] = requested["format"]
        delete!(actual, "custom")
        haskey(requested, "custom") && (actual["custom"] = requested["custom"])
    end
    for (key, dtype) in requested["data_types"]
        if dtype == "iso[$(get(actual["data_types"], key, nothing))]"
            data = read(io[key])
            width = startswith(dtype, "iso[complex[") ? 2 : 1
            value = data[1:min(width, end)]
            data == repeat(value, length(data) ÷ width) ||
                error("$key must have identical stored values to be written as $dtype")
            delete_object(io, key)
            io[key] = value
            actual["data_types"][key] = dtype
        end
    end
    delete_attribute(io, "binsparse")
    Finch.bspwrite_header(io, JSON.json(desc, 4))
end

# The transpose and a list of (level_desc, rank) pairs, with dense ranks split.
function layout(header)
    fmt = binsparse_format(header)
    transpose = Vector{Int}(get(fmt, "transpose", 0:(length(header["shape"]) - 1)))
    return (transpose, levels(fmt["level"]))
end

function levels(fmt)
    kind = fmt["level_desc"]
    kind == "element" && return [(kind, 0)]
    rank = fmt["rank"]
    here = kind == "dense" ? fill((kind, 1), rank) : [(kind, rank)]
    return [here; levels(fmt["level"])]
end

# ─── server loop ──────────────────────────────────────────────────────

const COMMANDS = Dict{String,Function}(
    "binsparse_to_npy" => cmd_binsparse_to_npy,
    "npy_to_binsparse" => cmd_npy_to_binsparse,
    "binsparse_to_binsparse" => cmd_binsparse_to_binsparse,
)

function write_response(response_path::String, exit_code::Int; error_msg::String="")
    resp = Dict{String,Any}("exit_code" => exit_code)
    if !isempty(error_msg)
        resp["error"] = error_msg
    end
    tmp_path = response_path * ".tmp"
    open(tmp_path, "w") do f
        JSON.print(f, resp)
    end
    mv(tmp_path, response_path; force=true)
end

function handle_request(line::AbstractString)
    req = JSON.parse(line)

    cmd = get(req, "cmd", nothing)
    cmd === nothing && return true  # ignore malformed, keep running

    # Shutdown command
    cmd == "shutdown" && return false

    args = get(req, "args", String[])
    response_path = get(req, "response", nothing)

    if response_path === nothing
        @warn "request missing 'response' field, ignoring"
        return true
    end

    if !haskey(COMMANDS, cmd)
        write_response(response_path, 1; error_msg="unknown command: $cmd")
        return true
    end

    try
        COMMANDS[cmd](args)
        write_response(response_path, 0)
    catch e
        msg = sprint(showerror, e, catch_backtrace())
        write_response(response_path, 1; error_msg=msg)
    end

    return true  # keep running
end

function server_main()
    if length(ARGS) < 1
        println(stderr, "usage: finch_server.jl <fifo_path> [ready_file]")
        exit(2)
    end

    fifo_path = ARGS[1]
    ready_file = length(ARGS) >= 2 ? ARGS[2] : nothing

    # Signal that we are ready (all packages loaded, server listening)
    if ready_file !== nothing
        open(ready_file, "w") do f
            println(f, "ready")
        end
    end

    # Main loop: keep reopening the FIFO.
    # Each time a writer closes, `readline` returns "" and we reopen.
    while true
        io = open(fifo_path, "r")
        try
            for line in eachline(io)
                stripped = strip(line)
                isempty(stripped) && continue
                keep_running = handle_request(stripped)
                keep_running || return nothing
            end
        finally
            close(io)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    server_main()
end
