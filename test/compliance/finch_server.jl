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
using SparseArrays

# ─── binsparse_to_npy ────────────────────────────────────────────────
# Read a Binsparse HDF5 file with Finch, then write its dense values,
# explicit-storage pattern, and fill value as separate .npy files.

function cmd_binsparse_to_npy(args)
    length(args) == 4 || error(
        "binsparse_to_npy requires 4 args: tensor_in tensor_out pattern_out fill_value_out"
    )
    tensor_in, tensor_out, pattern_out, fill_value_out = args

    tns = h5open(tensor_in, "r") do io
        Finch.bspread(io)
    end

    Vf = Finch.fill_value(tns)

    # Dense values — Finch Array() materialises to a Julia dense array
    dense = Array(tns)
    npzwrite(tensor_out, dense)

    # Explicit-storage pattern (Bool → UInt8 for numpy bint8 compat)
    pat = UInt8.(Array(pattern!(tns)))
    npzwrite(pattern_out, pat)

    # Fill value as 0-D array
    npzwrite(fill_value_out, fill(Vf))

    return nothing
end

# ─── npy_to_binsparse ────────────────────────────────────────────────
# Read dense .npy, pattern .npy, fill-value .npy, and a partial JSON
# header, then write a Binsparse HDF5 file using Finch.
#
# The header JSON may contain "format", "shape", "data_types", etc.
# We honour the format if Finch recognises it, otherwise fall back to
# letting bspwrite choose.

function cmd_npy_to_binsparse(args)
    length(args) == 5 || error(
        "npy_to_binsparse requires 5 args: tensor_in pattern_in fill_value_in header_in tensor_out (got $(length(args)): $args)"
    )
    tensor_in, pattern_in, fill_value_in, header_in, tensor_out = args

    dense = npzread(tensor_in)
    pat = npzread(pattern_in)
    fill_arr = npzread(fill_value_in)
    # fill_value is stored as a 0-D npy array
    fill_val = ndims(fill_arr) == 0 ? fill_arr[] : fill_arr[1]

    header = JSON.parsefile(header_in)
    fmt = get(header, "format", nothing)
    version = get(header, "version", "0.1")

    tns = _construct_tensor_by_format(dense, fill_val, fmt)

    h5open(tensor_out, "w") do io
        Finch.bspwrite_tensor(io, tns, DataStructures.OrderedDict(), version)
    end

    return nothing
end

function _construct_tensor_by_format(dense, fill_val, fmt)
    T = eltype(dense)
    N = ndims(dense)

    if N == 0
        return Tensor(Element{fill_val,T,Int}([dense[]]))
    end

    if fmt == "CSR" && N == 2
        tns = Tensor(Dense(SparseList(Element(fill_val))))
        copyto!(tns, dense)
        return tns
    elseif fmt == "CSC" && N == 2
        tns = swizzle(Tensor(Dense(SparseList(Element(fill_val)))), 2, 1)
        copyto!(tns, dense)
        return tns
    elseif (fmt == "COO" || fmt == "COOR") && N == 2
        tns = Tensor(SparseCOO{2}(Element(fill_val)))
        copyto!(tns, dense)
        return tns
    elseif fmt == "COOC" && N == 2
        tns = swizzle(Tensor(SparseCOO{2}(Element(fill_val))), 2, 1)
        copyto!(tns, dense)
        return tns
    elseif fmt == "CVEC" && N == 1
        tns = Tensor(SparseList(Element(fill_val)))
        copyto!(tns, dense)
        return tns
    elseif fmt == "DVEC" && N == 1
        tns = Tensor(Dense(Element(fill_val)))
        copyto!(tns, dense)
        return tns
    elseif (fmt == "DMAT" || fmt == "DMATR") && N == 2
        tns = Tensor(Dense(Dense(Element(fill_val))))
        copyto!(tns, dense)
        return tns
    elseif fmt == "DMATC" && N == 2
        tns = swizzle(Tensor(Dense(Dense(Element(fill_val)))), 2, 1)
        copyto!(tns, dense)
        return tns
    elseif fmt == "DCSR" && N == 2
        tns = Tensor(SparseList(SparseList(Element(fill_val))))
        copyto!(tns, dense)
        return tns
    elseif fmt == "DCSC" && N == 2
        tns = swizzle(Tensor(SparseList(SparseList(Element(fill_val)))), 2, 1)
        copyto!(tns, dense)
        return tns
    else
        tns = Tensor(_build_dense_levels(T, fill_val, size(dense)...))
        copyto!(tns, dense)
        return tns
    end
end

# Build nested Dense(Dense(...Element)) levels for an N-D tensor.
function _build_dense_levels(T, fill_val, dims...)
    if isempty(dims)
        return Element{fill_val,T,Int}(T[])
    end
    inner = _build_dense_levels(T, fill_val, dims[2:end]...)
    return DenseLevel(inner, dims[1])
end

# ─── binsparse_to_binsparse ──────────────────────────────────────────
# Read a Binsparse file with Finch, then write it back out.
# The roundtrip forces Finch to fully materialise the tensor through
# its own internal representation.

function cmd_binsparse_to_binsparse(args)
    length(args) == 2 ||
        error("binsparse_to_binsparse requires 2 args: tensor_in tensor_out")
    tensor_in, tensor_out = args

    tns = h5open(tensor_in, "r") do io
        Finch.bspread(io)
    end

    h5open(tensor_out, "w") do io
        Finch.bspwrite(io, tns)
    end

    return nothing
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

server_main()
