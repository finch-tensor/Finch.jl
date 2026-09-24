# SPDX-FileCopyrightText: 2026 Finch Developers
# SPDX-License-Identifier: MIT

# Quick checks of the compliance server's converters, without the Python harness.

include("finch_server.jl")

element = Dict("level_desc" => "element")
dense(lvl, rank=1) = Dict("level_desc" => "dense", "rank" => rank, "level" => lvl)
sparse(lvl, rank=1) = Dict("level_desc" => "sparse", "rank" => rank, "level" => lvl)

pattern_3d = falses(2, 3, 2)
pattern_3d[[1, 4, 8, 11]] .= true
values_3d = ifelse.(pattern_3d, reshape(Int64.(1:12), 2, 3, 2), Int64(7))
values_3d[8] = 7 # a stored entry equal to the fill value

cases = [
    # (values, pattern, fill value, format, custom, values data type)
    (fill(true), fill(true), false, "custom", Dict("level" => element), "bint8"),
    ([1 0 3; 0 5 0], [true false true; false true true], 0, "CSR", nothing, "int64"),
    ([1 0 3; 0 5 0], trues(2, 3), 0, "DMAT", nothing, "int64"),
    ([1 0 3; 0 5 0], [true false true; false true true], 0, "COO", nothing, "int64"),
    (fill(2.0 + 3.0im, 4), trues(4), -1.0 + 2.0im, "custom",
        Dict("level" => dense(element)), "iso[complex[float64]]"),
    ([1 2 3; 4 5 6], trues(2, 3), 0, "custom",
        Dict("transpose" => [0, 1], "level" => dense(element, 2)), "int64"),
    (values_3d, pattern_3d, Int64(7), "custom",
        Dict("transpose" => [2, 0, 1], "level" => sparse(dense(sparse(element)))), "int64"),
    (values_3d, pattern_3d, Int64(7), "custom",
        Dict("transpose" => [1, 2, 0], "level" => sparse(element, 3)), "int64"),
]

mktempdir() do dir
    path(name) = joinpath(dir, name)
    for (values, pat, fill_value, format, custom, dtype) in cases
        @testset "converters $format $(size(values)) $dtype" begin
            header = Dict(
                "format" => format,
                "shape" => collect(size(values)),
                "data_types" => Dict("values" => dtype),
            )
            custom === nothing || (header["custom"] = custom)
            npzwrite(path("values.npy"), values)
            npzwrite(path("pattern.npy"), pat)
            npzwrite(path("fill.npy"), fill(fill_value))
            write(path("header.json"), JSON.json(header))

            cmd_npy_to_binsparse([
                path("values.npy"), path("pattern.npy"), path("fill.npy"),
                path("header.json"), path("output.h5"),
            ])
            cmd_binsparse_to_npy([
                path("output.h5"), path("values_out.npy"), path("pattern_out.npy"),
                path("fill_out.npy"),
            ])
            cmd_binsparse_to_binsparse([path("output.h5"), path("roundtrip.h5")])

            @test read_array(path("values_out.npy")) == values
            @test read_array(path("pattern_out.npy")) == pat
            @test npzread(path("fill_out.npy")) == fill_value

            output, roundtrip = map(("output.h5", "roundtrip.h5")) do name
                h5open(path(name), "r") do io
                    desc = Finch.bspread_header(io)["binsparse"]
                    desc, Dict(key => read(io[key]) for key in keys(desc["data_types"]))
                end
            end
            desc = output[1]
            @test desc["format"] == format
            @test get(desc, "custom", nothing) == custom
            @test desc["number_of_stored_values"] == count(pat)
            @test desc["data_types"]["values"] == dtype
            @test roundtrip[1]["format"] == get(Dict("DMAT" => "DMATR", "COO" => "COOR"), format, format)
            roundtrip[1]["format"] = format
            @test roundtrip == output
        end
    end
end
