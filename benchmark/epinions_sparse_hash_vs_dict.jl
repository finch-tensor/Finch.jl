#!/usr/bin/env julia

using BenchmarkTools
using Finch
using MatrixDepot
using Printf
using SparseArrays
using Statistics

const MATRIX_NAME = "SNAP/soc-Epinions1"

function human_bytes(n)
    units = ("B", "KiB", "MiB", "GiB")
    x = Float64(n)
    unit = first(units)
    for u in units
        unit = u
        x < 1024 && break
        u == last(units) && break
        x /= 1024
    end
    return @sprintf("%.2f %s", x, unit)
end

function human_time(ns)
    ns < 1_000 && return @sprintf("%.2f ns", ns)
    ns < 1_000_000 && return @sprintf("%.2f us", ns / 1_000)
    ns < 1_000_000_000 && return @sprintf("%.2f ms", ns / 1_000_000)
    return @sprintf("%.2f s", ns / 1_000_000_000)
end

function level_parts(::Type{SparseDictLevel}, lvl)
    return (
        "ptr" => lvl.ptr,
        "idx" => lvl.idx,
        "val" => lvl.val,
        "tbl" => lvl.tbl,
        "pool" => lvl.pool,
    )
end

function level_parts(::Type{SparseHashLevel}, lvl)
    return (
        "ptr" => lvl.ptr,
        "tbl_pos" => lvl.tbl_pos,
        "tbl_idx" => lvl.tbl_idx,
        "tbl_ctrl" => lvl.tbl_ctrl,
        "tbl_val" => lvl.tbl_val,
        "perm" => lvl.perm,
    )
end

function collect_level_parts!(rows, ::Type{Format}, lvl, name) where {Format}
    if lvl isa Format
        for (part, xs) in level_parts(Format, lvl)
            push!(rows, (name, part, length(xs), Base.summarysize(xs)))
        end
    end
    if hasproperty(lvl, :lvl)
        collect_level_parts!(rows, Format, lvl.lvl, "$name.lvl")
    end
    return rows
end

function print_storage(::Type{Format}, matrix) where {Format}
    tensor = Tensor(Format(Format(Element(0.0))), matrix)
    rows = collect_level_parts!(Tuple{String,String,Int,Int}[], Format, tensor.lvl, "lvl")
    total = Base.summarysize(tensor)
    @printf("%-10s tensor summarysize: %s\n", string(Format), human_bytes(total))
    for (name, part, len, bytes) in rows
        @printf(
            "  %-7s %-7s length=%10d  size=%s\n",
            name,
            part,
            len,
            human_bytes(bytes),
        )
    end
    return tensor
end

function spgemm_outer_with(::Type{Format}, A, B) where {Format}
    z = fill_value(A) * fill_value(B) + false
    C = Tensor(Dense(SparseList(Element(z))))
    w = Tensor(Format(Format(Element(z))))
    BT = Tensor(Dense(SparseList(Element(z))))
    @finch mode = :fast begin
        w .= 0
        for j in _, k in _
            w[j, k] = B[k, j]
        end
    end
    @finch begin
        BT .= 0
        for k in _, j in _
            BT[j, k] = w[j, k]
        end
    end
    @finch begin
        w .= 0
        for k in _, j in _, i in _
            w[i, j] += A[i, k] * BT[j, k]
        end
    end
    @finch begin
        C .= 0
        for j in _, i in _
            C[i, j] = w[i, j]
        end
    end
    return C
end

function run_case(::Type{Format}, A) where {Format}
    println()
    println("== $(Format) ==")
    GC.gc()
    warm = spgemm_outer_with(Format, A, A)
    @printf("warmup result: %s, stored=%d\n", summary(warm), countstored(warm))
    GC.gc()
    bench = @benchmarkable spgemm_outer_with($Format, $A, $A) evals = 1
    trial = run(bench)
    best = minimum(trial)
    med_time = median(trial.times)
    @printf(
        "best:   %s, memory=%s, allocs=%d\n",
        human_time(best.time),
        human_bytes(best.memory),
        best.allocs,
    )
    @printf("median: %s across %d sample(s)\n", human_time(med_time), length(trial.times))
    return trial
end

function main()
    println("SparseHash vs SparseDict on $(MATRIX_NAME)")
    println("BenchmarkTools default samples and seconds, evals=1")
    matrix = SparseMatrixCSC(matrixdepot(MATRIX_NAME))
    @printf(
        "matrix: %d x %d, nnz=%d, density=%.6g\n",
        size(matrix, 1),
        size(matrix, 2),
        nnz(matrix),
        nnz(matrix) / prod(size(matrix)),
    )

    println()
    println("Storage for a direct 2-level tensor loaded from Epinions:")
    print_storage(SparseDictLevel, matrix)
    print_storage(SparseHashLevel, matrix)

    A = Tensor(matrix)
    dict_trial = run_case(SparseDictLevel, A)
    hash_trial = run_case(SparseHashLevel, A)

    dict_best = minimum(dict_trial).time
    hash_best = minimum(hash_trial).time
    @printf(
        "\nSparseHash best / SparseDict best: %.3fx\n",
        hash_best / dict_best,
    )
end

main()
