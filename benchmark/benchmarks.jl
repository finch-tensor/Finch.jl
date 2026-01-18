using Pkg
#tempdir = mktempdir()
#Pkg.activate(tempdir)
#Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..")))
#Pkg.add(["BenchmarkTools", "PkgBenchmark", "MatrixDepot"])
#Pkg.resolve()

using Finch
using BenchmarkTools
using MatrixDepot
using SparseArrays
using Random
using ArgParse

s = ArgParseSettings(
    "Run Finch.jl benchmarks. By default, all tests are run unless --include or --exclude options are provided. 
If the environment variable FINCH_BENCHMARK_ARGS is set, it will override the given arguments.",
)

@add_arg_table! s begin
    "--include", "-i"
    nargs = '*'
    default = []
    help = "list of benchmark suites to include, e.g., --include high-level structures"

    "--exclude", "-e"
    nargs = '*'
    default = []
    help = "list of benchmark suites to exclude, e.g., --exclude compile graphs"
end

if "FINCH_BENCHMARK_ARGS" in keys(ENV)
    ARGS = split(ENV["FINCH_BENCHMARK_ARGS"], " ")
end
parsed_args = parse_args([], s) # TODO https://github.com/carlobaldassi/ArgParse.jl/issues/135

include(joinpath(@__DIR__, "../docs/examples/bfs.jl"))
include(joinpath(@__DIR__, "../docs/examples/pagerank.jl"))
include(joinpath(@__DIR__, "../docs/examples/shortest_paths.jl"))
include(joinpath(@__DIR__, "../docs/examples/spgemm.jl"))
include(joinpath(@__DIR__, "../docs/examples/triangle_counting.jl"))

SUITE = BenchmarkGroup()

SUITE["high-level"] = BenchmarkGroup()

for (scheduler_name, scheduler) in [
    "default_scheduler" => Finch.default_scheduler(),
    "galley_scheduler" => Finch.galley_scheduler(),
]
    let
        A = Tensor(Dense(Sparse(Element(0.0))), fsprand(10000, 10000, 0.01))
        SUITE["high-level"]["permutedims(Dense(Sparse()))"][scheduler_name] = @benchmarkable(
            permutedims($A, (2, 1)), setup = (Finch.set_scheduler!($scheduler))
        )
    end

    let
        A = Tensor(Dense(Dense(Element(0.0))), rand(10000, 10000))
        SUITE["high-level"]["permutedims(Dense(Dense()))"][scheduler_name] = @benchmarkable(
            permutedims($A, (2, 1)), setup = (Finch.set_scheduler!($scheduler))
        )
    end

    let
        k = Ref(0.0)
        x = rand(1)
        y = rand(1)
        SUITE["high-level"]["einsum_spmv_compile_overhead"][scheduler_name] = @benchmarkable(
            begin
                A, x, y = (A, $x, $y)
                @einsum y[i] += A[i, j] * x[j]
            end,
            setup = (
                Finch.set_scheduler!($scheduler);
                A = Tensor(
                    Dense(SparseList(Element($k[] += 1))), fsprand(1, 1, 1)
                )
            )
        )
    end

    let
        N = 100000
        function generate_kernel_defs()
            for nnz1 in reverse([4, 4^2, 4^3, 4^4])
                for nnz2 in reverse([4, 4^2, 4^3, 4^4])
                    for nnz3 in reverse([4, 4^2, 4^3, 4^4])
                        A = lazy(fsprand(N, N, nnz1))
                        B = lazy(fsprand(N, N, nnz2))
                        C = lazy(fsprand(N, N, nnz3))
                        compute(A * B * C)
                    end
                end
            end
        end

        SUITE["high-level"]["matchain_adaptive_overhead"][scheduler_name] = @benchmarkable(
            begin
                compute(A * B * C)
            end,
            setup = begin
                Finch.set_scheduler!($scheduler)
                N = $N
                $generate_kernel_defs()
                A = lazy(fsprand(N, N, 4))
                B = lazy(fsprand(N, N, 4))
                C = lazy(fsprand(N, N, 4))
            end,
            evals = 1
        )
    end

    let
        A = Tensor(Dense(SparseList(Element(0.0))), fsprand(1, 1, 1))
        x = rand(1)
        y = rand(1)
        SUITE["high-level"]["einsum_spmv_call_overhead"][scheduler_name] = @benchmarkable(
            begin
                A, x, y = ($A, $x, $y)
                @einsum y[i] += A[i, j] * x[j]
            end,
            setup = (Finch.set_scheduler!($scheduler)),
            evals = 1
        )
    end

    let
        A = Tensor(Dense(SparseList(Element(0.0))), fsprand(1, 1, 1))
        x = rand(1)
        SUITE["high-level"]["compute_spmv_call_overhead"][scheduler_name] = @benchmarkable(
            begin
                A, x = (lazy($A), lazy($x))
                compute(A * x)
            end,
            setup = (Finch.set_scheduler!($scheduler)),
            evals = 1
        )
    end

    let
        N = 1_000
        K = 1_000
        p = 0.001
        A = Tensor(Dense(Dense(Element(0.0))), rand(N, K))
        B = Tensor(Dense(Dense(Element(0.0))), rand(K, N))
        M = Tensor(Dense(SparseList(Element(0.0))), fsprand(N, N, p))

        SUITE["high-level"]["sddmm_fused"][scheduler_name] = @benchmarkable(
            begin
                M = lazy($M)
                A = lazy($A)
                B = lazy($B)
                compute(M .* (A * B))
            end,
            setup = (Finch.set_scheduler!($scheduler)),
        )

        SUITE["high-level"]["sddmm_unfused"][scheduler_name] = @benchmarkable(
            begin
                M = $M
                A = $A
                B = $B
                M .* (A * B)
            end,
            setup = (Finch.set_scheduler!($scheduler)),
        )
    end
end

eval(
    let
        A = Tensor(Dense(SparseList(Element(0.0))), fsprand(1, 1, 1))
        x = rand(1)
        y = rand(1)
        @finch_kernel function spmv(y, A, x)
            for j in _, i in _
                y[i] += A[i, j] * x[j]
            end
        end
    end,
)

let
    A = Tensor(Dense(SparseList(Element(0.0))), fsprand(1, 1, 1))
    x = rand(1)
    y = rand(1)
    SUITE["high-level"]["einsum_spmv_baremetal"] = @benchmarkable(
        begin
            A, x, y = ($A, $x, $y)
            spmv(y, A, x)
        end,
        evals = 1000
    )
end

SUITE["compile"] = BenchmarkGroup()

code = """
using Finch
A = Tensor(Dense(SparseList(Element(0.0))))
B = Tensor(Dense(SparseList(Element(0.0))))
C = Tensor(Dense(SparseList(Element(0.0))))

@finch (C .= 0; for i=_, j=_, k=_; C[j, i] += A[k, i] * B[k, i] end)
"""
cmd = pipeline(
    `$(Base.julia_cmd()) --project=$(Base.active_project()) --eval $code`;
    stdout=IOBuffer(),
)

SUITE["compile"]["time_to_first_SpGeMM"] = @benchmarkable run(cmd)

let
    A = Tensor(Dense(SparseList(Element(0.0))))
    B = Tensor(Dense(SparseList(Element(0.0))))
    C = Tensor(Dense(SparseList(Element(0.0))))

    SUITE["compile"]["compile_SpGeMM"] = @benchmarkable begin
        A, B, C = ($A, $B, $C)
        Finch.execute_code(
            :ex,
            typeof(
                Finch.@finch_program_instance begin
                    C .= 0
                    for i in _, j in _, k in _
                        C[j, i] += A[k, i] * B[k, j]
                    end
                    return C
                end
            ),
        )
    end
end

let
    A = Tensor(SparseList(SparseList(Element(0.0))))
    c = Scalar(0.0)

    SUITE["compile"]["compile_pretty_triangle"] = @benchmarkable begin
        A, c = ($A, $c)
        @finch_code begin
            c .= 0
            for i in _, j in _, k in _
                c[] += A[i, j] * A[j, k] * A[i, k]
            end
            return c
        end
    end
end

SUITE["graphs"] = BenchmarkGroup()

SUITE["graphs"]["pagerank"] = BenchmarkGroup()
for mtx in ["SNAP/soc-Epinions1", "SNAP/soc-LiveJournal1"]
    SUITE["graphs"]["pagerank"][mtx] = @benchmarkable pagerank(
        $(pattern!(Tensor(SparseMatrixCSC(matrixdepot(mtx)))))
    )
end

SUITE["graphs"]["bfs"] = BenchmarkGroup()
for mtx in ["SNAP/soc-Epinions1", "SNAP/soc-LiveJournal1"]
    SUITE["graphs"]["bfs"][mtx] = @benchmarkable bfs(
        $(Tensor(SparseMatrixCSC(matrixdepot(mtx))))
    )
end

SUITE["graphs"]["bellmanford"] = BenchmarkGroup()
for mtx in ["Newman/netscience", "SNAP/roadNet-CA"]
    A = set_fill_value!(Tensor(SparseMatrixCSC(matrixdepot(mtx))), Inf)
    SUITE["graphs"]["bellmanford"][mtx] = @benchmarkable bellmanford($A)
end

SUITE["matrices"] = BenchmarkGroup()

SUITE["matrices"]["ATA_spgemm_inner"] = BenchmarkGroup()
for mtx in []#"SNAP/soc-Epinions1", "SNAP/soc-LiveJournal1"]
    A = Tensor(permutedims(SparseMatrixCSC(matrixdepot(mtx))))
    SUITE["matrices"]["ATA_spgemm_inner"][mtx] = @benchmarkable spgemm_inner($A, $A)
end

SUITE["matrices"]["ATA_spgemm_gustavson"] = BenchmarkGroup()
for mtx in ["SNAP/soc-Epinions1"]#], "SNAP/soc-LiveJournal1"]
    A = Tensor(SparseMatrixCSC(matrixdepot(mtx)))
    SUITE["matrices"]["ATA_spgemm_gustavson"][mtx] = @benchmarkable spgemm_gustavson($A, $A)
end

SUITE["matrices"]["ATA_spgemm_outer"] = BenchmarkGroup()
for mtx in ["SNAP/soc-Epinions1"]#, "SNAP/soc-LiveJournal1"]
    A = Tensor(SparseMatrixCSC(matrixdepot(mtx)))
    SUITE["matrices"]["ATA_spgemm_outer"][mtx] = @benchmarkable spgemm_outer($A, $A)
end

SUITE["indices"] = BenchmarkGroup()

function spmv32(A, x)
    y = Tensor(Dense{Int32}(Element{0.0,Float64,Int32}()))
    @finch begin
        y .= 0
        for i in _, j in _
            y[i] += A[j, i] * x[j]
        end
    end
    return y
end

SUITE["indices"]["SpMV_32"] = BenchmarkGroup()
for mtx in ["SNAP/soc-Epinions1"]#, "SNAP/soc-LiveJournal1"]
    A = SparseMatrixCSC(matrixdepot(mtx))
    A = Tensor(Dense{Int32}(SparseList{Int32}(Element{0.0,Float64,Int32}())), A)
    x = Tensor(Dense{Int32}(Element{0.0,Float64,Int32}()), rand(size(A)[2]))
    SUITE["indices"]["SpMV_32"][mtx] = @benchmarkable spmv32($A, $x)
end

function spmv_p1(A, x)
    y = Tensor(Dense(Element(0.0)))
    @finch begin
        y .= 0
        for i in _, j in _
            y[i] += A[j, i] * x[j]
        end
    end
    return y
end

SUITE["indices"]["SpMV_p1"] = BenchmarkGroup()
for mtx in ["SNAP/soc-Epinions1"]#, "SNAP/soc-LiveJournal1"]
    A = SparseMatrixCSC(matrixdepot(mtx))
    (m, n) = size(A)
    ptr = A.colptr .- 1
    idx = A.rowval .- 1
    A = Tensor(
        Dense(
            SparseList(
                Element(0.0, A.nzval), m, Finch.PlusOneVector(ptr), Finch.PlusOneVector(idx)
            ),
            n,
        ),
    )
    x = Tensor(Dense(Element(0.0)), rand(n))
    SUITE["indices"]["SpMV_p1"][mtx] = @benchmarkable spmv_p1($A, $x)
end

function spmv64(A, x)
    y = Tensor(Dense{Int64}(Element{0.0,Float64,Int64}()))
    @finch begin
        y .= 0
        for i in _, j in _
            y[i] += A[j, i] * x[j]
        end
    end
    return y
end

SUITE["indices"]["SpMV_64"] = BenchmarkGroup()
for mtx in ["SNAP/soc-Epinions1"]#, "SNAP/soc-LiveJournal1"]
    A = SparseMatrixCSC(matrixdepot(mtx))
    A = Tensor(Dense{Int64}(SparseList{Int64}(Element{0.0,Float64,Int64}())), A)
    x = Tensor(Dense{Int64}(Element{0.0,Float64,Int64}()), rand(size(A)[2]))
    SUITE["indices"]["SpMV_64"][mtx] = @benchmarkable spmv64($A, $x)
end

SUITE["parallel"] = BenchmarkGroup()

function spmv_serial(A, x)
    y = Tensor(Dense{Int64}(Element{0.0,Float64}()))
    @finch begin
        y .= 0
        for i in _
            for j in _
                y[i] += A[j, i] * x[j]
            end
        end
        return y
    end
end

function spmv_noinit(y, A, x)
    @finch begin
        for i in _
            for j in _
                y[i] += A[j, i] * x[j]
            end
        end
        return y
    end
end

function spmv_threaded(A, x)
    y = Tensor(Dense{Int64}(Element{0.0,Float64}()))
    @finch begin
        y .= 0
        for i in parallel(_)
            for j in _
                y[i] += A[j, i] * x[j]
            end
        end
        return y
    end
end

function spmv_atomic_element(A, x)
    y = Tensor(Dense{Int64}(AtomicElement{0.0,Float64}()))
    @finch begin
        y .= 0
        for j in parallel(_)
            for i in _
                y[i] += A[i, j] * x[j]
            end
        end
        return y
    end
end

function spmv_mutex(A, x)
    y = Tensor(Dense{Int64}(Mutex(Element{0.0,Float64}())))
    @finch begin
        y .= 0
        for j in parallel(_)
            for i in _
                y[i] += A[i, j] * x[j]
            end
        end
        return y
    end
end

SUITE["parallel"]["SpMV_serial"] = BenchmarkGroup()
SUITE["parallel"]["SpMV_threaded"] = BenchmarkGroup()
SUITE["parallel"]["SpMV_atomic_element"] = BenchmarkGroup()
SUITE["parallel"]["SpMV_mutex"] = BenchmarkGroup()
for (key, mtx) in [
    "SNAP/soc-Epinions1" => SparseMatrixCSC(matrixdepot("SNAP/soc-Epinions1")),
    "fsprand(10_000, 10_000, 0.01)" => fsprand(10_000, 10_000, 0.01)]
    A = Tensor(Dense{Int64}(SparseList{Int64}(Element{0.0,Float64,Int64}())), mtx)
    A_T = Tensor(
        Dense{Int64}(SparseList{Int64}(Element{0.0,Float64,Int64}())), permutedims(A)
    )
    x = Tensor(Dense{Int64}(Element{0.0,Float64,Int64}()), rand(size(A)[2]))
    SUITE["parallel"]["SpMV_serial"][key] = @benchmarkable spmv_serial($A_T, $x)
    SUITE["parallel"]["SpMV_threaded"][key] = @benchmarkable spmv_threaded($A_T, $x)
    SUITE["parallel"]["SpMV_atomic_element"][key] = @benchmarkable spmv_atomic_element(
        $A, $x
    )
    SUITE["parallel"]["SpMV_mutex"][key] = @benchmarkable spmv_mutex($A, $x)
end

SUITE["structure"] = BenchmarkGroup()

N = 1_000_000

SUITE["structure"]["permutation"] = BenchmarkGroup()

perm = randperm(N)

A_ref = Tensor(Dense(SparseList(Element(0.0))), fsparse(collect(1:N), perm, ones(N)))

A = Tensor(Dense(SparsePoint(Element(0.0))), A_ref)

x = rand(N)

SUITE["structure"]["permutation"]["SparseList"] = @benchmarkable spmv_serial($A_ref, $x)
SUITE["structure"]["permutation"]["SparsePoint"] = @benchmarkable spmv_serial($A, $x)
SUITE["structure"]["permutation"]["baseline"] = @benchmarkable $x[$perm]

SUITE["structure"]["banded"] = BenchmarkGroup()

A_ref = Tensor(Dense(Sparse(Element(0.0))), N, N)

@finch for j in _, i in _
    if j - 2 < i < j + 2
        A_ref[i, j] = 1.0
    end
end

A = Tensor(Dense(SparseBand(Element(0.0))), A_ref)
A2 = Tensor(Dense(SparseRunList(Element(0.0))), A_ref)
A2 = Tensor(Dense(SparseInterval(Element(0.0))), A2)

x = rand(N)

SUITE["structure"]["banded"]["SparseList"] = @benchmarkable spmv_serial($A_ref, $x)
SUITE["structure"]["banded"]["SparseBand"] = @benchmarkable spmv_serial($A, $x)
SUITE["structure"]["banded"]["SparseInterval"] = @benchmarkable spmv_serial($A2, $x)

if !isempty(parsed_args["include"])
    inc = reduce((a, b) -> :($a || $b), parsed_args["include"])
    SUITE = eval(:(SUITE[@tagged $inc]))
end
if !isempty(parsed_args["exclude"])
    exc = reduce((a, b) -> :($a || $b), parsed_args["exclude"])
    SUITE = eval(:(SUITE[@tagged !$exc]))
end
