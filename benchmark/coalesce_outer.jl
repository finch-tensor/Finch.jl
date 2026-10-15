# Run just this case (without downloading the matrix datasets used by the full suite):
# julia --project=benchmark --threads=8 benchmark/coalesce_outer.jl [results.json]
using BenchmarkTools
using Finch
using LinearAlgebra
using Printf
using Random

# Output formats for the coalesced product. Formats a Finch version cannot build
# or merge correctly are skipped, so older versions still run the rest.
const COALESCE_OUTER_FORMATS = [
    "SparseList(SparseList)" => () -> SparseList(SparseList(Element(0.0))),
    "Dense(SparseHash)" => () -> Dense(SparseHash(Element(0.0))),
    "SparseHash(SparseHash)" => () -> SparseHash(SparseHash(Element(0.0))),
    "SparseHash(Dense)" => () -> SparseHash(Dense(Element(0.0))),
]

function coalesce_outer(fmt, A, BT, device)
    C = Tensor(Coalesce(device, fmt()))
    @finch begin
        C .= 0
        for k in parallel(_, device), j in _, i in _
            C[i, j] += A[i, k] * BT[j, k]
        end
    end
    return C
end

function coalesce_outer_benchmarks(; sizes=((1024, 8, 1024), (4096, 8, 4096)))
    density = 0.25
    seed = 42
    suite = BenchmarkGroup()
    for (m, k, n) in sizes
        rng = MersenneTwister(seed)
        # Small integer-valued Float64 inputs make the reference comparison exact,
        # including when parallel reduction changes the order of additions.
        a = Float64.(rand(rng, 1:9, m, k)) .* (rand(rng, m, k) .< density)
        bt = Float64.(rand(rng, 1:9, n, k)) .* (rand(rng, n, k) .< density)
        A = Tensor(Dense(SparseList(Element(0.0))), a)
        BT = Tensor(Dense(SparseList(Element(0.0))), bt)
        expected = a * transpose(bt)

        # One logical task per outer product keeps each shard's writes ordered.
        # The overlapping shards must be summed and coalesced afterward. Keep
        # this task count fixed even when benchmarking with fewer CPU threads.
        device = cpu(:k, k)

        for (name, fmt) in COALESCE_OUTER_FORMATS
            # Input conversion, reference multiplication, validation, and
            # compilation are excluded from timing. Output allocation,
            # normalization, and merging are included on every evaluation.
            Random.seed!(seed)
            supported = try
                Array(coalesce_outer(fmt, A, BT, device)) == expected
            catch
                false
            end
            supported || continue
            suite["$(m)x$(k)x$(n)_density$(density)"][name] = @benchmarkable begin
                coalesce_outer($fmt, $A, $BT, $device)
            end setup = (Random.seed!($seed)) evals = 1 samples = 100 seconds = 5
        end
    end
    return suite
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Julia ", VERSION, "; CPU threads: ", Threads.nthreads(), "; logical tasks: 8")
    println("Finch source: ", pathof(Finch))
    suite = coalesce_outer_benchmarks()
    results = run(suite; verbose=false)
    for (size, group) in sort(collect(results); by=first), (name, trial) in group
        best = minimum(trial)
        med = median(trial)
        @printf(
            "%-26s %-24s min=%9.3f ms  median=%9.3f ms  memory=%11d B  allocs=%7d  samples=%d\n",
            size, name, best.time / 1e6, med.time / 1e6, best.memory, best.allocs,
            length(trial),
        )
    end
    if !isempty(ARGS)
        BenchmarkTools.save(only(ARGS), results)
    end
end
