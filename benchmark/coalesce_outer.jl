# Run just this case (without downloading the matrix datasets used by the full suite):
# julia --project=benchmark --threads=8 benchmark/coalesce_outer.jl [results.json]
using BenchmarkTools
using Finch
using LinearAlgebra
using Printf
using Random

function coalesce_outer(A, BT, device)
    C = Tensor(Coalesce(device, SparseList(SparseList(Element(0.0)))))
    @finch begin
        C .= 0
        for k in parallel(_, device), j in _, i in _
            C[i, j] += A[i, k] * BT[j, k]
        end
    end
    return C
end

function coalesce_outer_benchmarks()
    m, k, n = 1024, 8, 1024
    density = 0.25
    seed = 42
    rng = MersenneTwister(seed)

    # Small integer-valued Float64 inputs make the reference comparison exact,
    # including when parallel reduction changes the order of additions.
    a = Float64.(rand(rng, 1:9, m, k)) .* (rand(rng, m, k) .< density)
    bt = Float64.(rand(rng, 1:9, n, k)) .* (rand(rng, n, k) .< density)
    A = Tensor(Dense(SparseList(Element(0.0))), a)
    BT = Tensor(Dense(SparseList(Element(0.0))), bt)

    # One logical task per outer product keeps each shard's SparseList writes
    # ordered. The overlapping shards must be summed and coalesced afterward.
    # Keep this task count fixed even when benchmarking with fewer CPU threads.
    device = cpu(:k, k)

    # Input conversion, reference multiplication, validation, and compilation
    # are excluded from timing. Output allocation, normalization, and merging
    # are included on every evaluation.
    Random.seed!(seed)
    result = coalesce_outer(A, BT, device)
    @assert Array(result) == a * transpose(bt)

    suite = BenchmarkGroup()
    suite["outer_1024x8x1024_density0.25"] = @benchmarkable begin
        coalesce_outer($A, $BT, $device)
    end setup = (Random.seed!($seed)) evals = 1 samples = 100 seconds = 5
    return suite
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Julia ", VERSION, "; CPU threads: ", Threads.nthreads(), "; logical tasks: 8")
    println("Finch source: ", pathof(Finch))
    suite = coalesce_outer_benchmarks()
    results = run(suite; verbose=true)
    for (name, trial) in results
        best = minimum(trial)
        med = median(trial)
        @printf(
            "%s: min=%.3f ms, median=%.3f ms, memory=%d bytes, allocs=%d, samples=%d\n",
            name, best.time / 1e6, med.time / 1e6, best.memory, best.allocs, length(trial),
        )
    end
    if !isempty(ARGS)
        BenchmarkTools.save(only(ARGS), results)
    end
end
