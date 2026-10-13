# Serial SparseHash workloads, without Coalesce:
# julia --project=benchmark benchmark/sparse_hash.jl [results.json]
using BenchmarkTools
using Finch
using Printf
using Random

const SPARSE_HASH_FORMATS = [
    "Dense(SparseHash)" => () -> Dense(SparseHash(Element(0.0))),
    "SparseHash(SparseHash)" => () -> SparseHash(SparseHash(Element(0.0))),
]

# Outer-product matmul: most updates hit existing keys.
function sparse_hash_accumulate(fmt, A, BT)
    w = Tensor(fmt())
    @finch begin
        w .= 0
        for k in _, j in _, i in _
            w[i, j] += A[i, k] * BT[j, k]
        end
    end
    return w
end

# Iterate every stored entry in order.
function sparse_hash_iterate(w)
    C = Tensor(Dense(SparseList(Element(0.0))))
    @finch begin
        C .= 0
        for j in _, i in _
            C[i, j] = w[i, j]
        end
    end
    return C
end

# Random access: probe the table once per entry of a sparse mask.
function sparse_hash_lookup(w, X)
    s = Scalar(0.0)
    @finch begin
        s .= 0
        for j in _, i in _
            s[] += X[i, j] * w[follow(i), follow(j)]
        end
    end
    return s[]
end

function sparse_hash_benchmarks(; m=4096, k=8, n=4096, density=0.25, seed=42)
    rng = MersenneTwister(seed)
    a = Float64.(rand(rng, 1:9, m, k)) .* (rand(rng, m, k) .< density)
    bt = Float64.(rand(rng, 1:9, n, k)) .* (rand(rng, n, k) .< density)
    x = Float64.(rand(rng, m, n) .< 0.05)
    A = Tensor(Dense(SparseList(Element(0.0))), a)
    BT = Tensor(Dense(SparseList(Element(0.0))), bt)
    X = Tensor(Dense(SparseList(Element(0.0))), x)
    expected = a * transpose(bt)

    suite = BenchmarkGroup()
    for (name, fmt) in SPARSE_HASH_FORMATS
        # Compile and validate outside timing.
        w = sparse_hash_accumulate(fmt, A, BT)
        @assert Array(w) == expected
        @assert Array(sparse_hash_iterate(w)) == expected
        @assert sparse_hash_lookup(w, X) == sum(x .* expected)
        group = suite["$(m)x$(k)x$(n)_density$(density)"][name]
        group["accumulate"] = @benchmarkable(
            sparse_hash_accumulate($fmt, $A, $BT), evals = 1, samples = 20, seconds = 10
        )
        group["iterate"] = @benchmarkable(
            sparse_hash_iterate($w), evals = 1, samples = 20, seconds = 10
        )
        group["lookup"] = @benchmarkable(
            sparse_hash_lookup($w, $X), evals = 1, samples = 20, seconds = 10
        )
    end
    return suite
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Julia ", VERSION, "; CPU threads: ", Threads.nthreads())
    println("Finch source: ", pathof(Finch))
    results = run(sparse_hash_benchmarks(); verbose=false)
    for (size, formats) in results, (fmt, group) in sort(collect(formats); by=first),
        (phase, trial) in sort(collect(group); by=first)

        @printf(
            "%-24s %-38s %-10s min=%9.3f ms  median=%9.3f ms  allocs=%d\n",
            size, fmt, phase, minimum(trial).time / 1e6, median(trial).time / 1e6,
            minimum(trial).allocs,
        )
    end
    if !isempty(ARGS)
        BenchmarkTools.save(only(ARGS), results)
    end
end
