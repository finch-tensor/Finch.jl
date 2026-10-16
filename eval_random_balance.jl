using Galley, Random, Statistics

CAPTURED = Vector{Tuple}(undef, 0)

import Galley: balance, decrement_idxs, MergeRandom, build_sampler, sample

@inbounds function Galley.balance(
    sampler::Vector{NTuple{m,Int}}, tid, P, shapes, style::MergeRandom
) where {m}
    neg = ntuple(_ -> -1, m)
    start = searchsortedlast(sampler, neg; by=reverse) + 1
    n = length(sampler) - start + 1
    base = div(n, P)
    remainder = n % P
    function lb_at(t)
        t == 1 ? ntuple(_ -> 1, m) : sampler[start + (t - 1) * base + min(t - 1, remainder)]
    end
    lb = Tuple(lb_at(tid))
    ub = tid == P ? Tuple(shapes) : Tuple(decrement_idxs(collect(lb_at(tid + 1)), shapes))
    CAPTURED[tid] = (lb, ub)
    return (lb, ub)
end

const RATE = Ref(1000)
function Galley.build_sampler(lvl::Galley.AbstractLevel, P, nnz, tsize)
    elvl = lvl
    while !(elvl isa Galley.ElementLevel)
        elvl = elvl.lvl
    end
    sampler = Vector{NTuple{tsize,Int}}(undef, 0)
    for p in 1:P
        active = round(Int, RATE[] * P * length(elvl.val.data[p]) / nnz)
        for _ in 1:active
            push!(sampler, sample(p, lvl))
        end
    end
    sort!(sampler; by=reverse)
    sampler
end

function collect_tuples(outer)
    inner = outer.lvl
    tuples = NTuple{2,Int}[]
    for q in 1:length(outer.idx)
        j = outer.idx[q]
        for r in inner.ptr[q]:(inner.ptr[q + 1] - 1)
            push!(tuples, (inner.idx[r], j))
        end
    end
    tuples
end

contained(lb, ub, t) = reverse(lb) <= reverse(t) <= reverse(ub)

function run_once(n, m, P, density_fn, seed)
    Random.seed!(seed)
    ncpu = cpu(:t, P)
    tens = Tensor(Coalesce(ncpu, SparseList(SparseList(Element(0.0)))), n, m)

    dense_acc = zeros(n, m)
    for j in 1:m, i in 1:n
        if rand() < density_fn(j)
            dense_acc[i, j] = round(rand() * 10; digits=2)
        end
    end
    acc = Tensor(SparseList(SparseList(Element(0.0))), dense_acc)

    cols_per_task = m ÷ P
    shard_nnz = [
        count(
            !=(0.0), @view dense_acc[:, ((p - 1) * cols_per_task + 1):(p * cols_per_task)]
        ) for p in 1:P
    ]

    global CAPTURED = Vector{Tuple}(undef, P)
    @finch begin
        tens .= 0
        for j in parallel(_, ncpu)
            for i in _
                tens[i, j] = acc[i, j] + acc[i, j]
            end
        end
    end

    total_nnz = count(!=(0.0), dense_acc)
    tuples = collect_tuples(tens.lvl.coalescent)
    @assert length(tuples) == total_nnz "ground truth extraction mismatch: got $(length(tuples)), expected $total_nnz"

    counts = zeros(Int, P)
    for tid in 1:P
        lb, ub = CAPTURED[tid]
        counts[tid] = count(t -> contained(lb, ub, t), tuples)
    end
    @assert sum(counts) == total_nnz "partition does not cover ground truth: sum=$(sum(counts)) vs total=$total_nnz"

    ideal = total_nnz / P
    fairness = sum(counts)^2 / (P * sum(abs2, counts)) ##Jain's fairness index, 1.0 = perfect
    efficiency = ideal / maximum(counts)
    (efficiency=efficiency, fairness=fairness, min_shard=minimum(shard_nnz))
end

function eval_case(name, n, m, P, density_fn; nseeds=6)
    results = [run_once(n, m, P, density_fn, seed) for seed in 1:nseeds]
    effs = [r.efficiency for r in results]
    fairs = [r.fairness for r in results]
    min_shard = minimum(r.min_shard for r in results)
    println("=== $name (n=$n, m=$m, P=$P), rate=$(RATE[]) ===")
    println(
        "  efficiency: ", round.(effs, digits=3), "  mean=", round(mean(effs); digits=3)
    )
    println(
        "  Jain:       ", round.(fairs, digits=3), "  mean=", round(mean(fairs); digits=3)
    )
    println()
    (name=name, mean_eff=mean(effs), mean_fair=mean(fairs))
end

const SCENARIOS = [
    ##name, n, m, P, density_fn
    ("uniform density, P=8", 8000, 80, 8, j -> 0.5),
    ("linear ramp, P=10", 8000, 100, 10, j -> 0.05 + 0.9 * (j - 1) / 99),
    ("linear ramp, P=16 (max scale)", 8000, 160, 16, j -> 0.05 + 0.9 * (j - 1) / 159),
    ("linear ramp, P=32", 8000, 320, 32, j -> 0.05 + 0.9 * (j - 1) / 319),
    ("single hotspot column, P=8", 24000, 40, 8, j -> j == 20 ? 1.0 : 0.05),
    ("single hotspot column, P=16 (max scale)", 24000, 80, 16, j -> j == 40 ? 1.0 : 0.05),
]

function go(rate)
    RATE[] = rate
    [eval_case(name, n, m, P, dfn) for (name, n, m, P, dfn) in SCENARIOS]
end

go(1000)
