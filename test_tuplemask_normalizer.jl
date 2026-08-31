using Finch
using Random

include(joinpath(@__DIR__, "normalizer.jl"))

# ---------------------------------------------------------------------------
# Question: given the (lb, ub) tuples produced by the normalizer's balance()
# for a loop nest `for j in _; for i in _; ...A[i,j]...`, can those bounds be
# fed directly into `tuplemask(lb, ub)` to get a deterministic, correct
# lexicographic membership test for (i,j)?
#
# balance() returns tuples in the SAME convention as level_size/countstored_level:
# idxs[1] = inner dim (i), idxs[end] = outer dim (j), i.e. plain array-index
# order matching A[i,j]. TupleMask reverses its lb/ub internally and compares
# most-significant-last, so tuplemask's dim=1 (post-reverse) is the ORIGINAL
# last tuple entry -- which is exactly the outermost loop variable (j) in
# Finch's iteration order. So the conventions match up *by construction* --
# this test verifies that claim empirically, for both
# Dense(SparseList(Element)) and SparseList(SparseList(Element)).
# ---------------------------------------------------------------------------

key(c) = (c[2], c[1])  # (j, i) lexicographic order, j primary (outer/most significant)

# expected partition membership by brute-force rank comparison, matching the
# same ordering used throughout test_normalizer(_dense).jl
function expected_member(i, j, lb, ub)
    return key(lb) <= (j, i) <= key(ub)
end

# countstored_level SUMS each level's per-processor contribution across all P
# shards, so replicating the same data into every shard would multiply the
# apparent nnz by P. Instead, put the real (non-sharded) Tensor's data in
# shard 1 only, and give every other shard an empty (but structurally valid)
# copy -- i.e. "one worker holds the whole merged structure and everyone
# (including it) needs to figure out its own [lb,ub] slice of the global
# nnz," matching how countstored_level's per-processor sum is meant to behave
# for a real, disjoint sharding.
function wrap_solo_idx(v::Vector{Int}, P)
    device = Finch.CPU{:test}(P)
    data = [p == 1 ? copy(v) : Int[] for p in 1:P]
    Finch.CPULocalArray{Vector{Int},typeof(device)}(device, data)
end

function wrap_solo_ptr(v::Vector{Int}, P)
    device = Finch.CPU{:test}(P)
    data = [p == 1 ? copy(v) : ones(Int, length(v)) for p in 1:P]
    Finch.CPULocalArray{Vector{Int},typeof(device)}(device, data)
end

function wrap_dense_sparse(t, P)
    inner = t.lvl.lvl
    ptr = wrap_solo_ptr(inner.ptr, P)
    idx = wrap_solo_idx(inner.idx, P)
    elem = Finch.ElementLevel(0.0)
    lvl1 = Finch.SparseListLevel{Int}(elem, inner.shape, ptr, idx)
    lvl2 = Finch.DenseLevel{Int}(lvl1, t.lvl.shape)
    return lvl2
end

function wrap_sparse_sparse(t, P)
    outer = t.lvl
    inner = outer.lvl
    ptr1 = wrap_solo_ptr(inner.ptr, P)
    idx1 = wrap_solo_idx(inner.idx, P)
    ptr2 = wrap_solo_ptr(outer.ptr, P)
    idx2 = wrap_solo_idx(outer.idx, P)
    elem = Finch.ElementLevel(0.0)
    lvl1 = Finch.SparseListLevel{Int}(elem, inner.shape, ptr1, idx1)
    lvl2 = Finch.SparseListLevel{Int}(lvl1, outer.shape, ptr2, idx2)
    return lvl2
end

# ---------------------------------------------------------------------------
# 1. Direct membership test: materialize the FULL (i,j) boolean grid via
#    tuplemask in an actual @finch loop, and compare it against the
#    brute-force rank definition, for every processor's partition.
# ---------------------------------------------------------------------------
function test_full_grid_membership(name, lvl_wrapped, nnz, S1, S2, P)
    issues = String[]

    parts = NamedTuple[]
    for tid in 1:P
        lb, ub = balance(lvl_wrapped, tid, P, nnz, MergeNormalization())
        push!(parts, (tid=tid, lb=lb, ub=ub))
    end

    for part in parts
        m = Finch.tuplemask(part.lb, part.ub)
        M = Tensor(Dense(Dense(Element(false))), S1, S2)
        @finch begin
            M .= false
            for j in _
                for i in _
                    M[i, j] = m[i, j]
                end
            end
        end
        got = Array(M)

        for j in 1:S2, i in 1:S1
            want = expected_member(i, j, part.lb, part.ub)
            if got[i, j] != want
                push!(
                    issues,
                    "[$name] tid=$(part.tid) (lb=$(part.lb),ub=$(part.ub)) at (i=$i,j=$j): tuplemask=$(got[i, j]) expected=$want",
                )
            end
        end
    end

    # every (i,j) in the full grid must be claimed by EXACTLY one partition
    for j in 1:S2, i in 1:S1
        owners = [part.tid for part in parts if expected_member(i, j, part.lb, part.ub)]
        if length(owners) != 1
            push!(issues, "[$name] (i=$i,j=$j) claimed by $(length(owners)) partitions: $owners")
        end
    end

    return issues
end

# ---------------------------------------------------------------------------
# 2. End-to-end test: use tuplemask as a filter combined with a REAL Finch
#    tensor (not just a synthetic grid) inside a compiled @finch reduction,
#    and check the partial sums add up to the correct total and match a
#    ground truth computed independently from the dense array.
# ---------------------------------------------------------------------------
function test_real_reduction(name, t, lvl_wrapped, nnz, S1, S2, P)
    issues = String[]
    Amat = Array(t)

    parts = NamedTuple[]
    for tid in 1:P
        lb, ub = balance(lvl_wrapped, tid, P, nnz, MergeNormalization())
        push!(parts, (tid=tid, lb=lb, ub=ub))
    end

    total_check = 0.0
    for part in parts
        m = Finch.tuplemask(part.lb, part.ub)
        y = Scalar(0.0)
        @finch begin
            y .= 0
            for j in _
                for i in _
                    if m[i, j]
                        y[] += t[i, j]
                    end
                end
            end
        end

        want = sum(
            Amat[i, j] for j in 1:S2, i in 1:S1 if expected_member(i, j, part.lb, part.ub)
        )
        if !isapprox(y(), want)
            push!(issues, "[$name] tid=$(part.tid): masked reduction $(y()) != expected $want")
        end
        total_check += y()
    end

    full = sum(Amat)
    if !isapprox(total_check, full)
        push!(issues, "[$name] sum of partitioned reductions $total_check != full sum $full")
    end

    return issues
end

function run_case(name, build_tensor, wrap_lvl, S1, S2, nnz_target, P; seed)
    Random.seed!(seed)
    A = zeros(S1, S2)
    universe = [(i, j) for i in 1:S1 for j in 1:S2]
    coords = Random.shuffle(universe)[1:nnz_target]
    for (i, j) in coords
        A[i, j] = rand(1.0:100.0)
    end

    t = build_tensor(A)
    lvl_wrapped = wrap_lvl(t, P)
    nnz = length(coords)

    issues = String[]
    append!(issues, test_full_grid_membership(name, lvl_wrapped, nnz, S1, S2, P))
    append!(issues, test_real_reduction(name, t, lvl_wrapped, nnz, S1, S2, P))
    return issues
end

trials = [
    (S1=6, S2=5, nnz=20, P=3),
    (S1=6, S2=5, nnz=20, P=4),
    (S1=10, S2=10, nnz=60, P=5),
    (S1=10, S2=10, nnz=60, P=8),
    (S1=20, S2=3, nnz=40, P=6),
    (S1=3, S2=20, nnz=40, P=6),
    (S1=8, S2=8, nnz=63, P=8),  # near-full, tight packing -- may hit the
                                 # pre-existing known edge case documented at
                                 # the top of normalizer.jl (unrelated to
                                 # tuplemask); handled separately below.
]

const KNOWN_BALANCE_BUG = "nnz too small to load balance across P processors"

cases = [
    ("Dense(SparseList)", A -> Tensor(Dense(SparseList(Element(0.0))), A), wrap_dense_sparse),
    ("SparseList(SparseList)", A -> Tensor(SparseList(SparseList(Element(0.0))), A), wrap_sparse_sparse),
]

total_issues = 0
total_errors = 0
total_known_bug = 0
for (name, build_tensor, wrap_lvl) in cases
    for (idx, tr) in enumerate(trials)
        for seed in 1:5
            try
                issues = run_case(
                    name, build_tensor, wrap_lvl, tr.S1, tr.S2, tr.nnz, tr.P;
                    seed=seed * 1000 + idx,
                )
                if isempty(issues)
                    println("[$name] TRIAL $tr seed=$seed: OK")
                else
                    global total_issues += length(issues)
                    println("[$name] TRIAL $tr seed=$seed: ISSUES")
                    for iss in issues
                        println("  $iss")
                    end
                end
            catch e
                if sprint(showerror, e) == KNOWN_BALANCE_BUG
                    global total_known_bug += 1
                    println("[$name] TRIAL $tr seed=$seed: SKIPPED (pre-existing normalizer.jl edge case, not a tuplemask issue)")
                else
                    global total_errors += 1
                    println("[$name] TRIAL $tr seed=$seed: ERROR -- $(sprint(showerror, e))")
                end
            end
        end
    end
end

println()
println("total_issues=$total_issues total_errors=$total_errors total_known_bug_skips=$total_known_bug")
println((total_issues == 0 && total_errors == 0) ? "ALL TUPLEMASK CHECKS PASSED" : "SEE ABOVE")
