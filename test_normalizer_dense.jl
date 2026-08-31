using Finch
using Random

include(joinpath(@__DIR__, "normalizer.jl"))

# ---------------------------------------------------------------------------
# Load-balance tests for the countstored_level API extended to DenseLevel,
# exercising the two possible nestings of Dense and SparseList:
#
#   Dense(SparseList(Element))  -- outer dim (j) is Dense, inner dim (i) is
#                                   SparseList. Arbitrary (i,j) pairs may be
#                                   sharded to any processor, since sparsity
#                                   lives entirely in the inner SparseList's
#                                   per-processor ptr/idx arrays; the outer
#                                   Dense level itself is a stateless index
#                                   multiplier (it stores no ptr/idx).
#
#   SparseList(Dense(Element))  -- outer dim (j) is SparseList, inner dim (i)
#                                   is Dense. Once a column j is "stored" for
#                                   a processor, its *entire* dense column (all
#                                   i in 1:S1) counts as stored for that
#                                   processor -- Dense has no way to record
#                                   partial-column sparsity. So sharding must
#                                   assign whole columns to processors, not
#                                   individual (i,j) pairs.
#
# Dimension convention matches test_normalizer.jl: idxs[1] = inner (i, rows),
# idxs[2] = outer (j, cols).
# ---------------------------------------------------------------------------

key(c) = (c[2], c[1])  # (j, i) -- matches idxs[2]=outer, idxs[1]=inner

function full_grid_ranks(S1, S2)
    grid = [(i, j) for j in 1:S2 for i in 1:S1]  # already in (j,i) lex order
    Dict(zip(grid, 1:length(grid))), length(grid)
end

# ---------------------------------------------------------------------------
# Builder 1: Dense(SparseList(Element)) -- outer Dense (S2), inner SparseList
# (S1). Each processor gets a full ptr array of length S2+1 (one boundary per
# real column, since Dense forces the inner level's `pos` to range over all
# of 1:S2), populated only with the (i,j) pairs assigned to that processor.
# ---------------------------------------------------------------------------
function build_dense_sparse_shard(coords_by_proc, S1, S2)
    P = length(coords_by_proc)
    ptr_data = Vector{Vector{Int}}(undef, P)
    idx_data = Vector{Vector{Int}}(undef, P)

    for p in 1:P
        coords = sort(coords_by_proc[p]; by=c -> (c[2], c[1])) # (j,i) lex order
        ptr = Int[1]
        idx = Int[]
        pos = 1
        k = 1
        n = length(coords)
        for j in 1:S2
            while k <= n && coords[k][2] == j
                push!(idx, coords[k][1])
                pos += 1
                k += 1
            end
            push!(ptr, pos)
        end
        ptr_data[p] = ptr
        idx_data[p] = idx
    end

    device = Finch.CPU{:test}(P)
    ptr = Finch.CPULocalArray{Vector{Int},typeof(device)}(device, ptr_data)
    idx = Finch.CPULocalArray{Vector{Int},typeof(device)}(device, idx_data)

    elem = Finch.ElementLevel(0.0)
    inner = Finch.SparseListLevel{Int}(elem, S1, ptr, idx)
    outer = Finch.DenseLevel{Int}(inner, S2)
    return outer
end

# ---------------------------------------------------------------------------
# Builder 2: SparseList(Dense(Element)) -- outer SparseList (S2), inner Dense
# (S1). Sharding is by whole column: owner_of_col[j] gives the processor that
# owns column j (or 0 if the column is unstored entirely). Each processor's
# outer level uses the standard single-root-position ptr = [1, ncols+1], with
# idx listing the (sorted) columns it owns.
# ---------------------------------------------------------------------------
function build_sparse_dense_shard(owner_of_col, S1, S2, P)
    ptr_data = Vector{Vector{Int}}(undef, P)
    idx_data = Vector{Vector{Int}}(undef, P)

    for p in 1:P
        cols = sort([j for j in 1:S2 if owner_of_col[j] == p])
        idx_data[p] = cols
        ptr_data[p] = Int[1, length(cols) + 1]
    end

    device = Finch.CPU{:test}(P)
    ptr = Finch.CPULocalArray{Vector{Int},typeof(device)}(device, ptr_data)
    idx = Finch.CPULocalArray{Vector{Int},typeof(device)}(device, idx_data)

    elem = Finch.ElementLevel(0.0)
    inner = Finch.DenseLevel{Int}(elem, S1)
    outer = Finch.SparseListLevel{Int}(inner, S2, ptr, idx)
    return outer
end

# ---------------------------------------------------------------------------
# Shared validation: given a level `lvl`, the full list of "stored" (i,j)
# coordinate units, and P, check that balance() produces adjacent,
# non-overlapping, roughly-equal partitions.
# ---------------------------------------------------------------------------
function validate_balance(lvl, coords, S1, S2, P)
    nnz = length(coords)
    shapes = collect(Finch.level_size(lvl))
    @assert shapes == [S1, S2]

    rankmap, ngrid = full_grid_ranks(S1, S2)

    parts = NamedTuple[]
    for tid in 1:P
        lb, ub = balance(lvl, tid, P, nnz, MergeNormalization())
        push!(parts, (tid=tid, lb=lb, ub=ub))
    end

    issues = String[]

    for tid in 1:P
        lbr = rankmap[parts[tid].lb]
        ubr = rankmap[parts[tid].ub]
        if lbr > ubr
            push!(issues, "tid=$tid: lb rank ($lbr) > ub rank ($ubr) -- inverted range (lb=$(parts[tid].lb), ub=$(parts[tid].ub))")
        end
        if tid == 1 && lbr != 1
            push!(issues, "tid=1: lb rank is $lbr, expected 1")
        end
        if tid == P && ubr != ngrid
            push!(issues, "tid=$P: ub rank is $ubr, expected $ngrid")
        end
        if tid > 1
            prev_ubr = rankmap[parts[tid - 1].ub]
            if prev_ubr + 1 != lbr
                push!(issues, "gap/overlap between tid=$(tid - 1) (ub rank $prev_ubr) and tid=$tid (lb rank $lbr)")
            end
        end
    end

    base = nnz ÷ P
    counts = Int[]
    for tid in 1:P
        lo = key(parts[tid].lb)
        hi = key(parts[tid].ub)
        cnt = count(c -> lo <= key(c) <= hi, coords)
        push!(counts, cnt)
        if abs(cnt - base) > 2P - 1
            push!(issues, "tid=$tid: actual nnz $cnt deviates from ideal $base by more than 2P-1=$(2P - 1)")
        end
    end
    if sum(counts) != nnz
        push!(issues, "sum of partition counts $(sum(counts)) != nnz $nnz")
    end

    return (issues=issues, counts=counts, base=base, nnz=nnz)
end

function run_trial_dense_sparse(S1, S2, nnz_target, P; seed)
    Random.seed!(seed)
    universe = [(i, j) for i in 1:S1 for j in 1:S2]
    coords = Random.shuffle(universe)[1:nnz_target]
    shard_of = [rand(1:P) for _ in coords]
    coords_by_proc = [Tuple{Int,Int}[] for _ in 1:P]
    for (c, p) in zip(coords, shard_of)
        push!(coords_by_proc[p], c)
    end

    lvl = build_dense_sparse_shard(coords_by_proc, S1, S2)
    validate_balance(lvl, coords, S1, S2, P)
end

function run_trial_sparse_dense(S1, S2, ncols_target, P; seed)
    Random.seed!(seed)
    cols = Random.shuffle(1:S2)[1:ncols_target]
    owner_of_col = zeros(Int, S2)
    for j in cols
        owner_of_col[j] = rand(1:P)
    end

    lvl = build_sparse_dense_shard(owner_of_col, S1, S2, P)
    coords = [(i, j) for j in cols for i in 1:S1]  # each owned column is fully dense
    validate_balance(lvl, coords, S1, S2, P)
end

# ---------------------------------------------------------------------------
trials = [
    (S1=6, S2=5, nnz=20, P=3),
    (S1=6, S2=5, nnz=20, P=4),
    (S1=10, S2=10, nnz=60, P=5),
    (S1=10, S2=10, nnz=60, P=8),
    (S1=20, S2=3, nnz=40, P=6),
    (S1=3, S2=20, nnz=40, P=6),
    (S1=8, S2=8, nnz=8, P=8),   # tight, ~1 per proc
    (S1=8, S2=8, nnz=63, P=8),  # near-full
]

# for SparseList(Dense), "nnz" is expressed as number of owned columns (each
# contributing S1 stored units), so pick col counts that keep total nnz in a
# comparable range to the trials above.
col_trials = [
    (S1=6, S2=5, ncols=4, P=3),
    (S1=6, S2=5, ncols=5, P=4),
    (S1=10, S2=10, ncols=6, P=5),
    (S1=10, S2=10, ncols=9, P=8),
    (S1=4, S2=20, ncols=15, P=6),
    (S1=2, S2=30, ncols=20, P=6),
    (S1=3, S2=8, ncols=8, P=8),
    (S1=8, S2=8, ncols=8, P=8),
]

function run_suite(name, trials, runner)
    total_issues = 0
    total_errors = 0
    for (idx, t) in enumerate(trials)
        for seed in 1:5
            try
                r = runner(t, seed * 1000 + idx)
                if !isempty(r.issues)
                    total_issues += length(r.issues)
                    println("[$name] TRIAL $t seed=$seed: base=$(r.base) counts=$(r.counts)")
                    for iss in r.issues
                        println("  ISSUE: $iss")
                    end
                else
                    println("[$name] TRIAL $t seed=$seed: OK  base=$(r.base) counts=$(r.counts)")
                end
            catch e
                total_errors += 1
                println("[$name] TRIAL $t seed=$seed: ERROR -- $(sprint(showerror, e))")
            end
        end
    end
    println()
    println("[$name] total_issues=$total_issues total_errors=$total_errors")
    println((total_issues == 0 && total_errors == 0) ? "[$name] ALL TRIALS PASSED" : "[$name] SEE ABOVE")
    return total_issues == 0 && total_errors == 0
end

ok1 = run_suite("Dense(SparseList)", trials, (t, seed) -> run_trial_dense_sparse(t.S1, t.S2, t.nnz, t.P; seed=seed))
ok2 = run_suite("SparseList(Dense)", col_trials, (t, seed) -> run_trial_sparse_dense(t.S1, t.S2, t.ncols, t.P; seed=seed))

println()
println("OVERALL: ", (ok1 && ok2) ? "ALL PASSED" : "FAILURES PRESENT")
