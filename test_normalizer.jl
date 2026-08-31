using Finch
using Random

include(joinpath(@__DIR__, "normalizer.jl"))

# ---------------------------------------------------------------------------
# Build a P-way sharded 2D SparseList(SparseList(Element)) tensor by hand,
# matching the per-processor CPULocalArray{ptr,idx} layout that
# countstored_level expects: lvl.ptr.data[proc], lvl.idx.data[proc] at
# every nesting level. Dimension convention (confirmed from level_size and
# find_normalizer_split): idxs[1] = inner/least-significant dim (rows, i),
# idxs[2] = outer/most-significant dim (cols, j).
# ---------------------------------------------------------------------------
function build_sharded_2d(coords_by_proc, S1, S2)
    P = length(coords_by_proc)
    ptr1_data = Vector{Vector{Int}}(undef, P)
    idx1_data = Vector{Vector{Int}}(undef, P)
    ptr2_data = Vector{Vector{Int}}(undef, P)
    idx2_data = Vector{Vector{Int}}(undef, P)

    for p in 1:P
        coords = sort(coords_by_proc[p]; by = c -> (c[2], c[1])) # (j,i) lexicographic
        idx2 = Int[]
        ptr1 = Int[1]; idx1 = Int[]
        i = 1
        pos = 1
        n = length(coords)
        while i <= n
            j = coords[i][2]
            push!(idx2, j)
            k = i
            while k <= n && coords[k][2] == j
                push!(idx1, coords[k][1])
                k += 1
            end
            pos += (k - i)
            push!(ptr1, pos)
            i = k
        end
        # outer level has exactly one top-level "row" (pos=1): the whole
        # tensor for this processor lives at ptr2[1]:ptr2[2]-1
        ptr2 = Int[1, length(idx2) + 1]
        ptr1_data[p] = ptr1; idx1_data[p] = idx1
        ptr2_data[p] = ptr2; idx2_data[p] = idx2
    end

    device = Finch.CPU{:test}(P)
    ptr1 = Finch.CPULocalArray{Vector{Int},typeof(device)}(device, ptr1_data)
    idx1 = Finch.CPULocalArray{Vector{Int},typeof(device)}(device, idx1_data)
    ptr2 = Finch.CPULocalArray{Vector{Int},typeof(device)}(device, ptr2_data)
    idx2 = Finch.CPULocalArray{Vector{Int},typeof(device)}(device, idx2_data)

    elem = Finch.ElementLevel(0.0)
    lvl1 = Finch.SparseListLevel{Int}(elem, S1, ptr1, idx1)
    lvl2 = Finch.SparseListLevel{Int}(lvl1, S2, ptr2, idx2)
    return lvl2
end

key(c) = (c[2], c[1])  # (j, i) — matches idxs[2]=outer, idxs[1]=inner

function full_grid_ranks(S1, S2)
    grid = [(i, j) for j in 1:S2 for i in 1:S1]  # already in (j,i) lex order
    Dict(zip(grid, 1:length(grid))) , length(grid)
end

function run_trial(S1, S2, nnz_target, P; seed)
    Random.seed!(seed)
    universe = [(i, j) for i in 1:S1 for j in 1:S2]
    coords = Random.shuffle(universe)[1:nnz_target]
    # scatter coords across P processors round-robin-ish (random shard sizes)
    shard_of = [rand(1:P) for _ in coords]
    coords_by_proc = [Tuple{Int,Int}[] for _ in 1:P]
    for (c, p) in zip(coords, shard_of)
        push!(coords_by_proc[p], c)
    end

    lvl = build_sharded_2d(coords_by_proc, S1, S2)
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

    # 1. adjacency / no gap / no overlap in the FULL index grid (rank-based)
    #    ub(tid) must be exactly 1 rank before lb(tid+1)
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
            prev_ubr = rankmap[parts[tid-1].ub]
            if prev_ubr + 1 != lbr
                push!(issues, "gap/overlap between tid=$(tid-1) (ub rank $prev_ubr) and tid=$tid (lb rank $lbr)")
            end
        end
    end

    # 2. actual nnz balance vs ideal, tolerance 2P-1 (additive, per partition)
    base = nnz ÷ P
    counts = Int[]
    for tid in 1:P
        lo = key(parts[tid].lb); hi = key(parts[tid].ub)
        cnt = count(c -> lo <= key(c) <= hi, coords)
        push!(counts, cnt)
        if abs(cnt - base) > 2P - 1
            push!(issues, "tid=$tid: actual nnz $cnt deviates from ideal $base by more than 2P-1=$(2P-1)")
        end
    end
    if sum(counts) != nnz
        push!(issues, "sum of partition counts $(sum(counts)) != nnz $nnz")
    end

    return (issues=issues, counts=counts, base=base, nnz=nnz)
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

total_issues = 0
total_errors = 0
for (idx, t) in enumerate(trials)
    for seed in 1:5
        try
            r = run_trial(t.S1, t.S2, t.nnz, t.P; seed=seed*1000+idx)
            if !isempty(r.issues)
                global total_issues += length(r.issues)
                println("TRIAL $t seed=$seed: base=$(r.base) counts=$(r.counts)")
                for iss in r.issues
                    println("  ISSUE: $iss")
                end
            else
                println("TRIAL $t seed=$seed: OK  base=$(r.base) counts=$(r.counts)")
            end
        catch e
            global total_errors += 1
            println("TRIAL $t seed=$seed: ERROR -- $(sprint(showerror, e))")
        end
    end
end

println()
println("total_issues=$total_issues total_errors=$total_errors")
println((total_issues == 0 && total_errors == 0) ? "ALL TRIALS PASSED" : "SEE ABOVE")
