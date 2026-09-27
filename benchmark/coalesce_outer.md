# Coalesce outer-product benchmark

Run the standalone case without downloading the full benchmark suite's datasets:

```sh
julia --project=benchmark --threads=8 benchmark/coalesce_outer.jl results.json
```

The case is also registered as `SUITE["coalesce"]` in `benchmarks.jl`.

It multiplies a 1024×8 matrix by an 8×1024 matrix, with independently generated
25%-dense inputs. Eight logical tasks each build one outer product in
`Coalesce(SparseList(SparseList(Element(0.0))))`. Their overlapping outputs
exercise normalization, reduction, and merging through two sparse levels.
One outer product per task keeps each shard's writes ordered. Use eight CPU
threads for the parallel case; the logical task count stays eight on other
thread counts.

Input conversion, compilation, and an exact comparison against a dense reference
multiply happen before timing. Every timed evaluation includes fresh output
allocation and the full multiply and coalesce operation. The normalization RNG
is reset outside timing, and each sample performs one evaluation.

## Local comparison, 2026-09-27

The same benchmark script was run sequentially against an isolated copy of
`266418612` and the working tree containing the separate-offset/shared-flag
refactor. Both used Julia 1.10.11, BenchmarkTools 1.8.0, eight CPU threads, seed 42,
and 100 samples. Both passed the reference check. Julia's normal JIT compilation
was enabled; package precompilation was disabled for these runs.

| Metric | `266418612` | Working tree |
| --- | ---: | ---: |
| Minimum | 5.238 ms | 5.013 ms |
| Median | 5.916 ms | 6.039 ms |
| Allocated bytes | 138,589,792 | 138,587,872 |
| Allocations | 30,825 | 30,819 |

The median increased 2.1% while the minimum decreased 4.3%; this single local
comparison does not establish a timing improvement or regression. The working
tree used six fewer allocations and 1,920 fewer allocated bytes per multiply.
