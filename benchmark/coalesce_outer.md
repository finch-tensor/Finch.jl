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

## Hash output formats, 2026-09-30

`coalesce_outer.jl` now also times hash output formats and a 4096×8×4096 case.
Formats a version cannot run correctly are skipped. The baselines below ran the
same script against isolated copies of `main` (`2dc58af1d`), `desc`
(`266418612`), and this branch before optimizing the hash merge (`56660150d`
plus uncommitted changes), with Julia 1.12.6, eight CPU threads, and seed 42.
Times are minima; at 4096 the slow hash cases fit only one or two samples.

| Case | Format | `main` | `desc` | branch, before |
| --- | --- | ---: | ---: | ---: |
| 1024×8×1024 | `SparseList(SparseList)` | 8.262 ms | 7.343 ms | 7.449 ms |
| 1024×8×1024 | `Dense(SparseHash)` | 154.962 ms | unsupported | 277.493 ms |
| 1024×8×1024 | `SparseHash(SparseHash)` | fails | unsupported | 292.731 ms |
| 1024×8×1024 | `SparseHash(Dense)` | fails | unsupported | 15.006 ms |
| 4096×8×4096 | `SparseList(SparseList)` | 192.244 ms | 222.630 ms | 213.439 ms |
| 4096×8×4096 | `Dense(SparseHash)` | 3598.136 ms | unsupported | 16148.051 ms |
| 4096×8×4096 | `SparseHash(SparseHash)` | fails | unsupported | 15606.296 ms |
| 4096×8×4096 | `SparseHash(Dense)` | fails | unsupported | 379.206 ms |

`desc` lacks `sample_dims` for hashes; `main` merges `SparseHash(SparseHash)` and
`SparseHash(Dense)` out of bounds.

## Compact hash layout, 2026-10-01

Hashes now store keys by child position, `key[q] == (p, i)`, so sampling a hash
looks a child's entry up directly instead of scanning `perm`, which dominated
the hash cases above. `MemoryChannel` transfers of multichannel buffers are also
type-stable now; the old union result made three-buffer hash helpers dispatch
dynamically. Both trees below ran back to back twice, at `6e7d927ea` and with
these changes, with Julia 1.12.6, eight CPU threads, and seed 42. Times are the
better minimum of the two rounds.

| Case | Format | `6e7d927ea` | Compact layout |
| --- | --- | ---: | ---: |
| 1024×8×1024 | `SparseList(SparseList)` | 7.182 ms | 6.168 ms |
| 1024×8×1024 | `Dense(SparseHash)` | 290.466 ms | 51.579 ms |
| 1024×8×1024 | `SparseHash(SparseHash)` | 285.228 ms | 53.663 ms |
| 1024×8×1024 | `SparseHash(Dense)` | 15.096 ms | 16.523 ms |
| 4096×8×4096 | `SparseList(SparseList)` | 219.119 ms | 166.085 ms |
| 4096×8×4096 | `Dense(SparseHash)` | 15077.538 ms | 1170.066 ms |
| 4096×8×4096 | `SparseHash(SparseHash)` | 15265.900 ms | 1258.536 ms |
| 4096×8×4096 | `SparseHash(Dense)` | 371.969 ms | 378.357 ms |

`Dense(SparseHash)` now beats `main` (154.962 ms and 3598.136 ms above). Hash
formats still allocate about 20 times per output entry at 4096: values written
inside the parallel accumulation's `@barrier` closure are boxed, as on `main`.

## No closure boxing, 2026-10-01

Generated parallel loops now bind their free variables once, just before the
thread closure, and never rebind the `Finch` module as a local, which had made
every `Finch.f(...)` call inside parallel bodies dynamic. Same script, Julia
1.12.6, eight CPU threads, seed 42; better minimum of two back-to-back rounds.

| Case | Format | `6e7d927ea` | Compact layout | No boxing |
| --- | --- | ---: | ---: | ---: |
| 1024×8×1024 | `SparseList(SparseList)` | 6.967 ms | 6.168 ms | 5.930 ms |
| 1024×8×1024 | `Dense(SparseHash)` | 290.466 ms | 51.579 ms | 12.127 ms |
| 1024×8×1024 | `SparseHash(SparseHash)` | 285.228 ms | 53.663 ms | 12.813 ms |
| 1024×8×1024 | `SparseHash(Dense)` | 15.096 ms | 16.523 ms | 2.960 ms |
| 4096×8×4096 | `SparseList(SparseList)` | 204.188 ms | 166.085 ms | 179.771 ms |
| 4096×8×4096 | `Dense(SparseHash)` | 15077.538 ms | 1170.066 ms | 397.129 ms |
| 4096×8×4096 | `SparseHash(SparseHash)` | 15265.900 ms | 1258.536 ms | 396.599 ms |
| 4096×8×4096 | `SparseHash(Dense)` | 371.969 ms | 378.357 ms | 58.925 ms |

At 1024, allocations fell from 7.8 million to 12 thousand for
`Dense(SparseHash)` and from 4.3 million to 11 thousand for `SparseHash(Dense)`.
