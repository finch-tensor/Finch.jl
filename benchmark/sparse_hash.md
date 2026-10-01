# Serial SparseHash benchmark

Run without Coalesce:

```sh
julia --project=benchmark benchmark/sparse_hash.jl results.json
```

It builds a 4096×8×4096 outer-product matmul with 25%-dense inputs into each
hash format (`accumulate`, mostly hits on existing keys), copies the result out
in order (`iterate`), and probes it once per entry of a 5%-dense mask
(`lookup`). Compilation and validation happen before timing.

## Compact hash layout, 2026-10-01

Both trees ran back to back twice, at `6e7d927ea` and with keys stored by child
position, with Julia 1.12.6 on one thread. Times are the better minimum of the
two rounds.

| Format | Phase | `6e7d927ea` | Compact layout |
| --- | --- | ---: | ---: |
| `Dense(SparseHash)` | accumulate | 562.334 ms | 585.821 ms |
| `Dense(SparseHash)` | iterate | 89.760 ms | 26.325 ms |
| `Dense(SparseHash)` | lookup | 21.084 ms | 23.219 ms |
| `SparseHash(SparseHash)` | accumulate | 540.892 ms | 588.406 ms |
| `SparseHash(SparseHash)` | iterate | 91.532 ms | 26.142 ms |
| `SparseHash(SparseHash)` | lookup | 20.376 ms | 21.750 ms |
| `SparseHash{false}(SparseHash{false})` | accumulate | 681.942 ms | 706.945 ms |
| `SparseHash{false}(SparseHash{false})` | iterate | 90.871 ms | 26.925 ms |
| `SparseHash{false}(SparseHash{false})` | lookup | 20.337 ms | 21.357 ms |

A hit now loads the slot's child position and then its key, one more dependent
load than the old inline `(p, i, q)` slots, which costs 4–9% on accumulation and
5–10% on lookup. Iteration reads keys straight from `perm`'s child positions.
