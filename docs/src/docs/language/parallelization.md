```@meta
CurrentModule = Finch
```

# (Experimental) Parallelization in Finch

## Formats

Finch levels usually cannot be updated concurrently from multiple threads.
Sparse and structured formats typically store their data buffers contiguously
across different columns, making parallel updates difficult to implement
correctly and efficiently. However, Finch provides a few specialized concurrent
level types, and a few levels which can reinterpret other level formats in a
concurrent way.

```@docs
Finch.AtomicElementLevel
Finch.MutexLevel
Finch.SeparateLevel
```

## Parallel Loops

A loop can be run in parallel with a `parallel` dimension. A dimension can be
wrapped in the `parallel()` modifier to indicate that it should run in parallel.

```@docs
Finch.parallel
Finch.CPU
Finch.Serial
```
