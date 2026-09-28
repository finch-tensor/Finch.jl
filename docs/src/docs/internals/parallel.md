```@meta
CurrentModule = Finch
```

# Parallel Processing in Finch

## Modelling the Architecture

Finch uses a simple, hierarchical representation of devices and tasks to model
different kind of parallel processing. An [`AbstractDevice`](@ref) is a physical or
virtual device on which we can execute tasks, which may each be represented by
an [`AbstractTask`](@ref).

```@docs
AbstractTask
AbstractDevice
```

The current task in a compilation context can be queried with
[`get_task`](@ref). Each device has a set of numbered child
tasks, and each task has a parent task.

```@docs
get_num_tasks
get_task_num
get_device
get_parent_task
```

## Data Transfer

Before entering a parallel loop, a tensor may reside on a single task, or
represent a single view of data distributed across multiple tasks, or represent
multiple separate tensors local to multiple tasks. A tensor's data must be
resident in the current task to process operations on that tensor, such as loops
over the indices, accesses to the tensor, or `declare`, `freeze`, or `thaw`.
Upon entering a parallel loop, we must transfer the tensor to the tasks
where it is needed. Upon exiting the parallel loop, we may need to combine
the data from multiple tasks into a single tensor.

All tensor and buffer transfers are accomplished with the `transfer` function.

The `distribute` function is used by the compiler to orchestrate data distribution before and after a parallel region, with

different `style` objects signaling the type of transfer.

Note: After distributing a tensor, we must also update any in-progress
traversals over the tensor that may appear throughout the program. This is done
with the `redistribute` function. Tensors are responsible for defining their own
redistribute behavior, but it should be guaranteed that `distribute(tns, diff) == redistribute(tns, diff)`. In general, this means that
any nested structure in the tensor should be preserved through transfers. Most
subtensors will store a list of property names describing how to reach the
subtensor from the root tensor.

```@docs
distribute
redistribute
```

The `distribute` function is called on the `Host` and on the `Device`, and is responsible
for distributing the tensor among tasks and collecting the results, if applicable.

If the tensor is a temporary tensor declared within the parallel loop, we
distribute the tensor to `Local` scope. If the tensor is declared outside the
parallel loop and is not modified, we distribute the tensor to `Global` scope.
If the tensor is declared outside the parallel loop and is modified, we distribute
the tensor to `Shared` scope. Depending on the architecture, several of these operations
may be no-ops.

```@docs
HostLocal
HostGlobal
HostShared
DeviceLocal
DeviceGlobal
DeviceShared
```

The `transfer` function is used to distribute tensors and their constituent
buffers to different memory spaces.  We can ask for the default local, shared,
or global memory spaces of an `AbstractDevice` with the `local_memory`, etc.
trait functions.

```@docs
transfer
local_memory
shared_memory
global_memory
```

## Coalescing task-local output

When a `CoalesceLevel` freezes, it merges its `P` task shards with
`coalesce_shards!(src, dst, P, max_pos, bands)`. The shards must be ordered and
disjoint: everything shard `p` stores precedes, in outermost-first index order,
everything shard `p + 1` stores, so the result is their concatenation. The only
overlap is at a band boundary, where neighboring shards can both store the same
parent entry. The earlier shard owns it, and both shards' children land under it.

In the default `:normalize` mode, task `tid` first sums every shard's entries in
its band `bands[tid] = (lb, ub)` into its own accumulator, so the shards meet
this contract. Dense levels store fill values outside their band, so the merge
clips dense blocks on the band's edges. In `:fast` mode, tasks write their shards
directly and `bands` is `nothing`; dense blocks then merge by copying only stored
non-fill values.

The merge makes two passes over the levels:

- `setup_coalesce!(lvl, max_pos, dst, P, shift)` runs once. It sizes each
  destination level and returns a plan recording where each shard's positions
  land (`dst_pos = pos + shift[p]`) and whether its first entry is shared with
  the previous shard. Levels that compute child positions from their parent
  (`Dense`, `SparseByteMap`) scale the shift by their shape; levels that store
  children in their own slots (`SparseList`) shift by the entries earlier shards
  own. A merge in which every shard shares one position space is the case
  where every shift is zero.
- `coalesce_shard!(tid, plan, lvl, dst, runs, band)` then runs in parallel,
  copying shard `tid`. `runs` are the shard's parent positions to merge, and
  `band` is a `CoalesceBand` giving the positions on the band's edges, or
  `nothing`. Sparse levels merge all their shard's entries; each pointer entry
  is written by the one shard owning the first entry at or after it.
  `SparseByteMap` keeps the pointer layout `freeze_level!` produces, with bounds
  only around occupied positions, so the merge stays O(nnz).
