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

Hash levels in the destination, source shards, and normalization accumulators
inside `Coalesce` use `nextpow(2, get_num_tasks(device))`
subtables, rounding the configured merge worker count up to a power of two.
For example, `cpu(:k, 5)` uses eight subtables regardless of `Threads.nthreads()`.
Explicit `SparseHash` bucket counts must still be positive powers of two.
`coalesce_similar_level` constructs empty levels with the requested bucket
count; it does not copy or rehash the supplied level's contents.

For parent position `p` and index `i`, bucket routing uses
`(a * UInt(p) + hash(i)) & UInt(B - 1)`, with a shared random odd `a` and
`B` buckets. A uniform parent shift `delta` therefore rotates buckets by
`a * delta` modulo `B`. Bucket routing reads the low bits of
the hash word `x = a * UInt(p) + hash(i)`. Linear probing clusters on linear
hashes of structured parents, so the control-byte fingerprint (low seven bits)
and the starting slot within a bucket (the bits above those) come from
`hash(x)`.
Assembly sizes the table for its busiest bucket, including
pending keys; routing alone does not guarantee balanced occupancy.

Every hash inserts tentative entries directly into the table using the same
writer protocol. Each child record stores `(parent, index, state)`: state `0x00`
means free, `0x01:0x7f` counts outstanding writers, and `0x80` means retained.
Attempting to add a 128th pending writer throws an error without changing the
count. Every occupied table slot keeps its fingerprint, including tentative
entries, so probes compare keys only after a fingerprint match.

The first writer whose child retains data marks the record retained; later
writers cannot discard it. If all writers decline to retain data, the last one
locates the table slot, removes it with backward-shift deletion, marks the
record free, and recycles the child position. Writer counts are accessed through
stable child positions, so growth and deletion do not require repairing cached
slots. These are overlapping generated access scopes within one task, not
concurrent CPU writes to a shard. Freeze requires every such access to finish.

Growth rebuilds the table by scanning initialized child records in position
order, skipping free records and regenerating fingerprints. It does not read or
copy the old table, even when records have pending writers or holes. Freeze
also collects live children directly from their records. It trims unused child
positions beyond the last live entry and preserves interior holes in the free
pool, so an empty hash also has an empty child. Each coalesce worker fills
its output child range in position order, writing each record once as retained
or free, including holes and discarded shared duplicates.

Declaring a hash empty retains its table capacity, so reusing an accumulator
does not repeat table growth. With at most one live entry per 1,024 slots, reset
uses the frozen permutation to locate and clear just the occupied control bytes;
otherwise, it clears the whole control array. Sparse clearing probes past slots
already cleared, since clearing an entry must not hide later entries in its
collision chain. Slot words need no initialization: an empty control byte makes
their old contents inaccessible. Child records, free positions, and counts reset
for the new accumulation.

When a `CoalesceLevel` freezes, it merges its `P` task shards with
`coalesce_shards!(src, dst, P, max_pos, bands)`. The shards must be ordered and
disjoint: everything shard `p` stores precedes, in outermost-first index order,
everything shard `p + 1` stores, so the result is their concatenation. The only
overlap is at a band boundary, where neighboring shards can both store the same
parent entry. The earlier shard owns it, and both shards' children land under it.
Dense blocks under a shared entry overlap too, since every shard stores fill
values outside its band; there, values merge by copying only non-fill values into
a destination of fill.

In the default `:normalize` mode, task `tid` first sums every shard's entries in
its band `(lb, ub)` into its own accumulator, so the shards meet this contract,
and `bands[tid]` is that band as a range of flat indices. Bands keep each shard
from scanning dense storage outside its band. In `:fast` mode, tasks write their
shards directly and `bands` is `nothing`, so all dense storage may overlap.

The merge plans storage, initializes it, then copies entries:

- `setup_coalesce!(lvl, max_pos, dst, P, shift, overlap)` runs once. It sizes
  each destination level and returns a plan recording where each shard's
  parent positions land (`dst_pos = pos + shift[p]`), the local child position
  shared with an earlier shard, and whether shards' leaves can overlap below it.
  Levels that compute child positions from their parent (`Dense`,
  `SparseByteMap`) scale the shift by their shape; levels that store children in
  their own slots (`SparseList`) shift by the entries earlier shards own. A merge
  in which every shard shares one position space is the case where every shift
  is zero.
- The plan's `init` tuples describe `(buffer, start, value)` ranges, including
  child storage. Allocation and resizing remain serial. `coalesce_shards!`
  partitions initialization across workers, then waits for every worker before
  starting the copies. This barrier prevents initialization from overwriting
  another shard's contributions under a shared boundary entry.
- `coalesce_shard!(tid, plan, lvl, dst, runs)` then runs in parallel, copying
  shard `tid`. `runs` iterates ranges of leaf positions (positions at the
  `Element` level) under which the shard stores values. `Dense` levels leave leaf
  positions unchanged, so they pass `runs` through. Sparse levels merge all
  their shard's entries and pass on the leaves under them; each pointer entry is
  written by the one shard owning the first entry at or after it.
  `SparseByteMap` keeps the pointer layout `freeze_level!` produces, with bounds
  only around occupied positions, so the merge stays O(nnz).

Element and byte-map setup record newly allocated ranges instead of filling
them. Byte maps use the first and last entries of their sorted dirty list,
`srt`, to find boundaries; they do not scan the bitmap during setup. Declaration
already uses that dirty list to clear reused bitmap entries and parent bounds,
so coalesce initialization only needs to fill newly allocated storage. An empty
sparse list records its whole pointer array for parallel initialization to `1`.

Buffers whose contents are completely rewritten are emptied before resizing:
sparse-list indices and pointers, the byte-map dirty list, and hash pointers,
keys, and permutations. This avoids preserving old contents if growth reallocates
storage. Buffers written only at occupied positions retain their cleared
regions; setup records only the newly added range for initialization.

Setup's metadata work is O(P) per level, except for sparse-list boundary
searches, which take O(P log M) for M parent positions, and hash setup, which
takes O(PB) for B buckets. With B equal to the worker count rounded up to a
power of two, hash setup is O(P²). These bounds exclude buffer
allocation/resizing. Setup does not scan entries or initialize buffers.

### Shared positions and ownership

Sparse plans use integer positions for boundary ownership:

- `shared[p] == 0` means shard `p` writes all its index metadata.
- `shared[p] == q` means the metadata for local child position `q` belongs to
  an earlier shard. Skip that index, but still merge its children.
- `shared_dst[p]` gives the destination child position for that shared entry,
  or `0` when there is no shared entry. Empty shards between owners and later
  contributors do not break the relationship.

The position is format-specific, not the entry's rank in traversal order:

| Level | Meaning of `shared[p]` |
|:------|:-----------------------|
| `SparseList` | Position in `idx` (currently `1` at a shared boundary) |
| `SparseByteMap` | Flattened child position stored in `srt` |
| `SparseHash` | Child position `q` of the entry keyed `key[q] == (parent, index)` |

For example, a byte-map shard whose first `srt` entry is `7` reports
`shared[p] = 7` when that entry is shared. A hash whose first frozen `perm`
entry is child position `3`, keyed `(2, 5)`, reports `shared[p] = 3`. Neither reports `1` merely
because the shared entry comes first in traversal order. Counts and offsets
subtract `shared[p] != 0`, not `shared[p]`.

`Dense` and `Element` do not own sparse index metadata, so their plans do not
need a shared-position field. Dense levels pass through to their child plan;
elements copy non-fill values when their positions overlap.

A hash stores its keys by child position, `key[q] == (parent, index)`, and its
table slots and `perm` hold child positions. A frozen `key` spans exactly the
child positions, pooled vacancies included, so its length is the child extent.
Hash levels retain the assembly bucket-count array, `tbl_count`, through freeze.
Setup reads at most the first and last frozen entries of each shard that is
not below a hash. It rotates that shard's counts by `a * offset mod B`, then
subtracts its shared boundary entry from its output bucket. The busiest output
bucket determines the common subtable capacity, keeping every bucket at most
half full.

Hash children keep their arbitrary positions: shard `p`'s child `q` lands at
`q + child_shift[p]`, where `child_shift` concatenates the lengths of the shards' `key`.
The one exception is the shared entry, whose child lands at the earlier owner's
`shared_dst[p]`. The hash passes that exception down as a `ShardShift`, which
levels apply like an integer shift (`pos + shift[p]`, `shift[p] * shape`):

- Dense and element levels need nothing more. Every run a hash passes down is
  one entry's block, so no run straddles the exception.
- A byte map or hash below a hash has one run of entries under the moved
  parents, found through `ptr`. Bands cut in traversal order, so that run
  belongs right after the owner's entries under the same parents.
  Setup orders the index metadata (`srt`, or the hash's `perm`) as pieces,
  runs of one shard's entries: each shard's entries up to that block, the runs
  moved into it, then the rest. No child moves.
- Lists are rejected below a hash: their children are addressed by entry, so
  filling a gap would move children.

A hash merge needs no second phase. Worker `tid` copies shard `tid`'s keys,
`perm`, pointers, and children: `perm` holds child positions, so a shard places
its entries in traversal order without knowing their table slots. Worker `tid`
also owns output buckets `tid:P:B`. For each owned bucket, it visits the rotated
bucket of every frozen source shard, plus that shard's moved entries, and places
each entry's child position in the first empty slot of its probe. Keys are
distinct once shared entries are dropped, so placement never compares keys or
resizes. Each bucket has one writer, including when `B > P`. Since bucket work
is tied to the worker, every worker must reach every level, even with an empty
shard.

Pointers need the parent of the entry before each piece; setup records it from
the previous piece's last entry, so shards write pointers independently. Byte
maps write their pointers the same way.

## Closures in generated code

`Threads.@threads` turns its loop body into a closure. A closure that captures a
variable assigned more than once boxes it, and inference then sees `Any`
wherever the closure reads it. Generated code keeps captures single-assignment:

- Every parallel loop is wrapped in `Finch.@barrier`, which binds the loop's
  free variables in a `let` just before it. The thread closure captures those
  fresh bindings, so code outside may rebind a buffer, for example when
  `distribute_buffer` writes it back, without boxing anything.
- `@barrier` never binds a constant global, such as the `Finch` module. As a
  local, `Finch` would hide every `Finch.f(...)` from inference and make each
  call dynamic.
- Distributed levels freshen their scalar assembly state (`qos_stop` and the
  like), so each task assigns its own locals instead of a shared variable.
- Other generated loops are plain loops over fixed-size buffers, not
  comprehensions, so they create no closures.
