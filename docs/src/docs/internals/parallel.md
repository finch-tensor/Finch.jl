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

## Data Movement

Before entering a parallel loop, a tensor may reside on a single task, or
represent a single view of data distributed across multiple tasks, or represent 
multiple separate tensors local to multiple tasks. A tensor's data must be
resident in the current task to process operations on that tensor, such as loops
over the indices, accesses to the tensor, or `declare`, `freeze`, or `thaw`.
Upon entering a parallel loop, we must transfer the tensor to the tasks
where it is needed. Upon exiting the parallel loop, we may need to combine
the data from multiple tasks into a single tensor.

There are two cases, depending on whether the tensor is declared outside the
parallel loop or is a temporary tensor declared within the parallel loop.

If the tensor is a temporary tensor declared within the parallel loop, we call
`bcast` to broadcast the tensor to all tasks.

If the tensor is declared outside the parallel loop, we call `scatter` to 
send it to the tasks where it is needed. Note that if the tensor is in `read` mode,
`scatter` may simply `bcast` the entire tensor to all tasks. If the device has global
memory, `scatter` may also be a no-op. When the parallel loop is exited, we call
`gather` to reconcile the data from multiple tasks back into a single tensor.

Each of these operations begins with a `_send` variant on one task, and 
finishes with a `_recv` variant on the recieving task.

```@docs
bcast
bcast_send
bcast_recv
scatter
scatter_send
scatter_recv
gather
gather_send
gather_recv
```