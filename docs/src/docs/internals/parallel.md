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
or global memory spaces of an `AbstractDevice` with the `localmemory`, etc.
trait functions.

```@docs
transfer
localmemory
sharedmemory
globalmemory
```
