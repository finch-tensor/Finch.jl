```@meta
CurrentModule = Finch
```

# Finch Logic (High-Level IR)

Finch Logic is an internal high-level intermediate representation (IR) that allows us to
fuse and optimize successive calls to array operations such as `map`, `reduce`,
and `broadcast`. It is reminiscent to database query notation, representing the
a sequence of tensor expressions bound to variables. Values in the program are tensors,
with named indices. The order of indices is semantically meaningful.

The nodes are as follows:

```@docs
immediate
deferred
field
alias
table
mapjoin
aggregate
reorder
relabel
reformat
subquery
query
produces
plan
```

## Finch Logic Internals

```@docs
FinchLogic.LogicNode
FinchLogic.logic_leaf
isimmediate
isdeferred
isalias
isfield
```

## Executing FinchLogic
