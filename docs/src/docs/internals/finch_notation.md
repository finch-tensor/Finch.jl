```@meta
CurrentModule = Finch
```

# Finch Notation Internals

Finch IR is a tree structure that represents a finch program. Different types of nodes are delineated by a `FinchKind` enum, for type stability. There are a few useful functions to be aware of:

```@docs
FinchNode
cached
finch_leaf
FinchNotation.isstateful
isliteral
isvalue
isconstant
isvirtual
isvariable
isindex
```
