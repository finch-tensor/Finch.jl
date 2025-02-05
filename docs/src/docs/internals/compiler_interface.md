```@meta
CurrentModule = Finch
```

# Compiler Internals

Finch has several compiler modules with separate interfaces.

## SymbolicContexts

SymbolicContexts are used to represent the symbolic information of a program. They are used to reason about the bounds of loops and the symbolic information of the program, and are defined on an algebra

```@docs
StaticHash
SymbolicContext
get_algebra
get_static_hash
prove
simplify
```

## ScopeContexts

ScopeContexts are used to represent the scope of a program. They are used to reason about values bound to variables and also the modes of tensor variables.

```@docs
ScopeContext
get_binding
has_binding
set_binding!
set_declared!
set_frozen!
set_thawed!
get_tensor_mode
open_scope
```

## JuliaContexts

JuliaContexts are used to represent the execution environment of a program, including variables and tasks. They are used to generate code.

```@docs
Namespace
JuliaContext
push_preamble!
push_epilogue!
get_task
freshen
contain
```

## AbstractCompiler

The AbstractCompiler interface requires all of the functionality of the above contexts, as well as the following two methods:

```@docs
FinchCompiler
get_result
get_mode_flag
```
