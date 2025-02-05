```@meta
CurrentModule = Finch
```

### Program Instances

Finch relies heavily on Julia's
[metaprogramming capabilities](https://docs.julialang.org/en/v1/manual/metaprogramming/) (
[macros](https://docs.julialang.org/en/v1/manual/metaprogramming/#Macros) and
[generated functions](https://docs.julialang.org/en/v1/manual/metaprogramming/#Generated-functions)
in particular) to produce code. To review briefly, a macro allows us to inspect
the syntax of it's arguments and generate replacement syntax. A generated
function allows us to inspect the type of the function arguments and produce
code for a function body.

In normal Finch usage, we might call Finch as follows:

```jldoctest example1; setup = :(using Finch)
julia> C = Tensor(SparseList(Element(0)));

julia> A = Tensor(SparseList(Element(0)), [0, 2, 0, 0, 3]);

julia> B = Tensor(Dense(Element(0)), [11, 12, 13, 14, 15]);

julia> @finch (C .= 0;
       for i in _
           C[i] = A[i] * B[i]
       end);

julia> C
5 Tensor{SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0, Int64, Int64, Vector{Int64}}}}:
  0
 24
  0
  0
 45

julia> tensor_tree(C)
5-Tensor
└─ SparseList (0) [1:5]
   ├─ [2]: 24
   └─ [5]: 45
```

The
[`@macroexpand`](https://docs.julialang.org/en/v1/base/base/#Base.macroexpand)
macro allows us to see the result of applying a macro. Let's examine what
happens when we use the `@finch` macro (we've stripped line numbers from the
result to clean it up):

```jldoctest example1; filter=r"Finch.FinchNotation."
julia> Finch.regensym(Finch.striplines((@macroexpand @finch (C .= 0;
       for i in _
           C[i] = A[i] * B[i]
       end))))
quote
    _res_1 = (Finch.execute)((Finch.FinchNotation.block_instance)((Finch.FinchNotation.block_instance)((Finch.FinchNotation.declare_instance)((Finch.FinchNotation.tag_instance)(variable_instance(:C), (Finch.FinchNotation.finch_leaf_instance)(C)), literal_instance(0), (Finch.FinchNotation.literal_instance)(Finch.auto)), begin
                        let i = index_instance(i)
                            (Finch.FinchNotation.loop_instance)(i, Finch.FinchNotation.Auto(), (Finch.FinchNotation.assign_instance)((Finch.FinchNotation.access_instance)((Finch.FinchNotation.tag_instance)(variable_instance(:C), (Finch.FinchNotation.finch_leaf_instance)(C)), (Finch.FinchNotation.updater_instance)((Finch.FinchNotation.literal_instance)(initwrite)), (Finch.FinchNotation.tag_instance)(variable_instance(:i), (Finch.FinchNotation.finch_leaf_instance)(i))), (Finch.FinchNotation.literal_instance)(initwrite), (Finch.FinchNotation.call_instance)((Finch.FinchNotation.tag_instance)(variable_instance(:*), (Finch.FinchNotation.finch_leaf_instance)(*)), (Finch.FinchNotation.access_instance)((Finch.FinchNotation.tag_instance)(variable_instance(:A), (Finch.FinchNotation.finch_leaf_instance)(A)), (Finch.FinchNotation.reader_instance)(), (Finch.FinchNotation.tag_instance)(variable_instance(:i), (Finch.FinchNotation.finch_leaf_instance)(i))), (Finch.FinchNotation.access_instance)((Finch.FinchNotation.tag_instance)(variable_instance(:B), (Finch.FinchNotation.finch_leaf_instance)(B)), (Finch.FinchNotation.reader_instance)(), (Finch.FinchNotation.tag_instance)(variable_instance(:i), (Finch.FinchNotation.finch_leaf_instance)(i))))))
                        end
                    end), (Finch.FinchNotation.yieldbind_instance)(variable_instance(:C))); )
    begin
        C = _res_1[:C]
    end
    begin
        _res_1
    end
end
```

In the above output, `@finch` creates an AST of program instances, then calls
`Finch.execute` on it. A program instance is a struct that contains the program
to be executed along with its arguments. Although we can use the above
constructors (e.g. `loop_instance`) to make our own program instance, it is most
convenient to use the unexported macro `Finch.finch_program_instance`:

```jldoctest example1
julia> using Finch: @finch_program_instance

julia> prgm = Finch.@finch_program_instance (C .= 0;
       for i in _
           C[i] = A[i] * B[i]
       end;
       return C)
Finch program instance: begin
  tag(C, Tensor(SparseList(Element(0)))) .= 0
  for i = Auto()
    tag(C, Tensor(SparseList(Element(0))))[tag(i, i)] <<initwrite>>= tag(*, *)(tag(A, Tensor(SparseList(Element(0))))[tag(i, i)], tag(B, Tensor(Dense(Element(0))))[tag(i, i)])
  end
  return (tag(C, Tensor(SparseList(Element(0)))))
end
```

As we can see, our program instance contains not only the AST to be executed,
but also the data to execute the program with. The type of the program instance
contains only the program portion; there may be many program instances with
different inputs, but the same program type. We can run our program using
`Finch.execute`, which returns a `NamedTuple` of outputs.

```jldoctest example1; filter=r"Finch.FinchNotation."
julia> typeof(prgm)
Finch.FinchNotation.BlockInstance{Tuple{Finch.FinchNotation.DeclareInstance{Finch.FinchNotation.TagInstance{Finch.FinchNotation.VariableInstance{:C}, Tensor{SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0, Int64, Int64, Vector{Int64}}}}}, Finch.FinchNotation.LiteralInstance{0}, Finch.FinchNotation.LiteralInstance{Finch.FinchNotation.Auto()}}, Finch.FinchNotation.LoopInstance{Finch.FinchNotation.IndexInstance{:i}, Finch.FinchNotation.Auto, Finch.FinchNotation.AssignInstance{Finch.FinchNotation.AccessInstance{Finch.FinchNotation.TagInstance{Finch.FinchNotation.VariableInstance{:C}, Tensor{SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0, Int64, Int64, Vector{Int64}}}}}, Finch.FinchNotation.UpdaterInstance{Finch.FinchNotation.LiteralInstance{initwrite}}, Tuple{Finch.FinchNotation.TagInstance{Finch.FinchNotation.VariableInstance{:i}, Finch.FinchNotation.IndexInstance{:i}}}}, Finch.FinchNotation.LiteralInstance{initwrite}, Finch.FinchNotation.CallInstance{Finch.FinchNotation.TagInstance{Finch.FinchNotation.VariableInstance{:*}, Finch.FinchNotation.LiteralInstance{*}}, Tuple{Finch.FinchNotation.AccessInstance{Finch.FinchNotation.TagInstance{Finch.FinchNotation.VariableInstance{:A}, Tensor{SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0, Int64, Int64, Vector{Int64}}}}}, Finch.FinchNotation.ReaderInstance, Tuple{Finch.FinchNotation.TagInstance{Finch.FinchNotation.VariableInstance{:i}, Finch.FinchNotation.IndexInstance{:i}}}}, Finch.FinchNotation.AccessInstance{Finch.FinchNotation.TagInstance{Finch.FinchNotation.VariableInstance{:B}, Tensor{DenseLevel{Int64, ElementLevel{0, Int64, Int64, Vector{Int64}}}}}, Finch.FinchNotation.ReaderInstance, Tuple{Finch.FinchNotation.TagInstance{Finch.FinchNotation.VariableInstance{:i}, Finch.FinchNotation.IndexInstance{:i}}}}}}}}, Finch.FinchNotation.YieldBindInstance{Tuple{Finch.FinchNotation.TagInstance{Finch.FinchNotation.VariableInstance{:C}, Tensor{SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0, Int64, Int64, Vector{Int64}}}}}}}}}

julia> C = Finch.execute(prgm).C
5 Tensor{SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0, Int64, Int64, Vector{Int64}}}}:
  0
 24
  0
  0
 45
```

This functionality is sufficient for building finch kernels programatically. For
example, if we wish to define a function `pointwise_sum()` that takes the
pointwise sum of a variable number of vector inputs, we might implement it as
follows:

```jldoctest example1
julia> function pointwise_sum(As...)
           B = Tensor(Dense(Element(0)))
           isempty(As) && return B
           i = Finch.FinchNotation.index_instance(:i)
           A_vars = [
               Finch.FinchNotation.tag_instance(
                   Finch.FinchNotation.variable_instance(Symbol(:A, n)), As[n]
               ) for n in 1:length(As)
           ]
           #create a list of variable instances with different names to hold the input tensors
           ex = @finch_program_instance 0
           for A_var in A_vars
               ex = @finch_program_instance $A_var[i] + $ex
           end
           prgm = @finch_program_instance (B .= 0;
           for i in _
               B[i] = $ex
           end;
           return B)
           return Finch.execute(prgm).B
       end
pointwise_sum (generic function with 1 method)

julia> pointwise_sum([1, 2], [3, 4])
2 Tensor{DenseLevel{Int64, ElementLevel{0, Int64, Int64, Vector{Int64}}}}:
 4
 6
```

## Virtualization

Finch generates different code depending on the types of the arguments to the
program. For example, in the following program, Finch generates different code
depending on the types of `A` and `B`. In order to execute a program, Finch
builds a typed AST (Abstract Syntax Tree), then calls `Finch.execute` on it. The
AST object is just an instance of a program to execute, and contains the program
to execute along with the data to execute it.  The type of the program instance
contains only the program portion; there may be many program instances with
different inputs, but the same program type. During compilation, Finch uses the
type of the program to construct a more ergonomic representation, which is then
used to generate code. This process is called "virtualization".  All of the
Finch AST nodes have both instance and virtual representations. For example, the
literal `42` is represented as `Finch.FinchNotation.LiteralInstance(42)` and
then virtualized to `literal(42)`.  The virtualization process is implemented by
the `virtualize` function.

```jldoctest example2; setup = :(using Finch)
julia> A = Tensor(SparseList(Element(0)), [0, 2, 0, 0, 3]);

julia> B = Tensor(Dense(Element(0)), [11, 12, 13, 14, 15]);

julia> s = Scalar(0);

julia> typeof(A)
Tensor{SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0, Int64, Int64, Vector{Int64}}}}

julia> typeof(B)
Tensor{DenseLevel{Int64, ElementLevel{0, Int64, Int64, Vector{Int64}}}}

julia> inst = Finch.@finch_program_instance begin
           for i in _
               s[] += A[i]
           end
       end
Finch program instance: for i = Auto()
  tag(s, Scalar{0, Int64})[] <<tag(+, +)>>= tag(A, Tensor(SparseList(Element(0))))[tag(i, i)]
end

julia> typeof(inst)
Finch.FinchNotation.LoopInstance{Finch.FinchNotation.IndexInstance{:i}, Finch.FinchNotation.Auto, Finch.FinchNotation.AssignInstance{Finch.FinchNotation.AccessInstance{Finch.FinchNotation.TagInstance{Finch.FinchNotation.VariableInstance{:s}, Scalar{0, Int64}}, Finch.FinchNotation.UpdaterInstance{Finch.FinchNotation.TagInstance{Finch.FinchNotation.VariableInstance{:+}, Finch.FinchNotation.LiteralInstance{+}}}, Tuple{}}, Finch.FinchNotation.TagInstance{Finch.FinchNotation.VariableInstance{:+}, Finch.FinchNotation.LiteralInstance{+}}, Finch.FinchNotation.AccessInstance{Finch.FinchNotation.TagInstance{Finch.FinchNotation.VariableInstance{:A}, Tensor{SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0, Int64, Int64, Vector{Int64}}}}}, Finch.FinchNotation.ReaderInstance, Tuple{Finch.FinchNotation.TagInstance{Finch.FinchNotation.VariableInstance{:i}, Finch.FinchNotation.IndexInstance{:i}}}}}}

julia> Finch.virtualize(Finch.JuliaContext(), :inst, typeof(inst))
Finch program: for i = virtual(Finch.FinchNotation.Auto)
  tag(s, virtual(Finch.VirtualScalar))[] <<tag(+, +)>>= tag(A, virtual(Finch.VirtualFiber{Finch.VirtualSparseListLevel}))[tag(i, i)]
end

julia> @finch_code begin
           for i in _
               s[] += A[i]
           end
       end
quote
    s = (ex.bodies[1]).body.lhs.tns.bind
    s_val = s.val
    A_lvl = (ex.bodies[1]).body.rhs.tns.bind.lvl
    A_lvl_ptr = A_lvl.ptr
    A_lvl_idx = A_lvl.idx
    A_lvl_val = A_lvl.lvl.val
    A_lvl_q = A_lvl_ptr[1]
    A_lvl_q_stop = A_lvl_ptr[1 + 1]
    if A_lvl_q < A_lvl_q_stop
        A_lvl_i1 = A_lvl_idx[A_lvl_q_stop - 1]
    else
        A_lvl_i1 = 0
    end
    phase_stop = min(A_lvl_i1, A_lvl.shape)
    if phase_stop >= 1
        if A_lvl_idx[A_lvl_q] < 1
            A_lvl_q = Finch.scansearch(A_lvl_idx, 1, A_lvl_q, A_lvl_q_stop - 1)
        end
        while true
            A_lvl_i = A_lvl_idx[A_lvl_q]
            if A_lvl_i < phase_stop
                A_lvl_2_val = A_lvl_val[A_lvl_q]
                s_val = A_lvl_2_val + s_val
                A_lvl_q += 1
            else
                phase_stop_3 = min(phase_stop, A_lvl_i)
                if A_lvl_i == phase_stop_3
                    A_lvl_2_val = A_lvl_val[A_lvl_q]
                    s_val += A_lvl_2_val
                    A_lvl_q += 1
                end
                break
            end
        end
    end
    result = ()
    s.val = s_val
    result
end

julia> @finch_code begin
           for i in _
               s[] += B[i]
           end
       end
quote
    s = (ex.bodies[1]).body.lhs.tns.bind
    s_val = s.val
    B_lvl = (ex.bodies[1]).body.rhs.tns.bind.lvl
    B_lvl_val = B_lvl.lvl.val
    for i_3 = 1:B_lvl.shape
        B_lvl_q = (1 - 1) * B_lvl.shape + i_3
        B_lvl_2_val = B_lvl_val[B_lvl_q]
        s_val = B_lvl_2_val + s_val
    end
    result = ()
    s.val = s_val
    result
end
```

### The "virtual" IR Node

Users can also create their own virtual nodes to represent their custom types.
While most calls to virtualize result in a Finch IR Node, some objects, such as
tensors and dimensions, are virtualized to a `virtual` object, which holds the
custom virtual type.  These types may contain constants and other virtuals, as
well as reference variables in the scope of the executing context. Any aspect of
virtuals visible to Finch should be considered immutable, but virtuals may
reference mutable variables in the scope of the executing context.

```@docs
virtualize
FinchNotation.virtual
```

### Virtual Methods

Many methods have analogues we can call on the virtual version of the object.
For example, we can call `size` an an array, and `virtual_size` on a virtual
array. The virtual methods are used to generate code, so if they are pure they
may return an expression which computes the results, and if they have side
effects they may accept a context argument into which they can emit their
side-effecting code.

In addition to the special compiler methods which are prefixed `virtual_`, there
is also a function `virtual_call`, which is used to evaluate function calls on
Finch IR when it would result in a virtual object. The behavior should mirror
the concrete behavior of the corresponding function.

```@docs
virtual_call
```

## Working with Finch IR

Calling print on a finch program or program instance will print the
structure of the program as one would call constructors to build it. For
example,

```jldoctest example2; setup = :(using Finch)
julia> prgm_inst = Finch.@finch_program_instance for i in _
           s[] += A[i]
       end;

julia> println(prgm_inst)
loop_instance(index_instance(i), Finch.FinchNotation.Auto(), assign_instance(access_instance(tag_instance(variable_instance(:s), Scalar{0, Int64}(0)), updater_instance(tag_instance(variable_instance(:+), literal_instance(+)))), tag_instance(variable_instance(:+), literal_instance(+)), access_instance(tag_instance(variable_instance(:A), Tensor(SparseList{Int64}(Element{0, Int64, Int64}([2, 3]), 5, [1, 3], [2, 5]))), reader_instance(), tag_instance(variable_instance(:i), index_instance(i)))))

julia> prgm_inst
Finch program instance: for i = Auto()
  tag(s, Scalar{0, Int64})[] <<tag(+, +)>>= tag(A, Tensor(SparseList(Element(0))))[tag(i, i)]
end

julia> prgm = Finch.@finch_program for i in _
           s[] += A[i]
       end;

julia> println(prgm)
loop(index(i), virtual(Finch.FinchNotation.Auto()), assign(access(literal(Scalar{0, Int64}(0)), updater(literal(+))), literal(+), access(literal(Tensor(SparseList{Int64}(Element{0, Int64, Int64}([2, 3]), 5, [1, 3], [2, 5]))), reader(), index(i))))

julia> prgm
Finch program: for i = virtual(Finch.FinchNotation.Auto)
  Scalar{0, Int64}(0)[] <<+>>= Tensor(SparseList{Int64}(Element{0, Int64, Int64}([2, 3]), 5, [1, 3], [2, 5]))[i]
end
```

Both the virtual and instance representations of Finch IR define
[SyntaxInterface.jl](https://github.com/willow-ahrens/SyntaxInterface.jl) and
[AbstractTrees.jl](https://github.com/JuliaCollections/AbstractTrees.jl)
representations, so you can use the standard `operation`, `arguments`, `istree`, and `children` functions to inspect the structure of the program, as well as the rewriters defined by [RewriteTools.jl](https://github.com/willow-ahrens/RewriteTools.jl)

```jldoctest example2; setup = :(using Finch, AbstractTrees, SyntaxInterface, RewriteTools)
julia> using Finch.FinchNotation;

julia> PostOrderDFS(prgm)
PostOrderDFS{FinchNode}(loop(index(i), virtual(Auto()), assign(access(literal(Scalar{0, Int64}(0)), updater(literal(+))), literal(+), access(literal(Tensor(SparseList{Int64}(Element{0, Int64, Int64}([2, 3]), 5, [1, 3], [2, 5]))), reader(), index(i)))))

julia> (@capture prgm loop(~idx, ~ext, ~val))
true

julia> idx
Finch program: i
```
