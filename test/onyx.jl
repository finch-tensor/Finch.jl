using Finch

A = Tensor(CSCFormat(), fsprand(10, 10, 0.2))
B = Tensor(rand(10, 10))
C = Tensor(rand(10, 10))


A = lazy(A)
B = lazy(B)
C = lazy(C)

D = B * C
E = A .* D

ctx = Finch.LogicInterpreter(verbose=true)
ctx_2 = Finch.DefaultOptimizer(ctx)
E = compute!(E, ctx=ctx_2)