using Finch

# scalar tensor
T1 = swizzle(Tensor(Element(0, 0)))
# one-element tensor
T2 = swizzle(Tensor([0]), 1)

println("Testing T1")
@show T1

println("reshape(T1) = ", reshape(T1))
println("reshape(T1, 1) = ", reshape(T1, 1))
println("reshape(T1, 1, 1) = ", reshape(T1, 1, 1))

println("\nTesting T2")
@show T2

println("reshape(T2) = ", reshape(T2))
println("reshape(T2, 1) = ", reshape(T2, 1))
println("reshape(T2, 1, 1) = ", reshape(T2, 1, 1))