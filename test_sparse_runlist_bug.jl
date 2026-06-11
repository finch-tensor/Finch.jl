using Finch

# Create a simple test case
m, n = 3, 3
A = Float64[1.0 2.0 3.0; 4.0 0.0 6.0; 7.0 8.0 9.0]
B = Float64[9.0 8.0 7.0; 0.0 0.0 6.0; 3.0 2.0 1.0]

_A = Tensor(Dense(SparseRunList(Element(0.0))), A)
_B = Tensor(Dense(SparseRunList(Element(0.0))), B)

# Sequential version
_C_s = Tensor(Dense(SparseRunList(Element(0.0))))
@finch mode = :fast begin
    _C_s .= 0.0
    for j in _, i in _
        _C_s[i, j] = _A[i, j] + _B[i, j]
    end
end

println("Sequential result:")
for i in 1:m, j in 1:n
    println("_C_s[$i, $j] = $(_C_s[i, j])")
end

expected = A + B
println("\nExpected result:")
for i in 1:m, j in 1:n
    println("expected[$i, $j] = $(expected[i, j])")
end

# Check if it matches
println("\n\nDo they match?")
mismatch_count = 0
for i in 1:m, j in 1:n
    if _C_s[i, j] != expected[i, j]
        mismatch_count += 1
        println("Mismatch at ($i, $j): got $(_C_s[i, j]), expected $(expected[i, j])")
    end
end
println("Total mismatches: $mismatch_count")
