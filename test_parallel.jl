using Finch

# Create a simple test case with some structure
m, n = 5, 5
A = Float64[1.0 2.0 3.0 0.0 5.0;
    4.0 0.0 6.0 7.0 0.0;
    0.0 8.0 9.0 0.0 10.0;
    11.0 0.0 12.0 13.0 0.0;
    0.0 14.0 0.0 15.0 16.0]

B = Float64[9.0 8.0 7.0 0.0 6.0;
    5.0 0.0 4.0 3.0 0.0;
    0.0 2.0 1.0 0.0 0.0;
    19.0 0.0 20.0 21.0 0.0;
    0.0 22.0 0.0 23.0 24.0]

_A = Tensor(Dense(SparseRunList(Element(0.0))), A)
_B = Tensor(Dense(SparseRunList(Element(0.0))), B)

# Test just the sequential version for now
_C_s = Tensor(Dense(SparseRunList(Element(0.0))))

println("Building sequential version...")
@finch mode = :fast begin
    _C_s .= 0.0
    for j in _, i in _
        _C_s[i, j] = _A[i, j] + _B[i, j]
    end
end

expected = A + B
println("\nSequential result check:")
mismatch_count = 0
for i in 1:m, j in 1:n
    if _C_s[i, j] != expected[i, j]
        mismatch_count += 1
        if mismatch_count <= 10
            println("Mismatch at ($i, $j): got $(_C_s[i, j]), expected $(expected[i, j])")
        end
    end
end
println("Total mismatches: $mismatch_count")

if mismatch_count == 0
    println("Sequential version works correctly!")
end
