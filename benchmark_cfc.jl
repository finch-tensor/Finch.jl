using BenchmarkTools;
using Finch;


## Imagine: Three {0-3}-D matrices
## Operations: + and *? Maybe more? (Ex: min)

function matmul(B, C)
    return B*C
end

function matmul_einsum(B, C)
    return @einsum A[i, j] += B[i, k] * C[k, j]
end

function matadd(B, C)
    return B + C
end

function matadd_einsum(B, C)
    return @einsum A[i, j] += B[i, j] + C[i, j]
end

function matvectmul(B, x)
    return B * x 
end

function matvectmul_einsum(B, x)
    return @einsum y[i] += B[i, k] * x[k]
end

function tensormatmul(B, C)
    return B * C 
end

# relative to this guy

function tensormatmul_einsum(B, C)
    return @einsum A[i, j, k] += B[i, j, l] * C[l, k]
end

# 4 kernels
# compare this guy

function tensormatmul_einsum_reshape(B, C)
    B_p = reshape(B, (size(B)[1]*size(B)[2], size(B)[3]))
    @einsum A_p[ij, k] += B_p[ij, l] * C[l, k]
    A = reshape(A_p, (size(B)[1], size(B)[2], size(C)[2]))
    return A
end

sparsity = 0.01
row = 1_000
col = 1_000
tube = 100
# TODO: different size + sparsity!
A_tensor = fsprand(row, col, tube, sparsity)
B = fsprand(row, tube, col, sparsity)
C = fsprand(col, row, sparsity)
x = fsprand(row, sparsity)

display(sparsity)
display(@benchmark tensormatmul_einsum(B, C))
display(@benchmark tensormatmul_einsum_reshape(B, C))
# display(@benchmark matmul(B, C))
# display(@benchmark matmul_einsum(B, C))
# display(@benchmark matadd(B, C))
# display(@benchmark matadd_einsum(B, C))
# display(@benchmark matvectmul(B, x))
# display(@benchmark matvectmul_einsum(B, x))
# display(@benchmark tensormatmul(A_tensor, C))
# display(@benchmark tensormatmul_einsum(A_tensor, C))

# sparsity = 0.5
# # TODO: different size + sparsity!
# A_tensor = fsprand(6, 6, 6, sparsity)
# B = fsprand(6, 6, sparsity)
# C = fsprand(6, 6, sparsity)
# x = fsprand(6, sparsity)
# 
# display(sparsity)
# display(@benchmark matmul(B, C))
# display(@benchmark matmul_einsum(B, C))
# display(@benchmark matadd(B, C))
# display(@benchmark matadd_einsum(B, C))
# display(@benchmark matvectmul(B, x))
# display(@benchmark matvectmul_einsum(B, x))
# display(@benchmark tensormatmul(A_tensor, C))
# display(@benchmark tensormatmul_einsum(A_tensor, C))
# 
# sparsity = 0.75
# # TODO: different size + sparsity!
# A_tensor = fsprand(6, 6, 6, sparsity)
# B = fsprand(6, 6, sparsity)
# C = fsprand(6, 6, sparsity)
# x = fsprand(6, sparsity)
# 
# display(sparsity)
# display(@benchmark matmul(B, C))
# display(@benchmark matmul_einsum(B, C))
# display(@benchmark matadd(B, C))
# display(@benchmark matadd_einsum(B, C))
# display(@benchmark matvectmul(B, x))
# display(@benchmark matvectmul_einsum(B, x))
# display(@benchmark tensormatmul(A_tensor, C))
# display(@benchmark tensormatmul_einsum(A_tensor, C))
# 
# sparsity = 1.0
# # TODO: different size + sparsity!
# A_tensor = fsprand(6, 6, 6, sparsity)
# B = fsprand(6, 6, sparsity)
# C = fsprand(6, 6, sparsity)
# x = fsprand(6, sparsity)
# 
# display(sparsity)
# display(@benchmark matmul(B, C))
# display(@benchmark matmul_einsum(B, C))
# display(@benchmark matadd(B, C))
# display(@benchmark matadd_einsum(B, C))
# display(@benchmark matvectmul(B, x))
# display(@benchmark matvectmul_einsum(B, x))
# display(@benchmark tensormatmul(A_tensor, C))
# display(@benchmark tensormatmul_einsum(A_tensor, C))
# 
