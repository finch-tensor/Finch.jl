using Pkg

Pkg.add("FileIO")

using Finch
using FileIO

A_orig = load("www.abalip.com.jpg")

m, n = size(A_orig)
A = zeros(m, n)

for i in 1:m
    for j in 1:n
        c = A_orig[i, j]
        A[i, j] = c.r * 0.299 + c.g * 0.587 + c.b * 0.114
    end
end

B = reverse(A, dims=2)

_A = Tensor(Dense(SparseRunList(Element(0.0))), A)
_B = Tensor(Dense(SparseRunList(Element(0.0))), B)

dev = cpu(:t, 2)
_C = Tensor(Dense(Shard(dev, SparseRunList(Element(0.0)))))
_C_s = Tensor(Dense(SparseRunList(Element(0.0))))

@finch mode = :fast begin
    _C .= 0.0
    for j = parallel(_, dev), i = _
        _C[i, j] = _A[i, j] + _B[i, j]
    end
end

@finch mode = :fast begin
    _C_s .= 0.0
    for j = _, i = _
        _C_s[i, j] = _A[i, j] + _B[i, j]
    end
end

cnt = 0
for i = 1:m
    for j = 1:n
        if _C[i, j] != _C_s[i, j] && cnt < 50
            cnt += 1
            print("Mismatch at ($i, $j):\n\t_A = $(_A[i, j])\t_B = $(_B[i, j])\n\t_C = \t$(_C[i, j])\n\t_C_s = \t$(_C_s[i, j])\n")
        end
    end
end

println("Total incorrect entries: $cnt")