using Pkg
Pkg.add("StatsBase")
Pkg.add("BenchmarkTools")
Pkg.add("SparseArrays")
Pkg.add("Finch")
Pkg.add("JSON3")
Pkg.add("HDF5")

using StatsBase
using BenchmarkTools
using SparseArrays
using Finch
using JSON3
using HDF5


function custom_coo_setup(M)
    # define the exact type of u to remove "Any"
    u = Vector{Tuple{Int, Int, Float64}}()
    i_d, j_d = size(M)

    for i in 1:i_d
        for j in 1:j_d
            if M[i, j] != 0
                push!(u, (i, j, M[i, j]))
            end
        end
    end

    return u
end

function custom_coo_mul(M, X)
    # use @inbound to avoid bound check when accessing indices (y[i, d], X[j, d])
    @inbounds begin
        # @fastmath to allow floating point optimizations that are correct for real numbers
        @fastmath begin
            y = zeros(size(X, 1), size(X, 2))
            s = size(X, 2)
            for (i, j, v) in M
                for d in 1:s
                    y[i, d] += v * X[j, d]
                end
            end
        end
    end
    return y
end


function custom_coo(M, X, sol)
    m_coo = custom_coo_setup(M)

    bench_results = @benchmark custom_coo_mul($m_coo, $X)
    temp = custom_coo_mul(m_coo, X)
    if isapprox(temp, sol; rtol=1e-8, atol=1e-12)
        return (minimum(bench_results.times))
    else
        return -1
    end
end


function custom_coo_sep_mul(I, J, V, X)
    @inbounds begin
        @fastmath begin
            y = zeros(size(X, 1), size(X, 2))
            size_m = size(I, 1)
            s = size(X, 2)
            for p in 1:size_m
                i_p = I[p]
                j_p = J[p]
                v_p = V[p]

                # add potential SIMD
                @simd for d in 1:s
                    y[i_p, d] += v_p * X[j_p, d]
                end
            end
        end
    end
    return y
end


function unzip(A)
    return map(x->getfield.(A, x), fieldnames(eltype(A)))
end


function custom_coo_sep(M, X, sol)
    m_coo = custom_coo_setup(M)
    I, J, V = unzip(m_coo)

    bench_results = @benchmark custom_coo_sep_mul($I, $J, $V, $X)
    temp = custom_coo_sep_mul(I, J, V, X)
    if isapprox(temp, sol; rtol=1e-8, atol=1e-12)
        return (minimum(bench_results.times))
    else
        return -1
    end
end


function custom_csc_mul(A::SparseMatrixCSC, X)
    @inbounds begin
        @fastmath begin
            # work with permuted matrix for row-major
            X_t = permutedims(X)
            ptr = A.colptr
            idx = A.rowval
            val = A.nzval

            m, n = size(A)
            n, d = size(X)

            Y_t = zeros(d, m)

            for j in 1:n
                st = ptr[j]
                ed = ptr[j+1] - 1
                for p in st:ed
                    i = idx[p]
                    v = val[p]
                    for k in 1:d
                        Y_t[k, i] += v * X_t[k, j]
                    end
                end
            end
        end
    end
    return permutedims(Y_t)
end


function custom_csc(M, X, sol)
    m_csc = sparse(M)

    bench_results = @benchmark custom_csc_mul($m_csc, $X)
    temp = custom_csc_mul(m_csc, X)
    if isapprox(temp, sol; rtol=1e-8, atol=1e-12)
        return (minimum(bench_results.times))
    else
        return -1
    end
end


function bytemap(M)
    i_d = size(M, 1)
    j_d = size(M, 2)
    B = zeros(Int8, i_d, j_d)
    val = Vector{Float64}()
    for i in 1:i_d
        for j in 1:j_d
            if M[i, j] != 0.0
                B[i, j] = 1
                push!(val, M[i, j])
            else
                B[i, j] = 0
            end
        end
    end
    return B, val 
end


function custom_bytemap_mul(B, val, X)
    i_d = size(B, 1)
    j_d = size(B, 2)
    k_d = size(X, 2)
    Y_t = zeros(k_d, i_d)
    X_t = transpose(X)

    cnt = 1
    @inbounds begin
        @fastmath begin
            for i in 1:i_d
                for j in 1:j_d
                    if B[i, j] == 1
                        v = val[cnt]
                        for k in 1:k_d
                            Y_t[k, i] += v * X_t[k, j]
                        end
                        cnt += 1
                    end
                end
            end
        end
    end

    return transpose(Y_t)
end


function custom_bytemap(M, X, sol)
    B, val = bytemap(M)

    bench_results = @benchmark custom_bytemap_mul($B, $val, $X)
    temp = custom_bytemap_mul(B, val, X)
    if isapprox(temp, sol; rtol=1e-8, atol=1e-12)
        return (minimum(bench_results.times))
    else
        return -1
    end
end


function bitmap(M)
    i_d = size(M, 1)
    j_d = size(M, 2)

    b_j = ceil(Int64, j_d / 64)

    B = zeros(UInt64, i_d, b_j)
    val = Vector{Float64}()
    for i in 1:i_d
        for j in 1:j_d
            if M[i, j] != 0.0
                push!(val, M[i, j])
                B[i, div(j - 1, 64) + 1] |= 1 << (63 - ((j - 1) % 64))
            end
        end
    end
    return B, val 
end


function custom_bitmap_mul(B, val, X)
    i_dim = size(B, 1)
    j_wdim = size(B, 2)
    k_dim = size(X, 2)
    Y_t = zeros(k_dim, i_dim)
    X_t = transpose(X)

    cnt = 1
    @inbounds begin
        @fastmath begin
            for i in 1:i_dim
                for j_w in 1:j_wdim
                    word = B[i, j_w]
                    if word == 0
                        continue
                    end

                    bit_id = 0
                    while word != 0
                        bit_id += 1
                        bit_msk = 1 << (64 - bit_id)
                        if word & bit_msk == 0
                            continue
                        else
                            word &= ~bit_msk
                            v = val[cnt]
                            j = (j_w - 1) * 64 + bit_id
                            for k in 1:k_dim
                                Y_t[k, i] += v * X_t[k, j]
                            end
                            cnt += 1
                        end
                    end
                end
            end
        end
    end

    return transpose(Y_t)
end


function custom_bitmap(M, X, sol)
    B, val = bitmap(M)

    bench_results = @benchmark custom_bitmap_mul($B, $val, $X)
    temp = custom_bitmap_mul(B, val, X)
    if isapprox(temp, sol; rtol=1e-8, atol=1e-12)
        return (minimum(bench_results.times))
    else
        return -1
    end
end


function main()
    N = 250
    eps = 1e-6

    indices = exp10.(range(log10(eps), log10(0.5), length=N))

    X = fread("ben/x.bsp.h5")


    # # custom coo
    # pairs_coo = []
    # println("For custom coo multiplication:")
    # k = -1
    # for i in indices
    #     # reduce number of runs
    #     k += 1
    #     if k % 10 != 0
    #         continue
    #     end

    #     M_dense = fread(string("ben/m", i, ".bsp.h5"))
    #     sol = Array(fread(string("ben/sol", i, ".bsp.h5")))
    #     X_dense = Array(X)

    #     println("\tDensity: ", i)
    #     M_dense = Array(M_dense)

    #     local custom_coo_val = custom_coo(M_dense, X_dense, sol)
    #     if custom_coo_val != -1
    #         println("\t\tTime: \t", custom_coo_val, "ns")
    #         push!(pairs_coo, (i, custom_coo_val))
    #     else
    #         println("\t\tcustom COO: matmul resulted in wrong answer.")
    #     end
    # end

    # open("res/1005/custom-coo-opt.json", "w") do f
    #     JSON3.write(f, pairs_coo)
    # end


    # # custom coo with decoupled tuples
    # pairs_coo_sep = []
    # println("For custom coo with decoupled tuples multiplication:")
    # k = -1
    # for i in indices
    #     # reduce number of runs
    #     k += 1
    #     if k % 10 != 0
    #         continue
    #     end

    #     M_dense = fread(string("ben/m", i, ".bsp.h5"))
    #     sol = Array(fread(string("ben/sol", i, ".bsp.h5")))
    #     X_dense = Array(X)

    #     println("\tDensity: ", i)
    #     M_dense = Array(M_dense)

    #     local custom_coo_sep_val = custom_coo_sep(M_dense, X_dense, sol)
    #     if custom_coo_sep_val != -1
    #         println("\t\tTime: \t", custom_coo_sep_val, "ns")
    #         push!(pairs_coo_sep, (i, custom_coo_sep_val))
    #     else
    #         println("\t\tcustom COO sep: matmul resulted in wrong answer.")
    #     end
    # end

    # open("res/1005/custom-coo-sep.json", "w") do f
    #     JSON3.write(f, pairs_coo_sep)
    # end


    # # custom csc
    # pairs_csc = []
    # println("For custom csc multiplication:")
    # # k = -1
    # for i in indices
    #     # # reduce number of runs
    #     # k += 1
    #     # if k % 10 != 0
    #     #     continue
    #     # end

    #     M_dense = fread(string("ben/m", i, ".bsp.h5"))
    #     sol = Array(fread(string("ben/sol", i, ".bsp.h5")))
    #     X_dense = Array(X)

    #     println("\tDensity: ", i)
    #     M_dense = Array(M_dense)

    #     local custom_csc_val = custom_csc(M_dense, X_dense, sol)
    #     if custom_csc_val != -1
    #         println("\t\tTime: \t", custom_csc_val, "ns")
    #         push!(pairs_csc, (i, custom_csc_val))
    #     else
    #         println("\t\tcustom CSC: matmul resulted in wrong answer.")
    #     end
    # end

    # open("res/1005/custom-csc.json", "w") do f
    #     JSON3.write(f, pairs_csc)
    # end


    # custom bytemap
    pairs_bm = []
    println("For custom bytemap multiplication:")
    k = -1
    for i in indices
        # reduce number of runs
        k += 1
        if k % 10 != 0
            continue
        end

        M_dense = fread(string("ben/m", i, ".bsp.h5"))
        sol = Array(fread(string("ben/sol", i, ".bsp.h5")))
        X_dense = Array(X)

        println("\tDensity: ", i)
        M_dense = Array(M_dense)

        local custom_bm_val = custom_bytemap(M_dense, X_dense, sol)
        if custom_bm_val != -1
            println("\t\tTime: \t", custom_bm_val, "ns")
            push!(pairs_bm, (i, custom_bm_val))
        else
            println("\t\tcustom bytemap: matmul resulted in wrong answer.")
        end
    end

    open("res/1010/custom-bytem.json", "w") do f
        JSON3.write(f, pairs_bm)
    end


    # custom bitmap
    pairs_bitm = []
    println("For custom bitmap multiplication:")
    k = -1
    for i in indices
        # reduce number of runs
        k += 1
        if k % 10 != 0
            continue
        end

        M_dense = fread(string("ben/m", i, ".bsp.h5"))
        sol = Array(fread(string("ben/sol", i, ".bsp.h5")))
        X_dense = Array(X)

        println("\tDensity: ", i)
        M_dense = Array(M_dense)

        local custom_bm_val = custom_bytemap(M_dense, X_dense, sol)
        if custom_bm_val != -1
            println("\t\tTime: \t", custom_bm_val, "ns")
            push!(pairs_bm, (i, custom_bm_val))
        else
            println("\t\tcustom bitmap: matmul resulted in wrong answer.")
        end
    end

    open("res/1019/custom-bitm.json", "w") do f
        JSON3.write(f, pairs_bitm)
    end    
end

main()