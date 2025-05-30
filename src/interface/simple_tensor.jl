
# TODO: possibly have it inherit ( <: ) from something
struct SimpleTensor{T}
    shape::Array{Int64}
    data::T
    dim::Array{Array{Int64}}
end

function combine_two_dim(A_old, B_old, shared_dim)

    A_shape = A_old.shape
    B_shape = B_old.shape

    A_dim_idx = Set() # Array()
    B_dim_idx = Set() # Array()

    A_new_dims = Array(collect(A_shape))
    B_new_dims = Array(collect(B_shape))

    combined_dim = 1
    offset = 0
    for dim in shared_dim
        push!(A_dim_idx, dim[1])
        push!(B_dim_idx, dim[2])
        @assert A_shape[dim[1]] == B_shape[dim[2]] 

        deleteat!(A_new_dims, dim[1] - offset)
        deleteat!(B_new_dims, dim[2] - offset)
        offset += 1

        # new size
        combined_dim *= A_shape[dim[1]]
    end

    push!(A_new_dims, combined_dim)
    push!(B_new_dims, combined_dim)

    _A_new_dims = Array(collect(setdiff(Set(1:length(A_shape)), Set(A_dim_idx))))
    A_new_dims = [[dim] for dim in _A_new_dims]
    push!(A_new_dims, Array(collect(A_dim_idx)))

    _B_new_dims = Array(collect(setdiff(Set(1:length(B_shape)), Set(B_dim_idx))))
    B_new_dims = [[dim] for dim in _B_new_dims]
    push!(B_new_dims, Array(collect(B_dim_idx)))

    A = reshape(A_old.data, combined_dim)
    B = reshape(B_old.data, combined_dim)
    A_new = SimpleTensor(A_shape, A, Array{Array{Int}}(collect(A_new_dims)))
    B_new = SimpleTensor(B_shape, B, Array{Array{Int}}(B_new_dims))

    return A_new, B_new
end

function combine_just_d(D_out, solution, unique_D)
    # basically a broadcast operation

end

function do_op_simple_tensor(A, B, C, D)
    # D = A B C
    # FOR NOW: assume A is a mask

    A_size = size(A)
    A_n_idx = length(A_size)
    B_size = size(B)
    B_n_idx = length(B_size)
    C_size = size(C)
    C_n_idx = length(C_size)
    D_size = size(D)
    D_n_idx = length(D_size)

    # in all
    ABCD_idx = Set()

    # in three
    ABC_idx = Set()
    ABD_idx = Set()
    ACD_idx = Set()
    BCD_idx = Set()

    # in two
    AB_idx = Set()
    AC_idx = Set()
    AD_idx = Set()
    BC_idx = Set()
    BD_idx = Set()
    CD_idx = Set()

    # single index
    A_idx = Set()
    B_idx = Set()
    C_idx = Set()
    D_idx = Set()

    # used indices
    A_used_idx = Set()
    B_used_idx = Set()
    C_used_idx = Set()
    D_used_idx = Set()

    # a TON of loops to populate these sets

    # in four
    for a_i in 1:A_n_idx
        for b_i in 1:B_n_idx
            for c_i in 1:C_n_idx
                for d_i in 1:D_n_idx
                    if !(a_i in A_used_idx || b_i in B_used_idx || c_i in C_used_idx || d_i in D_used_idx) && (A_size[a_i] == B_size[b_i] && B_size[b_i] == C_size[c_i] && C_size[c_i] == D_size[d_i])
                        push!(ABCD_idx, (a_i, b_i, c_i, d_i))
                        push!(A_used_idx, a_i)
                        push!(B_used_idx, b_i)
                        push!(C_used_idx, c_i)
                        push!(D_used_idx, d_i)
                    end
                end
            end
        end
    end

    # in three
    for a_i in 1:A_n_idx
        for b_i in 1:B_n_idx
            for c_i in 1:C_n_idx
                if !(a_i in A_used_idx || b_i in B_used_idx || c_i in C_used_idx) && (A_size[a_i] == B_size[b_i] && B_size[b_i] == C_size[c_i])
                    push!(ABC_idx, (a_i, b_i, c_i))
                    push!(A_used_idx, a_i)
                    push!(B_used_idx, b_i)
                    push!(C_used_idx, c_i)
                end
            end
        end
    end

    for b_i in 1:B_n_idx
        for c_i in 1:C_n_idx
            for d_i in 1:D_n_idx
                if !(b_i in B_used_idx || c_i in C_used_idx || d_i in D_used_idx) && (B_size[b_i] == C_size[c_i] && C_size[c_i] == D_size[d_i])
                    push!(BCD_idx, (b_i, c_i, d_i))
                    push!(B_used_idx, b_i)
                    push!(C_used_idx, c_i)
                    push!(D_used_idx, d_i)
                end
            end
        end
    end


    for a_i in 1:A_n_idx
        for b_i in 1:B_n_idx
            for d_i in 1:D_n_idx
                if !(a_i in A_used_idx || b_i in B_used_idx || d_i in D_used_idx) && (A_size[a_i] == B_size[b_i] && B_size[b_i] == D_size[d_i])
                    push!(ABD_idx, (a_i, b_i, d_i))
                    push!(A_used_idx, a_i)
                    push!(B_used_idx, b_i)
                    push!(D_used_idx, d_i)
                end
            end
        end
    end

    for a_i in 1:A_n_idx
        for c_i in 1:C_n_idx
            for d_i in 1:D_n_idx
                if !(a_i in A_used_idx || c_i in C_used_idx || d_i in D_used_idx) && (A_size[a_i] == C_size[c_i] && C_size[c_i] == D_size[d_i])
                    push!(ACD_idx, (a_i, c_i, d_i))
                    push!(A_used_idx, a_i)
                    push!(C_used_idx, c_i)
                    push!(D_used_idx, d_i)
                end
            end
        end
    end

    # in two
    for a_i in 1:A_n_idx
        for b_i in 1:B_n_idx
            if !(a_i in A_used_idx || b_i in B_used_idx) && (A_size[a_i] == B_size[b_i])
                push!(AB_idx, (a_i, b_i))
                push!(A_used_idx, a_i)
                push!(B_used_idx, b_i)
            end
        end
    end

    for a_i in 1:A_n_idx
        for c_i in 1:C_n_idx
            if !(a_i in A_used_idx || c_i in C_used_idx) && (A_size[a_i] == C_size[c_i])
                push!(AC_idx, (a_i, c_i))
                push!(A_used_idx, a_i)
                push!(C_used_idx, c_i)
            end
        end
    end

    for a_i in 1:A_n_idx
        for d_i in 1:D_n_idx
            if !(a_i in A_used_idx || d_i in D_used_idx) && (A_size[a_i] == D_size[d_i])
                push!(AD_idx, (a_i, d_i))
                push!(A_used_idx, a_i)
                push!(D_used_idx, d_i)
            end
        end
    end

    for b_i in 1:B_n_idx
        for c_i in 1:C_n_idx
            if !(b_i in B_used_idx || c_i in C_used_idx) && (B_size[b_i] == C_size[c_i])
                push!(BC_idx, (b_i, c_i))
                push!(B_used_idx, b_i)
                push!(C_used_idx, c_i)
            end
        end
    end

    for b_i in 1:B_n_idx
        for d_i in 1:D_n_idx
            if !(b_i in B_used_idx || d_i in D_used_idx) && (B_size[b_i] == D_size[d_i])
                push!(BD_idx, (b_i, d_i))
                push!(B_used_idx, b_i)
                push!(D_used_idx, d_i)
            end
        end
    end

    for c_i in 1:C_n_idx
        for d_i in 1:D_n_idx
            if !(c_i in C_used_idx || d_i in D_used_idx) && (C_size[c_i] == D_size[d_i])
                push!(CD_idx, (c_i, d_i))
                push!(C_used_idx, c_i)
                push!(D_used_idx, d_i)
            end
        end
    end

    # single index
    for a_i in 1:A_n_idx
        if !(a_i in A_used_idx)
            push!(A_idx, a_i)
            push!(A_used_idx, a_i)
        end
    end

    for b_i in 1:B_n_idx
        if !(b_i in B_used_idx)
            push!(B_idx, b_i)
            push!(B_used_idx, b_i)
        end
    end

    for c_i in 1:C_n_idx
        if !(c_i in C_used_idx)
            push!(C_idx, c_i)
            push!(C_used_idx, c_i)
        end
    end

    for d_i in 1:D_n_idx
        if !(d_i in D_used_idx)
            push!(D_idx, d_i)
            push!(D_used_idx, d_i)
        end
    end

    #### TIME FOR THE ACTUAL COMPUTATION!

    A_SimpleTensor = SimpleTensor(Array(collect(A_size)), A, Array{Array{Int64}, 1}([ [dim] for dim in 1:A_n_idx ]))
    B_SimpleTensor = SimpleTensor(Array(collect(B_size)), B, Array{Array{Int64}, 1}([ [dim] for dim in 1:B_n_idx ]))
    C_SimpleTensor = SimpleTensor(Array(collect(C_size)), C, Array{Array{Int64}, 1}([ [dim] for dim in 1:C_n_idx ]))
    D_SimpleTensor = SimpleTensor(Array(collect(D_size)), D, Array{Array{Int64}, 1}([ [dim] for dim in 1:D_n_idx ]))


end
