begin
    B = ((ex.bodies[1]).bodies[1]).tns.bind
    B_ptr = B.colptr
    B_idx = B.rowval
    B_val = B.nzval
    A = ((ex.bodies[1]).bodies[2]).body.body.rhs.tns.bind
    A_m = A.m
    A_n = A.n
    A_ptr = A.colptr
    A_idx = A.rowval
    A_val = A.nzval
    B_qos_used = 0
    B_qos_alloc = 0
    resize!(B_ptr, A_n + 1)
    fill_range!(B_ptr, 0, 1, A_n + 1)
    B_ptr[1] = 1
    B_prev_pos = 0
    for j_4 = 1:A_n
        B_qos = B_qos_used + 1
        B_prev_pos < j_4 || throw(FinchProtocolError("SparseMatrixCSCs cannot be updated multiple times"))
        A_q = A_ptr[j_4]
        A_q_stop = A_ptr[j_4 + 1]
        if A_q < A_q_stop
            A_i1 = A_idx[A_q_stop - 1]
        else
            A_i1 = 0
        end
        phase_stop = min(A_i1, A_m)
        if phase_stop >= 1
            if A_idx[A_q] < 1
                A_q = Finch.scansearch(A_idx, 1, A_q, A_q_stop - 1)
            end
            while true
                A_i = A_idx[A_q]
                if A_i < phase_stop
                    A_val_2 = A_val[A_q]
                    if B_qos > B_qos_alloc
                        B_qos_alloc = max(B_qos_alloc << 1, 1)
                        Finch.resize_if_smaller!(B_idx, B_qos_alloc)
                        Finch.resize_if_smaller!(B_val, B_qos_alloc)
                    end
                    B_val[B_qos] = A_val_2
                    B_idx[B_qos] = A_i
                    B_qos += 1
                    B_prev_pos = j_4
                    A_q += 1
                else
                    phase_stop_3 = min(phase_stop, A_i)
                    if A_i == phase_stop_3
                        A_val_2 = A_val[A_q]
                        if B_qos > B_qos_alloc
                            B_qos_alloc = max(B_qos_alloc << 1, 1)
                            Finch.resize_if_smaller!(B_idx, B_qos_alloc)
                            Finch.resize_if_smaller!(B_val, B_qos_alloc)
                        end
                        B_val[B_qos] = A_val_2
                        B_idx[B_qos] = phase_stop_3
                        B_qos += 1
                        B_prev_pos = j_4
                        A_q += 1
                    end
                    break
                end
            end
        end
        B_ptr[j_4 + 1] += (B_qos - B_qos_used) - 1
        B_qos_used = B_qos - 1
    end
    resize!(B_ptr, A_n + 1)
    for p = 1:A_n
        B_ptr[p + 1] += B_ptr[p]
    end
    qos_alloc = B_ptr[A_n + 1] - 1
    resize!(B_idx, qos_alloc)
    resize!(B_val, qos_alloc)
    (B = (SparseMatrixCSC)(A_m, A_n, B_ptr, B_idx, B_val),)
end
