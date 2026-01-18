begin
    C_lvl = ((ex.bodies[1]).bodies[1]).tns.bind.lvl
    C_lvl_ptr = C_lvl.ptr
    C_lvl_idx = C_lvl.idx
    C_lvl_2 = C_lvl.lvl
    C_lvl_2_val = C_lvl_2.val
    A_lvl = ((ex.bodies[1]).bodies[2]).body.rhs.tns.bind.lvl
    A_lvl_ptr = A_lvl.ptr
    A_lvl_idx = A_lvl.idx
    A_lvl_2 = A_lvl.lvl
    A_lvl_2_val = A_lvl_2.val
    C_lvl_qos_stop = 0
    Finch.resize_if_smaller!(C_lvl_ptr, 1 + 1)
    Finch.fill_range!(C_lvl_ptr, 0, 1 + 1, 1 + 1)
    C_lvl_qos = 0 + 1
    0 < 1 || throw((Finch.FinchProtocolError)("SparseListLevels cannot be updated multiple times"))
    A_lvl_q = A_lvl_ptr[1]
    A_lvl_q_stop = A_lvl_ptr[1 + 1]
    if A_lvl_q < A_lvl_q_stop
        A_lvl_i1 = A_lvl_idx[A_lvl_q_stop - 1]
    else
        A_lvl_i1 = 0
    end
    phase_stop = min(3, A_lvl_i1 + -1)
    if phase_stop >= 1
        if A_lvl_idx[A_lvl_q] < 1 + 1
            A_lvl_q = Finch.scansearch(A_lvl_idx, 1 + 1, A_lvl_q, A_lvl_q_stop - 1)
        end
        while true
            A_lvl_i = A_lvl_idx[A_lvl_q]
            phase_stop_2 = -1 + A_lvl_i
            if phase_stop_2 < phase_stop
                A_lvl_2_val_2 = A_lvl_2_val[A_lvl_q]
                if C_lvl_qos > C_lvl_qos_stop
                    C_lvl_qos_stop = max(C_lvl_qos_stop << 1, 1)
                    Finch.resize_if_smaller!(C_lvl_idx, C_lvl_qos_stop)
                    Finch.resize_if_smaller!(C_lvl_2_val, C_lvl_qos_stop)
                    Finch.fill_range!(C_lvl_2_val, 0.0, C_lvl_qos, C_lvl_qos_stop)
                end
                C_lvl_2_val[C_lvl_qos] = A_lvl_2_val_2
                C_lvl_idx[C_lvl_qos] = phase_stop_2
                C_lvl_qos += 1
                A_lvl_q += 1
            else
                phase_stop_3 = min(phase_stop, -1 + A_lvl_i)
                if A_lvl_i == 1 + phase_stop_3
                    A_lvl_2_val_2 = A_lvl_2_val[A_lvl_q]
                    if C_lvl_qos > C_lvl_qos_stop
                        C_lvl_qos_stop = max(C_lvl_qos_stop << 1, 1)
                        Finch.resize_if_smaller!(C_lvl_idx, C_lvl_qos_stop)
                        Finch.resize_if_smaller!(C_lvl_2_val, C_lvl_qos_stop)
                        Finch.fill_range!(C_lvl_2_val, 0.0, C_lvl_qos, C_lvl_qos_stop)
                    end
                    C_lvl_2_val[C_lvl_qos] = A_lvl_2_val_2
                    C_lvl_idx[C_lvl_qos] = phase_stop_3
                    C_lvl_qos += 1
                    A_lvl_q += 1
                end
                break
            end
        end
    end
    C_lvl_ptr[1 + 1] += (C_lvl_qos - 0) - 1
    resize!(C_lvl_ptr, 1 + 1)
    for p = 1:1
        C_lvl_ptr[p + 1] += C_lvl_ptr[p]
    end
    qos_stop = C_lvl_ptr[1 + 1] - 1
    resize!(C_lvl_idx, qos_stop)
    resize!(C_lvl_2_val, qos_stop)
    (C = Tensor((SparseListLevel){Int64}(ElementLevel{0.0, Float64, Int64}(C_lvl_2_val), 3, C_lvl_ptr, C_lvl_idx)),)
end
