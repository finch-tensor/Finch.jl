begin
    C_lvl = (ex.bodies[1]).tns.tns.lvl
    C_lvl_2 = C_lvl.lvl
    A_lvl = (((ex.bodies[2]).body.body.rhs.args[1]).args[1]).tns.tns.lvl
    A_lvl_2 = A_lvl.lvl
    F_lvl = (((ex.bodies[2]).body.body.rhs.args[3]).args[1]).tns.tns.lvl
    F_lvl_2 = F_lvl.lvl
    j_stop = A_lvl.shape + -3
    C_lvl_qos_stop = 0
    resize_if_smaller!(C_lvl.ptr, 1 + 1)
    fill_range!(C_lvl.ptr, 0, 1 + 1, 1 + 1)
    C_lvl_qos = 0 + 1
    A_lvl_q_2 = A_lvl.ptr[1]
    A_lvl_q_stop_2 = A_lvl.ptr[1 + 1]
    if A_lvl_q_2 < A_lvl_q_stop_2
        A_lvl_i1_2 = A_lvl.idx[A_lvl_q_stop_2 - 1]
    else
        A_lvl_i1_2 = 0
    end
    phase_stop = min(A_lvl.shape, A_lvl_i1_2)
    if phase_stop >= 1
        i = 1
        if A_lvl.idx[A_lvl_q] < 1
            A_lvl_q = Finch.scansearch(A_lvl.idx, 1, A_lvl_q, A_lvl_q_stop - 1)
        end
        while i <= phase_stop
            A_lvl_i_2 = A_lvl.idx[A_lvl_q_2]
            phase_stop_2 = min(phase_stop, A_lvl_i_2)
            if A_lvl_i_2 == phase_stop_2
                A_lvl_2_val_2 = A_lvl_2.val[A_lvl_q_2]
                if C_lvl_qos > C_lvl_qos_stop
                    C_lvl_qos_stop = max(C_lvl_qos_stop << 1, 1)
                    resize_if_smaller!(C_lvl.idx, C_lvl_qos_stop)
                    resize_if_smaller!(C_lvl_2.val, C_lvl_qos_stop)
                    fill_range!(C_lvl_2.val, 0.0, C_lvl_qos, C_lvl_qos_stop)
                end
                C_lvldirty = false
                s_2 = -phase_stop_2
                phase_start_9 = max(1, -2, -s_2 + -2)
                phase_stop_9 = min(j_stop, A_lvl.shape + -3 + -s_2, F_lvl.shape)
                if phase_stop_9 >= phase_start_9
                    A_lvl_q = A_lvl.ptr[1]
                    A_lvl_q_stop = A_lvl.ptr[1 + 1]
                    if A_lvl_q < A_lvl_q_stop
                        A_lvl_i1 = A_lvl.idx[A_lvl_q_stop - 1]
                    else
                        A_lvl_i1 = 0
                    end
                    phase_stop_10 = min(-3 + -s_2 + A_lvl_i1, phase_stop_9)
                    if phase_stop_10 >= phase_start_9
                        j = phase_start_9
                        if A_lvl.idx[A_lvl_q_2] < phase_start_9 + -s_2
                            A_lvl_q_2 = Finch.scansearch(A_lvl.idx, phase_start_9 + -s_2, A_lvl_q_2, A_lvl_q_stop_2 - 1)
                        end
                        while j <= phase_stop_10
                            A_lvl_i = A_lvl.idx[A_lvl_q]
                            phase_stop_11 = min(phase_stop_10, -3 + -s_2 + A_lvl_i)
                            if A_lvl_i == (phase_stop_11 + s_2) + +3
                                A_lvl_2_val_3 = A_lvl_2.val[A_lvl_q]
                                F_lvl_q = (1 - 1) * F_lvl.shape + phase_stop_11
                                F_lvl_2_val_2 = F_lvl_2.val[F_lvl_q]
                                C_lvldirty = true
                                C_lvl_2.val[C_lvl_qos] = C_lvl_2.val[C_lvl_qos] + (A_lvl_2_val_2 != 0) * coalesce(((Scalar){0.0, Float64}(A_lvl_2_val_3))[], 0) * coalesce(((Scalar){0, Int64}(F_lvl_2_val_2))[], 0)
                                A_lvl_q += 1
                            end
                            j = phase_stop_11 + 1
                        end
                    end
                end
                if C_lvldirty
                    C_lvl.idx[C_lvl_qos] = phase_stop_2
                    C_lvl_qos += 1
                end
                A_lvl_q_2 += 1
            end
            i = phase_stop_2 + 1
        end
    end
    C_lvl.ptr[1 + 1] = (C_lvl_qos - 0) - 1
    for p = 2:1 + 1
        C_lvl.ptr[p] += C_lvl.ptr[p - 1]
    end
    resize!(C_lvl.ptr, 1 + 1)
    qos = C_lvl.ptr[end] - 1
    resize!(C_lvl.idx, qos)
    resize!(C_lvl_2.val, qos)
    (C = Fiber((SparseListLevel){Int64, Int64}(C_lvl_2, A_lvl.shape, C_lvl.ptr, C_lvl.idx)),)
end