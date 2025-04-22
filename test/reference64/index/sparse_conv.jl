begin
    C_lvl = ((ex.bodies[1]).bodies[1]).tns.bind.lvl
    C_lvl_ptr = C_lvl.ptr
    C_lvl_idx = C_lvl.idx
    C_lvl_2 = C_lvl.lvl
    C_lvl_2_val = C_lvl_2.val
    A_lvl = ((((ex.bodies[1]).bodies[2]).body.body.rhs.args[1]).args[1]).tns.bind.lvl
    A_lvl_ptr = A_lvl.ptr
    A_lvl_idx = A_lvl.idx
    A_lvl_stop = A_lvl.shape
    A_lvl_2 = A_lvl.lvl
    A_lvl_2_val = A_lvl_2.val
    F_lvl = (((ex.bodies[1]).bodies[2]).body.body.rhs.args[3]).tns.bind.lvl
    F_lvl_stop = F_lvl.shape
    F_lvl_2 = F_lvl.lvl
    F_lvl_2_val = F_lvl_2.val
    C_lvl_qos_stop = 0
    Finch.resize_if_smaller!(C_lvl_ptr, 1 + 1)
    Finch.fill_range!(C_lvl_ptr, 0, 1 + 1, 1 + 1)
    C_lvl_qos = 0 + 1
    0 < 1 || throw(FinchProtocolError("SparseListLevels cannot be updated multiple times"))
    A_lvl_q = A_lvl_ptr[1]
    A_lvl_q_stop = A_lvl_ptr[1 + 1]
    if A_lvl_q < A_lvl_q_stop
        A_lvl_i1 = A_lvl_idx[A_lvl_q_stop - 1]
    else
        A_lvl_i1 = 0
    end
    phase_stop = min(A_lvl_i1, A_lvl_stop)
    if phase_stop >= 1
        if A_lvl_idx[A_lvl_q] < 1
            A_lvl_q = Finch.scansearch(A_lvl_idx, 1, A_lvl_q, A_lvl_q_stop - 1)
        end
        while true
            A_lvl_i = A_lvl_idx[A_lvl_q]
            if A_lvl_i < phase_stop
                A_lvl_2_val_2 = A_lvl_2_val[A_lvl_q]
                if C_lvl_qos > C_lvl_qos_stop
                    C_lvl_qos_stop = max(C_lvl_qos_stop << 1, 1)
                    Finch.resize_if_smaller!(C_lvl_idx, C_lvl_qos_stop)
                    Finch.resize_if_smaller!(C_lvl_2_val, C_lvl_qos_stop)
                    Finch.fill_range!(C_lvl_2_val, 0.0, C_lvl_qos, C_lvl_qos_stop)
                end
                C_lvldirty = false
                v_3 = -A_lvl_i
                phase_start_3 = max(1, -v_3 + -2)
                phase_stop_4 = min(F_lvl_stop, A_lvl_stop + -v_3 + -3)
                if phase_stop_4 >= phase_start_3
                    A_lvl_q_2 = A_lvl_ptr[1]
                    A_lvl_q_stop_2 = A_lvl_ptr[1 + 1]
                    if A_lvl_q_2 < A_lvl_q_stop_2
                        A_lvl_i1_2 = A_lvl_idx[A_lvl_q_stop_2 - 1]
                    else
                        A_lvl_i1_2 = 0
                    end
                    phase_stop_5 = min(phase_stop_4, -v_3 + -3 + A_lvl_i1_2)
                    if phase_stop_5 >= phase_start_3
                        if A_lvl_idx[A_lvl_q_2] < 3 + v_3 + phase_start_3
                            A_lvl_q_2 = Finch.scansearch(A_lvl_idx, 3 + v_3 + phase_start_3, A_lvl_q_2, A_lvl_q_stop_2 - 1)
                        end
                        while true
                            A_lvl_i_2 = A_lvl_idx[A_lvl_q_2]
                            phase_stop_6 = -v_3 + -3 + A_lvl_i_2
                            if phase_stop_6 < phase_stop_5
                                A_lvl_2_val_3 = A_lvl_2_val[A_lvl_q_2]
                                F_lvl_q = (1 - 1) * F_lvl_stop + phase_stop_6
                                F_lvl_2_val_2 = F_lvl_2_val[F_lvl_q]
                                C_lvldirty = true
                                C_lvl_2_val[C_lvl_qos] = (A_lvl_2_val_2 != 0) * F_lvl_2_val_2 * coalesce(A_lvl_2_val_3, 0) + C_lvl_2_val[C_lvl_qos]
                                A_lvl_q_2 += 1
                            else
                                phase_stop_7 = min(phase_stop_5, -v_3 + -3 + A_lvl_i_2)
                                if A_lvl_i_2 == 3 + v_3 + phase_stop_7
                                    A_lvl_2_val_3 = A_lvl_2_val[A_lvl_q_2]
                                    F_lvl_q = (1 - 1) * F_lvl_stop + phase_stop_7
                                    F_lvl_2_val_3 = F_lvl_2_val[F_lvl_q]
                                    C_lvldirty = true
                                    C_lvl_2_val[C_lvl_qos] += (A_lvl_2_val_2 != 0) * F_lvl_2_val_3 * coalesce(A_lvl_2_val_3, 0)
                                    A_lvl_q_2 += 1
                                end
                                break
                            end
                        end
                    end
                end
                if C_lvldirty
                    C_lvl_idx[C_lvl_qos] = A_lvl_i
                    C_lvl_qos += 1
                end
                A_lvl_q += 1
            else
                phase_stop_10 = min(phase_stop, A_lvl_i)
                if A_lvl_i == phase_stop_10
                    A_lvl_2_val_2 = A_lvl_2_val[A_lvl_q]
                    if C_lvl_qos > C_lvl_qos_stop
                        C_lvl_qos_stop = max(C_lvl_qos_stop << 1, 1)
                        Finch.resize_if_smaller!(C_lvl_idx, C_lvl_qos_stop)
                        Finch.resize_if_smaller!(C_lvl_2_val, C_lvl_qos_stop)
                        Finch.fill_range!(C_lvl_2_val, 0.0, C_lvl_qos, C_lvl_qos_stop)
                    end
                    C_lvldirty = false
                    v_5 = -phase_stop_10
                    phase_start_11 = max(1, -2 + -v_5)
                    phase_stop_12 = min(F_lvl_stop, A_lvl_stop + -3 + -v_5)
                    if phase_stop_12 >= phase_start_11
                        A_lvl_q_3 = A_lvl_ptr[1]
                        A_lvl_q_stop_3 = A_lvl_ptr[1 + 1]
                        if A_lvl_q_3 < A_lvl_q_stop_3
                            A_lvl_i1_3 = A_lvl_idx[A_lvl_q_stop_3 - 1]
                        else
                            A_lvl_i1_3 = 0
                        end
                        phase_stop_13 = min(phase_stop_12, -3 + -v_5 + A_lvl_i1_3)
                        if phase_stop_13 >= phase_start_11
                            if A_lvl_idx[A_lvl_q_3] < 3 + v_5 + phase_start_11
                                A_lvl_q_3 = Finch.scansearch(A_lvl_idx, 3 + v_5 + phase_start_11, A_lvl_q_3, A_lvl_q_stop_3 - 1)
                            end
                            while true
                                A_lvl_i_3 = A_lvl_idx[A_lvl_q_3]
                                phase_stop_14 = -3 + -v_5 + A_lvl_i_3
                                if phase_stop_14 < phase_stop_13
                                    A_lvl_2_val_4 = A_lvl_2_val[A_lvl_q_3]
                                    F_lvl_q_2 = (1 - 1) * F_lvl_stop + phase_stop_14
                                    F_lvl_2_val_4 = F_lvl_2_val[F_lvl_q_2]
                                    C_lvldirty = true
                                    C_lvl_2_val[C_lvl_qos] += (A_lvl_2_val_2 != 0) * F_lvl_2_val_4 * coalesce(A_lvl_2_val_4, 0)
                                    A_lvl_q_3 += 1
                                else
                                    phase_stop_15 = min(phase_stop_13, -3 + -v_5 + A_lvl_i_3)
                                    if A_lvl_i_3 == 3 + v_5 + phase_stop_15
                                        A_lvl_2_val_4 = A_lvl_2_val[A_lvl_q_3]
                                        F_lvl_q_2 = (1 - 1) * F_lvl_stop + phase_stop_15
                                        F_lvl_2_val_5 = F_lvl_2_val[F_lvl_q_2]
                                        C_lvldirty = true
                                        C_lvl_2_val[C_lvl_qos] += (A_lvl_2_val_2 != 0) * F_lvl_2_val_5 * coalesce(A_lvl_2_val_4, 0)
                                        A_lvl_q_3 += 1
                                    end
                                    break
                                end
                            end
                        end
                    end
                    if C_lvldirty
                        C_lvl_idx[C_lvl_qos] = phase_stop_10
                        C_lvl_qos += 1
                    end
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
    (C = Tensor((SparseListLevel){Int64}(ElementLevel{0.0, Float64, Int64}(C_lvl_2_val), A_lvl_stop, C_lvl_ptr, C_lvl_idx)),)
end
