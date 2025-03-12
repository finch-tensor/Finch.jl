begin
    B_lvl = ((ex.bodies[1]).bodies[1]).tns.bind.lvl
    B_lvl_2 = B_lvl.lvl
    B_lvl_2_ptr = B_lvl_2.ptr
    B_lvl_2_idx = B_lvl_2.idx
    B_lvl_3 = B_lvl_2.lvl
    B_lvl_3_val = B_lvl_3.val
    w_lvl = (((ex.bodies[1]).bodies[2]).body.bodies[1]).tns.bind.lvl
    w_lvl_ptr = w_lvl.ptr
    w_lvl_tbl = w_lvl.tbl
    w_lvl_srt = w_lvl.srt
    w_lvl_qos_stop = (w_lvl_qos_fill = length(w_lvl.srt))
    w_lvl_2 = w_lvl.lvl
    w_lvl_2_val = w_lvl_2.val
    A_lvl = ((((ex.bodies[1]).bodies[2]).body.bodies[2]).body.body.rhs.args[1]).tns.bind.lvl
    A_lvl_stop = A_lvl.shape
    A_lvl_2 = A_lvl.lvl
    A_lvl_2_ptr = A_lvl_2.ptr
    A_lvl_2_idx = A_lvl_2.idx
    A_lvl_2_stop = A_lvl_2.shape
    A_lvl_3 = A_lvl_2.lvl
    A_lvl_3_val = A_lvl_3.val
    A_lvl_2_stop == A_lvl_stop || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl_2_stop) != $(A_lvl_stop))"))
    B_lvl_2_qos_fill = 0
    B_lvl_2_qos_stop = 0
    B_lvl_2_prev_pos = 0
    Finch.resize_if_smaller!(B_lvl_2_ptr, A_lvl_stop + 1)
    Finch.fill_range!(B_lvl_2_ptr, 0, 1 + 1, A_lvl_stop + 1)
    for j_4 = 1:A_lvl_stop
        A_lvl_q = (1 - 1) * A_lvl_stop + j_4
        B_lvl_q = (1 - 1) * A_lvl_stop + j_4
        for w_lvl_r = 1:w_lvl_qos_fill
            w_lvl_p = first(w_lvl_srt[w_lvl_r])
            w_lvl_ptr[w_lvl_p] = 0
            w_lvl_ptr[w_lvl_p + 1] = 0
            w_lvl_i = last(w_lvl_srt[w_lvl_r])
            w_lvl_q = (w_lvl_p - 1) * A_lvl_2_stop + w_lvl_i
            w_lvl_tbl[w_lvl_q] = false
            Finch.resize_if_smaller!(w_lvl_2_val, w_lvl_q)
            Finch.fill_range!(w_lvl_2_val, 0.0, w_lvl_q, w_lvl_q)
        end
        w_lvl_qos_fill = 0
        w_lvl_ptr[1] = 1
        w_lvlq_stop = 1A_lvl_2_stop
        Finch.resize_if_smaller!(w_lvl_ptr, 1 + 1)
        Finch.fill_range!(w_lvl_ptr, 0, 1 + 1, 1 + 1)
        w_lvlold = length(w_lvl_tbl) + 1
        Finch.resize_if_smaller!(w_lvl_tbl, w_lvlq_stop)
        Finch.fill_range!(w_lvl_tbl, false, w_lvlold, w_lvlq_stop)
        Finch.resize_if_smaller!(w_lvl_2_val, w_lvlq_stop)
        Finch.fill_range!(w_lvl_2_val, 0.0, w_lvlold, w_lvlq_stop)
        A_lvl_2_q = A_lvl_2_ptr[A_lvl_q]
        A_lvl_2_q_stop = A_lvl_2_ptr[A_lvl_q + 1]
        if A_lvl_2_q < A_lvl_2_q_stop
            A_lvl_2_i1 = A_lvl_2_idx[A_lvl_2_q_stop - 1]
        else
            A_lvl_2_i1 = 0
        end
        phase_stop = min(A_lvl_2_stop, A_lvl_2_i1)
        if phase_stop >= 1
            if A_lvl_2_idx[A_lvl_2_q] < 1
                A_lvl_2_q = Finch.scansearch(A_lvl_2_idx, 1, A_lvl_2_q, A_lvl_2_q_stop - 1)
            end
            while true
                A_lvl_2_i = A_lvl_2_idx[A_lvl_2_q]
                if A_lvl_2_i < phase_stop
                    A_lvl_3_val_2 = A_lvl_3_val[A_lvl_2_q]
                    A_lvl_q_2 = (1 - 1) * A_lvl_stop + A_lvl_2_i
                    A_lvl_2_q_2 = A_lvl_2_ptr[A_lvl_q_2]
                    A_lvl_2_q_stop_2 = A_lvl_2_ptr[A_lvl_q_2 + 1]
                    if A_lvl_2_q_2 < A_lvl_2_q_stop_2
                        A_lvl_2_i1_2 = A_lvl_2_idx[A_lvl_2_q_stop_2 - 1]
                    else
                        A_lvl_2_i1_2 = 0
                    end
                    phase_stop_3 = min(A_lvl_2_stop, A_lvl_2_i1_2)
                    if phase_stop_3 >= 1
                        if A_lvl_2_idx[A_lvl_2_q_2] < 1
                            A_lvl_2_q_2 = Finch.scansearch(A_lvl_2_idx, 1, A_lvl_2_q_2, A_lvl_2_q_stop_2 - 1)
                        end
                        while true
                            A_lvl_2_i_2 = A_lvl_2_idx[A_lvl_2_q_2]
                            if A_lvl_2_i_2 < phase_stop_3
                                A_lvl_3_val_3 = A_lvl_3_val[A_lvl_2_q_2]
                                w_lvl_q_2 = (1 - 1) * A_lvl_2_stop + A_lvl_2_i_2
                                w_lvl_2_val[w_lvl_q_2] = A_lvl_3_val_2 * A_lvl_3_val_3 + w_lvl_2_val[w_lvl_q_2]
                                if !(w_lvl_tbl[w_lvl_q_2])
                                    w_lvl_tbl[w_lvl_q_2] = true
                                    w_lvl_qos_fill += 1
                                    if w_lvl_qos_fill > w_lvl_qos_stop
                                        w_lvl_qos_stop = max(w_lvl_qos_stop << 1, 1)
                                        Finch.resize_if_smaller!(w_lvl_srt, w_lvl_qos_stop)
                                    end
                                    w_lvl_srt[w_lvl_qos_fill] = (1, A_lvl_2_i_2)
                                end
                                A_lvl_2_q_2 += 1
                            else
                                phase_stop_5 = min(phase_stop_3, A_lvl_2_i_2)
                                if A_lvl_2_i_2 == phase_stop_5
                                    A_lvl_3_val_3 = A_lvl_3_val[A_lvl_2_q_2]
                                    w_lvl_q_2 = (1 - 1) * A_lvl_2_stop + phase_stop_5
                                    w_lvl_2_val[w_lvl_q_2] += A_lvl_3_val_2 * A_lvl_3_val_3
                                    if !(w_lvl_tbl[w_lvl_q_2])
                                        w_lvl_tbl[w_lvl_q_2] = true
                                        w_lvl_qos_fill += 1
                                        if w_lvl_qos_fill > w_lvl_qos_stop
                                            w_lvl_qos_stop = max(w_lvl_qos_stop << 1, 1)
                                            Finch.resize_if_smaller!(w_lvl_srt, w_lvl_qos_stop)
                                        end
                                        w_lvl_srt[w_lvl_qos_fill] = (1, phase_stop_5)
                                    end
                                    A_lvl_2_q_2 += 1
                                end
                                break
                            end
                        end
                    end
                    A_lvl_2_q += 1
                else
                    phase_stop_7 = min(phase_stop, A_lvl_2_i)
                    if A_lvl_2_i == phase_stop_7
                        A_lvl_3_val_2 = A_lvl_3_val[A_lvl_2_q]
                        A_lvl_q_2 = (1 - 1) * A_lvl_stop + phase_stop_7
                        A_lvl_2_q_3 = A_lvl_2_ptr[A_lvl_q_2]
                        A_lvl_2_q_stop_3 = A_lvl_2_ptr[A_lvl_q_2 + 1]
                        if A_lvl_2_q_3 < A_lvl_2_q_stop_3
                            A_lvl_2_i1_3 = A_lvl_2_idx[A_lvl_2_q_stop_3 - 1]
                        else
                            A_lvl_2_i1_3 = 0
                        end
                        phase_stop_8 = min(A_lvl_2_stop, A_lvl_2_i1_3)
                        if phase_stop_8 >= 1
                            if A_lvl_2_idx[A_lvl_2_q_3] < 1
                                A_lvl_2_q_3 = Finch.scansearch(A_lvl_2_idx, 1, A_lvl_2_q_3, A_lvl_2_q_stop_3 - 1)
                            end
                            while true
                                A_lvl_2_i_3 = A_lvl_2_idx[A_lvl_2_q_3]
                                if A_lvl_2_i_3 < phase_stop_8
                                    A_lvl_3_val_4 = A_lvl_3_val[A_lvl_2_q_3]
                                    w_lvl_q_3 = (1 - 1) * A_lvl_2_stop + A_lvl_2_i_3
                                    w_lvl_2_val[w_lvl_q_3] = A_lvl_3_val_2 * A_lvl_3_val_4 + w_lvl_2_val[w_lvl_q_3]
                                    if !(w_lvl_tbl[w_lvl_q_3])
                                        w_lvl_tbl[w_lvl_q_3] = true
                                        w_lvl_qos_fill += 1
                                        if w_lvl_qos_fill > w_lvl_qos_stop
                                            w_lvl_qos_stop = max(w_lvl_qos_stop << 1, 1)
                                            Finch.resize_if_smaller!(w_lvl_srt, w_lvl_qos_stop)
                                        end
                                        w_lvl_srt[w_lvl_qos_fill] = (1, A_lvl_2_i_3)
                                    end
                                    A_lvl_2_q_3 += 1
                                else
                                    phase_stop_10 = min(phase_stop_8, A_lvl_2_i_3)
                                    if A_lvl_2_i_3 == phase_stop_10
                                        A_lvl_3_val_4 = A_lvl_3_val[A_lvl_2_q_3]
                                        w_lvl_q_3 = (1 - 1) * A_lvl_2_stop + phase_stop_10
                                        w_lvl_2_val[w_lvl_q_3] += A_lvl_3_val_2 * A_lvl_3_val_4
                                        if !(w_lvl_tbl[w_lvl_q_3])
                                            w_lvl_tbl[w_lvl_q_3] = true
                                            w_lvl_qos_fill += 1
                                            if w_lvl_qos_fill > w_lvl_qos_stop
                                                w_lvl_qos_stop = max(w_lvl_qos_stop << 1, 1)
                                                Finch.resize_if_smaller!(w_lvl_srt, w_lvl_qos_stop)
                                            end
                                            w_lvl_srt[w_lvl_qos_fill] = (1, phase_stop_10)
                                        end
                                        A_lvl_2_q_3 += 1
                                    end
                                    break
                                end
                            end
                        end
                        A_lvl_2_q += 1
                    end
                    break
                end
            end
        end
        resize!(w_lvl_ptr, 1 + 1)
        resize!(w_lvl_tbl, 1A_lvl_2_stop)
        resize!(w_lvl_srt, w_lvl_qos_fill)
        sort!(w_lvl_srt)
        w_lvl_p_prev = 0
        for w_lvl_r_2 = 1:w_lvl_qos_fill
            w_lvl_p_2 = first(w_lvl_srt[w_lvl_r_2])
            if w_lvl_p_2 != w_lvl_p_prev
                w_lvl_ptr[w_lvl_p_prev + 1] = w_lvl_r_2
                w_lvl_ptr[w_lvl_p_2] = w_lvl_r_2
            end
            w_lvl_p_prev = w_lvl_p_2
        end
        w_lvl_ptr[w_lvl_p_prev + 1] = w_lvl_qos_fill + 1
        w_lvl_qos_stop = w_lvl_qos_fill
        resize!(w_lvl_2_val, A_lvl_2_stop)
        B_lvl_2_qos = B_lvl_2_qos_fill + 1
        B_lvl_2_prev_pos < B_lvl_q || throw(FinchProtocolError("SparseListLevels cannot be updated multiple times"))
        w_lvl_r_3 = w_lvl_ptr[1]
        w_lvl_r_stop = w_lvl_ptr[1 + 1]
        if w_lvl_r_3 != 0 && w_lvl_r_3 < w_lvl_r_stop
            w_lvl_i_stop = last(w_lvl_srt[w_lvl_r_stop - 1])
        else
            w_lvl_i_stop = 0
        end
        phase_stop_13 = min(A_lvl_2_stop, w_lvl_i_stop)
        if phase_stop_13 >= 1
            while w_lvl_r_3 + 1 < w_lvl_r_stop && last(w_lvl_srt[w_lvl_r_3]) < 1
                w_lvl_r_3 += 1
            end
            while true
                w_lvl_i_2 = last(w_lvl_srt[w_lvl_r_3])
                if w_lvl_i_2 < phase_stop_13
                    w_lvl_q_4 = (1 - 1) * A_lvl_2_stop + w_lvl_i_2
                    w_lvl_2_val_2 = w_lvl_2_val[w_lvl_q_4]
                    if B_lvl_2_qos > B_lvl_2_qos_stop
                        B_lvl_2_qos_stop = max(B_lvl_2_qos_stop << 1, 1)
                        Finch.resize_if_smaller!(B_lvl_2_idx, B_lvl_2_qos_stop)
                        Finch.resize_if_smaller!(B_lvl_3_val, B_lvl_2_qos_stop)
                        Finch.fill_range!(B_lvl_3_val, 0.0, B_lvl_2_qos, B_lvl_2_qos_stop)
                    end
                    B_lvl_3_val[B_lvl_2_qos] = w_lvl_2_val_2
                    B_lvl_2_idx[B_lvl_2_qos] = w_lvl_i_2
                    B_lvl_2_qos += 1
                    B_lvl_2_prev_pos = B_lvl_q
                    w_lvl_r_3 += 1
                else
                    phase_stop_15 = min(phase_stop_13, w_lvl_i_2)
                    if w_lvl_i_2 == phase_stop_15
                        w_lvl_q_4 = (1 - 1) * A_lvl_2_stop + w_lvl_i_2
                        w_lvl_2_val_3 = w_lvl_2_val[w_lvl_q_4]
                        if B_lvl_2_qos > B_lvl_2_qos_stop
                            B_lvl_2_qos_stop = max(B_lvl_2_qos_stop << 1, 1)
                            Finch.resize_if_smaller!(B_lvl_2_idx, B_lvl_2_qos_stop)
                            Finch.resize_if_smaller!(B_lvl_3_val, B_lvl_2_qos_stop)
                            Finch.fill_range!(B_lvl_3_val, 0.0, B_lvl_2_qos, B_lvl_2_qos_stop)
                        end
                        B_lvl_3_val[B_lvl_2_qos] = w_lvl_2_val_3
                        B_lvl_2_idx[B_lvl_2_qos] = phase_stop_15
                        B_lvl_2_qos += 1
                        B_lvl_2_prev_pos = B_lvl_q
                        w_lvl_r_3 += 1
                    end
                    break
                end
            end
        end
        B_lvl_2_ptr[B_lvl_q + 1] += (B_lvl_2_qos - B_lvl_2_qos_fill) - 1
        B_lvl_2_qos_fill = B_lvl_2_qos - 1
    end
    resize!(B_lvl_2_ptr, A_lvl_stop + 1)
    for p = 1:A_lvl_stop
        B_lvl_2_ptr[p + 1] += B_lvl_2_ptr[p]
    end
    qos_stop = B_lvl_2_ptr[A_lvl_stop + 1] - 1
    resize!(B_lvl_2_idx, qos_stop)
    resize!(B_lvl_3_val, qos_stop)
    (B = Tensor((DenseLevel){Int64}((SparseListLevel){Int64}(ElementLevel{0.0, Float64, Int64}(B_lvl_3_val), A_lvl_2_stop, B_lvl_2_ptr, B_lvl_2_idx), A_lvl_stop)),)
end
