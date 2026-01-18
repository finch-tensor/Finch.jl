quote
    tmp_lvl = ((ex.bodies[1]).bodies[1]).tns.bind.lvl
    tmp_lvl_ptr = tmp_lvl.ptr
    tmp_lvl_right = tmp_lvl.right
    tmp_lvl_2 = tmp_lvl.lvl
    tmp_lvl_val = tmp_lvl.lvl.val
    tmp_lvl_3 = tmp_lvl.buf
    tmp_lvl_val_2 = tmp_lvl.buf.val
    ref_lvl = ((ex.bodies[1]).bodies[2]).body.rhs.tns.bind.lvl
    ref_lvl_ptr = ref_lvl.ptr
    ref_lvl_idx = ref_lvl.idx
    ref_lvl_val = ref_lvl.lvl.val
    tmp_lvl_qos_fill = 0
    tmp_lvl_qos_stop = 0
    tmp_lvl_i_prev = 1 - 1
    Finch.resize_if_smaller!(tmp_lvl_ptr, 1 + 1)
    Finch.fill_range!(tmp_lvl_ptr, 1, 1 + 1, 1 + 1)
    tmp_lvl_qos = 0 + 1
    1 <= 1 || throw(FinchProtocolError("RunListLevels cannot be updated multiple times"))
    tmp_lvl_i_prev_3 = tmp_lvl_i_prev
    if 1 < 1
        tmp_lvl_qos += (1 - 1) - 1
        tmp_lvl_qos += tmp_lvl_i_prev < ref_lvl.shape
        tmp_lvl_i_prev_3 = 1 - 1
    end
    tmp_lvl_qos_set = tmp_lvl_qos
    ref_lvl_q = ref_lvl_ptr[1]
    ref_lvl_q_stop = ref_lvl_ptr[1 + 1]
    if ref_lvl_q < ref_lvl_q_stop
        ref_lvl_i1 = ref_lvl_idx[ref_lvl_q_stop - 1]
    else
        ref_lvl_i1 = 0
    end
    phase_stop = min(ref_lvl_i1, ref_lvl.shape)
    if phase_stop >= 1
        if ref_lvl_idx[ref_lvl_q] < 1
            ref_lvl_q = Finch.scansearch(ref_lvl_idx, 1, ref_lvl_q, ref_lvl_q_stop - 1)
        end
        while true
            ref_lvl_i = ref_lvl_idx[ref_lvl_q]
            if ref_lvl_i < phase_stop
                ref_lvl_2_val = ref_lvl_val[ref_lvl_q]
                tmp_lvl_qos_3 = tmp_lvl_qos + (tmp_lvl_i_prev_3 < ref_lvl_i - 1)
                if tmp_lvl_qos_3 > tmp_lvl_qos_stop
                    tmp_lvl_qos_2 = tmp_lvl_qos_stop + 1
                    while tmp_lvl_qos_3 > tmp_lvl_qos_stop
                        tmp_lvl_qos_stop = max(tmp_lvl_qos_stop << 1, 1)
                    end
                    Finch.resize_if_smaller!(tmp_lvl_right, tmp_lvl_qos_stop)
                    Finch.fill_range!(tmp_lvl_right, ref_lvl.shape, tmp_lvl_qos_2, tmp_lvl_qos_stop)
                    Finch.resize_if_smaller!(tmp_lvl_val_2, tmp_lvl_qos_stop)
                    Finch.fill_range!(tmp_lvl_val_2, false, tmp_lvl_qos_2, tmp_lvl_qos_stop)
                end
                tmp_lvl_val_2[tmp_lvl_qos_3] = ref_lvl_2_val
                tmp_lvl_right[tmp_lvl_qos] = ref_lvl_i - 1
                tmp_lvl_right[tmp_lvl_qos_3] = ref_lvl_i
                tmp_lvl_qos = tmp_lvl_qos_3 + 1
                tmp_lvl_i_prev_3 = ref_lvl_i
                ref_lvl_q += 1
            else
                phase_stop_3 = min(phase_stop, ref_lvl_i)
                if ref_lvl_i == phase_stop_3
                    ref_lvl_2_val = ref_lvl_val[ref_lvl_q]
                    tmp_lvl_qos_3 = tmp_lvl_qos + (tmp_lvl_i_prev_3 < phase_stop_3 - 1)
                    if tmp_lvl_qos_3 > tmp_lvl_qos_stop
                        tmp_lvl_qos_2 = tmp_lvl_qos_stop + 1
                        while tmp_lvl_qos_3 > tmp_lvl_qos_stop
                            tmp_lvl_qos_stop = max(tmp_lvl_qos_stop << 1, 1)
                        end
                        Finch.resize_if_smaller!(tmp_lvl_right, tmp_lvl_qos_stop)
                        Finch.fill_range!(tmp_lvl_right, ref_lvl.shape, tmp_lvl_qos_2, tmp_lvl_qos_stop)
                        Finch.resize_if_smaller!(tmp_lvl_val_2, tmp_lvl_qos_stop)
                        Finch.fill_range!(tmp_lvl_val_2, false, tmp_lvl_qos_2, tmp_lvl_qos_stop)
                    end
                    tmp_lvl_val_2[tmp_lvl_qos_3] = ref_lvl_2_val
                    tmp_lvl_right[tmp_lvl_qos] = phase_stop_3 - 1
                    tmp_lvl_right[tmp_lvl_qos_3] = phase_stop_3
                    tmp_lvl_qos = tmp_lvl_qos_3 + 1
                    tmp_lvl_i_prev_3 = phase_stop_3
                    ref_lvl_q += 1
                end
                break
            end
        end
    end
    if tmp_lvl_qos - tmp_lvl_qos_set > 0
        tmp_lvl_ptr[1 + 1] += (tmp_lvl_qos - tmp_lvl_qos_set) - (tmp_lvl_i_prev_3 == ref_lvl.shape)
        tmp_lvl_i_prev = tmp_lvl_i_prev_3
        tmp_lvl_qos_fill = tmp_lvl_qos - 1
    end
    qos = tmp_lvl_qos_fill + (tmp_lvl_i_prev < ref_lvl.shape)
    qos += 1 - 1
    if qos > tmp_lvl_qos_stop
        Finch.resize_if_smaller!(tmp_lvl_right, qos)
        Finch.fill_range!(tmp_lvl_right, ref_lvl.shape, tmp_lvl_qos_fill + 1, qos)
        pos_start = 1 + tmp_lvl_qos_fill
        Finch.resize_if_smaller!(tmp_lvl_val_2, qos)
        Finch.fill_range!(tmp_lvl_val_2, false, pos_start, qos)
    end
    resize!(tmp_lvl_ptr, 1 + 1)
    for p = 1:1
        tmp_lvl_ptr[p + 1] += tmp_lvl_ptr[p]
    end
    tmp_lvl_qos_stop = tmp_lvl_ptr[1 + 1] - 1
    resize!(tmp_lvl_val_2, tmp_lvl_qos_stop)
    Finch.resize_if_smaller!(tmp_lvl_val, tmp_lvl_qos_stop)
    Finch.fill_range!(tmp_lvl_val, false, 1, tmp_lvl_qos_stop)
    q = 1
    q_2 = 1
    for p_2 = 1:1
        q_stop = tmp_lvl_ptr[p_2 + 1]
        while q < q_stop
            q_head = q
            while q + 1 < q_stop && tmp_lvl_right[q] == tmp_lvl_right[q + 1] - 1
                tmp_lvl_3_val = tmp_lvl_val_2[q_head]
                tmp_lvl_3_val_2 = tmp_lvl_val_2[1 + q]
                check = isequal(tmp_lvl_3_val, tmp_lvl_3_val_2) && true
                if !check
                    break
                else
                    q += 1
                end
            end
            tmp_lvl_right[q_2] = tmp_lvl_right[q]
            tmp_lvl_3_val_3 = tmp_lvl_val_2[q_head]
            tmp_lvl_val[q_2] = tmp_lvl_3_val_3
            q_2 += 1
            q += 1
        end
        tmp_lvl_ptr[p_2 + 1] = q_2
    end
    resize!(tmp_lvl_right, q_2 - 1)
    tmp_lvl_qos_stop = q_2 - 1
    resize!(tmp_lvl_val, tmp_lvl_qos_stop)
    resize!(tmp_lvl_val_2, 0)
    (tmp = Tensor((RunListLevel){Int64}(tmp_lvl_2, ref_lvl.shape, tmp_lvl_ptr, tmp_lvl_right, tmp_lvl_3; merge = true)),)
end
