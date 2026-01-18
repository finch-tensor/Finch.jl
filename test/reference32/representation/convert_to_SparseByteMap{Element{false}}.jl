quote
    tmp_lvl = ((ex.bodies[1]).bodies[1]).tns.bind.lvl
    tmp_lvl_ptr = ((ex.bodies[1]).bodies[1]).tns.bind.lvl.ptr
    tmp_lvl_tbl = ((ex.bodies[1]).bodies[1]).tns.bind.lvl.tbl
    tmp_lvl_srt = ((ex.bodies[1]).bodies[1]).tns.bind.lvl.srt
    tmp_lvl_qos_stop = (tmp_lvl_qos_fill = length(tmp_lvl.srt))
    tmp_lvl_2 = tmp_lvl.lvl
    tmp_lvl_val = tmp_lvl.lvl.val
    ref_lvl = ((ex.bodies[1]).bodies[2]).body.rhs.tns.bind.lvl
    ref_lvl_ptr = ref_lvl.ptr
    ref_lvl_idx = ref_lvl.idx
    ref_lvl_val = ref_lvl.lvl.val
    for tmp_lvl_r = 1:tmp_lvl_qos_fill
        tmp_lvl_p = first(tmp_lvl_srt[tmp_lvl_r])
        tmp_lvl_ptr[tmp_lvl_p] = 0
        tmp_lvl_ptr[tmp_lvl_p + 1] = 0
        tmp_lvl_i = last(tmp_lvl_srt[tmp_lvl_r])
        tmp_lvl_q = (tmp_lvl_p - 1) * ref_lvl.shape + tmp_lvl_i
        tmp_lvl_tbl[tmp_lvl_q] = false
        Finch.resize_if_smaller!(tmp_lvl_val, tmp_lvl_q)
        Finch.fill_range!(tmp_lvl_val, false, tmp_lvl_q, tmp_lvl_q)
    end
    tmp_lvl_qos_fill = 0
    tmp_lvl_ptr[1] = 1
    tmp_lvlq_stop = 1 * ref_lvl.shape
    Finch.resize_if_smaller!(tmp_lvl_ptr, 1 + 1)
    Finch.fill_range!(tmp_lvl_ptr, 0, 1 + 1, 1 + 1)
    tmp_lvlold = length(tmp_lvl_tbl) + 1
    Finch.resize_if_smaller!(tmp_lvl_tbl, tmp_lvlq_stop)
    Finch.fill_range!(tmp_lvl_tbl, false, tmp_lvlold, tmp_lvlq_stop)
    Finch.resize_if_smaller!(tmp_lvl_val, tmp_lvlq_stop)
    Finch.fill_range!(tmp_lvl_val, false, tmp_lvlold, tmp_lvlq_stop)
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
                tmp_lvl_q_2 = (1 - 1) * ref_lvl.shape + ref_lvl_i
                tmp_lvl_val[tmp_lvl_q_2] = ref_lvl_2_val
                if !(tmp_lvl_tbl[tmp_lvl_q_2])
                    tmp_lvl_tbl[tmp_lvl_q_2] = true
                    tmp_lvl_qos_fill += 1
                    if tmp_lvl_qos_fill > tmp_lvl_qos_stop
                        tmp_lvl_qos_stop = max(tmp_lvl_qos_stop << 1, 1)
                        Finch.resize_if_smaller!(tmp_lvl_srt, tmp_lvl_qos_stop)
                    end
                    tmp_lvl_srt[tmp_lvl_qos_fill] = (1, ref_lvl_i)
                end
                ref_lvl_q += 1
            else
                phase_stop_3 = min(phase_stop, ref_lvl_i)
                if ref_lvl_i == phase_stop_3
                    ref_lvl_2_val = ref_lvl_val[ref_lvl_q]
                    tmp_lvl_q_2 = (1 - 1) * ref_lvl.shape + phase_stop_3
                    tmp_lvl_val[tmp_lvl_q_2] = ref_lvl_2_val
                    if !(tmp_lvl_tbl[tmp_lvl_q_2])
                        tmp_lvl_tbl[tmp_lvl_q_2] = true
                        tmp_lvl_qos_fill += 1
                        if tmp_lvl_qos_fill > tmp_lvl_qos_stop
                            tmp_lvl_qos_stop = max(tmp_lvl_qos_stop << 1, 1)
                            Finch.resize_if_smaller!(tmp_lvl_srt, tmp_lvl_qos_stop)
                        end
                        tmp_lvl_srt[tmp_lvl_qos_fill] = (1, phase_stop_3)
                    end
                    ref_lvl_q += 1
                end
                break
            end
        end
    end
    resize!(tmp_lvl_ptr, 1 + 1)
    resize!(tmp_lvl_tbl, 1 * ref_lvl.shape)
    resize!(tmp_lvl_srt, tmp_lvl_qos_fill)
    sort!(tmp_lvl_srt)
    tmp_lvl_p_prev = 0
    for tmp_lvl_r_2 = 1:tmp_lvl_qos_fill
        tmp_lvl_p_2 = first(tmp_lvl_srt[tmp_lvl_r_2])
        if tmp_lvl_p_2 != tmp_lvl_p_prev
            tmp_lvl_ptr[tmp_lvl_p_prev + 1] = tmp_lvl_r_2
            tmp_lvl_ptr[tmp_lvl_p_2] = tmp_lvl_r_2
        end
        tmp_lvl_p_prev = tmp_lvl_p_2
    end
    tmp_lvl_ptr[tmp_lvl_p_prev + 1] = tmp_lvl_qos_fill + 1
    resize!(tmp_lvl_val, ref_lvl.shape)
    (tmp = Tensor((SparseByteMapLevel){Int32}(tmp_lvl_2, ref_lvl.shape, tmp_lvl_ptr, tmp_lvl_tbl, tmp_lvl_srt)),)
end
