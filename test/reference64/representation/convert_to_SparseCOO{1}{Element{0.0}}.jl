quote
    tmp_lvl = ((ex.bodies[1]).bodies[1]).tns.bind.lvl
    tmp_lvl_ptr = ((ex.bodies[1]).bodies[1]).tns.bind.lvl.ptr
    tmp_lvl_tbl1 = ((ex.bodies[1]).bodies[1]).tns.bind.lvl.tbl[1]
    tmp_lvl_2 = tmp_lvl.lvl
    tmp_lvl_val = tmp_lvl.lvl.val
    ref_lvl = ((ex.bodies[1]).bodies[2]).body.rhs.tns.bind.lvl
    ref_lvl_ptr = ref_lvl.ptr
    ref_lvl_idx = ref_lvl.idx
    ref_lvl_val = ref_lvl.lvl.val
    tmp_lvl_qos_stop = 0
    Finch.resize_if_smaller!(tmp_lvl_ptr, 1 + 1)
    Finch.fill_range!(tmp_lvl_ptr, 0, 1 + 1, 1 + 1)
    tmp_lvl_q = 0 + 1
    0 < 1 || throw(FinchProtocolError("SparseCOOLevels cannot be updated multiple times"))
    tmp_lvl_prev_coord_2 = ()
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
                if tmp_lvl_q > tmp_lvl_qos_stop
                    tmp_lvl_qos_stop = max(tmp_lvl_qos_stop << 1, 1)
                    Finch.resize_if_smaller!(tmp_lvl_tbl1, tmp_lvl_qos_stop)
                    Finch.resize_if_smaller!(tmp_lvl_val, tmp_lvl_qos_stop)
                    Finch.fill_range!(tmp_lvl_val, 0.0, tmp_lvl_q, tmp_lvl_qos_stop)
                end
                tmp_lvl_val[tmp_lvl_q] = ref_lvl_2_val
                tmp_lvl_prev_coord_2 < (ref_lvl_i,) || throw(FinchProtocolError("SparseCOOLevels cannot be updated multiple times"))
                tmp_lvl_prev_coord_2 = (ref_lvl_i,)
                tmp_lvl_tbl1[tmp_lvl_q] = ref_lvl_i
                tmp_lvl_q += 1
                ref_lvl_q += 1
            else
                phase_stop_3 = min(phase_stop, ref_lvl_i)
                if ref_lvl_i == phase_stop_3
                    ref_lvl_2_val = ref_lvl_val[ref_lvl_q]
                    if tmp_lvl_q > tmp_lvl_qos_stop
                        tmp_lvl_qos_stop = max(tmp_lvl_qos_stop << 1, 1)
                        Finch.resize_if_smaller!(tmp_lvl_tbl1, tmp_lvl_qos_stop)
                        Finch.resize_if_smaller!(tmp_lvl_val, tmp_lvl_qos_stop)
                        Finch.fill_range!(tmp_lvl_val, 0.0, tmp_lvl_q, tmp_lvl_qos_stop)
                    end
                    tmp_lvl_val[tmp_lvl_q] = ref_lvl_2_val
                    tmp_lvl_prev_coord_2 < (phase_stop_3,) || throw(FinchProtocolError("SparseCOOLevels cannot be updated multiple times"))
                    tmp_lvl_prev_coord_2 = (phase_stop_3,)
                    tmp_lvl_tbl1[tmp_lvl_q] = phase_stop_3
                    tmp_lvl_q += 1
                    ref_lvl_q += 1
                end
                break
            end
        end
    end
    tmp_lvl_ptr[1 + 1] = (tmp_lvl_q - 0) - 1
    resize!(tmp_lvl_ptr, 1 + 1)
    for p = 2:1 + 1
        tmp_lvl_ptr[p] += tmp_lvl_ptr[p - 1]
    end
    qos_stop = tmp_lvl_ptr[1 + 1] - 1
    resize!(tmp_lvl_tbl1, qos_stop)
    resize!(tmp_lvl_val, qos_stop)
    (tmp = Tensor((SparseCOOLevel){1, Tuple{Int64}}(tmp_lvl_2, (ref_lvl.shape,), tmp_lvl_ptr, (tmp_lvl_tbl1,))),)
end
