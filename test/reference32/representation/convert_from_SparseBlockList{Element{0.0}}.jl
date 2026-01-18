quote
    res_lvl = ((ex.bodies[1]).bodies[1]).tns.bind.lvl
    res_lvl_ptr = res_lvl.ptr
    res_lvl_idx = res_lvl.idx
    res_lvl_2 = res_lvl.lvl
    res_lvl_val = res_lvl.lvl.val
    tmp_lvl = ((ex.bodies[1]).bodies[2]).body.rhs.tns.bind.lvl
    tmp_lvl_ptr = tmp_lvl.ptr
    tmp_lvl_idx = tmp_lvl.idx
    tmp_lvl_ofs = tmp_lvl.ofs
    tmp_lvl_val = tmp_lvl.lvl.val
    res_lvl_qos_stop = 0
    Finch.resize_if_smaller!(res_lvl_ptr, 1 + 1)
    Finch.fill_range!(res_lvl_ptr, 0, 1 + 1, 1 + 1)
    res_lvl_qos = 0 + 1
    0 < 1 || throw(FinchProtocolError("SparseListLevels cannot be updated multiple times"))
    tmp_lvl_r = tmp_lvl_ptr[1]
    tmp_lvl_r_stop = tmp_lvl_ptr[1 + 1]
    if tmp_lvl_r < tmp_lvl_r_stop
        tmp_lvl_i1 = tmp_lvl_idx[tmp_lvl_r_stop - 1]
    else
        tmp_lvl_i1 = 0
    end
    phase_stop = min(tmp_lvl_i1, tmp_lvl.shape)
    if phase_stop >= 1
        i = 1
        if tmp_lvl_idx[tmp_lvl_r] < 1
            tmp_lvl_r = Finch.scansearch(tmp_lvl_idx, 1, tmp_lvl_r, tmp_lvl_r_stop - 1)
        end
        while true
            i_start_2 = i
            tmp_lvl_i = tmp_lvl_idx[tmp_lvl_r]
            tmp_lvl_q_stop = tmp_lvl_ofs[tmp_lvl_r + 1]
            tmp_lvl_i_2 = tmp_lvl_i - (tmp_lvl_q_stop - tmp_lvl_ofs[tmp_lvl_r])
            tmp_lvl_q_ofs = (tmp_lvl_q_stop - tmp_lvl_i) - 1
            if tmp_lvl_i < phase_stop
                phase_start_3 = max(i_start_2, 1 + tmp_lvl_i_2)
                if tmp_lvl_i >= phase_start_3
                    for i_8 = phase_start_3:tmp_lvl_i
                        if res_lvl_qos > res_lvl_qos_stop
                            res_lvl_qos_stop = max(res_lvl_qos_stop << 1, 1)
                            Finch.resize_if_smaller!(res_lvl_idx, res_lvl_qos_stop)
                            Finch.resize_if_smaller!(res_lvl_val, res_lvl_qos_stop)
                            Finch.fill_range!(res_lvl_val, 0.0, res_lvl_qos, res_lvl_qos_stop)
                        end
                        tmp_lvl_q = tmp_lvl_q_ofs + i_8
                        tmp_lvl_2_val = tmp_lvl_val[tmp_lvl_q]
                        res = (res_lvl_val[res_lvl_qos] = tmp_lvl_2_val)
                        res_lvl_idx[res_lvl_qos] = i_8
                        res_lvl_qos += 1
                    end
                end
                tmp_lvl_r += tmp_lvl_i == tmp_lvl_i
                i = tmp_lvl_i + 1
            else
                phase_start_4 = i
                phase_stop_5 = min(phase_stop, tmp_lvl_i)
                phase_start_6 = max(1 + tmp_lvl_i_2, phase_start_4)
                if phase_stop_5 >= phase_start_6
                    for i_11 = phase_start_6:phase_stop_5
                        if res_lvl_qos > res_lvl_qos_stop
                            res_lvl_qos_stop = max(res_lvl_qos_stop << 1, 1)
                            Finch.resize_if_smaller!(res_lvl_idx, res_lvl_qos_stop)
                            Finch.resize_if_smaller!(res_lvl_val, res_lvl_qos_stop)
                            Finch.fill_range!(res_lvl_val, 0.0, res_lvl_qos, res_lvl_qos_stop)
                        end
                        tmp_lvl_q = tmp_lvl_q_ofs + i_11
                        tmp_lvl_2_val_2 = tmp_lvl_val[tmp_lvl_q]
                        res_lvl_val[res_lvl_qos] = tmp_lvl_2_val_2
                        res_lvl_idx[res_lvl_qos] = i_11
                        res_lvl_qos += 1
                    end
                end
                tmp_lvl_r += phase_stop_5 == tmp_lvl_i
                i = phase_stop_5 + 1
                break
            end
        end
    end
    res_lvl_ptr[1 + 1] += (res_lvl_qos - 0) - 1
    resize!(res_lvl_ptr, 1 + 1)
    for p = 1:1
        res_lvl_ptr[p + 1] += res_lvl_ptr[p]
    end
    qos_stop = res_lvl_ptr[1 + 1] - 1
    resize!(res_lvl_idx, qos_stop)
    resize!(res_lvl_val, qos_stop)
    (res = Tensor((SparseListLevel){Int32}(res_lvl_2, tmp_lvl.shape, res_lvl_ptr, res_lvl_idx)),)
end
