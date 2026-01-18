quote
    res_lvl = ((ex.bodies[1]).bodies[1]).tns.bind.lvl
    res_lvl_ptr = res_lvl.ptr
    res_lvl_idx = res_lvl.idx
    res_lvl_2 = res_lvl.lvl
    res_lvl_val = res_lvl.lvl.val
    tmp_lvl = ((ex.bodies[1]).bodies[2]).body.rhs.tns.bind.lvl
    tmp_lvl_ptr = tmp_lvl.ptr
    tmp_lvl_right = tmp_lvl.right
    tmp_lvl_val = tmp_lvl.lvl.val
    res_lvl_qos_stop = 0
    Finch.resize_if_smaller!(res_lvl_ptr, 1 + 1)
    Finch.fill_range!(res_lvl_ptr, 0, 1 + 1, 1 + 1)
    res_lvl_qos = 0 + 1
    0 < 1 || throw(FinchProtocolError("SparseListLevels cannot be updated multiple times"))
    tmp_lvl_q = tmp_lvl_ptr[1]
    tmp_lvl_q_stop = tmp_lvl_ptr[1 + 1]
    i = 1
    if tmp_lvl_right[tmp_lvl_q] < 1
        tmp_lvl_q = Finch.scansearch(tmp_lvl_right, 1, tmp_lvl_q, tmp_lvl_q_stop - 1)
    end
    while true
        tmp_lvl_i = tmp_lvl_right[tmp_lvl_q]
        if tmp_lvl_i < tmp_lvl.shape
            tmp_lvl_2_val = tmp_lvl_val[tmp_lvl_q]
            for i_5 = i:tmp_lvl_i
                if res_lvl_qos > res_lvl_qos_stop
                    res_lvl_qos_stop = max(res_lvl_qos_stop << 1, 1)
                    Finch.resize_if_smaller!(res_lvl_idx, res_lvl_qos_stop)
                    Finch.resize_if_smaller!(res_lvl_val, res_lvl_qos_stop)
                    Finch.fill_range!(res_lvl_val, 0.0, res_lvl_qos, res_lvl_qos_stop)
                end
                res = (res_lvl_val[res_lvl_qos] = tmp_lvl_2_val)
                res_lvl_idx[res_lvl_qos] = i_5
                res_lvl_qos += 1
            end
            tmp_lvl_q += 1
            i = tmp_lvl_i + 1
        else
            phase_stop_2 = min(tmp_lvl.shape, tmp_lvl_i)
            if tmp_lvl_i == phase_stop_2
                tmp_lvl_2_val = tmp_lvl_val[tmp_lvl_q]
                for i_6 = i:phase_stop_2
                    if res_lvl_qos > res_lvl_qos_stop
                        res_lvl_qos_stop = max(res_lvl_qos_stop << 1, 1)
                        Finch.resize_if_smaller!(res_lvl_idx, res_lvl_qos_stop)
                        Finch.resize_if_smaller!(res_lvl_val, res_lvl_qos_stop)
                        Finch.fill_range!(res_lvl_val, 0.0, res_lvl_qos, res_lvl_qos_stop)
                    end
                    res_lvl_val[res_lvl_qos] = tmp_lvl_2_val
                    res_lvl_idx[res_lvl_qos] = i_6
                    res_lvl_qos += 1
                end
                tmp_lvl_q += 1
            else
                tmp_lvl_2_val = tmp_lvl_val[tmp_lvl_q]
                for i_7 = i:phase_stop_2
                    if res_lvl_qos > res_lvl_qos_stop
                        res_lvl_qos_stop = max(res_lvl_qos_stop << 1, 1)
                        Finch.resize_if_smaller!(res_lvl_idx, res_lvl_qos_stop)
                        Finch.resize_if_smaller!(res_lvl_val, res_lvl_qos_stop)
                        Finch.fill_range!(res_lvl_val, 0.0, res_lvl_qos, res_lvl_qos_stop)
                    end
                    res_lvl_val[res_lvl_qos] = tmp_lvl_2_val
                    res_lvl_idx[res_lvl_qos] = i_7
                    res_lvl_qos += 1
                end
            end
            i = phase_stop_2 + 1
            break
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
