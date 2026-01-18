quote
    res_lvl = ((ex.bodies[1]).bodies[1]).tns.bind.lvl
    res_lvl_ptr = res_lvl.ptr
    res_lvl_idx = res_lvl.idx
    res_lvl_2 = res_lvl.lvl
    res_lvl_val = res_lvl.lvl.val
    tmp_lvl = ((ex.bodies[1]).bodies[2]).body.rhs.tns.bind.lvl
    tmp_lvl_val = tmp_lvl.lvl.val
    res_lvl_qos_stop = 0
    Finch.resize_if_smaller!(res_lvl_ptr, 1 + 1)
    Finch.fill_range!(res_lvl_ptr, 0, 1 + 1, 1 + 1)
    res_lvl_qos = 0 + 1
    0 < 1 || throw(FinchProtocolError("SparseListLevels cannot be updated multiple times"))
    for i_4 = 1:tmp_lvl.shape
        if res_lvl_qos > res_lvl_qos_stop
            res_lvl_qos_stop = max(res_lvl_qos_stop << 1, 1)
            Finch.resize_if_smaller!(res_lvl_idx, res_lvl_qos_stop)
            Finch.resize_if_smaller!(res_lvl_val, res_lvl_qos_stop)
            Finch.fill_range!(res_lvl_val, false, res_lvl_qos, res_lvl_qos_stop)
        end
        tmp_lvl_q = (1 - 1) * tmp_lvl.shape + i_4
        tmp_lvl_2_val = tmp_lvl_val[tmp_lvl_q]
        res = (res_lvl_val[res_lvl_qos] = tmp_lvl_2_val)
        res_lvl_idx[res_lvl_qos] = i_4
        res_lvl_qos += 1
    end
    res_lvl_ptr[1 + 1] += (res_lvl_qos - 0) - 1
    resize!(res_lvl_ptr, 1 + 1)
    for p = 1:1
        res_lvl_ptr[p + 1] += res_lvl_ptr[p]
    end
    qos_stop = res_lvl_ptr[1 + 1] - 1
    resize!(res_lvl_idx, qos_stop)
    resize!(res_lvl_val, qos_stop)
    (res = Tensor((SparseListLevel){Int64}(res_lvl_2, tmp_lvl.shape, res_lvl_ptr, res_lvl_idx)),)
end
