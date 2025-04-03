quote
    res_lvl = ((ex.bodies[1]).bodies[1]).tns.bind.lvl
    res_lvl_ptr = res_lvl.ptr
    res_lvl_idx = res_lvl.idx
    res_lvl_2 = res_lvl.lvl
    res_lvl_val = res_lvl.lvl.val
    tmp_lvl_val = ((ex.bodies[1]).bodies[2]).body.rhs.tns.bind.lvl.val
    tmp_lvl_2 = ((ex.bodies[1]).bodies[2]).body.rhs.tns.bind.lvl.lvl
    pointer_to_lvl_2 = tmp_lvl_val[1]
    pointer_to_lvl_2_val = pointer_to_lvl_2.lvl.val
    res_lvl_qos_alloc = 0
    Finch.resize_if_smaller!(res_lvl_ptr, 1 + 1)
    Finch.fill_range!(res_lvl_ptr, 0, 1 + 1, 1 + 1)
    res_lvl_qos = 0 + 1
    0 < 1 || throw(FinchProtocolError("SparseListLevels cannot be updated multiple times"))
    for i_4 = 1:tmp_lvl_2.shape
        if res_lvl_qos > res_lvl_qos_alloc
            res_lvl_qos_alloc = max(res_lvl_qos_alloc << 1, 1)
            Finch.resize_if_smaller!(res_lvl_idx, res_lvl_qos_alloc)
            Finch.resize_if_smaller!(res_lvl_val, res_lvl_qos_alloc)
            Finch.fill_range!(res_lvl_val, 0.0, res_lvl_qos, res_lvl_qos_alloc)
        end
        pointer_to_lvl_2_q = (1 - 1) * pointer_to_lvl_2.shape + i_4
        pointer_to_lvl_3_val = pointer_to_lvl_2_val[pointer_to_lvl_2_q]
        res = (res_lvl_val[res_lvl_qos] = pointer_to_lvl_3_val)
        res_lvl_idx[res_lvl_qos] = i_4
        res_lvl_qos += 1
    end
    res_lvl_ptr[1 + 1] += (res_lvl_qos - 0) - 1
    resize!(res_lvl_ptr, 1 + 1)
    for p = 1:1
        res_lvl_ptr[p + 1] += res_lvl_ptr[p]
    end
    qos_asmbld = res_lvl_ptr[1 + 1] - 1
    resize!(res_lvl_idx, qos_asmbld)
    resize!(res_lvl_val, qos_asmbld)
    result = (res = Tensor((SparseListLevel){Int32}(res_lvl_2, tmp_lvl_2.shape, res_lvl_ptr, res_lvl_idx)),)
    result
end
