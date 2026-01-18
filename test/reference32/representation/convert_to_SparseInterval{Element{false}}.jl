quote
    tmp_lvl = ((ex.bodies[1]).bodies[1]).tns.bind.lvl
    tmp_lvl_left = tmp_lvl.left
    tmp_lvl_right = tmp_lvl.right
    tmp_lvl_2 = tmp_lvl.lvl
    tmp_lvl_val = tmp_lvl.lvl.val
    ref_lvl = ((ex.bodies[1]).bodies[2]).body.rhs.tns.bind.lvl
    ref_lvl_ptr = ref_lvl.ptr
    ref_lvl_idx = ref_lvl.idx
    ref_lvl_val = ref_lvl.lvl.val
    Finch.resize_if_smaller!(tmp_lvl_left, 1)
    Finch.resize_if_smaller!(tmp_lvl_right, 1)
    Finch.fill_range!(tmp_lvl_left, 1, 1, 1)
    Finch.fill_range!(tmp_lvl_right, 0, 1, 1)
    Finch.resize_if_smaller!(tmp_lvl_val, 1)
    Finch.fill_range!(tmp_lvl_val, false, 1, 1)
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
                tmp_lvl_val[1] = ref_lvl_2_val
                tmp_lvl_left[1] < tmp_lvl_right[1] && throw(FinchProtocolError("SparseIntervalLevels can only be updated once"))
                tmp_lvl_left[1] = ref_lvl_i
                tmp_lvl_right[1] = ref_lvl_i
                ref_lvl_q += 1
            else
                phase_stop_3 = min(phase_stop, ref_lvl_i)
                if ref_lvl_i == phase_stop_3
                    ref_lvl_2_val = ref_lvl_val[ref_lvl_q]
                    tmp_lvl_val[1] = ref_lvl_2_val
                    tmp_lvl_left[1] < tmp_lvl_right[1] && throw(FinchProtocolError("SparseIntervalLevels can only be updated once"))
                    tmp_lvl_left[1] = phase_stop_3
                    tmp_lvl_right[1] = phase_stop_3
                    ref_lvl_q += 1
                end
                break
            end
        end
    end
    resize!(tmp_lvl_left, 1)
    resize!(tmp_lvl_right, 1)
    resize!(tmp_lvl_val, 1)
    (tmp = Tensor((SparseIntervalLevel){Int32}(tmp_lvl_2, ref_lvl.shape, tmp_lvl.left, tmp_lvl.right)),)
end
