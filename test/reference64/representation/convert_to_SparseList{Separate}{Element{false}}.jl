quote
    tmp_lvl = ((ex.bodies[1]).bodies[1]).tns.bind.lvl
    tmp_lvl_ptr = tmp_lvl.ptr
    tmp_lvl_idx = tmp_lvl.idx
    tmp_lvl_2 = tmp_lvl.lvl
    tmp_lvl_val = tmp_lvl.lvl.val
    tmp_lvl_3 = tmp_lvl.lvl.lvl
    ref_lvl = ((ex.bodies[1]).bodies[2]).body.rhs.tns.bind.lvl
    ref_lvl_ptr = ref_lvl.ptr
    ref_lvl_idx = ref_lvl.idx
    ref_lvl_val = ref_lvl.lvl.val
    tmp_lvl_qos_stop = 0
    Finch.resize_if_smaller!(tmp_lvl_ptr, 1 + 1)
    Finch.fill_range!(tmp_lvl_ptr, 0, 1 + 1, 1 + 1)
    tmp_lvl_qos = 0 + 1
    0 < 1 || throw(FinchProtocolError("SparseListLevels cannot be updated multiple times"))
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
                if tmp_lvl_qos > tmp_lvl_qos_stop
                    tmp_lvl_qos_stop = max(tmp_lvl_qos_stop << 1, 1)
                    Finch.resize_if_smaller!(tmp_lvl_idx, tmp_lvl_qos_stop)
                    Finch.resize_if_smaller!(tmp_lvl_val, tmp_lvl_qos_stop)
                    for pos = tmp_lvl_qos:tmp_lvl_qos_stop
                        pointer_to_lvl = Finch.similar_level(tmp_lvl_2.lvl, Finch.level_fill_value(typeof(tmp_lvl_2.lvl)), Finch.level_eltype(typeof(tmp_lvl_2.lvl)))
                        pointer_to_lvl_val = pointer_to_lvl.val
                        Finch.resize_if_smaller!(pointer_to_lvl_val, 1)
                        Finch.fill_range!(pointer_to_lvl_val, false, 1, 1)
                        resize!(pointer_to_lvl_val, 1)
                        tmp_lvl_val[pos] = pointer_to_lvl
                    end
                end
                pointer_to_lvl_4 = tmp_lvl_val[tmp_lvl_qos]
                pointer_to_lvl_3_val = (tmp_lvl_val[tmp_lvl_qos]).val
                Finch.resize_if_smaller!(pointer_to_lvl_3_val, 1)
                Finch.fill_range!(pointer_to_lvl_3_val, false, 1, 1)
                pointer_to_lvl_3_val[1] = ref_lvl_2_val
                resize!(pointer_to_lvl_3_val, 1)
                tmp_lvl_val[tmp_lvl_qos] = pointer_to_lvl_4
                tmp_lvl_idx[tmp_lvl_qos] = ref_lvl_i
                tmp_lvl_qos += 1
                ref_lvl_q += 1
            else
                phase_stop_3 = min(phase_stop, ref_lvl_i)
                if ref_lvl_i == phase_stop_3
                    ref_lvl_2_val = ref_lvl_val[ref_lvl_q]
                    if tmp_lvl_qos > tmp_lvl_qos_stop
                        tmp_lvl_qos_stop = max(tmp_lvl_qos_stop << 1, 1)
                        Finch.resize_if_smaller!(tmp_lvl_idx, tmp_lvl_qos_stop)
                        Finch.resize_if_smaller!(tmp_lvl_val, tmp_lvl_qos_stop)
                        for pos_2 = tmp_lvl_qos:tmp_lvl_qos_stop
                            pointer_to_lvl_5 = Finch.similar_level(tmp_lvl_2.lvl, Finch.level_fill_value(typeof(tmp_lvl_2.lvl)), Finch.level_eltype(typeof(tmp_lvl_2.lvl)))
                            pointer_to_lvl_5_val = pointer_to_lvl_5.val
                            Finch.resize_if_smaller!(pointer_to_lvl_5_val, 1)
                            Finch.fill_range!(pointer_to_lvl_5_val, false, 1, 1)
                            resize!(pointer_to_lvl_5_val, 1)
                            tmp_lvl_val[pos_2] = pointer_to_lvl_5
                        end
                    end
                    pointer_to_lvl_8 = tmp_lvl_val[tmp_lvl_qos]
                    pointer_to_lvl_7_val = (tmp_lvl_val[tmp_lvl_qos]).val
                    Finch.resize_if_smaller!(pointer_to_lvl_7_val, 1)
                    Finch.fill_range!(pointer_to_lvl_7_val, false, 1, 1)
                    pointer_to_lvl_7_val[1] = ref_lvl_2_val
                    resize!(pointer_to_lvl_7_val, 1)
                    tmp_lvl_val[tmp_lvl_qos] = pointer_to_lvl_8
                    tmp_lvl_idx[tmp_lvl_qos] = phase_stop_3
                    tmp_lvl_qos += 1
                    ref_lvl_q += 1
                end
                break
            end
        end
    end
    tmp_lvl_ptr[1 + 1] += (tmp_lvl_qos - 0) - 1
    resize!(tmp_lvl_ptr, 1 + 1)
    for p = 1:1
        tmp_lvl_ptr[p + 1] += tmp_lvl_ptr[p]
    end
    qos_stop = tmp_lvl_ptr[1 + 1] - 1
    resize!(tmp_lvl_idx, qos_stop)
    (tmp = Tensor((SparseListLevel){Int64}((SeparateLevel){ElementLevel{false, Bool, Int64, Vector{Bool}}, Vector{ElementLevel{false, Bool, Int64, Vector{Bool}}}}(tmp_lvl_3, tmp_lvl_val), ref_lvl.shape, tmp_lvl_ptr, tmp_lvl_idx)),)
end
