quote
    tmp_lvl = ((ex.bodies[1]).bodies[1]).tns.bind.lvl
    tmp_lvl_ptr = tmp_lvl.ptr
    tmp_lvl_idx = tmp_lvl.idx
    tmp_lvl_val = tmp_lvl.val
    tmp_lvl_tbl = tmp_lvl.tbl
    tmp_lvl_pool = tmp_lvl.pool
    tmp_lvl_2 = tmp_lvl.lvl
    tmp_lvl_val_2 = tmp_lvl.lvl.val
    ref_lvl = ((ex.bodies[1]).bodies[2]).body.rhs.tns.bind.lvl
    ref_lvl_ptr = ref_lvl.ptr
    ref_lvl_idx = ref_lvl.idx
    ref_lvl_val = ref_lvl.lvl.val
    empty!(tmp_lvl_tbl)
    empty!(tmp_lvl_pool)
    tmp_lvl_qos_stop = 0
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
                tmp_lvl_qos = get(tmp_lvl_tbl, (1, ref_lvl_i), 0)
                if tmp_lvl_qos == 0
                    if !(isempty(tmp_lvl_pool))
                        tmp_lvl_qos = pop!(tmp_lvl_pool)
                    else
                        tmp_lvl_qos = length(tmp_lvl_tbl) + 1
                        if tmp_lvl_qos > tmp_lvl_qos_stop
                            tmp_lvl_qos_stop = max(tmp_lvl_qos_stop << 1, 1)
                            Finch.resize_if_smaller!(tmp_lvl_val_2, tmp_lvl_qos_stop)
                            Finch.fill_range!(tmp_lvl_val_2, false, tmp_lvl_qos, tmp_lvl_qos_stop)
                            Finch.resize_if_smaller!(tmp_lvl_val, tmp_lvl_qos_stop)
                            Finch.fill_range!(tmp_lvl_val, 0, tmp_lvl_qos, tmp_lvl_qos_stop)
                        end
                    end
                    tmp_lvl_tbl[(1, ref_lvl_i)] = tmp_lvl_qos
                end
                tmp_lvl_val_2[tmp_lvl_qos] = ref_lvl_2_val
                tmp_lvl_val[tmp_lvl_qos] = tmp_lvl_qos
                ref_lvl_q += 1
            else
                phase_stop_3 = min(phase_stop, ref_lvl_i)
                if ref_lvl_i == phase_stop_3
                    ref_lvl_2_val = ref_lvl_val[ref_lvl_q]
                    tmp_lvl_qos = get(tmp_lvl_tbl, (1, phase_stop_3), 0)
                    if tmp_lvl_qos == 0
                        if !(isempty(tmp_lvl_pool))
                            tmp_lvl_qos = pop!(tmp_lvl_pool)
                        else
                            tmp_lvl_qos = length(tmp_lvl_tbl) + 1
                            if tmp_lvl_qos > tmp_lvl_qos_stop
                                tmp_lvl_qos_stop = max(tmp_lvl_qos_stop << 1, 1)
                                Finch.resize_if_smaller!(tmp_lvl_val_2, tmp_lvl_qos_stop)
                                Finch.fill_range!(tmp_lvl_val_2, false, tmp_lvl_qos, tmp_lvl_qos_stop)
                                Finch.resize_if_smaller!(tmp_lvl_val, tmp_lvl_qos_stop)
                                Finch.fill_range!(tmp_lvl_val, 0, tmp_lvl_qos, tmp_lvl_qos_stop)
                            end
                        end
                        tmp_lvl_tbl[(1, phase_stop_3)] = tmp_lvl_qos
                    end
                    tmp_lvl_val_2[tmp_lvl_qos] = ref_lvl_2_val
                    tmp_lvl_val[tmp_lvl_qos] = tmp_lvl_qos
                    ref_lvl_q += 1
                end
                break
            end
        end
    end
    resize!(tmp_lvl_ptr, 1 + 1)
    tmp_lvl_ptr[1] = 1
    Finch.fill_range!(tmp_lvl_ptr, 0, 2, 1 + 1)
    pdx_tmp = Vector{Int64}(undef, length(tmp_lvl_tbl))
    resize!(tmp_lvl_idx, length(tmp_lvl_tbl))
    resize!(tmp_lvl_val, length(tmp_lvl_tbl))
    idx_tmp = Vector{Int64}(undef, length(tmp_lvl_tbl))
    val_tmp = Vector{Int64}(undef, length(tmp_lvl_tbl))
    q = 0
    for entry = pairs(tmp_lvl_tbl)
        sugar_2 = entry[1]
        p_2 = sugar_2[1]
        i_9 = sugar_2[2]
        v = entry[2]
        q += 1
        idx_tmp[q] = i_9
        val_tmp[q] = v
        pdx_tmp[q] = p_2
        tmp_lvl_ptr[p_2 + 1] += 1
    end
    for p_2 = 2:1 + 1
        tmp_lvl_ptr[p_2] += tmp_lvl_ptr[p_2 - 1]
    end
    perm = sortperm(idx_tmp)
    ptr_2 = copy(tmp_lvl_ptr)
    for q = perm
        p_2 = pdx_tmp[q]
        r = ptr_2[p_2]
        tmp_lvl_idx[r] = idx_tmp[q]
        tmp_lvl_val[r] = val_tmp[q]
        ptr_2[p_2] += 1
    end
    qos_stop = tmp_lvl_ptr[1 + 1] - 1
    resize!(tmp_lvl_val_2, qos_stop)
    (tmp = Tensor((SparseDictLevel){Int64}(tmp_lvl_2, ref_lvl.shape, tmp_lvl_ptr, tmp_lvl_idx, tmp_lvl_val, tmp_lvl_tbl, tmp_lvl_pool)),)
end
