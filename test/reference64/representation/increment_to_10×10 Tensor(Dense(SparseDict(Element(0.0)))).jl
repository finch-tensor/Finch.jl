begin
    fmt_lvl = (ex.bodies[1]).body.body.lhs.tns.bind.lvl
    fmt_lvl_stop = fmt_lvl.shape
    fmt_lvl_2 = fmt_lvl.lvl
    fmt_lvl_2_ptr = fmt_lvl_2.ptr
    fmt_lvl_2_idx = fmt_lvl_2.idx
    fmt_lvl_2_val = fmt_lvl_2.val
    fmt_lvl_2_tbl = fmt_lvl_2.tbl
    fmt_lvl_2_pool = fmt_lvl_2.pool
    fmt_lvl_2_stop = fmt_lvl_2.shape
    fmt_lvl_3 = fmt_lvl_2.lvl
    fmt_lvl_3_val = fmt_lvl_3.val
    arr_2_lvl = (ex.bodies[1]).body.body.rhs.tns.bind.lvl
    arr_2_lvl_ptr = arr_2_lvl.ptr
    arr_2_lvl_tbl1 = (ex.bodies[1]).body.body.rhs.tns.bind.lvl.tbl[1]
    arr_2_lvl_stop1 = (ex.bodies[1]).body.body.rhs.tns.bind.lvl.shape[1]
    arr_2_lvl_tbl2 = (ex.bodies[1]).body.body.rhs.tns.bind.lvl.tbl[2]
    arr_2_lvl_stop2 = (ex.bodies[1]).body.body.rhs.tns.bind.lvl.shape[2]
    arr_2_lvl_2 = arr_2_lvl.lvl
    arr_2_lvl_2_val = arr_2_lvl_2.val
    arr_2_lvl_stop1 == fmt_lvl_2_stop || throw(DimensionMismatch("mismatched dimension limits ($(arr_2_lvl_stop1) != $(fmt_lvl_2_stop))"))
    arr_2_lvl_stop2 == fmt_lvl_stop || throw(DimensionMismatch("mismatched dimension limits ($(arr_2_lvl_stop2) != $(fmt_lvl_stop))"))
    fmt_lvl_2_qos_stop = fmt_lvl_2_ptr[fmt_lvl_stop + 1] - 1
    arr_2_lvl_q = arr_2_lvl_ptr[1]
    arr_2_lvl_q_stop = arr_2_lvl_ptr[1 + 1]
    if arr_2_lvl_q < arr_2_lvl_q_stop
        arr_2_lvl_i_stop = arr_2_lvl_tbl2[arr_2_lvl_q_stop - 1]
    else
        arr_2_lvl_i_stop = 0
    end
    phase_stop = min(arr_2_lvl_stop2, arr_2_lvl_i_stop)
    if phase_stop >= 1
        if arr_2_lvl_tbl2[arr_2_lvl_q] < 1
            arr_2_lvl_q = Finch.scansearch(arr_2_lvl_tbl2, 1, arr_2_lvl_q, arr_2_lvl_q_stop - 1)
        end
        while true
            arr_2_lvl_i = arr_2_lvl_tbl2[arr_2_lvl_q]
            arr_2_lvl_q_step = arr_2_lvl_q
            if arr_2_lvl_tbl2[arr_2_lvl_q] == arr_2_lvl_i
                arr_2_lvl_q_step = Finch.scansearch(arr_2_lvl_tbl2, arr_2_lvl_i + 1, arr_2_lvl_q, arr_2_lvl_q_stop - 1)
            end
            if arr_2_lvl_i < phase_stop
                fmt_lvl_q = (1 - 1) * fmt_lvl_stop + arr_2_lvl_i
                arr_2_lvl_q_2 = arr_2_lvl_q
                if arr_2_lvl_q < arr_2_lvl_q_step
                    arr_2_lvl_i_stop_2 = arr_2_lvl_tbl1[arr_2_lvl_q_step - 1]
                else
                    arr_2_lvl_i_stop_2 = 0
                end
                phase_stop_3 = min(arr_2_lvl_stop1, arr_2_lvl_i_stop_2)
                if phase_stop_3 >= 1
                    if arr_2_lvl_tbl1[arr_2_lvl_q] < 1
                        arr_2_lvl_q_2 = Finch.scansearch(arr_2_lvl_tbl1, 1, arr_2_lvl_q, arr_2_lvl_q_step - 1)
                    end
                    while true
                        arr_2_lvl_i_2 = arr_2_lvl_tbl1[arr_2_lvl_q_2]
                        if arr_2_lvl_i_2 < phase_stop_3
                            arr_2_lvl_2_val_2 = arr_2_lvl_2_val[arr_2_lvl_q_2]
                            fmt_lvl_2_qos = get(fmt_lvl_2_tbl, (fmt_lvl_q, arr_2_lvl_i_2), 0)
                            if fmt_lvl_2_qos == 0
                                if !(isempty(fmt_lvl_2_pool))
                                    fmt_lvl_2_qos = pop!(fmt_lvl_2_pool)
                                else
                                    fmt_lvl_2_qos = length(fmt_lvl_2_tbl) + 1
                                    if fmt_lvl_2_qos > fmt_lvl_2_qos_stop
                                        fmt_lvl_2_qos_stop = max(fmt_lvl_2_qos_stop << 1, 1)
                                        Finch.resize_if_smaller!(fmt_lvl_3_val, fmt_lvl_2_qos_stop)
                                        Finch.fill_range!(fmt_lvl_3_val, 0.0, fmt_lvl_2_qos, fmt_lvl_2_qos_stop)
                                        Finch.resize_if_smaller!(fmt_lvl_2_val, fmt_lvl_2_qos_stop)
                                        Finch.fill_range!(fmt_lvl_2_val, 0, fmt_lvl_2_qos, fmt_lvl_2_qos_stop)
                                    end
                                end
                                fmt_lvl_2_tbl[(fmt_lvl_q, arr_2_lvl_i_2)] = fmt_lvl_2_qos
                            end
                            fmt_lvl_3_val[fmt_lvl_2_qos] = arr_2_lvl_2_val_2 + fmt_lvl_3_val[fmt_lvl_2_qos]
                            fmt_lvl_2_val[fmt_lvl_2_qos] = fmt_lvl_2_qos
                            arr_2_lvl_q_2 += 1
                        else
                            phase_stop_5 = min(phase_stop_3, arr_2_lvl_i_2)
                            if arr_2_lvl_i_2 == phase_stop_5
                                arr_2_lvl_2_val_2 = arr_2_lvl_2_val[arr_2_lvl_q_2]
                                fmt_lvl_2_qos = get(fmt_lvl_2_tbl, (fmt_lvl_q, phase_stop_5), 0)
                                if fmt_lvl_2_qos == 0
                                    if !(isempty(fmt_lvl_2_pool))
                                        fmt_lvl_2_qos = pop!(fmt_lvl_2_pool)
                                    else
                                        fmt_lvl_2_qos = length(fmt_lvl_2_tbl) + 1
                                        if fmt_lvl_2_qos > fmt_lvl_2_qos_stop
                                            fmt_lvl_2_qos_stop = max(fmt_lvl_2_qos_stop << 1, 1)
                                            Finch.resize_if_smaller!(fmt_lvl_3_val, fmt_lvl_2_qos_stop)
                                            Finch.fill_range!(fmt_lvl_3_val, 0.0, fmt_lvl_2_qos, fmt_lvl_2_qos_stop)
                                            Finch.resize_if_smaller!(fmt_lvl_2_val, fmt_lvl_2_qos_stop)
                                            Finch.fill_range!(fmt_lvl_2_val, 0, fmt_lvl_2_qos, fmt_lvl_2_qos_stop)
                                        end
                                    end
                                    fmt_lvl_2_tbl[(fmt_lvl_q, phase_stop_5)] = fmt_lvl_2_qos
                                end
                                fmt_lvl_3_val[fmt_lvl_2_qos] += arr_2_lvl_2_val_2
                                fmt_lvl_2_val[fmt_lvl_2_qos] = fmt_lvl_2_qos
                                arr_2_lvl_q_2 += 1
                            end
                            break
                        end
                    end
                end
                arr_2_lvl_q = arr_2_lvl_q_step
            else
                phase_stop_7 = min(phase_stop, arr_2_lvl_i)
                if arr_2_lvl_i == phase_stop_7
                    fmt_lvl_q = (1 - 1) * fmt_lvl_stop + phase_stop_7
                    arr_2_lvl_q_3 = arr_2_lvl_q
                    if arr_2_lvl_q < arr_2_lvl_q_step
                        arr_2_lvl_i_stop_3 = arr_2_lvl_tbl1[arr_2_lvl_q_step - 1]
                    else
                        arr_2_lvl_i_stop_3 = 0
                    end
                    phase_stop_8 = min(arr_2_lvl_stop1, arr_2_lvl_i_stop_3)
                    if phase_stop_8 >= 1
                        if arr_2_lvl_tbl1[arr_2_lvl_q] < 1
                            arr_2_lvl_q_3 = Finch.scansearch(arr_2_lvl_tbl1, 1, arr_2_lvl_q, arr_2_lvl_q_step - 1)
                        end
                        while true
                            arr_2_lvl_i_3 = arr_2_lvl_tbl1[arr_2_lvl_q_3]
                            if arr_2_lvl_i_3 < phase_stop_8
                                arr_2_lvl_2_val_3 = arr_2_lvl_2_val[arr_2_lvl_q_3]
                                fmt_lvl_2_qos_2 = get(fmt_lvl_2_tbl, (fmt_lvl_q, arr_2_lvl_i_3), 0)
                                if fmt_lvl_2_qos_2 == 0
                                    if !(isempty(fmt_lvl_2_pool))
                                        fmt_lvl_2_qos_2 = pop!(fmt_lvl_2_pool)
                                    else
                                        fmt_lvl_2_qos_2 = length(fmt_lvl_2_tbl) + 1
                                        if fmt_lvl_2_qos_2 > fmt_lvl_2_qos_stop
                                            fmt_lvl_2_qos_stop = max(fmt_lvl_2_qos_stop << 1, 1)
                                            Finch.resize_if_smaller!(fmt_lvl_3_val, fmt_lvl_2_qos_stop)
                                            Finch.fill_range!(fmt_lvl_3_val, 0.0, fmt_lvl_2_qos_2, fmt_lvl_2_qos_stop)
                                            Finch.resize_if_smaller!(fmt_lvl_2_val, fmt_lvl_2_qos_stop)
                                            Finch.fill_range!(fmt_lvl_2_val, 0, fmt_lvl_2_qos_2, fmt_lvl_2_qos_stop)
                                        end
                                    end
                                    fmt_lvl_2_tbl[(fmt_lvl_q, arr_2_lvl_i_3)] = fmt_lvl_2_qos_2
                                end
                                fmt_lvl_3_val[fmt_lvl_2_qos_2] = arr_2_lvl_2_val_3 + fmt_lvl_3_val[fmt_lvl_2_qos_2]
                                fmt_lvl_2_val[fmt_lvl_2_qos_2] = fmt_lvl_2_qos_2
                                arr_2_lvl_q_3 += 1
                            else
                                phase_stop_10 = min(phase_stop_8, arr_2_lvl_i_3)
                                if arr_2_lvl_i_3 == phase_stop_10
                                    arr_2_lvl_2_val_3 = arr_2_lvl_2_val[arr_2_lvl_q_3]
                                    fmt_lvl_2_qos_2 = get(fmt_lvl_2_tbl, (fmt_lvl_q, phase_stop_10), 0)
                                    if fmt_lvl_2_qos_2 == 0
                                        if !(isempty(fmt_lvl_2_pool))
                                            fmt_lvl_2_qos_2 = pop!(fmt_lvl_2_pool)
                                        else
                                            fmt_lvl_2_qos_2 = length(fmt_lvl_2_tbl) + 1
                                            if fmt_lvl_2_qos_2 > fmt_lvl_2_qos_stop
                                                fmt_lvl_2_qos_stop = max(fmt_lvl_2_qos_stop << 1, 1)
                                                Finch.resize_if_smaller!(fmt_lvl_3_val, fmt_lvl_2_qos_stop)
                                                Finch.fill_range!(fmt_lvl_3_val, 0.0, fmt_lvl_2_qos_2, fmt_lvl_2_qos_stop)
                                                Finch.resize_if_smaller!(fmt_lvl_2_val, fmt_lvl_2_qos_stop)
                                                Finch.fill_range!(fmt_lvl_2_val, 0, fmt_lvl_2_qos_2, fmt_lvl_2_qos_stop)
                                            end
                                        end
                                        fmt_lvl_2_tbl[(fmt_lvl_q, phase_stop_10)] = fmt_lvl_2_qos_2
                                    end
                                    fmt_lvl_3_val[fmt_lvl_2_qos_2] += arr_2_lvl_2_val_3
                                    fmt_lvl_2_val[fmt_lvl_2_qos_2] = fmt_lvl_2_qos_2
                                    arr_2_lvl_q_3 += 1
                                end
                                break
                            end
                        end
                    end
                    arr_2_lvl_q = arr_2_lvl_q_step
                end
                break
            end
        end
    end
    result = ()
    resize!(fmt_lvl_2_ptr, fmt_lvl_stop + 1)
    fmt_lvl_2_ptr[1] = 1
    Finch.fill_range!(fmt_lvl_2_ptr, 0, 2, fmt_lvl_stop + 1)
    pdx_tmp = Vector{Int64}(undef, length(fmt_lvl_2_tbl))
    resize!(fmt_lvl_2_idx, length(fmt_lvl_2_tbl))
    resize!(fmt_lvl_2_val, length(fmt_lvl_2_tbl))
    idx_tmp = Vector{Int64}(undef, length(fmt_lvl_2_tbl))
    val_tmp = Vector{Int64}(undef, length(fmt_lvl_2_tbl))
    q = 0
    for entry = pairs(fmt_lvl_2_tbl)
        sugar_2 = entry[1]
        p_3 = sugar_2[1]
        i_14 = sugar_2[2]
        v = entry[2]
        q += 1
        idx_tmp[q] = i_14
        val_tmp[q] = v
        pdx_tmp[q] = p_3
        fmt_lvl_2_ptr[p_3 + 1] += 1
    end
    for p_3 = 2:fmt_lvl_stop + 1
        fmt_lvl_2_ptr[p_3] += fmt_lvl_2_ptr[p_3 - 1]
    end
    perm = sortperm(idx_tmp)
    ptr_2 = copy(fmt_lvl_2_ptr)
    for q = perm
        p_3 = pdx_tmp[q]
        r = ptr_2[p_3]
        fmt_lvl_2_idx[r] = idx_tmp[q]
        fmt_lvl_2_val[r] = val_tmp[q]
        ptr_2[p_3] += 1
    end
    qos_stop = fmt_lvl_2_ptr[fmt_lvl_stop + 1] - 1
    resize!(fmt_lvl_3_val, qos_stop)
    result
end
