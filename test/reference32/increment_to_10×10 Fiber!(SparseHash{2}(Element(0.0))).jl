begin
    fmt_lvl = ex.body.body.lhs.tns.bind.lvl
    fmt_lvl_qos_fill = length(fmt_lvl.tbl)
    fmt_lvl_qos_stop = fmt_lvl_qos_fill
    fmt_lvl_ptr = ex.body.body.lhs.tns.bind.lvl.ptr
    fmt_lvl_tbl = ex.body.body.lhs.tns.bind.lvl.tbl
    fmt_lvl_srt = ex.body.body.lhs.tns.bind.lvl.srt
    fmt_lvl_2 = fmt_lvl.lvl
    fmt_lvl_val = fmt_lvl.lvl.val
    arr_2_lvl = ex.body.body.rhs.tns.bind.lvl
    arr_2_lvl_ptr = ex.body.body.rhs.tns.bind.lvl.ptr
    arr_2_lvl_tbl1 = ex.body.body.rhs.tns.bind.lvl.tbl[1]
    arr_2_lvl_tbl2 = ex.body.body.rhs.tns.bind.lvl.tbl[2]
    arr_2_lvl_val = arr_2_lvl.lvl.val
    arr_2_lvl.shape[1] == fmt_lvl.shape[1] || throw(DimensionMismatch("mismatched dimension limits ($(arr_2_lvl.shape[1]) != $(fmt_lvl.shape[1]))"))
    arr_2_lvl.shape[2] == fmt_lvl.shape[2] || throw(DimensionMismatch("mismatched dimension limits ($(arr_2_lvl.shape[2]) != $(fmt_lvl.shape[2]))"))
    for fmt_lvl_p = 1 + 1:-1:2
        fmt_lvl_ptr[fmt_lvl_p] = fmt_lvl_ptr[fmt_lvl_p] - fmt_lvl_ptr[fmt_lvl_p - 1]
    end
    fmt_lvl_ptr[1] = 1
    fmt_lvl_qos_fill = length(fmt_lvl_tbl)
    arr_2_lvl_q = arr_2_lvl_ptr[1]
    arr_2_lvl_q_stop = arr_2_lvl_ptr[1 + 1]
    if arr_2_lvl_q < arr_2_lvl_q_stop
        arr_2_lvl_i_stop = arr_2_lvl_tbl2[arr_2_lvl_q_stop - 1]
    else
        arr_2_lvl_i_stop = 0
    end
    phase_stop = min(arr_2_lvl.shape[2], arr_2_lvl_i_stop)
    if phase_stop >= 1
        j = 1
        if arr_2_lvl_tbl2[arr_2_lvl_q] < 1
            arr_2_lvl_q = Finch.scansearch(arr_2_lvl_tbl2, 1, arr_2_lvl_q, arr_2_lvl_q_stop - 1)
        end
        while j <= phase_stop
            arr_2_lvl_i = arr_2_lvl_tbl2[arr_2_lvl_q]
            arr_2_lvl_q_step = arr_2_lvl_q
            if arr_2_lvl_tbl2[arr_2_lvl_q] == arr_2_lvl_i
                arr_2_lvl_q_step = Finch.scansearch(arr_2_lvl_tbl2, arr_2_lvl_i + 1, arr_2_lvl_q, arr_2_lvl_q_stop - 1)
            end
            phase_stop_2 = min(phase_stop, arr_2_lvl_i)
            if arr_2_lvl_i == phase_stop_2
                arr_2_lvl_q_2 = arr_2_lvl_q
                if arr_2_lvl_q < arr_2_lvl_q_step
                    arr_2_lvl_i_stop_2 = arr_2_lvl_tbl1[arr_2_lvl_q_step - 1]
                else
                    arr_2_lvl_i_stop_2 = 0
                end
                phase_stop_3 = min(arr_2_lvl.shape[1], arr_2_lvl_i_stop_2)
                if phase_stop_3 >= 1
                    i = 1
                    if arr_2_lvl_tbl1[arr_2_lvl_q] < 1
                        arr_2_lvl_q_2 = Finch.scansearch(arr_2_lvl_tbl1, 1, arr_2_lvl_q, arr_2_lvl_q_step - 1)
                    end
                    while i <= phase_stop_3
                        arr_2_lvl_i_2 = arr_2_lvl_tbl1[arr_2_lvl_q_2]
                        phase_stop_4 = min(phase_stop_3, arr_2_lvl_i_2)
                        if arr_2_lvl_i_2 == phase_stop_4
                            arr_2_lvl_2_val = arr_2_lvl_val[arr_2_lvl_q_2]
                            fmt_lvl_key_2 = (1, (phase_stop_4, phase_stop_2))
                            fmt_lvl_q_2 = get(fmt_lvl_tbl, fmt_lvl_key_2, fmt_lvl_qos_fill + 1)
                            if fmt_lvl_q_2 > fmt_lvl_qos_stop
                                fmt_lvl_qos_stop = max(fmt_lvl_qos_stop << 1, 1)
                                Finch.resize_if_smaller!(fmt_lvl_val, fmt_lvl_qos_stop)
                                Finch.fill_range!(fmt_lvl_val, 0.0, fmt_lvl_q_2, fmt_lvl_qos_stop)
                            end
                            fmt_lvl_val[fmt_lvl_q_2] = arr_2_lvl_2_val + fmt_lvl_val[fmt_lvl_q_2]
                            if fmt_lvl_q_2 > fmt_lvl_qos_fill
                                fmt_lvl_qos_fill = fmt_lvl_q_2
                                fmt_lvl_tbl[fmt_lvl_key_2] = fmt_lvl_q_2
                                fmt_lvl_ptr[1 + 1] += 1
                            end
                            arr_2_lvl_q_2 += 1
                        end
                        i = phase_stop_4 + 1
                    end
                end
                arr_2_lvl_q = arr_2_lvl_q_step
            end
            j = phase_stop_2 + 1
        end
    end
    resize!(fmt_lvl_srt, length(fmt_lvl_tbl))
    copyto!(fmt_lvl_srt, pairs(fmt_lvl_tbl))
    sort!(fmt_lvl_srt, by = hashkeycmp)
    for p = 2:1 + 1
        fmt_lvl_ptr[p] += fmt_lvl_ptr[p - 1]
    end
    resize!(fmt_lvl_ptr, 1 + 1)
    qos = fmt_lvl_ptr[end] - 1
    resize!(fmt_lvl_srt, qos)
    resize!(fmt_lvl_val, qos)
    (fmt = Fiber((SparseHashLevel){2, Tuple{Int32, Int32}}(fmt_lvl_2, (fmt_lvl.shape[1], fmt_lvl.shape[2]), fmt_lvl_ptr, fmt_lvl_tbl, fmt_lvl_srt)),)
end
