begin
    C_lvl = (ex.bodies[1]).tns.bind.lvl
    C_lvl_2 = C_lvl.lvl
    A_lvl = (ex.bodies[2]).body.rhs.tns.bind.lvl
    A_lvl_2 = A_lvl.lvl
    I = ((ex.bodies[2]).body.rhs.idxs[1]).tns.bind
    i_stop = -(first(I)) + last(I) + 1
    C_lvl_qos_stop = 0
    Finch.resize_if_smaller!(C_lvl.ptr, 1 + 1)
    Finch.fill_range!(C_lvl.ptr, 0, 1 + 1, 1 + 1)
    C_lvl_qos = 0 + 1
    A_lvl_q = A_lvl.ptr[1]
    A_lvl_q_stop = A_lvl.ptr[1 + 1]
    if A_lvl_q < A_lvl_q_stop
        A_lvl_i1 = A_lvl.idx[A_lvl_q_stop - 1]
    else
        A_lvl_i1 = 0
    end
    phase_stop = min(i_stop, -(first(I)) + 1 + A_lvl_i1)
    if phase_stop >= 1
        i = 1
        if A_lvl.idx[A_lvl_q] < 1 + (first(I) - 1)
            A_lvl_q = Finch.scansearch(A_lvl.idx, 1 + (first(I) - 1), A_lvl_q, A_lvl_q_stop - 1)
        end
        while i <= phase_stop
            A_lvl_i = A_lvl.idx[A_lvl_q]
            phase_stop_2 = min(phase_stop, -(first(I)) + 1 + A_lvl_i)
            if A_lvl_i == phase_stop_2 + (first(I) - 1)
                A_lvl_2_val_2 = A_lvl_2.val[A_lvl_q]
                if C_lvl_qos > C_lvl_qos_stop
                    C_lvl_qos_stop = max(C_lvl_qos_stop << 1, 1)
                    Finch.resize_if_smaller!(C_lvl.idx, C_lvl_qos_stop)
                    Finch.resize_if_smaller!(C_lvl_2.val, C_lvl_qos_stop)
                    Finch.fill_range!(C_lvl_2.val, 0.0, C_lvl_qos, C_lvl_qos_stop)
                end
                C_lvl_2.val[C_lvl_qos] = A_lvl_2_val_2
                C_lvl.idx[C_lvl_qos] = phase_stop_2
                C_lvl_qos += 1
                A_lvl_q += 1
            end
            i = phase_stop_2 + 1
        end
    end
    C_lvl.ptr[1 + 1] = (C_lvl_qos - 0) - 1
    for p = 2:1 + 1
        C_lvl.ptr[p] += C_lvl.ptr[p - 1]
    end
    resize!(C_lvl.ptr, 1 + 1)
    qos = C_lvl.ptr[end] - 1
    resize!(C_lvl.idx, qos)
    resize!(C_lvl_2.val, qos)
    (C = Fiber((SparseListLevel){Int64, Int64}(C_lvl_2, last(I) + -((first(I) - 1)), C_lvl.ptr, C_lvl.idx)),)
end
