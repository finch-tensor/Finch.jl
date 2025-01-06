begin
    C_lvl = ((ex.bodies[1]).bodies[1]).tns.bind.lvl
    C_lvl_ptr = C_lvl.ptr
    C_lvl_idx = C_lvl.idx
    C_lvl_2 = C_lvl.lvl
    C_lvl_val = C_lvl.lvl.val
    A_lvl = (((ex.bodies[1]).bodies[2]).body.rhs.args[1]).tns.bind.lvl
    A_lvl_ptr = A_lvl.ptr
    A_lvl_idx = A_lvl.idx
    A_lvl_val = A_lvl.lvl.val
    B_lvl = (((ex.bodies[1]).bodies[2]).body.rhs.args[2]).tns.bind.lvl
    B_lvl_ptr = B_lvl.ptr
    B_lvl_idx = B_lvl.idx
    B_lvl_val = B_lvl.lvl.val
    C_lvl_qos_stop = 0
    Finch.resize_if_smaller!(C_lvl_ptr, 1 + 1)
    Finch.fill_range!(C_lvl_ptr, 0, 1 + 1, 1 + 1)
    C_lvl_qos = 0 + 1
    0 < 1 || throw(FinchProtocolError("SparseListLevels cannot be updated multiple times"))
    phase_stop = min(C_lvl.shape, 0)
    if phase_stop >= 1
        for i_6 = 1:phase_stop
            if C_lvl_qos > C_lvl_qos_stop
                C_lvl_qos_stop = max(C_lvl_qos_stop << 1, 1)
                Finch.resize_if_smaller!(C_lvl_idx, C_lvl_qos_stop)
                Finch.resize_if_smaller!(C_lvl_val, C_lvl_qos_stop)
                Finch.fill_range!(C_lvl_val, 0.0, C_lvl_qos, C_lvl_qos_stop)
            end
            C_lvl_val[C_lvl_qos] = missing
            C_lvl_idx[C_lvl_qos] = i_6
            C_lvl_qos += 1
        end
    end
    phase_stop_2 = min(C_lvl.shape, A_lvl.shape, 10)
    if phase_stop_2 >= 1
        A_lvl_q = A_lvl_ptr[1]
        A_lvl_q_stop = A_lvl_ptr[1 + 1]
        if A_lvl_q < A_lvl_q_stop
            A_lvl_i1 = A_lvl_idx[A_lvl_q_stop - 1]
        else
            A_lvl_i1 = 0
        end
        phase_stop_3 = min(A_lvl_i1, phase_stop_2)
        if phase_stop_3 >= 1
            if A_lvl_idx[A_lvl_q] < 1
                A_lvl_q = Finch.scansearch(A_lvl_idx, 1, A_lvl_q, A_lvl_q_stop - 1)
            end
            while true
                A_lvl_i = A_lvl_idx[A_lvl_q]
                if A_lvl_i < phase_stop_3
                    A_lvl_2_val = A_lvl_val[A_lvl_q]
                    if C_lvl_qos > C_lvl_qos_stop
                        C_lvl_qos_stop = max(C_lvl_qos_stop << 1, 1)
                        Finch.resize_if_smaller!(C_lvl_idx, C_lvl_qos_stop)
                        Finch.resize_if_smaller!(C_lvl_val, C_lvl_qos_stop)
                        Finch.fill_range!(C_lvl_val, 0.0, C_lvl_qos, C_lvl_qos_stop)
                    end
                    C_lvl_val[C_lvl_qos] = A_lvl_2_val
                    C_lvl_idx[C_lvl_qos] = A_lvl_i
                    C_lvl_qos += 1
                    A_lvl_q += 1
                else
                    phase_stop_5 = min(phase_stop_3, A_lvl_i)
                    if A_lvl_i == phase_stop_5
                        A_lvl_2_val = A_lvl_val[A_lvl_q]
                        if C_lvl_qos > C_lvl_qos_stop
                            C_lvl_qos_stop = max(C_lvl_qos_stop << 1, 1)
                            Finch.resize_if_smaller!(C_lvl_idx, C_lvl_qos_stop)
                            Finch.resize_if_smaller!(C_lvl_val, C_lvl_qos_stop)
                            Finch.fill_range!(C_lvl_val, 0.0, C_lvl_qos, C_lvl_qos_stop)
                        end
                        C_lvl_val[C_lvl_qos] = A_lvl_2_val
                        C_lvl_idx[C_lvl_qos] = phase_stop_5
                        C_lvl_qos += 1
                        A_lvl_q += 1
                    end
                    break
                end
            end
        end
    end
    phase_start_6 = max(1, 1 + A_lvl.shape)
    phase_stop_7 = min(C_lvl.shape, 10)
    if phase_stop_7 >= phase_start_6
        for i_14 = phase_start_6:phase_stop_7
            if C_lvl_qos > C_lvl_qos_stop
                C_lvl_qos_stop = max(C_lvl_qos_stop << 1, 1)
                Finch.resize_if_smaller!(C_lvl_idx, C_lvl_qos_stop)
                Finch.resize_if_smaller!(C_lvl_val, C_lvl_qos_stop)
                Finch.fill_range!(C_lvl_val, 0.0, C_lvl_qos, C_lvl_qos_stop)
            end
            C_lvl_val[C_lvl_qos] = missing
            C_lvl_idx[C_lvl_qos] = i_14
            C_lvl_qos += 1
        end
    end
    phase_stop_8 = min(C_lvl.shape, 0, 10 + B_lvl.shape)
    if phase_stop_8 >= 11
        B_lvl_q = B_lvl_ptr[1]
        B_lvl_q_stop = B_lvl_ptr[1 + 1]
        if B_lvl_q < B_lvl_q_stop
            B_lvl_i1 = B_lvl_idx[B_lvl_q_stop - 1]
        else
            B_lvl_i1 = 0
        end
        phase_stop_9 = min(phase_stop_8, 10 + B_lvl_i1)
        if phase_stop_9 >= 11
            if B_lvl_idx[B_lvl_q] < -10 + 11
                B_lvl_q = Finch.scansearch(B_lvl_idx, -10 + 11, B_lvl_q, B_lvl_q_stop - 1)
            end
            while true
                B_lvl_i = B_lvl_idx[B_lvl_q]
                phase_stop_10 = 10 + B_lvl_i
                if phase_stop_10 < phase_stop_9
                    B_lvl_2_val = B_lvl_val[B_lvl_q]
                    if C_lvl_qos > C_lvl_qos_stop
                        C_lvl_qos_stop = max(C_lvl_qos_stop << 1, 1)
                        Finch.resize_if_smaller!(C_lvl_idx, C_lvl_qos_stop)
                        Finch.resize_if_smaller!(C_lvl_val, C_lvl_qos_stop)
                        Finch.fill_range!(C_lvl_val, 0.0, C_lvl_qos, C_lvl_qos_stop)
                    end
                    C_lvl_val[C_lvl_qos] = B_lvl_2_val
                    C_lvl_idx[C_lvl_qos] = phase_stop_10
                    C_lvl_qos += 1
                    B_lvl_q += 1
                else
                    phase_stop_11 = min(phase_stop_9, 10 + B_lvl_i)
                    if B_lvl_i == -10 + phase_stop_11
                        B_lvl_2_val = B_lvl_val[B_lvl_q]
                        if C_lvl_qos > C_lvl_qos_stop
                            C_lvl_qos_stop = max(C_lvl_qos_stop << 1, 1)
                            Finch.resize_if_smaller!(C_lvl_idx, C_lvl_qos_stop)
                            Finch.resize_if_smaller!(C_lvl_val, C_lvl_qos_stop)
                            Finch.fill_range!(C_lvl_val, 0.0, C_lvl_qos, C_lvl_qos_stop)
                        end
                        C_lvl_val[C_lvl_qos] = B_lvl_2_val
                        C_lvl_idx[C_lvl_qos] = phase_stop_11
                        C_lvl_qos += 1
                        B_lvl_q += 1
                    end
                    break
                end
            end
        end
    end
    phase_stop_13 = min(C_lvl.shape, A_lvl.shape, 10 + B_lvl.shape)
    if phase_stop_13 >= 11
        A_lvl_q = A_lvl_ptr[1]
        A_lvl_q_stop = A_lvl_ptr[1 + 1]
        if A_lvl_q < A_lvl_q_stop
            A_lvl_i1 = A_lvl_idx[A_lvl_q_stop - 1]
        else
            A_lvl_i1 = 0
        end
        B_lvl_q = B_lvl_ptr[1]
        B_lvl_q_stop = B_lvl_ptr[1 + 1]
        if B_lvl_q < B_lvl_q_stop
            B_lvl_i1 = B_lvl_idx[B_lvl_q_stop - 1]
        else
            B_lvl_i1 = 0
        end
        phase_stop_14 = min(A_lvl_i1, 10 + B_lvl_i1, phase_stop_13)
        if phase_stop_14 >= 11
            i = 11
            if A_lvl_idx[A_lvl_q] < 11
                A_lvl_q = Finch.scansearch(A_lvl_idx, 11, A_lvl_q, A_lvl_q_stop - 1)
            end
            if B_lvl_idx[B_lvl_q] < -10 + 11
                B_lvl_q = Finch.scansearch(B_lvl_idx, -10 + 11, B_lvl_q, B_lvl_q_stop - 1)
            end
            while i <= phase_stop_14
                A_lvl_i = A_lvl_idx[A_lvl_q]
                B_lvl_i = B_lvl_idx[B_lvl_q]
                phase_stop_15 = min(A_lvl_i, 10 + B_lvl_i, phase_stop_14)
                if A_lvl_i == phase_stop_15 && B_lvl_i == -10 + phase_stop_15
                    A_lvl_2_val_2 = A_lvl_val[A_lvl_q]
                    B_lvl_2_val_2 = B_lvl_val[B_lvl_q]
                    if C_lvl_qos > C_lvl_qos_stop
                        C_lvl_qos_stop = max(C_lvl_qos_stop << 1, 1)
                        Finch.resize_if_smaller!(C_lvl_idx, C_lvl_qos_stop)
                        Finch.resize_if_smaller!(C_lvl_val, C_lvl_qos_stop)
                        Finch.fill_range!(C_lvl_val, 0.0, C_lvl_qos, C_lvl_qos_stop)
                    end
                    C_lvl_val[C_lvl_qos] = coalesce(A_lvl_2_val_2, B_lvl_2_val_2)
                    C_lvl_idx[C_lvl_qos] = phase_stop_15
                    C_lvl_qos += 1
                    A_lvl_q += 1
                    B_lvl_q += 1
                elseif B_lvl_i == -10 + phase_stop_15
                    B_lvl_q += 1
                elseif A_lvl_i == phase_stop_15
                    A_lvl_2_val_2 = A_lvl_val[A_lvl_q]
                    if C_lvl_qos > C_lvl_qos_stop
                        C_lvl_qos_stop = max(C_lvl_qos_stop << 1, 1)
                        Finch.resize_if_smaller!(C_lvl_idx, C_lvl_qos_stop)
                        Finch.resize_if_smaller!(C_lvl_val, C_lvl_qos_stop)
                        Finch.fill_range!(C_lvl_val, 0.0, C_lvl_qos, C_lvl_qos_stop)
                    end
                    C_lvl_val[C_lvl_qos] = coalesce(A_lvl_2_val_2, 0.0)
                    C_lvl_idx[C_lvl_qos] = phase_stop_15
                    C_lvl_qos += 1
                    A_lvl_q += 1
                end
                i = phase_stop_15 + 1
            end
        end
        phase_start_16 = max(11 + B_lvl_i1, 11)
        phase_stop_17 = min(A_lvl_i1, phase_stop_13)
        if phase_stop_17 >= phase_start_16
            if A_lvl_idx[A_lvl_q] < phase_start_16
                A_lvl_q = Finch.scansearch(A_lvl_idx, phase_start_16, A_lvl_q, A_lvl_q_stop - 1)
            end
            while true
                A_lvl_i = A_lvl_idx[A_lvl_q]
                if A_lvl_i < phase_stop_17
                    A_lvl_2_val_3 = A_lvl_val[A_lvl_q]
                    if C_lvl_qos > C_lvl_qos_stop
                        C_lvl_qos_stop = max(C_lvl_qos_stop << 1, 1)
                        Finch.resize_if_smaller!(C_lvl_idx, C_lvl_qos_stop)
                        Finch.resize_if_smaller!(C_lvl_val, C_lvl_qos_stop)
                        Finch.fill_range!(C_lvl_val, 0.0, C_lvl_qos, C_lvl_qos_stop)
                    end
                    C_lvl_val[C_lvl_qos] = coalesce(A_lvl_2_val_3, 0.0)
                    C_lvl_idx[C_lvl_qos] = A_lvl_i
                    C_lvl_qos += 1
                    A_lvl_q += 1
                else
                    phase_stop_19 = min(A_lvl_i, phase_stop_17)
                    if A_lvl_i == phase_stop_19
                        A_lvl_2_val_3 = A_lvl_val[A_lvl_q]
                        if C_lvl_qos > C_lvl_qos_stop
                            C_lvl_qos_stop = max(C_lvl_qos_stop << 1, 1)
                            Finch.resize_if_smaller!(C_lvl_idx, C_lvl_qos_stop)
                            Finch.resize_if_smaller!(C_lvl_val, C_lvl_qos_stop)
                            Finch.fill_range!(C_lvl_val, 0.0, C_lvl_qos, C_lvl_qos_stop)
                        end
                        C_lvl_val[C_lvl_qos] = coalesce(A_lvl_2_val_3, 0.0)
                        C_lvl_idx[C_lvl_qos] = phase_stop_19
                        C_lvl_qos += 1
                        A_lvl_q += 1
                    end
                    break
                end
            end
        end
    end
    phase_start_19 = max(11, 1 + A_lvl.shape)
    phase_stop_21 = min(C_lvl.shape, 10 + B_lvl.shape)
    if phase_stop_21 >= phase_start_19
        B_lvl_q = B_lvl_ptr[1]
        B_lvl_q_stop = B_lvl_ptr[1 + 1]
        if B_lvl_q < B_lvl_q_stop
            B_lvl_i1 = B_lvl_idx[B_lvl_q_stop - 1]
        else
            B_lvl_i1 = 0
        end
        phase_stop_22 = min(10 + B_lvl_i1, phase_stop_21)
        if phase_stop_22 >= phase_start_19
            if B_lvl_idx[B_lvl_q] < -10 + phase_start_19
                B_lvl_q = Finch.scansearch(B_lvl_idx, -10 + phase_start_19, B_lvl_q, B_lvl_q_stop - 1)
            end
            while true
                B_lvl_i = B_lvl_idx[B_lvl_q]
                phase_stop_23 = 10 + B_lvl_i
                if phase_stop_23 < phase_stop_22
                    B_lvl_2_val_4 = B_lvl_val[B_lvl_q]
                    if C_lvl_qos > C_lvl_qos_stop
                        C_lvl_qos_stop = max(C_lvl_qos_stop << 1, 1)
                        Finch.resize_if_smaller!(C_lvl_idx, C_lvl_qos_stop)
                        Finch.resize_if_smaller!(C_lvl_val, C_lvl_qos_stop)
                        Finch.fill_range!(C_lvl_val, 0.0, C_lvl_qos, C_lvl_qos_stop)
                    end
                    C_lvl_val[C_lvl_qos] = B_lvl_2_val_4
                    C_lvl_idx[C_lvl_qos] = phase_stop_23
                    C_lvl_qos += 1
                    B_lvl_q += 1
                else
                    phase_stop_24 = min(10 + B_lvl_i, phase_stop_22)
                    if B_lvl_i == -10 + phase_stop_24
                        B_lvl_2_val_4 = B_lvl_val[B_lvl_q]
                        if C_lvl_qos > C_lvl_qos_stop
                            C_lvl_qos_stop = max(C_lvl_qos_stop << 1, 1)
                            Finch.resize_if_smaller!(C_lvl_idx, C_lvl_qos_stop)
                            Finch.resize_if_smaller!(C_lvl_val, C_lvl_qos_stop)
                            Finch.fill_range!(C_lvl_val, 0.0, C_lvl_qos, C_lvl_qos_stop)
                        end
                        C_lvl_val[C_lvl_qos] = B_lvl_2_val_4
                        C_lvl_idx[C_lvl_qos] = phase_stop_24
                        C_lvl_qos += 1
                        B_lvl_q += 1
                    end
                    break
                end
            end
        end
    end
    phase_start_24 = max(1, 11 + B_lvl.shape)
    phase_stop_26 = min(C_lvl.shape, 0)
    if phase_stop_26 >= phase_start_24
        for i_39 = phase_start_24:phase_stop_26
            if C_lvl_qos > C_lvl_qos_stop
                C_lvl_qos_stop = max(C_lvl_qos_stop << 1, 1)
                Finch.resize_if_smaller!(C_lvl_idx, C_lvl_qos_stop)
                Finch.resize_if_smaller!(C_lvl_val, C_lvl_qos_stop)
                Finch.fill_range!(C_lvl_val, 0.0, C_lvl_qos, C_lvl_qos_stop)
            end
            C_lvl_val[C_lvl_qos] = missing
            C_lvl_idx[C_lvl_qos] = i_39
            C_lvl_qos += 1
        end
    end
    phase_start_25 = max(1, 11 + B_lvl.shape)
    phase_stop_27 = min(C_lvl.shape, A_lvl.shape)
    if phase_stop_27 >= phase_start_25
        A_lvl_q = A_lvl_ptr[1]
        A_lvl_q_stop = A_lvl_ptr[1 + 1]
        if A_lvl_q < A_lvl_q_stop
            A_lvl_i1 = A_lvl_idx[A_lvl_q_stop - 1]
        else
            A_lvl_i1 = 0
        end
        phase_stop_28 = min(A_lvl_i1, phase_stop_27)
        if phase_stop_28 >= phase_start_25
            if A_lvl_idx[A_lvl_q] < phase_start_25
                A_lvl_q = Finch.scansearch(A_lvl_idx, phase_start_25, A_lvl_q, A_lvl_q_stop - 1)
            end
            while true
                A_lvl_i = A_lvl_idx[A_lvl_q]
                if A_lvl_i < phase_stop_28
                    A_lvl_2_val_4 = A_lvl_val[A_lvl_q]
                    if C_lvl_qos > C_lvl_qos_stop
                        C_lvl_qos_stop = max(C_lvl_qos_stop << 1, 1)
                        Finch.resize_if_smaller!(C_lvl_idx, C_lvl_qos_stop)
                        Finch.resize_if_smaller!(C_lvl_val, C_lvl_qos_stop)
                        Finch.fill_range!(C_lvl_val, 0.0, C_lvl_qos, C_lvl_qos_stop)
                    end
                    C_lvl_val[C_lvl_qos] = A_lvl_2_val_4
                    C_lvl_idx[C_lvl_qos] = A_lvl_i
                    C_lvl_qos += 1
                    A_lvl_q += 1
                else
                    phase_stop_30 = min(A_lvl_i, phase_stop_28)
                    if A_lvl_i == phase_stop_30
                        A_lvl_2_val_4 = A_lvl_val[A_lvl_q]
                        if C_lvl_qos > C_lvl_qos_stop
                            C_lvl_qos_stop = max(C_lvl_qos_stop << 1, 1)
                            Finch.resize_if_smaller!(C_lvl_idx, C_lvl_qos_stop)
                            Finch.resize_if_smaller!(C_lvl_val, C_lvl_qos_stop)
                            Finch.fill_range!(C_lvl_val, 0.0, C_lvl_qos, C_lvl_qos_stop)
                        end
                        C_lvl_val[C_lvl_qos] = A_lvl_2_val_4
                        C_lvl_idx[C_lvl_qos] = phase_stop_30
                        C_lvl_qos += 1
                        A_lvl_q += 1
                    end
                    break
                end
            end
        end
    end
    phase_start_29 = max(1, 1 + A_lvl.shape, 11 + B_lvl.shape)
    phase_stop_32 = C_lvl.shape
    if phase_stop_32 >= phase_start_29
        for i_47 = phase_start_29:phase_stop_32
            if C_lvl_qos > C_lvl_qos_stop
                C_lvl_qos_stop = max(C_lvl_qos_stop << 1, 1)
                Finch.resize_if_smaller!(C_lvl_idx, C_lvl_qos_stop)
                Finch.resize_if_smaller!(C_lvl_val, C_lvl_qos_stop)
                Finch.fill_range!(C_lvl_val, 0.0, C_lvl_qos, C_lvl_qos_stop)
            end
            C_lvl_val[C_lvl_qos] = missing
            C_lvl_idx[C_lvl_qos] = i_47
            C_lvl_qos += 1
        end
    end
    C_lvl_ptr[1 + 1] += (C_lvl_qos - 0) - 1
    resize!(C_lvl_ptr, 1 + 1)
    for p = 1:1
        C_lvl_ptr[p + 1] += C_lvl_ptr[p]
    end
    qos_stop = C_lvl_ptr[1 + 1] - 1
    resize!(C_lvl_idx, qos_stop)
    resize!(C_lvl_val, qos_stop)
    (C = Tensor((SparseListLevel){Int64}(C_lvl_2, C_lvl.shape, C_lvl_ptr, C_lvl_idx)),)
end
