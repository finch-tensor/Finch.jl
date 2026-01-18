begin
    D_data = ((ex.bodies[1]).bodies[1]).tns.bind
    C_lvl = (((ex.bodies[1]).bodies[2]).body.body.body.rhs.args[1]).tns.bind.lvl
    C_lvl_ptr = C_lvl.ptr
    C_lvl_idx = C_lvl.idx
    C_lvl_stop = C_lvl.shape
    C_lvl_2 = C_lvl.lvl
    C_lvl_2_val = C_lvl_2.val
    B_lvl = (((ex.bodies[1]).bodies[2]).body.body.body.rhs.args[2]).tns.bind.lvl
    B_lvl_stop = B_lvl.shape
    B_lvl_2 = B_lvl.lvl
    B_lvl_2_ptr = B_lvl_2.ptr
    B_lvl_2_idx = B_lvl_2.idx
    B_lvl_2_stop = B_lvl_2.shape
    B_lvl_3 = B_lvl_2.lvl
    B_lvl_3_val = B_lvl_3.val
    A_lvl = (((ex.bodies[1]).bodies[2]).body.body.body.rhs.args[3]).tns.bind.lvl
    A_lvl_stop = A_lvl.shape
    A_lvl_2 = A_lvl.lvl
    A_lvl_2_ptr = A_lvl_2.ptr
    A_lvl_2_idx = A_lvl_2.idx
    A_lvl_2_stop = A_lvl_2.shape
    A_lvl_3 = A_lvl_2.lvl
    A_lvl_3_val = A_lvl_3.val
    B_lvl_stop == C_lvl_stop || throw(
        DimensionMismatch("mismatched dimension limits ($(B_lvl_stop) != $(C_lvl_stop))")
    )
    B_lvl_stop == A_lvl_2_stop || throw(
        DimensionMismatch("mismatched dimension limits ($(B_lvl_stop) != $(A_lvl_2_stop))")
    )
    D_val = 0
    for i_3 in 1:A_lvl_stop
        A_lvl_q = (1 - 1) * A_lvl_stop + i_3
        C_lvl_q = C_lvl_ptr[1]
        C_lvl_q_stop = C_lvl_ptr[1 + 1]
        if C_lvl_q < C_lvl_q_stop
            C_lvl_i1 = C_lvl_idx[C_lvl_q_stop - 1]
        else
            C_lvl_i1 = 0
        end
        A_lvl_2_q = A_lvl_2_ptr[A_lvl_q]
        A_lvl_2_q_stop = A_lvl_2_ptr[A_lvl_q + 1]
        if A_lvl_2_q < A_lvl_2_q_stop
            A_lvl_2_i1 = A_lvl_2_idx[A_lvl_2_q_stop - 1]
        else
            A_lvl_2_i1 = 0
        end
        phase_stop = min(B_lvl_stop, C_lvl_i1, A_lvl_2_i1)
        if phase_stop >= 1
            j = 1
            while j <= phase_stop
                if C_lvl_idx[C_lvl_q] < j
                    C_lvl_q = Finch.scansearch(C_lvl_idx, j, C_lvl_q, C_lvl_q_stop - 1)
                end
                C_lvl_i2 = C_lvl_idx[C_lvl_q]
                if A_lvl_2_idx[A_lvl_2_q] < j
                    A_lvl_2_q = Finch.scansearch(
                        A_lvl_2_idx, j, A_lvl_2_q, A_lvl_2_q_stop - 1
                    )
                end
                A_lvl_2_i2 = A_lvl_2_idx[A_lvl_2_q]
                phase_stop_2 = min(phase_stop, max(C_lvl_i2, A_lvl_2_i2))
                if C_lvl_i2 == phase_stop_2 && A_lvl_2_i2 == phase_stop_2
                    C_lvl_2_val_2 = C_lvl_2_val[C_lvl_q]
                    A_lvl_3_val_2 = A_lvl_3_val[A_lvl_2_q]
                    B_lvl_q = (1 - 1) * B_lvl_stop + phase_stop_2
                    B_lvl_2_q = B_lvl_2_ptr[B_lvl_q]
                    B_lvl_2_q_stop = B_lvl_2_ptr[B_lvl_q + 1]
                    if B_lvl_2_q < B_lvl_2_q_stop
                        B_lvl_2_i1 = B_lvl_2_idx[B_lvl_2_q_stop - 1]
                    else
                        B_lvl_2_i1 = 0
                    end
                    phase_stop_3 = min(B_lvl_2_i1, B_lvl_2_stop)
                    if phase_stop_3 >= 1
                        if B_lvl_2_idx[B_lvl_2_q] < 1
                            B_lvl_2_q = Finch.scansearch(
                                B_lvl_2_idx, 1, B_lvl_2_q, B_lvl_2_q_stop - 1
                            )
                        end
                        while true
                            B_lvl_2_i = B_lvl_2_idx[B_lvl_2_q]
                            if B_lvl_2_i < phase_stop_3
                                B_lvl_3_val_2 = B_lvl_3_val[B_lvl_2_q]
                                D_val =
                                    C_lvl_2_val_2 * A_lvl_3_val_2 * B_lvl_3_val_2 + D_val
                                B_lvl_2_q += 1
                            else
                                phase_stop_5 = min(phase_stop_3, B_lvl_2_i)
                                if B_lvl_2_i == phase_stop_5
                                    B_lvl_3_val_2 = B_lvl_3_val[B_lvl_2_q]
                                    D_val += C_lvl_2_val_2 * A_lvl_3_val_2 * B_lvl_3_val_2
                                    B_lvl_2_q += 1
                                end
                                break
                            end
                        end
                    end
                    C_lvl_q += 1
                    A_lvl_2_q += 1
                elseif A_lvl_2_i2 == phase_stop_2
                    A_lvl_3_val_2 = A_lvl_3_val[A_lvl_2_q]
                    if C_lvl_idx[C_lvl_q] < phase_stop_2
                        C_lvl_q = Finch.scansearch(
                            C_lvl_idx, phase_stop_2, C_lvl_q, C_lvl_q_stop - 1
                        )
                    end
                    C_lvl_i2 = C_lvl_idx[C_lvl_q]
                    phase_stop_7 = min(C_lvl_i2, phase_stop_2)
                    if C_lvl_i2 == phase_stop_7
                        C_lvl_2_val_2 = C_lvl_2_val[C_lvl_q]
                        B_lvl_q = (1 - 1) * B_lvl_stop + phase_stop_7
                        B_lvl_2_q_2 = B_lvl_2_ptr[B_lvl_q]
                        B_lvl_2_q_stop_2 = B_lvl_2_ptr[B_lvl_q + 1]
                        if B_lvl_2_q_2 < B_lvl_2_q_stop_2
                            B_lvl_2_i1_2 = B_lvl_2_idx[B_lvl_2_q_stop_2 - 1]
                        else
                            B_lvl_2_i1_2 = 0
                        end
                        phase_stop_8 = min(B_lvl_2_stop, B_lvl_2_i1_2)
                        if phase_stop_8 >= 1
                            if B_lvl_2_idx[B_lvl_2_q_2] < 1
                                B_lvl_2_q_2 = Finch.scansearch(
                                    B_lvl_2_idx, 1, B_lvl_2_q_2, B_lvl_2_q_stop_2 - 1
                                )
                            end
                            while true
                                B_lvl_2_i_2 = B_lvl_2_idx[B_lvl_2_q_2]
                                if B_lvl_2_i_2 < phase_stop_8
                                    B_lvl_3_val_3 = B_lvl_3_val[B_lvl_2_q_2]
                                    D_val += A_lvl_3_val_2 * C_lvl_2_val_2 * B_lvl_3_val_3
                                    B_lvl_2_q_2 += 1
                                else
                                    phase_stop_10 = min(phase_stop_8, B_lvl_2_i_2)
                                    if B_lvl_2_i_2 == phase_stop_10
                                        B_lvl_3_val_3 = B_lvl_3_val[B_lvl_2_q_2]
                                        D_val +=
                                            A_lvl_3_val_2 * C_lvl_2_val_2 * B_lvl_3_val_3
                                        B_lvl_2_q_2 += 1
                                    end
                                    break
                                end
                            end
                        end
                        C_lvl_q += 1
                    end
                    A_lvl_2_q += 1
                elseif C_lvl_i2 == phase_stop_2
                    C_lvl_2_val_2 = C_lvl_2_val[C_lvl_q]
                    if A_lvl_2_idx[A_lvl_2_q] < phase_stop_2
                        A_lvl_2_q = Finch.scansearch(
                            A_lvl_2_idx, phase_stop_2, A_lvl_2_q, A_lvl_2_q_stop - 1
                        )
                    end
                    A_lvl_2_i2 = A_lvl_2_idx[A_lvl_2_q]
                    phase_stop_12 = min(A_lvl_2_i2, phase_stop_2)
                    if A_lvl_2_i2 == phase_stop_12
                        A_lvl_3_val_2 = A_lvl_3_val[A_lvl_2_q]
                        B_lvl_q = (1 - 1) * B_lvl_stop + phase_stop_12
                        B_lvl_2_q_3 = B_lvl_2_ptr[B_lvl_q]
                        B_lvl_2_q_stop_3 = B_lvl_2_ptr[B_lvl_q + 1]
                        if B_lvl_2_q_3 < B_lvl_2_q_stop_3
                            B_lvl_2_i1_3 = B_lvl_2_idx[B_lvl_2_q_stop_3 - 1]
                        else
                            B_lvl_2_i1_3 = 0
                        end
                        phase_stop_13 = min(B_lvl_2_stop, B_lvl_2_i1_3)
                        if phase_stop_13 >= 1
                            if B_lvl_2_idx[B_lvl_2_q_3] < 1
                                B_lvl_2_q_3 = Finch.scansearch(
                                    B_lvl_2_idx, 1, B_lvl_2_q_3, B_lvl_2_q_stop_3 - 1
                                )
                            end
                            while true
                                B_lvl_2_i_3 = B_lvl_2_idx[B_lvl_2_q_3]
                                if B_lvl_2_i_3 < phase_stop_13
                                    B_lvl_3_val_4 = B_lvl_3_val[B_lvl_2_q_3]
                                    D_val += C_lvl_2_val_2 * A_lvl_3_val_2 * B_lvl_3_val_4
                                    B_lvl_2_q_3 += 1
                                else
                                    phase_stop_15 = min(phase_stop_13, B_lvl_2_i_3)
                                    if B_lvl_2_i_3 == phase_stop_15
                                        B_lvl_3_val_4 = B_lvl_3_val[B_lvl_2_q_3]
                                        D_val +=
                                            C_lvl_2_val_2 * A_lvl_3_val_2 * B_lvl_3_val_4
                                        B_lvl_2_q_3 += 1
                                    end
                                    break
                                end
                            end
                        end
                        A_lvl_2_q += 1
                    end
                    C_lvl_q += 1
                else
                    if C_lvl_idx[C_lvl_q] < j
                        C_lvl_q = Finch.scansearch(C_lvl_idx, j, C_lvl_q, C_lvl_q_stop - 1)
                    end
                    if A_lvl_2_idx[A_lvl_2_q] < j
                        A_lvl_2_q = Finch.scansearch(
                            A_lvl_2_idx, j, A_lvl_2_q, A_lvl_2_q_stop - 1
                        )
                    end
                    while j <= phase_stop_2
                        C_lvl_i2 = C_lvl_idx[C_lvl_q]
                        A_lvl_2_i2 = A_lvl_2_idx[A_lvl_2_q]
                        phase_stop_17 = min(C_lvl_i2, A_lvl_2_i2, phase_stop_2)
                        if C_lvl_i2 == phase_stop_17 && A_lvl_2_i2 == phase_stop_17
                            C_lvl_2_val_2 = C_lvl_2_val[C_lvl_q]
                            A_lvl_3_val_2 = A_lvl_3_val[A_lvl_2_q]
                            B_lvl_q = (1 - 1) * B_lvl_stop + phase_stop_17
                            B_lvl_2_q_4 = B_lvl_2_ptr[B_lvl_q]
                            B_lvl_2_q_stop_4 = B_lvl_2_ptr[B_lvl_q + 1]
                            if B_lvl_2_q_4 < B_lvl_2_q_stop_4
                                B_lvl_2_i1_4 = B_lvl_2_idx[B_lvl_2_q_stop_4 - 1]
                            else
                                B_lvl_2_i1_4 = 0
                            end
                            phase_stop_18 = min(B_lvl_2_stop, B_lvl_2_i1_4)
                            if phase_stop_18 >= 1
                                if B_lvl_2_idx[B_lvl_2_q_4] < 1
                                    B_lvl_2_q_4 = Finch.scansearch(
                                        B_lvl_2_idx, 1, B_lvl_2_q_4, B_lvl_2_q_stop_4 - 1
                                    )
                                end
                                while true
                                    B_lvl_2_i_4 = B_lvl_2_idx[B_lvl_2_q_4]
                                    if B_lvl_2_i_4 < phase_stop_18
                                        B_lvl_3_val_5 = B_lvl_3_val[B_lvl_2_q_4]
                                        D_val +=
                                            C_lvl_2_val_2 * A_lvl_3_val_2 * B_lvl_3_val_5
                                        B_lvl_2_q_4 += 1
                                    else
                                        phase_stop_20 = min(phase_stop_18, B_lvl_2_i_4)
                                        if B_lvl_2_i_4 == phase_stop_20
                                            B_lvl_3_val_5 = B_lvl_3_val[B_lvl_2_q_4]
                                            D_val +=
                                                C_lvl_2_val_2 * A_lvl_3_val_2 *
                                                B_lvl_3_val_5
                                            B_lvl_2_q_4 += 1
                                        end
                                        break
                                    end
                                end
                            end
                            C_lvl_q += 1
                            A_lvl_2_q += 1
                        elseif A_lvl_2_i2 == phase_stop_17
                            A_lvl_2_q += 1
                        elseif C_lvl_i2 == phase_stop_17
                            C_lvl_q += 1
                        end
                        j = phase_stop_17 + 1
                    end
                end
                j = phase_stop_2 + 1
            end
        end
    end
    D_data.val = D_val
    (D=D_data,)
end
