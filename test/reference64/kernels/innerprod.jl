begin
    B_lvl = ((ex.bodies[1]).bodies[1]).tns.bind.lvl
    B_lvl_2 = B_lvl.lvl
    B_lvl_2_ptr = B_lvl_2.ptr
    B_lvl_2_idx = B_lvl_2.idx
    B_lvl_3 = B_lvl_2.lvl
    B_lvl_3_val = B_lvl_3.val
    A_lvl = (((ex.bodies[1]).bodies[2]).body.body.body.rhs.args[1]).tns.bind.lvl
    A_lvl_stop = A_lvl.shape
    A_lvl_2 = A_lvl.lvl
    A_lvl_2_ptr = A_lvl_2.ptr
    A_lvl_2_idx = A_lvl_2.idx
    A_lvl_2_stop = A_lvl_2.shape
    A_lvl_3 = A_lvl_2.lvl
    A_lvl_3_val = A_lvl_3.val
    B_lvl_2_qos_fill = 0
    B_lvl_2_qos_stop = 0
    B_lvl_2_prev_pos = 0
    Finch.resize_if_smaller!(B_lvl_2_ptr, A_lvl_stop + 1)
    Finch.fill_range!(B_lvl_2_ptr, 0, 1 + 1, A_lvl_stop + 1)
    for j_4 = 1:A_lvl_stop
        B_lvl_q = (1 - 1) * A_lvl_stop + j_4
        A_lvl_q = (1 - 1) * A_lvl_stop + j_4
        B_lvl_2_qos = B_lvl_2_qos_fill + 1
        B_lvl_2_prev_pos < B_lvl_q || throw(FinchProtocolError("SparseListLevels cannot be updated multiple times"))
        for i_4 = 1:A_lvl_stop
            if B_lvl_2_qos > B_lvl_2_qos_stop
                B_lvl_2_qos_stop = max(B_lvl_2_qos_stop << 1, 1)
                Finch.resize_if_smaller!(B_lvl_2_idx, B_lvl_2_qos_stop)
                Finch.resize_if_smaller!(B_lvl_3_val, B_lvl_2_qos_stop)
                Finch.fill_range!(B_lvl_3_val, 0.0, B_lvl_2_qos, B_lvl_2_qos_stop)
            end
            B_lvl_2dirty = false
            A_lvl_q_2 = (1 - 1) * A_lvl_stop + i_4
            A_lvl_2_q = A_lvl_2_ptr[A_lvl_q]
            A_lvl_2_q_stop = A_lvl_2_ptr[A_lvl_q + 1]
            if A_lvl_2_q < A_lvl_2_q_stop
                A_lvl_2_i1 = A_lvl_2_idx[A_lvl_2_q_stop - 1]
            else
                A_lvl_2_i1 = 0
            end
            A_lvl_2_q_2 = A_lvl_2_ptr[A_lvl_q_2]
            A_lvl_2_q_stop_2 = A_lvl_2_ptr[A_lvl_q_2 + 1]
            if A_lvl_2_q_2 < A_lvl_2_q_stop_2
                A_lvl_2_i1_2 = A_lvl_2_idx[A_lvl_2_q_stop_2 - 1]
            else
                A_lvl_2_i1_2 = 0
            end
            phase_stop = min(A_lvl_2_stop, A_lvl_2_i1, A_lvl_2_i1_2)
            if phase_stop >= 1
                k = 1
                if A_lvl_2_idx[A_lvl_2_q] < 1
                    A_lvl_2_q = Finch.scansearch(A_lvl_2_idx, 1, A_lvl_2_q, A_lvl_2_q_stop - 1)
                end
                if A_lvl_2_idx[A_lvl_2_q_2] < 1
                    A_lvl_2_q_2 = Finch.scansearch(A_lvl_2_idx, 1, A_lvl_2_q_2, A_lvl_2_q_stop_2 - 1)
                end
                while k <= phase_stop
                    A_lvl_2_i = A_lvl_2_idx[A_lvl_2_q]
                    A_lvl_2_i_2 = A_lvl_2_idx[A_lvl_2_q_2]
                    phase_stop_2 = min(A_lvl_2_i_2, phase_stop, A_lvl_2_i)
                    if A_lvl_2_i == phase_stop_2 && A_lvl_2_i_2 == phase_stop_2
                        A_lvl_3_val_2 = A_lvl_3_val[A_lvl_2_q]
                        A_lvl_3_val_3 = A_lvl_3_val[A_lvl_2_q_2]
                        B_lvl_2dirty = true
                        B_lvl_3_val[B_lvl_2_qos] = A_lvl_3_val_3 * A_lvl_3_val_2 + B_lvl_3_val[B_lvl_2_qos]
                        A_lvl_2_q += 1
                        A_lvl_2_q_2 += 1
                    elseif A_lvl_2_i_2 == phase_stop_2
                        A_lvl_2_q_2 += 1
                    elseif A_lvl_2_i == phase_stop_2
                        A_lvl_2_q += 1
                    end
                    k = phase_stop_2 + 1
                end
            end
            if B_lvl_2dirty
                B_lvl_2_idx[B_lvl_2_qos] = i_4
                B_lvl_2_qos += 1
                B_lvl_2_prev_pos = B_lvl_q
            end
        end
        B_lvl_2_ptr[B_lvl_q + 1] += (B_lvl_2_qos - B_lvl_2_qos_fill) - 1
        B_lvl_2_qos_fill = B_lvl_2_qos - 1
    end
    resize!(B_lvl_2_ptr, A_lvl_stop + 1)
    for p = 1:A_lvl_stop
        B_lvl_2_ptr[p + 1] += B_lvl_2_ptr[p]
    end
    qos_stop = B_lvl_2_ptr[A_lvl_stop + 1] - 1
    resize!(B_lvl_2_idx, qos_stop)
    resize!(B_lvl_3_val, qos_stop)
    (B = Tensor((DenseLevel){Int64}((SparseListLevel){Int64}(ElementLevel{0.0, Float64, Int64}(B_lvl_3_val), A_lvl_stop, B_lvl_2_ptr, B_lvl_2_idx), A_lvl_stop)),)
end
