begin
    P_lvl = ((ex.bodies[1]).bodies[1]).tns.bind.lvl
    P_lvl_2 = P_lvl.lvl
    P_lvl_2_val = P_lvl_2.val
    p_data = (((ex.bodies[1]).bodies[2]).body.bodies[1]).tns.bind
    A_lvl = (((((ex.bodies[1]).bodies[2]).body.bodies[2]).body.rhs.args[1]).args[1]).tns.bind.lvl
    A_lvl_stop = A_lvl.shape
    A_lvl_2 = A_lvl.lvl
    A_lvl_2_ptr = A_lvl_2.ptr
    A_lvl_2_idx = A_lvl_2.idx
    A_lvl_2_stop = A_lvl_2.shape
    A_lvl_3 = A_lvl_2.lvl
    A_lvl_3_val = A_lvl_3.val
    y_lvl = (((((ex.bodies[1]).bodies[2]).body.bodies[2]).body.rhs.args[1]).args[2]).tns.bind.lvl
    y_lvl_ptr = y_lvl.ptr
    y_lvl_idx = y_lvl.idx
    y_lvl_stop = y_lvl.shape
    y_lvl_2 = y_lvl.lvl
    y_lvl_2_val = y_lvl_2.val
    y_lvl_stop == A_lvl_2_stop || throw(DimensionMismatch("mismatched dimension limits ($(y_lvl_stop) != $(A_lvl_2_stop))"))
    Finch.resize_if_smaller!(P_lvl_2_val, A_lvl_stop)
    Finch.fill_range!(P_lvl_2_val, 0, 1, A_lvl_stop)
    for j_4 = 1:A_lvl_stop
        A_lvl_q = (1 - 1) * A_lvl_stop + j_4
        P_lvl_q = (1 - 1) * A_lvl_stop + j_4
        p_val = false
        y_lvl_q = y_lvl_ptr[1]
        y_lvl_q_stop = y_lvl_ptr[1 + 1]
        if y_lvl_q < y_lvl_q_stop
            y_lvl_i1 = y_lvl_idx[y_lvl_q_stop - 1]
        else
            y_lvl_i1 = 0
        end
        A_lvl_2_q = A_lvl_2_ptr[A_lvl_q]
        A_lvl_2_q_stop = A_lvl_2_ptr[A_lvl_q + 1]
        if A_lvl_2_q < A_lvl_2_q_stop
            A_lvl_2_i1 = A_lvl_2_idx[A_lvl_2_q_stop - 1]
        else
            A_lvl_2_i1 = 0
        end
        phase_stop = min(y_lvl_stop, y_lvl_i1, A_lvl_2_i1)
        if phase_stop >= 1
            i = 1
            if y_lvl_idx[y_lvl_q] < 1
                y_lvl_q = Finch.scansearch(y_lvl_idx, 1, y_lvl_q, y_lvl_q_stop - 1)
            end
            if A_lvl_2_idx[A_lvl_2_q] < 1
                A_lvl_2_q = Finch.scansearch(A_lvl_2_idx, 1, A_lvl_2_q, A_lvl_2_q_stop - 1)
            end
            while i <= phase_stop
                if Finch.isannihilator(Finch.DefaultAlgebra(), Finch.Chooser{0}(), p_val)
                    break
                end
                y_lvl_i = y_lvl_idx[y_lvl_q]
                A_lvl_2_i = A_lvl_2_idx[A_lvl_2_q]
                phase_stop_2 = min(A_lvl_2_i, phase_stop, y_lvl_i)
                if y_lvl_i == phase_stop_2 && A_lvl_2_i == phase_stop_2
                    A_lvl_3_val_2 = A_lvl_3_val[A_lvl_2_q]
                    y_lvl_2_val_2 = y_lvl_2_val[y_lvl_q]
                    p_val = (Finch.Chooser{0}())(p_val, ifelse(y_lvl_2_val_2 && A_lvl_3_val_2, phase_stop_2, 0))
                    y_lvl_q += 1
                    A_lvl_2_q += 1
                elseif A_lvl_2_i == phase_stop_2
                    A_lvl_2_q += 1
                elseif y_lvl_i == phase_stop_2
                    y_lvl_q += 1
                end
                i = phase_stop_2 + 1
            end
        end
        p_data.val = p_val
        P_lvl_2_val[P_lvl_q] = p_val
    end
    resize!(P_lvl_2_val, A_lvl_stop)
    (P = Tensor((DenseLevel){Int64}(ElementLevel{0, Int64, Int64}(P_lvl_2_val), A_lvl_stop)),)
end
