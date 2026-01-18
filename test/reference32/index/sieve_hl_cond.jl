begin
    B_data = ((ex.bodies[1]).bodies[1]).tns.bind
    A_lvl = ((ex.bodies[1]).bodies[2]).body.body.rhs.tns.bind.lvl
    A_lvl_ptr = A_lvl.ptr
    A_lvl_idx = A_lvl.idx
    A_lvl_stop = A_lvl.shape
    A_lvl_2 = A_lvl.lvl
    A_lvl_2_val = A_lvl_2.val
    B_val = 0
    A_lvl_q = A_lvl_ptr[1]
    A_lvl_q_stop = A_lvl_ptr[1 + 1]
    if A_lvl_q < A_lvl_q_stop
        A_lvl_i1 = A_lvl_idx[A_lvl_q_stop - 1]
    else
        A_lvl_i1 = 0
    end
    phase_start_2 = max(1, 1 + (1 - 1))
    phase_stop_2 = min(A_lvl_stop, A_lvl_i1, 1)
    if phase_stop_2 >= phase_start_2
        if A_lvl_idx[A_lvl_q] < phase_start_2
            A_lvl_q = Finch.scansearch(A_lvl_idx, phase_start_2, A_lvl_q, A_lvl_q_stop - 1)
        end
        while true
            A_lvl_i = A_lvl_idx[A_lvl_q]
            if A_lvl_i < phase_stop_2
                A_lvl_2_val_3 = A_lvl_2_val[A_lvl_q]
                B_val = A_lvl_2_val_3 + B_val
                A_lvl_q += 1
            else
                phase_stop_4 = min(phase_stop_2, A_lvl_i)
                if A_lvl_i == phase_stop_4
                    A_lvl_2_val_3 = A_lvl_2_val[A_lvl_q]
                    B_val += A_lvl_2_val_3
                    A_lvl_q += 1
                end
                break
            end
        end
    end
    B_data.val = B_val
    (B=B_data,)
end
