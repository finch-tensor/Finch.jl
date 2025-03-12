begin
    B_data = (ex.bodies[1]).lhs.tns.bind
    B_val = B_data.val
    A_lvl = (ex.bodies[1]).rhs.tns.bind.lvl
    A_lvl_ptr = A_lvl.ptr
    A_lvl_idx = A_lvl.idx
    A_lvl_2 = A_lvl.lvl
    A_lvl_2_val = A_lvl_2.val
    A_lvl_q = A_lvl_ptr[1]
    A_lvl_q_stop = A_lvl_ptr[1 + 1]
    if A_lvl_q < A_lvl_q_stop
        A_lvl_i1 = A_lvl_idx[A_lvl_q_stop - 1]
    else
        A_lvl_i1 = 0
    end
    phase_stop = min(5, A_lvl_i1)
    if phase_stop >= 5
        if A_lvl_idx[A_lvl_q] < 5
            A_lvl_q = Finch.scansearch(A_lvl_idx, 5, A_lvl_q, A_lvl_q_stop - 1)
        end
        while true
            A_lvl_i = A_lvl_idx[A_lvl_q]
            if A_lvl_i < phase_stop
                A_lvl_2_val_2 = A_lvl_2_val[A_lvl_q]
                B_val = A_lvl_2_val_2 + B_val
                A_lvl_q += 1
            else
                phase_stop_3 = min(phase_stop, A_lvl_i)
                if A_lvl_i == phase_stop_3
                    A_lvl_2_val_2 = A_lvl_2_val[A_lvl_q]
                    B_val += A_lvl_2_val_2
                    A_lvl_q += 1
                end
                break
            end
        end
    end
    result = ()
    B_data.val = B_val
    result
end
