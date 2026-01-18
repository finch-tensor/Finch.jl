begin
    s_data = (ex.bodies[1]).body.lhs.tns.bind
    s_val = s_data.val
    x_lvl = ((ex.bodies[1]).body.rhs.args[1]).tns.bind.lvl
    x_lvl_ptr = x_lvl.ptr
    x_lvl_idx = x_lvl.idx
    x_lvl_stop = x_lvl.shape
    x_lvl_2 = x_lvl.lvl
    x_lvl_2_val = x_lvl_2.val
    y_lvl = ((ex.bodies[1]).body.rhs.args[2]).tns.bind.lvl
    y_lvl_ptr = y_lvl.ptr
    y_lvl_idx = y_lvl.idx
    y_lvl_stop = y_lvl.shape
    y_lvl_2 = y_lvl.lvl
    y_lvl_2_val = y_lvl_2.val
    y_lvl_stop == x_lvl_stop || throw(
        DimensionMismatch("mismatched dimension limits ($(y_lvl_stop) != $(x_lvl_stop))")
    )
    y_lvl_q = y_lvl_ptr[1]
    y_lvl_q_stop = y_lvl_ptr[1 + 1]
    if y_lvl_q < y_lvl_q_stop
        y_lvl_i1 = y_lvl_idx[y_lvl_q_stop - 1]
    else
        y_lvl_i1 = 0
    end
    x_lvl_q = x_lvl_ptr[1]
    x_lvl_q_stop = x_lvl_ptr[1 + 1]
    if x_lvl_q < x_lvl_q_stop
        x_lvl_i1 = x_lvl_idx[x_lvl_q_stop - 1]
    else
        x_lvl_i1 = 0
    end
    phase_stop = min(y_lvl_stop, y_lvl_i1, x_lvl_i1)
    if phase_stop >= 1
        i = 1
        if y_lvl_idx[y_lvl_q] < 1
            y_lvl_q = Finch.scansearch(y_lvl_idx, 1, y_lvl_q, y_lvl_q_stop - 1)
        end
        if x_lvl_idx[x_lvl_q] < 1
            x_lvl_q = Finch.scansearch(x_lvl_idx, 1, x_lvl_q, x_lvl_q_stop - 1)
        end
        while i <= phase_stop
            if Finch.isannihilator(Finch.DefaultAlgebra(), |, s_val)
                break
            end
            y_lvl_i = y_lvl_idx[y_lvl_q]
            x_lvl_i = x_lvl_idx[x_lvl_q]
            phase_stop_2 = min(x_lvl_i, phase_stop, y_lvl_i)
            if y_lvl_i == phase_stop_2 && x_lvl_i == phase_stop_2
                x_lvl_2_val_2 = x_lvl_2_val[x_lvl_q]
                y_lvl_2_val_2 = y_lvl_2_val[y_lvl_q]
                s_val = (y_lvl_2_val_2 && x_lvl_2_val_2) | s_val
                y_lvl_q += 1
                x_lvl_q += 1
            elseif x_lvl_i == phase_stop_2
                x_lvl_q += 1
            elseif y_lvl_i == phase_stop_2
                y_lvl_q += 1
            end
            i = phase_stop_2 + 1
        end
    end
    result = ()
    s_data.val = s_val
    result
end
