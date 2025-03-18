begin
    x_lvl = (ex.bodies[1]).body.rhs.tns.bind.lvl
    x_lvl_ptr = x_lvl.ptr
    x_lvl_idx = x_lvl.idx
    x_lvl_stop = x_lvl.shape
    x_lvl_2 = x_lvl.lvl
    x_lvl_2_val = x_lvl_2.val
    s_data = ((ex.bodies[1]).body.body.bodies[1]).lhs.tns.bind
    s_val = s_data.val
    y_lvl = (((ex.bodies[1]).body.body.bodies[1]).rhs.args[2]).tns.bind.lvl
    y_lvl_ptr = y_lvl.ptr
    y_lvl_idx = y_lvl.idx
    y_lvl_stop = y_lvl.shape
    y_lvl_2 = y_lvl.lvl
    y_lvl_2_val = y_lvl_2.val
    c_data = ((ex.bodies[1]).body.body.bodies[2]).lhs.tns.bind
    c_val = c_data.val
    y_lvl_stop == x_lvl_stop || throw(DimensionMismatch("mismatched dimension limits ($(y_lvl_stop) != $(x_lvl_stop))"))
    x_lvl_q = x_lvl_ptr[1]
    x_lvl_q_stop = x_lvl_ptr[1 + 1]
    if x_lvl_q < x_lvl_q_stop
        x_lvl_i1 = x_lvl_idx[x_lvl_q_stop - 1]
    else
        x_lvl_i1 = 0
    end
    y_lvl_q = y_lvl_ptr[1]
    y_lvl_q_stop = y_lvl_ptr[1 + 1]
    if y_lvl_q < y_lvl_q_stop
        y_lvl_i1 = y_lvl_idx[y_lvl_q_stop - 1]
    else
        y_lvl_i1 = 0
    end
    phase_stop = min(y_lvl_stop, x_lvl_i1, y_lvl_i1)
    if phase_stop >= 1
        i = 1
        if x_lvl_idx[x_lvl_q] < 1
            x_lvl_q = Finch.scansearch(x_lvl_idx, 1, x_lvl_q, x_lvl_q_stop - 1)
        end
        if y_lvl_idx[y_lvl_q] < 1
            y_lvl_q = Finch.scansearch(y_lvl_idx, 1, y_lvl_q, y_lvl_q_stop - 1)
        end
        while i <= phase_stop
            if Finch.isannihilator(Finch.DefaultAlgebra(), |, s_val)
                if x_lvl_idx[x_lvl_q] < i
                    x_lvl_q = Finch.scansearch(x_lvl_idx, i, x_lvl_q, x_lvl_q_stop - 1)
                end
                while true
                    x_lvl_i = x_lvl_idx[x_lvl_q]
                    if x_lvl_i < phase_stop
                        x_lvl_2_val_2 = x_lvl_2_val[x_lvl_q]
                        for i_9 = 1:phase_stop
                            c_val = x_lvl_2_val_2 + c_val
                        end
                        x_lvl_q += 1
                    else
                        phase_stop_4 = min(phase_stop, x_lvl_i)
                        if x_lvl_i == phase_stop_4
                            x_lvl_2_val_2 = x_lvl_2_val[x_lvl_q]
                            for i_11 = 1:phase_stop
                                c_val = x_lvl_2_val_2 + c_val
                            end
                            x_lvl_q += 1
                        end
                        break
                    end
                end
                break
            end
            x_lvl_i = x_lvl_idx[x_lvl_q]
            y_lvl_i = y_lvl_idx[y_lvl_q]
            phase_stop_2 = min(y_lvl_i, phase_stop, x_lvl_i)
            if x_lvl_i == phase_stop_2 && y_lvl_i == phase_stop_2
                x_lvl_2_val_2 = x_lvl_2_val[x_lvl_q]
                y_lvl_2_val_2 = y_lvl_2_val[y_lvl_q]
                s_val = (x_lvl_2_val_2 && y_lvl_2_val_2) | s_val
                c_val = x_lvl_2_val_2 + c_val
                x_lvl_q += 1
                y_lvl_q += 1
            elseif y_lvl_i == phase_stop_2
                y_lvl_q += 1
            elseif x_lvl_i == phase_stop_2
                x_lvl_2_val_2 = x_lvl_2_val[x_lvl_q]
                c_val = x_lvl_2_val_2 + c_val
                x_lvl_q += 1
            end
            i = phase_stop_2 + 1
        end
    end
    phase_start_5 = max(1, 1 + y_lvl_i1)
    phase_stop_6 = min(y_lvl_stop, x_lvl_i1)
    if phase_stop_6 >= phase_start_5
        if x_lvl_idx[x_lvl_q] < phase_start_5
            x_lvl_q = Finch.scansearch(x_lvl_idx, phase_start_5, x_lvl_q, x_lvl_q_stop - 1)
        end
        while true
            x_lvl_i = x_lvl_idx[x_lvl_q]
            if x_lvl_i < phase_stop_6
                x_lvl_2_val_3 = x_lvl_2_val[x_lvl_q]
                c_val = x_lvl_2_val_3 + c_val
                x_lvl_q += 1
            else
                phase_stop_8 = min(x_lvl_i, phase_stop_6)
                if x_lvl_i == phase_stop_8
                    x_lvl_2_val_3 = x_lvl_2_val[x_lvl_q]
                    c_val = x_lvl_2_val_3 + c_val
                    x_lvl_q += 1
                end
                break
            end
        end
    end
    result = ()
    c_data.val = c_val
    s_data.val = s_val
    result
end
