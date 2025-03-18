begin
    X_data = (ex.bodies[1]).body.body.body.rhs.tns.bind
    sugar_1 = size((ex.bodies[1]).body.body.body.rhs.tns.bind)
    X_mode1_stop = sugar_1[1]
    X_mode2_stop = sugar_1[2]
    A_data = (ex.bodies[1]).body.body.body.body.body.rhs.tns.bind
    sugar_2 = size((ex.bodies[1]).body.body.body.body.body.rhs.tns.bind)
    A_mode1_stop = sugar_2[1]
    A_mode2_stop = sugar_2[2]
    A_mode3_stop = sugar_2[3]
    C_data = (ex.bodies[1]).body.body.body.body.body.body.body.lhs.tns.bind
    sugar_3 = size((ex.bodies[1]).body.body.body.body.body.body.body.lhs.tns.bind)
    C_mode1_stop = sugar_3[1]
    C_mode2_stop = sugar_3[2]
    C_mode3_stop = sugar_3[3]
    A_mode1_stop == X_mode1_stop || throw(DimensionMismatch("mismatched dimension limits ($(A_mode1_stop) != $(X_mode1_stop))"))
    A_mode1_stop == C_mode1_stop || throw(DimensionMismatch("mismatched dimension limits ($(A_mode1_stop) != $(C_mode1_stop))"))
    C_mode2_stop == X_mode2_stop || throw(DimensionMismatch("mismatched dimension limits ($(C_mode2_stop) != $(X_mode2_stop))"))
    C_mode3_stop == A_mode3_stop || throw(DimensionMismatch("mismatched dimension limits ($(C_mode3_stop) != $(A_mode3_stop))"))
    for k_4 = 1:C_mode3_stop
        for j_4 = 1:C_mode2_stop
            for i_6 = 1:A_mode1_stop
                val = X_data[i_6, j_4]
                for l_4 = 1:A_mode2_stop
                    val_3 = A_data[i_6, l_4, k_4]
                    phase_stop = min(i_6, l_4 + -1)
                    if phase_stop >= i_6
                        for s_5 = i_6:phase_stop
                            C_data[i_6, j_4, k_4] = val_3 * val + C_data[i_6, j_4, k_4]
                        end
                    end
                end
            end
        end
    end
    ()
end
