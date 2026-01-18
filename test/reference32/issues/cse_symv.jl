begin
    y_lvl = ((ex.bodies[1]).body.body.bodies[1]).lhs.tns.bind.lvl
    y_lvl_stop = y_lvl.shape
    y_lvl_2 = y_lvl.lvl
    y_lvl_2_val = y_lvl_2.val
    A_lvl = (((ex.bodies[1]).body.body.bodies[1]).rhs.args[1]).tns.bind.lvl
    A_lvl_stop = A_lvl.shape
    A_lvl_2 = A_lvl.lvl
    A_lvl_2_stop = A_lvl_2.shape
    A_lvl_3 = A_lvl_2.lvl
    A_lvl_3_val = A_lvl_3.val
    x_lvl = (((ex.bodies[1]).body.body.bodies[1]).rhs.args[2]).tns.bind.lvl
    x_lvl_stop = x_lvl.shape
    x_lvl_2 = x_lvl.lvl
    x_lvl_2_val = x_lvl_2.val
    y_lvl_stop == A_lvl_2_stop || throw(DimensionMismatch("mismatched dimension limits ($(y_lvl_stop) != $(A_lvl_2_stop))"))
    x_lvl_stop == A_lvl_stop || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl_stop) != $(A_lvl_stop))"))
    y_lvl_stop == x_lvl_stop || throw(DimensionMismatch("mismatched dimension limits ($(y_lvl_stop) != $(x_lvl_stop))"))
    y_lvl_stop == x_lvl_stop || throw(DimensionMismatch("mismatched dimension limits ($(y_lvl_stop) != $(x_lvl_stop))"))
    @warn "Performance Warning: non-concordant traversal of A[i, j] (hint: most arrays prefer column major or first index fast, run in fast mode to ignore this warning)"
    for i_6 = 1:y_lvl_stop
        y_lvl_q = (1 - 1) * y_lvl_stop + i_6
        x_lvl_q = (1 - 1) * x_lvl_stop + i_6
        x_lvl_2_val_2 = x_lvl_2_val[x_lvl_q]
        for j_6 = 1:y_lvl_stop
            A_lvl_q = (1 - 1) * A_lvl_stop + j_6
            x_lvl_q_2 = (1 - 1) * x_lvl_stop + j_6
            y_lvl_q_2 = (1 - 1) * y_lvl_stop + j_6
            x_lvl_2_val_3 = x_lvl_2_val[x_lvl_q_2]
            A_lvl_2_q = (A_lvl_q - 1) * A_lvl_2_stop + i_6
            A_lvl_3_val_2 = A_lvl_3_val[A_lvl_2_q]
            y_lvl_2_val[y_lvl_q] = A_lvl_3_val_2 * x_lvl_2_val_3 + y_lvl_2_val[y_lvl_q]
            y_lvl_2_val[y_lvl_q_2] = A_lvl_3_val_2 * x_lvl_2_val_2 + y_lvl_2_val[y_lvl_q_2]
        end
    end
    result = ()
    resize!(y_lvl_2_val, y_lvl_stop)
    result
end
