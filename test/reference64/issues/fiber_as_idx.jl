begin
    B_lvl = ((ex.bodies[1]).bodies[1]).tns.bind.lvl
    B_lvl_2 = B_lvl.lvl
    B_lvl_2_val = B_lvl_2.val
    A_lvl = ((ex.bodies[1]).bodies[2]).body.rhs.tns.bind.lvl
    A_lvl_stop = A_lvl.shape
    A_lvl_2 = A_lvl.lvl
    A_lvl_2_stop = A_lvl_2.shape
    A_lvl_3 = A_lvl_2.lvl
    A_lvl_3_val = A_lvl_3.val
    I_lvl = (((ex.bodies[1]).bodies[2]).body.rhs.idxs[1]).tns.bind.lvl
    I_lvl_ptr = I_lvl.ptr
    I_lvl_right = I_lvl.right
    I_lvl_stop = I_lvl.shape
    I_lvl_2 = I_lvl.lvl
    I_lvl_2_val = I_lvl_2.val
    A_lvl_stop == I_lvl_stop || throw(
        DimensionMismatch("mismatched dimension limits ($(A_lvl_stop) != $(I_lvl_stop))")
    )
    Finch.resize_if_smaller!(B_lvl_2_val, A_lvl_stop)
    Finch.fill_range!(B_lvl_2_val, 0, 1, A_lvl_stop)
    I_lvl_q = I_lvl_ptr[1]
    I_lvl_q_stop = I_lvl_ptr[1 + 1]
    i = 1
    if I_lvl_right[I_lvl_q] < 1
        I_lvl_q = Finch.scansearch(I_lvl_right, 1, I_lvl_q, I_lvl_q_stop - 1)
    end
    while true
        I_lvl_i = I_lvl_right[I_lvl_q]
        if I_lvl_i < A_lvl_stop
            I_lvl_2_val_2 = I_lvl_2_val[I_lvl_q]
            for i_6 in i:I_lvl_i
                B_lvl_q = (1 - 1) * A_lvl_stop + i_6
                A_lvl_q = (1 - 1) * A_lvl_stop + i_6
                A_lvl_2_q = (A_lvl_q - 1) * A_lvl_2_stop + I_lvl_2_val_2
                A_lvl_3_val_2 = A_lvl_3_val[A_lvl_2_q]
                B_lvl_2_val[B_lvl_q] = A_lvl_3_val_2
            end
            I_lvl_q += 1
            i = I_lvl_i + 1
        else
            phase_stop_2 = min(A_lvl_stop, I_lvl_i)
            if I_lvl_i == phase_stop_2
                I_lvl_2_val_2 = I_lvl_2_val[I_lvl_q]
                for i_7 in i:phase_stop_2
                    B_lvl_q = (1 - 1) * A_lvl_stop + i_7
                    A_lvl_q = (1 - 1) * A_lvl_stop + i_7
                    A_lvl_2_q_2 = (A_lvl_q - 1) * A_lvl_2_stop + I_lvl_2_val_2
                    A_lvl_3_val_3 = A_lvl_3_val[A_lvl_2_q_2]
                    B_lvl_2_val[B_lvl_q] = A_lvl_3_val_3
                end
                I_lvl_q += 1
            else
                I_lvl_2_val_2 = I_lvl_2_val[I_lvl_q]
                for i_8 in i:phase_stop_2
                    B_lvl_q = (1 - 1) * A_lvl_stop + i_8
                    A_lvl_q = (1 - 1) * A_lvl_stop + i_8
                    A_lvl_2_q_3 = (A_lvl_q - 1) * A_lvl_2_stop + I_lvl_2_val_2
                    A_lvl_3_val_4 = A_lvl_3_val[A_lvl_2_q_3]
                    B_lvl_2_val[B_lvl_q] = A_lvl_3_val_4
                end
            end
            i = phase_stop_2 + 1
            break
        end
    end
    resize!(B_lvl_2_val, A_lvl_stop)
    (B=Tensor((DenseLevel){Int64}(ElementLevel{0,Int64,Int64}(B_lvl_2_val), A_lvl_stop)),)
end
