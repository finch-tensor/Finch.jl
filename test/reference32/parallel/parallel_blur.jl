begin
    output_lvl = ((ex.bodies[1]).bodies[1]).tns.bind.lvl
    output_lvl_2 = output_lvl.lvl
    output_lvl_3 = output_lvl_2.lvl
    output_lvl_2_val = output_lvl_2.lvl.val
    cpu = (((ex.bodies[1]).bodies[2]).ext.args[2]).bind
    tmp_lvl = (((ex.bodies[1]).bodies[2]).body.bodies[1]).tns.bind.lvl
    tmp_lvl_val = tmp_lvl.lvl.val
    input_lvl = ((((ex.bodies[1]).bodies[2]).body.bodies[2]).body.rhs.args[1]).tns.bind.lvl
    input_lvl_2 = input_lvl.lvl
    input_lvl_2_val = input_lvl_2.lvl.val
    1 == 2 || throw(DimensionMismatch("mismatched dimension limits ($(1) != $(2))"))
    input_lvl_2.shape == 1 + input_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(input_lvl_2.shape) != $(1 + input_lvl_2.shape))"))
    input_lvl.shape == input_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(input_lvl.shape) != $(input_lvl.shape))"))
    1 == 1 || throw(DimensionMismatch("mismatched dimension limits ($(1) != $(1))"))
    1 == 0 || throw(DimensionMismatch("mismatched dimension limits ($(1) != $(0))"))
    input_lvl_2.shape == input_lvl_2.shape + -1 || throw(DimensionMismatch("mismatched dimension limits ($(input_lvl_2.shape) != $(input_lvl_2.shape + -1))"))
    1 == 1 || throw(DimensionMismatch("mismatched dimension limits ($(1) != $(1))"))
    y_stop = input_lvl.shape
    pos_stop = input_lvl_2.shape * input_lvl.shape
    Finch.resize_if_smaller!(output_lvl_2_val, pos_stop)
    Finch.fill_range!(output_lvl_2_val, 0.0, 1, pos_stop)
    input_lvl_2_val = (Finch).transfer(cpu, input_lvl_2_val)
    val_2 = output_lvl_2_val
    output_lvl_2_val = (Finch).transfer(cpu, output_lvl_2_val)
    Threads.@threads for i = 1:cpu.n
            Finch.@barrier begin
                    @inbounds @fastmath(begin
                                val_3 = tmp_lvl_val
                                tmp_lvl_val = (Finch).transfer(tmp_lvl_val, CPUThread(i, cpu, Serial()), style)
                                res_6 = begin
                                        phase_start_2 = max(1, 1 + fld(y_stop * (-1 + i), cpu.n))
                                        phase_stop_2 = min(y_stop, fld(y_stop * i, cpu.n))
                                        if phase_stop_2 >= phase_start_2
                                            for y_8 = phase_start_2:phase_stop_2
                                                input_lvl_q_2 = (1 - 1) * input_lvl.shape + y_8
                                                input_lvl_q = (1 - 1) * input_lvl.shape + y_8
                                                input_lvl_q_3 = (1 - 1) * input_lvl.shape + y_8
                                                output_lvl_q = (1 - 1) * input_lvl.shape + y_8
                                                Finch.resize_if_smaller!(tmp_lvl_val, input_lvl_2.shape)
                                                Finch.fill_range!(tmp_lvl_val, 0, 1, input_lvl_2.shape)
                                                for x_9 = 1:input_lvl_2.shape
                                                    tmp_lvl_q = (1 - 1) * input_lvl_2.shape + x_9
                                                    input_lvl_2_q = (input_lvl_q_2 - 1) * input_lvl_2.shape + (-1 + x_9)
                                                    input_lvl_2_q_2 = (input_lvl_q - 1) * input_lvl_2.shape + x_9
                                                    input_lvl_2_q_3 = (input_lvl_q_3 - 1) * input_lvl_2.shape + (1 + x_9)
                                                    input_lvl_3_val = input_lvl_2_val[input_lvl_2_q]
                                                    input_lvl_3_val_2 = input_lvl_2_val[input_lvl_2_q_2]
                                                    input_lvl_3_val_3 = input_lvl_2_val[input_lvl_2_q_3]
                                                    tmp_lvl_val[tmp_lvl_q] = input_lvl_3_val_2 + input_lvl_3_val + input_lvl_3_val_3 + tmp_lvl_val[tmp_lvl_q]
                                                end
                                                resize!(tmp_lvl_val, input_lvl_2.shape)
                                                for x_10 = 1:input_lvl_2.shape
                                                    output_lvl_2_q = (output_lvl_q - 1) * input_lvl_2.shape + x_10
                                                    tmp_lvl_q_2 = (1 - 1) * input_lvl_2.shape + x_10
                                                    tmp_lvl_2_val = tmp_lvl_val[tmp_lvl_q_2]
                                                    output_lvl_2_val[output_lvl_2_q] = tmp_lvl_2_val
                                                end
                                            end
                                        end
                                        phase_start_3 = max(1, 1 + fld(y_stop * i, cpu.n))
                                        if y_stop >= phase_start_3
                                            y_stop + 1
                                        end
                                    end
                                tmp_lvl_val = val_3
                                res_6
                            end)
                    nothing
                end
        end
    resize!(val_2, input_lvl_2.shape * input_lvl.shape)
    (output = Tensor((DenseLevel){Int32}((DenseLevel){Int32}(output_lvl_3, input_lvl_2.shape), input_lvl.shape)),)
end
