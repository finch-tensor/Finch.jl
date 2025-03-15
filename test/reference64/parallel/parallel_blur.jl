begin
    output_lvl = ((ex.bodies[1]).bodies[1]).tns.bind.lvl
    output_lvl_2 = output_lvl.lvl
    output_lvl_3 = output_lvl_2.lvl
    output_lvl_3_val = output_lvl_3.val
    cpu = (((ex.bodies[1]).bodies[2]).ext.args[2]).bind
    tmp_lvl = (((ex.bodies[1]).bodies[2]).body.bodies[1]).tns.bind.lvl
    tmp_lvl_2 = tmp_lvl.lvl
    tmp_lvl_2_val = tmp_lvl_2.val
    input_lvl = ((((ex.bodies[1]).bodies[2]).body.bodies[2]).body.rhs.args[1]).tns.bind.lvl
    input_lvl_stop = input_lvl.shape
    input_lvl_2 = input_lvl.lvl
    input_lvl_2_stop = input_lvl_2.shape
    input_lvl_3 = input_lvl_2.lvl
    input_lvl_3_val = input_lvl_3.val
    1 == 2 || throw(DimensionMismatch("mismatched dimension limits ($(1) != $(2))"))
    input_lvl_2_stop == 1 + input_lvl_2_stop || throw(DimensionMismatch("mismatched dimension limits ($(input_lvl_2_stop) != $(1 + input_lvl_2_stop))"))
    input_lvl_stop == input_lvl_stop || throw(DimensionMismatch("mismatched dimension limits ($(input_lvl_stop) != $(input_lvl_stop))"))
    1 == 1 || throw(DimensionMismatch("mismatched dimension limits ($(1) != $(1))"))
    1 == 0 || throw(DimensionMismatch("mismatched dimension limits ($(1) != $(0))"))
    input_lvl_2_stop == input_lvl_2_stop + -1 || throw(DimensionMismatch("mismatched dimension limits ($(input_lvl_2_stop) != $(input_lvl_2_stop + -1))"))
    1 == 1 || throw(DimensionMismatch("mismatched dimension limits ($(1) != $(1))"))
    pos_stop = input_lvl_2_stop * input_lvl_stop
    Finch.resize_if_smaller!(output_lvl_3_val, pos_stop)
    Finch.fill_range!(output_lvl_3_val, 0.0, 1, pos_stop)
    buf = (Finch).transfer(CPULocalMemory(cpu), tmp_lvl_2_val)
    buf_3 = (Finch).transfer(CPUSharedMemory(cpu), output_lvl_3_val)
    Threads.@threads for tid = 1:cpu.n
            Finch.@barrier begin
                    @inbounds @fastmath(begin
                                buf_4 = (Finch).transfer(CPUThread(tid, cpu, Serial()), buf)
                                phase_start_2 = max(1, 1 + fld(input_lvl_stop * (-1 + tid), cpu.n))
                                phase_stop_2 = min(input_lvl_stop, fld(input_lvl_stop * tid, cpu.n))
                                if phase_stop_2 >= phase_start_2
                                    for y_8 = phase_start_2:phase_stop_2
                                        input_lvl_q_2 = (1 - 1) * input_lvl_stop + y_8
                                        input_lvl_q = (1 - 1) * input_lvl_stop + y_8
                                        input_lvl_q_3 = (1 - 1) * input_lvl_stop + y_8
                                        output_lvl_q = (1 - 1) * input_lvl_stop + y_8
                                        Finch.resize_if_smaller!(buf_4, input_lvl_2_stop)
                                        Finch.fill_range!(buf_4, 0, 1, input_lvl_2_stop)
                                        for x_9 = 1:input_lvl_2_stop
                                            tmp_lvl_q = (1 - 1) * input_lvl_2_stop + x_9
                                            input_lvl_2_q = (input_lvl_q_2 - 1) * input_lvl_2_stop + (-1 + x_9)
                                            input_lvl_2_q_2 = (input_lvl_q - 1) * input_lvl_2_stop + x_9
                                            input_lvl_2_q_3 = (input_lvl_q_3 - 1) * input_lvl_2_stop + (1 + x_9)
                                            input_lvl_3_val_2 = input_lvl_3_val[input_lvl_2_q]
                                            input_lvl_3_val_3 = input_lvl_3_val[input_lvl_2_q_2]
                                            input_lvl_3_val_4 = input_lvl_3_val[input_lvl_2_q_3]
                                            buf_4[tmp_lvl_q] = input_lvl_3_val_3 + input_lvl_3_val_2 + input_lvl_3_val_4 + buf_4[tmp_lvl_q]
                                        end
                                        resize!(buf_4, input_lvl_2_stop)
                                        for x_10 = 1:input_lvl_2_stop
                                            output_lvl_2_q = (output_lvl_q - 1) * input_lvl_2_stop + x_10
                                            tmp_lvl_q_2 = (1 - 1) * input_lvl_2_stop + x_10
                                            tmp_lvl_2_val_2 = buf_4[tmp_lvl_q_2]
                                            output_lvl_3_val[output_lvl_2_q] = tmp_lvl_2_val_2
                                        end
                                    end
                                end
                                phase_start_3 = max(1, 1 + fld(input_lvl_stop * tid, cpu.n))
                                if input_lvl_stop >= phase_start_3
                                    input_lvl_stop + 1
                                end
                            end)
                    nothing
                end
        end
    resize!(buf_3, input_lvl_2_stop * input_lvl_stop)
    (output = Tensor((DenseLevel){Int64}((DenseLevel){Int64}(ElementLevel{0.0, Float64, Int64}(buf_3), input_lvl_2_stop), input_lvl_stop)),)
end
