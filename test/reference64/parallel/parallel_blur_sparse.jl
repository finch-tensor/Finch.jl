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
    input_lvl_2_ptr = input_lvl_2.ptr
    input_lvl_2_idx = input_lvl_2.idx
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
    buf_5 = (Finch).transfer(CPUSharedMemory(cpu), output_lvl_3_val)
    Threads.@threads for tid = 1:cpu.n
            Finch.@barrier begin
                    @inbounds @fastmath(begin
                                buf_6 = (Finch).transfer(CPUThread(tid, cpu, Serial()), buf)
                                phase_start_2 = max(1, 1 + fld(input_lvl_stop * (-1 + tid), cpu.n))
                                phase_stop_2 = min(input_lvl_stop, fld(input_lvl_stop * tid, cpu.n))
                                if phase_stop_2 >= phase_start_2
                                    for y_8 = phase_start_2:phase_stop_2
                                        input_lvl_q_2 = (1 - 1) * input_lvl_stop + y_8
                                        input_lvl_q = (1 - 1) * input_lvl_stop + y_8
                                        input_lvl_q_3 = (1 - 1) * input_lvl_stop + y_8
                                        output_lvl_q = (1 - 1) * input_lvl_stop + y_8
                                        Finch.resize_if_smaller!(buf_6, input_lvl_2_stop)
                                        Finch.fill_range!(buf_6, 0, 1, input_lvl_2_stop)
                                        input_lvl_2_q = input_lvl_2_ptr[input_lvl_q_2]
                                        input_lvl_2_q_stop = input_lvl_2_ptr[input_lvl_q_2 + 1]
                                        if input_lvl_2_q < input_lvl_2_q_stop
                                            input_lvl_2_i1 = input_lvl_2_idx[input_lvl_2_q_stop - 1]
                                        else
                                            input_lvl_2_i1 = 0
                                        end
                                        input_lvl_2_q_2 = input_lvl_2_ptr[input_lvl_q]
                                        input_lvl_2_q_stop_2 = input_lvl_2_ptr[input_lvl_q + 1]
                                        if input_lvl_2_q_2 < input_lvl_2_q_stop_2
                                            input_lvl_2_i1_2 = input_lvl_2_idx[input_lvl_2_q_stop_2 - 1]
                                        else
                                            input_lvl_2_i1_2 = 0
                                        end
                                        input_lvl_2_q_3 = input_lvl_2_ptr[input_lvl_q_3]
                                        input_lvl_2_q_stop_3 = input_lvl_2_ptr[input_lvl_q_3 + 1]
                                        if input_lvl_2_q_3 < input_lvl_2_q_stop_3
                                            input_lvl_2_i1_3 = input_lvl_2_idx[input_lvl_2_q_stop_3 - 1]
                                        else
                                            input_lvl_2_i1_3 = 0
                                        end
                                        phase_stop_3 = min(input_lvl_2_stop, input_lvl_2_i1_2, -1 + input_lvl_2_i1_3, 1 + input_lvl_2_i1)
                                        if phase_stop_3 >= 1
                                            x = 1
                                            if input_lvl_2_idx[input_lvl_2_q] < -1 + 1
                                                input_lvl_2_q = Finch.scansearch(input_lvl_2_idx, -1 + 1, input_lvl_2_q, input_lvl_2_q_stop - 1)
                                            end
                                            if input_lvl_2_idx[input_lvl_2_q_2] < 1
                                                input_lvl_2_q_2 = Finch.scansearch(input_lvl_2_idx, 1, input_lvl_2_q_2, input_lvl_2_q_stop_2 - 1)
                                            end
                                            if input_lvl_2_idx[input_lvl_2_q_3] < 1 + 1
                                                input_lvl_2_q_3 = Finch.scansearch(input_lvl_2_idx, 1 + 1, input_lvl_2_q_3, input_lvl_2_q_stop_3 - 1)
                                            end
                                            while x <= phase_stop_3
                                                input_lvl_2_i = input_lvl_2_idx[input_lvl_2_q]
                                                input_lvl_2_i_2 = input_lvl_2_idx[input_lvl_2_q_2]
                                                input_lvl_2_i_3 = input_lvl_2_idx[input_lvl_2_q_3]
                                                phase_stop_4 = min(phase_stop_3, input_lvl_2_i_2, -1 + input_lvl_2_i_3, 1 + input_lvl_2_i)
                                                if (input_lvl_2_i == -1 + phase_stop_4 && input_lvl_2_i_2 == phase_stop_4) && input_lvl_2_i_3 == 1 + phase_stop_4
                                                    input_lvl_3_val_3 = input_lvl_3_val[input_lvl_2_q_2]
                                                    input_lvl_3_val_2 = input_lvl_3_val[input_lvl_2_q]
                                                    input_lvl_3_val_4 = input_lvl_3_val[input_lvl_2_q_3]
                                                    tmp_lvl_q = (1 - 1) * input_lvl_2_stop + phase_stop_4
                                                    buf_6[tmp_lvl_q] = input_lvl_3_val_2 + input_lvl_3_val_3 + input_lvl_3_val_4 + buf_6[tmp_lvl_q]
                                                    input_lvl_2_q += 1
                                                    input_lvl_2_q_2 += 1
                                                    input_lvl_2_q_3 += 1
                                                elseif input_lvl_2_i_2 == phase_stop_4 && input_lvl_2_i_3 == 1 + phase_stop_4
                                                    input_lvl_3_val_3 = input_lvl_3_val[input_lvl_2_q_2]
                                                    input_lvl_3_val_4 = input_lvl_3_val[input_lvl_2_q_3]
                                                    tmp_lvl_q = (1 - 1) * input_lvl_2_stop + phase_stop_4
                                                    buf_6[tmp_lvl_q] = buf_6[tmp_lvl_q] + input_lvl_3_val_4 + input_lvl_3_val_3
                                                    input_lvl_2_q_2 += 1
                                                    input_lvl_2_q_3 += 1
                                                elseif input_lvl_2_i == -1 + phase_stop_4 && input_lvl_2_i_3 == 1 + phase_stop_4
                                                    input_lvl_3_val_2 = input_lvl_3_val[input_lvl_2_q]
                                                    input_lvl_3_val_4 = input_lvl_3_val[input_lvl_2_q_3]
                                                    tmp_lvl_q = (1 - 1) * input_lvl_2_stop + phase_stop_4
                                                    buf_6[tmp_lvl_q] = buf_6[tmp_lvl_q] + input_lvl_3_val_4 + input_lvl_3_val_2
                                                    input_lvl_2_q += 1
                                                    input_lvl_2_q_3 += 1
                                                elseif input_lvl_2_i_3 == 1 + phase_stop_4
                                                    input_lvl_3_val_4 = input_lvl_3_val[input_lvl_2_q_3]
                                                    tmp_lvl_q = (1 - 1) * input_lvl_2_stop + phase_stop_4
                                                    buf_6[tmp_lvl_q] += input_lvl_3_val_4
                                                    input_lvl_2_q_3 += 1
                                                elseif input_lvl_2_i == -1 + phase_stop_4 && input_lvl_2_i_2 == phase_stop_4
                                                    input_lvl_3_val_3 = input_lvl_3_val[input_lvl_2_q_2]
                                                    input_lvl_3_val_2 = input_lvl_3_val[input_lvl_2_q]
                                                    tmp_lvl_q = (1 - 1) * input_lvl_2_stop + phase_stop_4
                                                    buf_6[tmp_lvl_q] = buf_6[tmp_lvl_q] + input_lvl_3_val_2 + input_lvl_3_val_3
                                                    input_lvl_2_q += 1
                                                    input_lvl_2_q_2 += 1
                                                elseif input_lvl_2_i_2 == phase_stop_4
                                                    input_lvl_3_val_3 = input_lvl_3_val[input_lvl_2_q_2]
                                                    tmp_lvl_q = (1 - 1) * input_lvl_2_stop + phase_stop_4
                                                    buf_6[tmp_lvl_q] += input_lvl_3_val_3
                                                    input_lvl_2_q_2 += 1
                                                elseif input_lvl_2_i == -1 + phase_stop_4
                                                    input_lvl_3_val_2 = input_lvl_3_val[input_lvl_2_q]
                                                    tmp_lvl_q = (1 - 1) * input_lvl_2_stop + phase_stop_4
                                                    buf_6[tmp_lvl_q] += input_lvl_3_val_2
                                                    input_lvl_2_q += 1
                                                end
                                                x = phase_stop_4 + 1
                                            end
                                        end
                                        phase_start_5 = max(1, 2 + input_lvl_2_i1)
                                        phase_stop_5 = min(input_lvl_2_stop, input_lvl_2_i1_2, -1 + input_lvl_2_i1_3)
                                        if phase_stop_5 >= phase_start_5
                                            x = phase_start_5
                                            if input_lvl_2_idx[input_lvl_2_q_2] < phase_start_5
                                                input_lvl_2_q_2 = Finch.scansearch(input_lvl_2_idx, phase_start_5, input_lvl_2_q_2, input_lvl_2_q_stop_2 - 1)
                                            end
                                            if input_lvl_2_idx[input_lvl_2_q_3] < 1 + phase_start_5
                                                input_lvl_2_q_3 = Finch.scansearch(input_lvl_2_idx, 1 + phase_start_5, input_lvl_2_q_3, input_lvl_2_q_stop_3 - 1)
                                            end
                                            while x <= phase_stop_5
                                                input_lvl_2_i_2 = input_lvl_2_idx[input_lvl_2_q_2]
                                                input_lvl_2_i_3 = input_lvl_2_idx[input_lvl_2_q_3]
                                                phase_stop_6 = min(input_lvl_2_i_2, -1 + input_lvl_2_i_3, phase_stop_5)
                                                if input_lvl_2_i_2 == phase_stop_6 && input_lvl_2_i_3 == 1 + phase_stop_6
                                                    input_lvl_3_val_6 = input_lvl_3_val[input_lvl_2_q_3]
                                                    input_lvl_3_val_5 = input_lvl_3_val[input_lvl_2_q_2]
                                                    tmp_lvl_q = (1 - 1) * input_lvl_2_stop + phase_stop_6
                                                    buf_6[tmp_lvl_q] = buf_6[tmp_lvl_q] + input_lvl_3_val_5 + input_lvl_3_val_6
                                                    input_lvl_2_q_2 += 1
                                                    input_lvl_2_q_3 += 1
                                                elseif input_lvl_2_i_3 == 1 + phase_stop_6
                                                    input_lvl_3_val_6 = input_lvl_3_val[input_lvl_2_q_3]
                                                    tmp_lvl_q = (1 - 1) * input_lvl_2_stop + phase_stop_6
                                                    buf_6[tmp_lvl_q] += input_lvl_3_val_6
                                                    input_lvl_2_q_3 += 1
                                                elseif input_lvl_2_i_2 == phase_stop_6
                                                    input_lvl_3_val_5 = input_lvl_3_val[input_lvl_2_q_2]
                                                    tmp_lvl_q = (1 - 1) * input_lvl_2_stop + phase_stop_6
                                                    buf_6[tmp_lvl_q] += input_lvl_3_val_5
                                                    input_lvl_2_q_2 += 1
                                                end
                                                x = phase_stop_6 + 1
                                            end
                                        end
                                        phase_start_7 = max(1, 1 + input_lvl_2_i1_2)
                                        phase_stop_7 = min(input_lvl_2_stop, -1 + input_lvl_2_i1_3, 1 + input_lvl_2_i1)
                                        if phase_stop_7 >= phase_start_7
                                            x = phase_start_7
                                            if input_lvl_2_idx[input_lvl_2_q] < -1 + phase_start_7
                                                input_lvl_2_q = Finch.scansearch(input_lvl_2_idx, -1 + phase_start_7, input_lvl_2_q, input_lvl_2_q_stop - 1)
                                            end
                                            if input_lvl_2_idx[input_lvl_2_q_3] < 1 + phase_start_7
                                                input_lvl_2_q_3 = Finch.scansearch(input_lvl_2_idx, 1 + phase_start_7, input_lvl_2_q_3, input_lvl_2_q_stop_3 - 1)
                                            end
                                            while x <= phase_stop_7
                                                input_lvl_2_i = input_lvl_2_idx[input_lvl_2_q]
                                                input_lvl_2_i_3 = input_lvl_2_idx[input_lvl_2_q_3]
                                                phase_stop_8 = min(-1 + input_lvl_2_i_3, 1 + input_lvl_2_i, phase_stop_7)
                                                if input_lvl_2_i == -1 + phase_stop_8 && input_lvl_2_i_3 == 1 + phase_stop_8
                                                    input_lvl_3_val_8 = input_lvl_3_val[input_lvl_2_q_3]
                                                    input_lvl_3_val_7 = input_lvl_3_val[input_lvl_2_q]
                                                    tmp_lvl_q = (1 - 1) * input_lvl_2_stop + phase_stop_8
                                                    buf_6[tmp_lvl_q] = buf_6[tmp_lvl_q] + input_lvl_3_val_7 + input_lvl_3_val_8
                                                    input_lvl_2_q += 1
                                                    input_lvl_2_q_3 += 1
                                                elseif input_lvl_2_i_3 == 1 + phase_stop_8
                                                    input_lvl_3_val_8 = input_lvl_3_val[input_lvl_2_q_3]
                                                    tmp_lvl_q = (1 - 1) * input_lvl_2_stop + phase_stop_8
                                                    buf_6[tmp_lvl_q] += input_lvl_3_val_8
                                                    input_lvl_2_q_3 += 1
                                                elseif input_lvl_2_i == -1 + phase_stop_8
                                                    input_lvl_3_val_7 = input_lvl_3_val[input_lvl_2_q]
                                                    tmp_lvl_q = (1 - 1) * input_lvl_2_stop + phase_stop_8
                                                    buf_6[tmp_lvl_q] += input_lvl_3_val_7
                                                    input_lvl_2_q += 1
                                                end
                                                x = phase_stop_8 + 1
                                            end
                                        end
                                        phase_start_9 = max(1, 2 + input_lvl_2_i1, 1 + input_lvl_2_i1_2)
                                        phase_stop_9 = min(input_lvl_2_stop, -1 + input_lvl_2_i1_3)
                                        if phase_stop_9 >= phase_start_9
                                            if input_lvl_2_idx[input_lvl_2_q_3] < 1 + phase_start_9
                                                input_lvl_2_q_3 = Finch.scansearch(input_lvl_2_idx, 1 + phase_start_9, input_lvl_2_q_3, input_lvl_2_q_stop_3 - 1)
                                            end
                                            while true
                                                input_lvl_2_i_3 = input_lvl_2_idx[input_lvl_2_q_3]
                                                phase_stop_10 = -1 + input_lvl_2_i_3
                                                if phase_stop_10 < phase_stop_9
                                                    input_lvl_3_val_9 = input_lvl_3_val[input_lvl_2_q_3]
                                                    tmp_lvl_q = (1 - 1) * input_lvl_2_stop + phase_stop_10
                                                    buf_6[tmp_lvl_q] += input_lvl_3_val_9
                                                    input_lvl_2_q_3 += 1
                                                else
                                                    phase_stop_11 = min(-1 + input_lvl_2_i_3, phase_stop_9)
                                                    if input_lvl_2_i_3 == 1 + phase_stop_11
                                                        input_lvl_3_val_9 = input_lvl_3_val[input_lvl_2_q_3]
                                                        tmp_lvl_q = (1 - 1) * input_lvl_2_stop + phase_stop_11
                                                        buf_6[tmp_lvl_q] += input_lvl_3_val_9
                                                        input_lvl_2_q_3 += 1
                                                    end
                                                    break
                                                end
                                            end
                                        end
                                        phase_start_12 = max(1, input_lvl_2_i1_3)
                                        phase_stop_12 = min(input_lvl_2_stop, input_lvl_2_i1_2, 1 + input_lvl_2_i1)
                                        if phase_stop_12 >= phase_start_12
                                            x = phase_start_12
                                            if input_lvl_2_idx[input_lvl_2_q_2] < phase_start_12
                                                input_lvl_2_q_2 = Finch.scansearch(input_lvl_2_idx, phase_start_12, input_lvl_2_q_2, input_lvl_2_q_stop_2 - 1)
                                            end
                                            if input_lvl_2_idx[input_lvl_2_q] < -1 + phase_start_12
                                                input_lvl_2_q = Finch.scansearch(input_lvl_2_idx, -1 + phase_start_12, input_lvl_2_q, input_lvl_2_q_stop - 1)
                                            end
                                            while x <= phase_stop_12
                                                input_lvl_2_i_2 = input_lvl_2_idx[input_lvl_2_q_2]
                                                input_lvl_2_i = input_lvl_2_idx[input_lvl_2_q]
                                                phase_stop_13 = min(input_lvl_2_i_2, 1 + input_lvl_2_i, phase_stop_12)
                                                if input_lvl_2_i_2 == phase_stop_13 && input_lvl_2_i == -1 + phase_stop_13
                                                    input_lvl_3_val_10 = input_lvl_3_val[input_lvl_2_q]
                                                    input_lvl_3_val_11 = input_lvl_3_val[input_lvl_2_q_2]
                                                    tmp_lvl_q = (1 - 1) * input_lvl_2_stop + phase_stop_13
                                                    buf_6[tmp_lvl_q] = buf_6[tmp_lvl_q] + input_lvl_3_val_11 + input_lvl_3_val_10
                                                    input_lvl_2_q_2 += 1
                                                    input_lvl_2_q += 1
                                                elseif input_lvl_2_i == -1 + phase_stop_13
                                                    input_lvl_3_val_10 = input_lvl_3_val[input_lvl_2_q]
                                                    tmp_lvl_q = (1 - 1) * input_lvl_2_stop + phase_stop_13
                                                    buf_6[tmp_lvl_q] += input_lvl_3_val_10
                                                    input_lvl_2_q += 1
                                                elseif input_lvl_2_i_2 == phase_stop_13
                                                    input_lvl_3_val_11 = input_lvl_3_val[input_lvl_2_q_2]
                                                    tmp_lvl_q = (1 - 1) * input_lvl_2_stop + phase_stop_13
                                                    buf_6[tmp_lvl_q] += input_lvl_3_val_11
                                                    input_lvl_2_q_2 += 1
                                                end
                                                x = phase_stop_13 + 1
                                            end
                                        end
                                        phase_start_14 = max(1, input_lvl_2_i1_3, 2 + input_lvl_2_i1)
                                        phase_stop_14 = min(input_lvl_2_stop, input_lvl_2_i1_2)
                                        if phase_stop_14 >= phase_start_14
                                            if input_lvl_2_idx[input_lvl_2_q_2] < phase_start_14
                                                input_lvl_2_q_2 = Finch.scansearch(input_lvl_2_idx, phase_start_14, input_lvl_2_q_2, input_lvl_2_q_stop_2 - 1)
                                            end
                                            while true
                                                input_lvl_2_i_2 = input_lvl_2_idx[input_lvl_2_q_2]
                                                if input_lvl_2_i_2 < phase_stop_14
                                                    input_lvl_3_val_12 = input_lvl_3_val[input_lvl_2_q_2]
                                                    tmp_lvl_q = (1 - 1) * input_lvl_2_stop + input_lvl_2_i_2
                                                    buf_6[tmp_lvl_q] += input_lvl_3_val_12
                                                    input_lvl_2_q_2 += 1
                                                else
                                                    phase_stop_16 = min(input_lvl_2_i_2, phase_stop_14)
                                                    if input_lvl_2_i_2 == phase_stop_16
                                                        input_lvl_3_val_12 = input_lvl_3_val[input_lvl_2_q_2]
                                                        tmp_lvl_q = (1 - 1) * input_lvl_2_stop + phase_stop_16
                                                        buf_6[tmp_lvl_q] += input_lvl_3_val_12
                                                        input_lvl_2_q_2 += 1
                                                    end
                                                    break
                                                end
                                            end
                                        end
                                        phase_start_16 = max(1, input_lvl_2_i1_3, 1 + input_lvl_2_i1_2)
                                        phase_stop_17 = min(input_lvl_2_stop, 1 + input_lvl_2_i1)
                                        if phase_stop_17 >= phase_start_16
                                            if input_lvl_2_idx[input_lvl_2_q] < -1 + phase_start_16
                                                input_lvl_2_q = Finch.scansearch(input_lvl_2_idx, -1 + phase_start_16, input_lvl_2_q, input_lvl_2_q_stop - 1)
                                            end
                                            while true
                                                input_lvl_2_i = input_lvl_2_idx[input_lvl_2_q]
                                                phase_stop_18 = 1 + input_lvl_2_i
                                                if phase_stop_18 < phase_stop_17
                                                    input_lvl_3_val_13 = input_lvl_3_val[input_lvl_2_q]
                                                    tmp_lvl_q = (1 - 1) * input_lvl_2_stop + phase_stop_18
                                                    buf_6[tmp_lvl_q] += input_lvl_3_val_13
                                                    input_lvl_2_q += 1
                                                else
                                                    phase_stop_19 = min(1 + input_lvl_2_i, phase_stop_17)
                                                    if input_lvl_2_i == -1 + phase_stop_19
                                                        input_lvl_3_val_13 = input_lvl_3_val[input_lvl_2_q]
                                                        tmp_lvl_q = (1 - 1) * input_lvl_2_stop + phase_stop_19
                                                        buf_6[tmp_lvl_q] += input_lvl_3_val_13
                                                        input_lvl_2_q += 1
                                                    end
                                                    break
                                                end
                                            end
                                        end
                                        resize!(buf_6, input_lvl_2_stop)
                                        for x_46 = 1:input_lvl_2_stop
                                            output_lvl_2_q = (output_lvl_q - 1) * input_lvl_2_stop + x_46
                                            tmp_lvl_q_2 = (1 - 1) * input_lvl_2_stop + x_46
                                            tmp_lvl_2_val_2 = buf_6[tmp_lvl_q_2]
                                            output_lvl_3_val[output_lvl_2_q] = tmp_lvl_2_val_2
                                        end
                                    end
                                end
                                phase_start_20 = max(1, 1 + fld(input_lvl_stop * tid, cpu.n))
                                if input_lvl_stop >= phase_start_20
                                    input_lvl_stop + 1
                                end
                            end)
                    nothing
                end
        end
    resize!(buf_5, input_lvl_2_stop * input_lvl_stop)
    (output = Tensor((DenseLevel){Int64}((DenseLevel){Int64}(ElementLevel{0.0, Float64, Int64}(buf_5), input_lvl_2_stop), input_lvl_stop)),)
end
