using Finch

dev = cpu(1, 2)
tens = Tensor(Dense(Shard(dev, SparseList(Element(0.0)))), 4, 4)

q = :(
    function run(tens, dev)
        @inbounds @fastmath(
            begin
                tens_lvl = tens.lvl
                tens_lvl_stop = tens_lvl.shape
                tens_lvl_2 = tens_lvl.lvl
                tens_lvl_2_ptr = tens_lvl_2.ptr
                tens_lvl_2_task = tens_lvl_2.task
                tens_lvl_2_qos_fill = tens_lvl_2.used
                tens_lvl_2_qos_stop = tens_lvl_2.alloc
                tens_lvl_3 = tens_lvl_2.lvl
                tens_lvl_3_ptr = tens_lvl_3.ptr
                tens_lvl_3_idx = tens_lvl_3.idx
                tens_lvl_3_stop = tens_lvl_3.shape
                tens_lvl_4 = tens_lvl_3.lvl
                tens_lvl_4_val = tens_lvl_4.val
                n_2 = dev.n
                tens_lvl_3_stop == 4 || throw(
                    DimensionMismatch(
                        "mismatched dimension limits ($(tens_lvl_3_stop) != $(4))"
                    ),
                )
                4 == tens_lvl_stop || throw(
                    DimensionMismatch(
                        "mismatched dimension limits ($(4) != $(tens_lvl_stop))"
                    ),
                )
                tens_lvl_4_val_2 = (Finch).transfer(
                    Finch.CPUSharedMemory(Finch.CPU{1}(n_2)), tens_lvl_4_val
                )
                tens_lvl_3_ptr_2 = (Finch).transfer(
                    Finch.CPUSharedMemory(Finch.CPU{1}(n_2)), tens_lvl_3_ptr
                )
                tens_lvl_3_idx_2 = (Finch).transfer(
                    Finch.CPUSharedMemory(Finch.CPU{1}(n_2)), tens_lvl_3_idx
                )
                tens_lvl_2_ptr_2 = (Finch).transfer(
                    Finch.CPUSharedMemory(Finch.CPU{1}(n_2)), tens_lvl_2_ptr
                )
                tens_lvl_2_task_2 = (Finch).transfer(
                    Finch.CPUSharedMemory(Finch.CPU{1}(n_2)), tens_lvl_2_task
                )
                tens_lvl_2_qos_fill_2 = (Finch).transfer(
                    Finch.CPUSharedMemory(Finch.CPU{1}(n_2)), tens_lvl_2_qos_fill
                )
                tens_lvl_2_qos_stop_2 = (Finch).transfer(
                    Finch.CPUSharedMemory(Finch.CPU{1}(n_2)), tens_lvl_2_qos_stop
                )
                Threads.@threads :dynamic for tid in 1:n_2
                    Finch.@barrier begin
                        @inbounds @fastmath(
                            begin
                                tens_lvl_2_qos_fill_3 = tens_lvl_2_qos_fill_2[tid]
                                tens_lvl_2_qos_stop_3 = tens_lvl_2_qos_stop_2[tid]
                                tens_lvl_4_val_3 = (Finch).transfer(
                                    MemoryChannel(
                                        tid,
                                        MultiChannelMemory(Finch.CPU{1}(n_2), n_2),
                                        Finch.CPUThread(
                                            tid, Finch.CPU{1}(n_2), SerialTask()
                                        ),
                                    ),
                                    tens_lvl_4_val_2,
                                )
                                tens_lvl_3_ptr_3 = (Finch).transfer(
                                    MemoryChannel(
                                        tid,
                                        MultiChannelMemory(Finch.CPU{1}(n_2), n_2),
                                        Finch.CPUThread(
                                            tid, Finch.CPU{1}(n_2), SerialTask()
                                        ),
                                    ),
                                    tens_lvl_3_ptr_2,
                                )
                                tens_lvl_3_idx_3 = (Finch).transfer(
                                    MemoryChannel(
                                        tid,
                                        MultiChannelMemory(Finch.CPU{1}(n_2), n_2),
                                        Finch.CPUThread(
                                            tid, Finch.CPU{1}(n_2), SerialTask()
                                        ),
                                    ),
                                    tens_lvl_3_idx_2,
                                )
                                tens_lvl_3_qos_fill =
                                    tens_lvl_3_ptr_3[tens_lvl_2_qos_stop_3 + 1] - 1
                                tens_lvl_3_qos_stop = tens_lvl_3_qos_fill
                                tens_lvl_3_prev_pos =
                                    Finch.scansearch(
                                        tens_lvl_3_ptr_3,
                                        tens_lvl_3_qos_fill + 1,
                                        1,
                                        tens_lvl_2_qos_stop_3,
                                    ) - 1
                                for p in tens_lvl_2_qos_stop_3:-1:1
                                    tens_lvl_3_ptr_3[p + 1] =
                                        tens_lvl_3_ptr_3[p + 1] - tens_lvl_3_ptr_3[p]
                                end
                                tens_lvl_2_ptr_3 = (Finch).transfer(
                                    Finch.CPUThread(tid, Finch.CPU{1}(n_2), SerialTask()),
                                    tens_lvl_2_ptr_2,
                                )
                                tens_lvl_2_task_3 = (Finch).transfer(
                                    Finch.CPUThread(tid, Finch.CPU{1}(n_2), SerialTask()),
                                    tens_lvl_2_task_2,
                                )
                                Finch.CPU{1}(n_2)
                                Finch.CPU{1}(n_2)
                                res_5 = begin
                                    phase_start_2 = max(1, 1 + fld(4 * (tid + -1), n_2))
                                    phase_stop_2 = min(4, fld(4tid, n_2))
                                    if phase_stop_2 >= phase_start_2
                                        for j_6 in phase_start_2:phase_stop_2
                                            tens_lvl_q = (1 - 1) * tens_lvl_stop + j_6
                                            qos = tens_lvl_2_ptr_3[tens_lvl_q]
                                            if qos == 0
                                                qos = (tens_lvl_2_qos_fill_3 += 1)
                                                tens_lvl_2_task_3[tens_lvl_q] = tid
                                                tens_lvl_2_ptr_3[tens_lvl_q] =
                                                    tens_lvl_2_qos_fill_3
                                                if tens_lvl_2_qos_fill_3 >
                                                    tens_lvl_2_qos_stop_3
                                                    tens_lvl_2_qos_stop_3 = max(
                                                        tens_lvl_2_qos_stop_3 << 1,
                                                        1,
                                                    )
                                                    Finch.resize_if_smaller!(
                                                        tens_lvl_3_ptr_3,
                                                        tens_lvl_2_qos_stop_3 + 1,
                                                    )
                                                    Finch.fill_range!(
                                                        tens_lvl_3_ptr_3,
                                                        0,
                                                        tens_lvl_2_qos_fill_3 + 1,
                                                        tens_lvl_2_qos_stop_3 + 1,
                                                    )
                                                end
                                            else
                                                @assert tens_lvl_2_task_3[tens_lvl_q] == tid "Task mismatch in ShardLevel"
                                            end
                                            tens_lvl_3_qos = tens_lvl_3_qos_fill + 1
                                            tens_lvl_3_prev_pos < qos || throw(
                                                FinchProtocolError(
                                                    "SparseListLevels cannot be updated multiple times"
                                                ),
                                            )
                                            for i_3 in 1:tens_lvl_3_stop
                                                if tens_lvl_3_qos > tens_lvl_3_qos_stop
                                                    tens_lvl_3_qos_stop = max(
                                                        tens_lvl_3_qos_stop << 1, 1
                                                    )
                                                    Finch.resize_if_smaller!(
                                                        tens_lvl_3_idx_3,
                                                        tens_lvl_3_qos_stop,
                                                    )
                                                    Finch.resize_if_smaller!(
                                                        tens_lvl_4_val_3,
                                                        tens_lvl_3_qos_stop,
                                                    )
                                                    Finch.fill_range!(
                                                        tens_lvl_4_val_3,
                                                        0.0,
                                                        tens_lvl_3_qos,
                                                        tens_lvl_3_qos_stop,
                                                    )
                                                end
                                                tens_lvl_4_val_3[tens_lvl_3_qos] = j_6
                                                tens_lvl_3_idx_3[tens_lvl_3_qos] = i_3
                                                tens_lvl_3_qos += 1
                                                tens_lvl_3_prev_pos = qos
                                            end
                                            tens_lvl_3_ptr_3[qos + 1] +=
                                                (tens_lvl_3_qos - tens_lvl_3_qos_fill) - 1
                                            tens_lvl_3_qos_fill = tens_lvl_3_qos - 1
                                        end
                                    end
                                    phase_start_3 = max(1, 1 + fld(4tid, n_2))
                                    if 4 >= phase_start_3
                                        4 + 1
                                    end
                                end
                                resize!(tens_lvl_3_ptr_3, tens_lvl_2_qos_stop_3 + 1)
                                for p_2 in 1:tens_lvl_2_qos_stop_3
                                    tens_lvl_3_ptr_3[p_2 + 1] += tens_lvl_3_ptr_3[p_2]
                                end
                                qos_stop_2 =
                                    tens_lvl_3_ptr_3[tens_lvl_2_qos_stop_3 + 1] - 1
                                resize!(tens_lvl_3_idx_3, qos_stop_2)
                                resize!(tens_lvl_4_val_3, qos_stop_2)
                                res_5
                            end
                        )
                        nothing
                    end
                end
                ()
            end
        )
    end
)

eval(q)
run(tens, dev)
display(tens)