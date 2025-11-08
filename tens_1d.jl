using Finch
using InteractiveUtils

dev = cpu(1, 2)
tens = Tensor(Dense(Shard(dev, Element(0.0))), 4)

f = :(function run(tens::Tensor{DenseLevel{Int64, ShardLevel{CPU{1}, ElementLevel{0.0, Float64, Int64, Finch.MultiChannelBuffer{Any}}, Vector{Int64}, Vector{Int64}, Vector{Int64}, Vector{Int64}, Finch.FinchStaticSchedule{:dynamic}}}})
      @inbounds @fastmath(begin
                  tens_lvl = tens.lvl
                  tens_lvl_2 = tens_lvl.lvl
                  tens_lvl_2_ptr = tens_lvl_2.ptr
                  tens_lvl_2_task = tens_lvl_2.task
                  tens_lvl_2_qos_fill = tens_lvl_2.used
                  tens_lvl_2_qos_stop = tens_lvl_2.alloc
                  n = tens_lvl_2.device.n
                  tens_lvl_3 = tens_lvl_2.lvl
                  tens_lvl_3_val = tens_lvl_3.val
                  n_3 = 2
                  tens_lvl_3_val_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU{1}(n)), tens_lvl_3_val)
                  tens_lvl_2_qos_fill_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU{1}(n)), tens_lvl_2_qos_fill)
                  tens_lvl_2_qos_stop_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU{1}(n)), tens_lvl_2_qos_stop)
                  Threads.@threads :dynamic for tid = 1:n
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              tens_lvl_3_val_3 = (Finch).transfer(Finch.MemoryChannel(tid, Finch.MultiChannelMemory(Finch.CPU{1}(n), n), Finch.CPUThread(tid, Finch.CPU{1}(n), Finch.SerialTask())), tens_lvl_3_val)
                                              println(@which (Finch).transfer(Finch.MemoryChannel(tid, Finch.MultiChannelMemory(Finch.CPU{1}(n), n), Finch.CPUThread(tid, Finch.CPU{1}(n), Finch.SerialTask())), tens_lvl_3_val))
                                              tens_lvl_2_qos_fill_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU{1}(n), Finch.SerialTask()), tens_lvl_2_qos_fill)
                                              tens_lvl_2_qos_stop_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU{1}(n), Finch.SerialTask()), tens_lvl_2_qos_stop)
                                              resize!(tens_lvl_3_val_3, 1)
                                              tens_lvl_2_qos_fill_3[tid] = 0
                                              tens_lvl_2_qos_stop_3[tid] = max(tens_lvl_2_qos_stop_3[tid], 1)
                                          end)
                                  nothing
                              end
                      end
                  tens_lvl_3_val = (Finch).transfer(tens_lvl_3_val, tens_lvl_3_val_2)
                  tens_lvl_2_qos_fill = (Finch).transfer(tens_lvl_2_qos_fill, tens_lvl_2_qos_fill_2)
                  tens_lvl_2_qos_stop = (Finch).transfer(tens_lvl_2_qos_stop, tens_lvl_2_qos_stop_2)
                  Finch.resize_if_smaller!(tens_lvl_2_task, 4)
                  Finch.resize_if_smaller!(tens_lvl_2_ptr, 4)
                  Finch.fill_range!(tens_lvl_2_ptr, 0, 1, 4)
                  tens_lvl_3_val_4 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU{1}(n_3)), tens_lvl_3_val)
                  tens_lvl_2_ptr_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU{1}(n_3)), tens_lvl_2_ptr)
                  tens_lvl_2_task_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU{1}(n_3)), tens_lvl_2_task)
                  tens_lvl_2_qos_fill_4 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU{1}(n_3)), tens_lvl_2_qos_fill)
                  tens_lvl_2_qos_stop_4 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU{1}(n_3)), tens_lvl_2_qos_stop)
                  Threads.@threads :dynamic for tid_2 = 1:n_3
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              tens_lvl_2_qos_fill_5 = tens_lvl_2_qos_fill_4[tid_2]
                                              tens_lvl_2_qos_stop_5 = tens_lvl_2_qos_stop_4[tid_2]
                                              tens_lvl_3_val_5 = (Finch).transfer(Finch.MemoryChannel(tid_2, Finch.MultiChannelMemory(Finch.CPU{1}(n_3), n_3), Finch.CPUThread(tid_2, Finch.CPU{1}(n_3), Finch.SerialTask())), tens_lvl_3_val_4)
                                              tens_lvl_2_ptr_3 = (Finch).transfer(Finch.CPUThread(tid_2, Finch.CPU{1}(n_3), Finch.SerialTask()), tens_lvl_2_ptr_2)
                                              tens_lvl_2_task_3 = (Finch).transfer(Finch.CPUThread(tid_2, Finch.CPU{1}(n_3), Finch.SerialTask()), tens_lvl_2_task_2)
                                              Finch.CPU{1}(n_3)
                                              Finch.CPU{1}(n_3)
                                              res_4 = begin
                                                      phase_start_2 = max(1, 1 + fld(4 * (tid_2 + -1), n_3))
                                                      phase_stop_2 = min(4, fld(4tid_2, n_3))
                                                      if phase_stop_2 >= phase_start_2
                                                          for j_6 = phase_start_2:phase_stop_2
                                                              tens_lvl_q = (1 - 1) * 4 + j_6
                                                              qos = tens_lvl_2_ptr_3[tens_lvl_q]
                                                              if qos == 0
                                                                  qos = (tens_lvl_2_qos_fill_5 += 1)
                                                                  tens_lvl_2_task_3[tens_lvl_q] = tid_2
                                                                  tens_lvl_2_ptr_3[tens_lvl_q] = tens_lvl_2_qos_fill_5
                                                                  if tens_lvl_2_qos_fill_5 > tens_lvl_2_qos_stop_5
                                                                      tens_lvl_2_qos_stop_5 = max(tens_lvl_2_qos_stop_5 << 1, 1)
                                                                      Finch.resize_if_smaller!(tens_lvl_3_val_5, tens_lvl_2_qos_stop_5)
                                                                      Finch.fill_range!(tens_lvl_3_val_5, 0.0, tens_lvl_2_qos_fill_5, tens_lvl_2_qos_stop_5)
                                                                  end
                                                              else
                                                                  @assert tens_lvl_2_task_3[tens_lvl_q] == tid_2 "Task mismatch in ShardLevel"
                                                              end
                                                              tens_lvl_3_val_5[qos] = j_6
                                                          end
                                                      end
                                                      phase_start_3 = max(1, 1 + fld(4tid_2, n_3))
                                                      if 4 >= phase_start_3
                                                          4 + 1
                                                      end
                                                  end
                                              resize!(tens_lvl_3_val_5, tens_lvl_2_qos_stop_5)
                                              res_4
                                          end)
                                  nothing
                              end
                      end
                  ()
              end)
  end)

eval(f)

run(tens)
display(tens)