using Finch

dev = cpu(:t, 2)
dev2 = cpu(:q, 2)
tens2 = Tensor(Dense(Shard(dev, Dense(Shard(dev2, Element(0.0))))), 3, 3)

code = :(function run(tens2::Tensor{DenseLevel{Int64, ShardLevel{CPU{:t}, DenseLevel{Int64, ShardLevel{CPU{:q}, ElementLevel{0.0, Float64, Int64, Finch.MultiChannelBuffer{Finch.MultiChannelBuffer{Vector{Float64}}}}, Finch.MultiChannelBuffer{Vector{Int64}}, Finch.MultiChannelBuffer{Vector{Int64}}, Finch.MultiChannelBuffer{Vector{Int64}}, Finch.MultiChannelBuffer{Vector{Int64}}, Finch.FinchStaticSchedule{:dynamic}}}, Vector{Int64}, Vector{Int64}, Vector{Int64}, Vector{Int64}, Finch.FinchStaticSchedule{:dynamic}}}}, dev::CPU{:t}, dev2::CPU{:q})
      @inbounds @fastmath(begin
                  tens2_lvl = tens2.lvl
                  tens2_lvl_2 = tens2_lvl.lvl
                  tens2_lvl_2_ptr = tens2_lvl_2.ptr
                  tens2_lvl_2_task = tens2_lvl_2.task
                  tens2_lvl_2_qos_fill = tens2_lvl_2.used
                  tens2_lvl_2_qos_stop = tens2_lvl_2.alloc
                  n = tens2_lvl_2.device.n
                  tens2_lvl_3 = tens2_lvl_2.lvl
                  tens2_lvl_4 = tens2_lvl_3.lvl
                  tens2_lvl_4_ptr = tens2_lvl_4.ptr
                  tens2_lvl_4_task = tens2_lvl_4.task
                  tens2_lvl_4_qos_fill = tens2_lvl_4.used
                  tens2_lvl_4_qos_stop = tens2_lvl_4.alloc
                  n_2 = tens2_lvl_4.device.n
                  tens2_lvl_5 = tens2_lvl_4.lvl
                  tens2_lvl_5_val = tens2_lvl_5.val
                  n_3 = dev.n
                  n_4 = dev2.n
                  tens2_lvl_5_val_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens2_lvl_5_val)
                  tens2_lvl_4_ptr_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens2_lvl_4_ptr)
                  tens2_lvl_4_task_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens2_lvl_4_task)
                  tens2_lvl_4_qos_fill_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens2_lvl_4_qos_fill)
                  tens2_lvl_4_qos_stop_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens2_lvl_4_qos_stop)
                  tens2_lvl_2_qos_fill_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens2_lvl_2_qos_fill)
                  tens2_lvl_2_qos_stop_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens2_lvl_2_qos_stop)
                  Threads.@threads :dynamic for tid = 1:n
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              alloced_pos = tens2_lvl_2_qos_stop_2[tid]
                                              tens2_lvl_5_val_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens2_lvl_5_val)
                                              (Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)()))
                                              (Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)()))
                                              tens2_lvl_4_qos_fill_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens2_lvl_4_qos_fill)
                                              tens2_lvl_4_qos_stop_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens2_lvl_4_qos_stop)
                                              tens2_lvl_2_qos_fill_3 = (Finch).transfer((Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)()), tens2_lvl_2_qos_fill)
                                              tens2_lvl_2_qos_stop_3 = (Finch).transfer((Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)()), tens2_lvl_2_qos_stop)
                                              (Finch.CPUSharedMemory)(Finch.CPU{:q}(n_2))
                                              (Finch.CPUSharedMemory)(Finch.CPU{:q}(n_2))
                                              tens2_lvl_4_qos_stop_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_2)), tens2_lvl_4_qos_stop_3)
                                              Threads.@threads :dynamic for tid_2 = 1:n_2
                                                      Finch.@barrier begin
                                                              @inbounds @fastmath(begin
                                                                          alloced_pos_2 = tens2_lvl_4_qos_stop_4[tid_2]
                                                                          tens2_lvl_5_val_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_2), n_2), (Finch.CPUThread)(tid_2, Finch.CPU{:q}(n_2), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)()))), tens2_lvl_5_val_3)
                                                                          tens2_lvl_4_qos_fill_5 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:q}(n_2), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens2_lvl_4_qos_fill_3)
                                                                          tens2_lvl_4_qos_stop_5 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:q}(n_2), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens2_lvl_4_qos_stop_3)
                                                                          resize!(tens2_lvl_5_val_5, alloced_pos_2)
                                                                          tens2_lvl_4_qos_fill_5[tid_2] = 0
                                                                          tens2_lvl_4_qos_stop_5[tid_2] = alloced_pos_2
                                                                      end)
                                                              nothing
                                                          end
                                                  end
                                              tens2_lvl_2_qos_fill_3[tid] = 0
                                              tens2_lvl_2_qos_stop_3[tid] = alloced_pos
                                          end)
                                  nothing
                              end
                      end
                  tens2_lvl_5_val = (Finch).transfer(tens2_lvl_5_val, tens2_lvl_5_val_2)
                  tens2_lvl_4_ptr = (Finch).transfer(tens2_lvl_4_ptr, tens2_lvl_4_ptr_2)
                  tens2_lvl_4_task = (Finch).transfer(tens2_lvl_4_task, tens2_lvl_4_task_2)
                  tens2_lvl_4_qos_fill = (Finch).transfer(tens2_lvl_4_qos_fill, tens2_lvl_4_qos_fill_2)
                  tens2_lvl_4_qos_stop = (Finch).transfer(tens2_lvl_4_qos_stop, tens2_lvl_4_qos_stop_2)
                  tens2_lvl_2_qos_fill = (Finch).transfer(tens2_lvl_2_qos_fill, tens2_lvl_2_qos_fill_2)
                  tens2_lvl_2_qos_stop = (Finch).transfer(tens2_lvl_2_qos_stop, tens2_lvl_2_qos_stop_2)
                  Finch.resize_if_smaller!(tens2_lvl_2_task, 3)
                  Finch.resize_if_smaller!(tens2_lvl_2_ptr, 3)
                  Finch.fill_range!(tens2_lvl_2_ptr, 0, 1, 3)
                  tens2_lvl_5_val_6 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), tens2_lvl_5_val)
                  tens2_lvl_4_ptr_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), tens2_lvl_4_ptr)
                  tens2_lvl_4_task_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), tens2_lvl_4_task)
                  tens2_lvl_4_qos_fill_6 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), tens2_lvl_4_qos_fill)
                  tens2_lvl_4_qos_stop_6 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), tens2_lvl_4_qos_stop)
                  tens2_lvl_2_ptr_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), tens2_lvl_2_ptr)
                  tens2_lvl_2_task_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), tens2_lvl_2_task)
                  tens2_lvl_2_qos_fill_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), tens2_lvl_2_qos_fill)
                  tens2_lvl_2_qos_stop_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), tens2_lvl_2_qos_stop)
                  Threads.@threads :dynamic for tid_3 = 1:n_3
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              tens2_lvl_2_ptr_3 = (Finch).transfer((Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), tens2_lvl_2_ptr_2)
                                              tens2_lvl_2_task_3 = (Finch).transfer((Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), tens2_lvl_2_task_2)
                                              tens2_lvl_2_qos_fill_6 = (Finch).transfer((Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), tens2_lvl_2_qos_fill_4)
                                              tens2_lvl_2_qos_stop_6 = (Finch).transfer((Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), tens2_lvl_2_qos_stop_4)
                                              tens2_lvl_2_qos_fill_5 = tens2_lvl_2_qos_fill_6[tid_3]
                                              tens2_lvl_2_qos_stop_5 = tens2_lvl_2_qos_stop_6[tid_3]
                                              tens2_lvl_5_val_7 = (Finch).transfer((Finch.MemoryChannel)(tid_3, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), tens2_lvl_5_val_6)
                                              tens2_lvl_4_ptr_5 = (Finch).transfer((Finch.MemoryChannel)(tid_3, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), tens2_lvl_4_ptr_4)
                                              tens2_lvl_4_task_5 = (Finch).transfer((Finch.MemoryChannel)(tid_3, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), tens2_lvl_4_task_4)
                                              tens2_lvl_4_qos_fill_7 = (Finch).transfer((Finch.MemoryChannel)(tid_3, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), tens2_lvl_4_qos_fill_6)
                                              tens2_lvl_4_qos_stop_7 = (Finch).transfer((Finch.MemoryChannel)(tid_3, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), tens2_lvl_4_qos_stop_6)
                                              res_9 = begin
                                                      phase_start_2 = max(1, 1 + fld(3 * (tid_3 + -1), n_3))
                                                      phase_stop_2 = min(3, fld(3tid_3, n_3))
                                                      if phase_stop_2 >= phase_start_2
                                                          for j_6 = phase_start_2:phase_stop_2
                                                              tens2_lvl_q = (1 - 1) * 3 + j_6
                                                              qos = tens2_lvl_2_ptr_3[tens2_lvl_q]
                                                              if qos == 0
                                                                  qos = (tens2_lvl_2_qos_fill_5 += 1)
                                                                  tens2_lvl_2_task_3[tens2_lvl_q] = tid_3
                                                                  tens2_lvl_2_ptr_3[tens2_lvl_q] = tens2_lvl_2_qos_fill_5
                                                                  if tens2_lvl_2_qos_fill_5 > tens2_lvl_2_qos_stop_5
                                                                      tens2_lvl_2_qos_stop_5 = max(tens2_lvl_2_qos_stop_5 << 1, 1)
                                                                      pos_start = 1 + 3 * (-1 + tens2_lvl_2_qos_fill_5)
                                                                      pos_stop = 3tens2_lvl_2_qos_stop_5
                                                                    #   fval = length(tens2_lvl_4_ptr_5) > 0 ? tens2_lvl_4_ptr_5[length(tens2_lvl_4_ptr_5)] : 0
                                                                      Finch.resize_if_smaller!(tens2_lvl_4_task_5, pos_stop)
                                                                      Finch.resize_if_smaller!(tens2_lvl_4_ptr_5, pos_stop)
                                                                      Finch.fill_range!(tens2_lvl_4_ptr_5, 0, pos_start, pos_stop)
                                                                  end
                                                              else
                                                                  @assert tens2_lvl_2_task_3[tens2_lvl_q] == tid_3 "Task mismatch in ShardLevel"
                                                              end
                                                              tens2_lvl_5_val_8 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), tens2_lvl_5_val_7)
                                                              tens2_lvl_4_ptr_6 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), tens2_lvl_4_ptr_5)
                                                              tens2_lvl_4_task_6 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), tens2_lvl_4_task_5)
                                                              tens2_lvl_4_qos_fill_8 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), tens2_lvl_4_qos_fill_7)
                                                              tens2_lvl_4_qos_stop_8 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), tens2_lvl_4_qos_stop_7)
                                                              tens2_lvl_2_ptr_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), tens2_lvl_2_ptr_3)
                                                              tens2_lvl_2_task_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), tens2_lvl_2_task_3)
                                                              (Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4))
                                                              (Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4))
                                                              Threads.@threads :dynamic for tid_4 = 1:n_4
                                                                      Finch.@barrier begin
                                                                              @inbounds @fastmath(begin
                                                                                          tens2_lvl_4_ptr_7 = (Finch).transfer((Finch.CPUThread)(tid_4, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), tens2_lvl_4_ptr_6)
                                                                                          tens2_lvl_4_task_7 = (Finch).transfer((Finch.CPUThread)(tid_4, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), tens2_lvl_4_task_6)
                                                                                          tens2_lvl_4_qos_fill_10 = (Finch).transfer((Finch.CPUThread)(tid_4, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), tens2_lvl_4_qos_fill_8)
                                                                                          tens2_lvl_4_qos_stop_10 = (Finch).transfer((Finch.CPUThread)(tid_4, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), tens2_lvl_4_qos_stop_8)
                                                                                          tens2_lvl_4_qos_fill_9 = tens2_lvl_4_qos_fill_10[tid_4]
                                                                                          tens2_lvl_4_qos_stop_9 = tens2_lvl_4_qos_stop_10[tid_4]
                                                                                          tens2_lvl_5_val_9 = (Finch).transfer((Finch.MemoryChannel)(tid_4, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_2), n_2), (Finch.CPUThread)(tid_4, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))), tens2_lvl_5_val_8)
                                                                                          (Finch.CPUThread)(tid_4, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))
                                                                                          (Finch.CPUThread)(tid_4, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))
                                                                                          (Finch.CPUThread)(tid_4, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))
                                                                                          (Finch.CPUThread)(tid_4, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))
                                                                                          res_5 = begin
                                                                                                  phase_start_4 = max(1, 1 + fld(3 * (-1 + tid_4), n_4))
                                                                                                  phase_stop_4 = min(3, fld(3tid_4, n_4))
                                                                                                  if phase_stop_4 >= phase_start_4
                                                                                                      for i_5 = phase_start_4:phase_stop_4
                                                                                                          tens2_lvl_3_q = (qos - 1) * 3 + i_5
                                                                                                          qos_2 = tens2_lvl_4_ptr_7[tens2_lvl_3_q]
                                                                                                          println((i_5, j_6))
                                                                                                          println(tens2_lvl_4_ptr.data)
                                                                                                          println("------------------")
                                                                                                          if qos_2 == 0
                                                                                                              qos_2 = (tens2_lvl_4_qos_fill_9 += 1)
                                                                                                              tens2_lvl_4_task_7[tens2_lvl_3_q] = tid_4
                                                                                                              tens2_lvl_4_ptr_7[tens2_lvl_3_q] = tens2_lvl_4_qos_fill_9
                                                                                                              if tens2_lvl_4_qos_fill_9 > tens2_lvl_4_qos_stop_9
                                                                                                                  tens2_lvl_4_qos_stop_9 = max(tens2_lvl_4_qos_stop_9 << 1, 1)
                                                                                                                  Finch.resize_if_smaller!(tens2_lvl_5_val_9, tens2_lvl_4_qos_stop_9)
                                                                                                                  Finch.fill_range!(tens2_lvl_5_val_9, 0.0, tens2_lvl_4_qos_fill_9, tens2_lvl_4_qos_stop_9)
                                                                                                              end
                                                                                                          else
                                                                                                              @assert tens2_lvl_4_task_7[tens2_lvl_3_q] == tid_4 "Task mismatch in ShardLevel"
                                                                                                          end
                                                                                                          
                                                                                                          tens2_lvl_5_val_9[qos_2] = j_6
                                                                                                      end
                                                                                                  end
                                                                                                  phase_start_5 = max(1, 1 + fld(3tid_4, n_4))
                                                                                                  if 3 >= phase_start_5
                                                                                                      3 + 1
                                                                                                  end
                                                                                              end
                                                                                            tens2_lvl_4_qos_fill_10[tid_4] = tens2_lvl_4_qos_fill_9
                                                                                                          tens2_lvl_4_qos_stop_10[tid_4] = tens2_lvl_4_qos_stop_9
                                                                                          resize!(tens2_lvl_5_val_9, tens2_lvl_4_qos_stop_9)
                                                                                          res_5
                                                                                      end)
                                                                              nothing
                                                                          end
                                                                  end
                                                              tens2_lvl_5_val_7 = (Finch).transfer(tens2_lvl_5_val_7, tens2_lvl_5_val_8)
                                                              tens2_lvl_4_ptr_5 = (Finch).transfer(tens2_lvl_4_ptr_5, tens2_lvl_4_ptr_6)
                                                              tens2_lvl_4_task_5 = (Finch).transfer(tens2_lvl_4_task_5, tens2_lvl_4_task_6)
                                                              tens2_lvl_4_qos_fill_7 = (Finch).transfer(tens2_lvl_4_qos_fill_7, tens2_lvl_4_qos_fill_8)
                                                              tens2_lvl_4_qos_stop_7 = (Finch).transfer(tens2_lvl_4_qos_stop_7, tens2_lvl_4_qos_stop_8)
                                                              tens2_lvl_2_ptr_3 = (Finch).transfer(tens2_lvl_2_ptr_3, tens2_lvl_2_ptr_4)
                                                              tens2_lvl_2_task_3 = (Finch).transfer(tens2_lvl_2_task_3, tens2_lvl_2_task_4)
                                                          end
                                                      end
                                                      phase_start_6 = max(1, 1 + fld(3tid_3, n_3))
                                                      if 3 >= phase_start_6
                                                          3 + 1
                                                      end
                                                  end
                                                  println(tens2_lvl_4_qos_fill)
                                                  println(tens2_lvl_4_qos_stop)
                                                  display(tens2)
                                          end)
                                  nothing
                              end
                      end
                  ()
              end)
  end)

eval(code)
run(tens2, dev, dev2)

