using Finch
using InteractiveUtils

dev = cpu(:t, 2)
acc = Tensor(Dense(SparseList(Element(0.0))), [1 2; 3 4])
tens = Tensor(Dense(Coalesce(dev, SparseList(Element(0.0)))), 2, 2)

f = :(function run(tens::Tensor{DenseLevel{Int64, CoalesceLevel{CPU{:t}, SparseListLevel{Int64, Finch.MultiChannelBuffer{Vector{Int64}}, Finch.MultiChannelBuffer{Vector{Int64}}, ElementLevel{0.0, Float64, Int64, Finch.MultiChannelBuffer{Vector{Float64}}}}, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}, Finch.FinchStaticSchedule{:dynamic}}}}, acc::Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}, dev::CPU{:t})
      @inbounds @fastmath(begin
                  tens_lvl = tens.lvl
                  tens_lvl_2 = tens_lvl.lvl
                  n = tens_lvl_2.device.n
                  tens_lvl_3 = tens_lvl_2.lvl
                  tens_lvl_3_ptr = tens_lvl_3.ptr
                  tens_lvl_3_idx = tens_lvl_3.idx
                  tens_lvl_4 = tens_lvl_3.lvl
                  tens_lvl_4_val = tens_lvl_4.val
                  tens_lvl_5 = tens_lvl_2.coalescent
                  tens_lvl_5_ptr = tens_lvl_5.ptr
                  tens_lvl_5_idx = tens_lvl_5.idx
                  tens_lvl_5_stop = tens_lvl_5.shape
                  tens_lvl_6 = tens_lvl_5.lvl
                  tens_lvl_6_val = tens_lvl_6.val
                  acc_lvl = acc.lvl
                  acc_lvl_stop = acc_lvl.shape
                  acc_lvl_2 = acc_lvl.lvl
                  acc_lvl_2_ptr = acc_lvl_2.ptr
                  acc_lvl_2_idx = acc_lvl_2.idx
                  acc_lvl_2_stop = acc_lvl_2.shape
                  acc_lvl_3 = acc_lvl_2.lvl
                  acc_lvl_3_val = acc_lvl_3.val
                  n_2 = dev.n
                  tens_lvl_4_val_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_4_val)
                  tens_lvl_3_ptr_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_3_ptr)
                  tens_lvl_3_idx_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_3_idx)
                  Threads.@threads :dynamic for tid = 1:n
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              tens_lvl_4_val_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_4_val)
                                              tens_lvl_3_ptr_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_3_ptr)
                                              tens_lvl_3_idx_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_3_idx)
                                              resize!(tens_lvl_3_ptr_3, acc_lvl_stop + 1)
                                              for p = 1:acc_lvl_stop
                                                  tens_lvl_3_ptr_3[p + 1] += tens_lvl_3_ptr_3[p]
                                              end
                                              qos_stop = tens_lvl_3_ptr_3[acc_lvl_stop + 1] - 1
                                              resize!(tens_lvl_3_idx_3, qos_stop)
                                              resize!(tens_lvl_4_val_3, qos_stop)
                                              Finch.VirtualSparseListLevel(:tens_lvl_3, Finch.VirtualElementLevel(:tens_lvl_4, 0.0, Float64, Int64, :tens_lvl_4_val_3), Int64, :tens_lvl_3_ptr_3, :tens_lvl_3_idx_3, value(acc_lvl_2_stop, Int64), :tens_lvl_3_qos_fill, :tens_lvl_3_qos_stop, :tens_lvl_3_prev_pos)
                                          end)
                                  nothing
                              end
                      end
                  tens_lvl_4_val = (Finch).transfer(tens_lvl_4_val, tens_lvl_4_val_2)
                  tens_lvl_3_ptr = (Finch).transfer(tens_lvl_3_ptr, tens_lvl_3_ptr_2)
                  tens_lvl_3_idx = (Finch).transfer(tens_lvl_3_idx, tens_lvl_3_idx_2)
                  resize!(tens_lvl_5_ptr, acc_lvl_stop + 1)
                  for p_2 = 1:acc_lvl_stop
                      tens_lvl_5_ptr[p_2 + 1] += tens_lvl_5_ptr[p_2]
                  end
                  qos_stop_2 = tens_lvl_5_ptr[acc_lvl_stop + 1] - 1
                  resize!(tens_lvl_5_idx, qos_stop_2)
                  resize!(tens_lvl_6_val, qos_stop_2)
                  tens_lvl_4_val_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_4_val)
                  tens_lvl_3_ptr_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_3_ptr)
                  tens_lvl_3_idx_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_3_idx)
                  Threads.@threads :dynamic for tid_2 = 1:n
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              tens_lvl_4_val_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_4_val)
                                              tens_lvl_3_ptr_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_3_ptr)
                                              tens_lvl_3_idx_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_3_idx)
                                              Finch.resize_if_smaller!(tens_lvl_3_ptr_5, acc_lvl_stop + 1)
                                              Finch.fill_range!(tens_lvl_3_ptr_5, 0, 1 + 1, acc_lvl_stop + 1)
                                              resize!(tens_lvl_3_ptr_5, acc_lvl_stop + 1)
                                              for p_3 = 1:acc_lvl_stop
                                                  tens_lvl_3_ptr_5[p_3 + 1] += tens_lvl_3_ptr_5[p_3]
                                              end
                                              qos_stop_3 = tens_lvl_3_ptr_5[acc_lvl_stop + 1] - 1
                                              resize!(tens_lvl_3_idx_5, qos_stop_3)
                                              resize!(tens_lvl_4_val_5, qos_stop_3)
                                              Finch.VirtualSparseListLevel(:tens_lvl_3, Finch.VirtualElementLevel(:tens_lvl_4, 0.0, Float64, Int64, :tens_lvl_4_val_5), Int64, :tens_lvl_3_ptr_5, :tens_lvl_3_idx_5, value(acc_lvl_2_stop, Int64), :tens_lvl_3_qos_fill, :tens_lvl_3_qos_stop, :tens_lvl_3_prev_pos)
                                          end)
                                  nothing
                              end
                      end
                  Finch.resize_if_smaller!(tens_lvl_5_ptr, acc_lvl_stop + 1)
                  Finch.fill_range!(tens_lvl_5_ptr, 0, 1 + 1, acc_lvl_stop + 1)
                  resize!(tens_lvl_5_ptr, acc_lvl_stop + 1)
                  for p_4 = 1:acc_lvl_stop
                      tens_lvl_5_ptr[p_4 + 1] += tens_lvl_5_ptr[p_4]
                  end
                  qos_stop_4 = tens_lvl_5_ptr[acc_lvl_stop + 1] - 1
                  resize!(tens_lvl_5_idx, qos_stop_4)
                  resize!(tens_lvl_6_val, qos_stop_4)
                  tens_lvl_4_val = (Finch).transfer(tens_lvl_4_val, tens_lvl_4_val_4)
                  tens_lvl_3_ptr = (Finch).transfer(tens_lvl_3_ptr, tens_lvl_3_ptr_4)
                  tens_lvl_3_idx = (Finch).transfer(tens_lvl_3_idx, tens_lvl_3_idx_4)
                  acc_lvl_3_val_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), acc_lvl_3_val)
                  acc_lvl_2_ptr_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), acc_lvl_2_ptr)
                  acc_lvl_2_idx_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), acc_lvl_2_idx)
                  tens_lvl_4_val_6 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), tens_lvl_4_val)
                  tens_lvl_3_ptr_6 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), tens_lvl_3_ptr)
                  tens_lvl_3_idx_6 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), tens_lvl_3_idx)
                  Threads.@threads :dynamic for tid_3 = 1:n_2
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              acc_lvl_3_val_3 = (Finch).transfer((Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_2), (Finch.SerialTask)()), acc_lvl_3_val_2)
                                              acc_lvl_2_ptr_3 = (Finch).transfer((Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_2), (Finch.SerialTask)()), acc_lvl_2_ptr_2)
                                              acc_lvl_2_idx_3 = (Finch).transfer((Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_2), (Finch.SerialTask)()), acc_lvl_2_idx_2)
                                              tens_lvl_4_val_7 = (Finch).transfer((Finch.MemoryChannel)(tid_3, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n_2), n_2), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_2), (Finch.SerialTask)())), tens_lvl_4_val_6)
                                              tens_lvl_3_ptr_7 = (Finch).transfer((Finch.MemoryChannel)(tid_3, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n_2), n_2), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_2), (Finch.SerialTask)())), tens_lvl_3_ptr_6)
                                              tens_lvl_3_idx_7 = (Finch).transfer((Finch.MemoryChannel)(tid_3, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n_2), n_2), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_2), (Finch.SerialTask)())), tens_lvl_3_idx_6)

                                              
                                              tens_lvl_3_qos_fill = tens_lvl_3_ptr_7[acc_lvl_stop + 1] - 1
                                              tens_lvl_3_qos_stop = tens_lvl_3_qos_fill
                                              tens_lvl_3_prev_pos = Finch.scansearch(tens_lvl_3_ptr_7, tens_lvl_3_qos_fill + 1, 1, acc_lvl_stop) - 1
                                              for p_5 = acc_lvl_stop:-1:1
                                                  tens_lvl_3_ptr_7[p_5 + 1] = tens_lvl_3_ptr_7[p_5 + 1] - tens_lvl_3_ptr_7[p_5]
                                              end

                                              tens_lvl_4_val_8 = (Finch).transfer((Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_2), (Finch.SerialTask)()), tens_lvl_4_val_6)
                                              tens_lvl_3_ptr_8 = (Finch).transfer((Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_2), (Finch.SerialTask)()), tens_lvl_3_ptr_6)
                                              println(@which(tens_lvl_3_ptr_8 = (Finch).transfer((Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_2), (Finch.SerialTask)()), tens_lvl_3_ptr_6)))
                                              tens_lvl_3_idx_8 = (Finch).transfer((Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_2), (Finch.SerialTask)()), tens_lvl_3_idx_6)
                                              res_12 = begin
                                                      phase_start_2 = max(1, 1 + fld(acc_lvl_stop * (tid_3 + -1), n_2))
                                                      phase_stop_2 = min(acc_lvl_stop, fld(acc_lvl_stop * tid_3, n_2))
                                                      if phase_stop_2 >= phase_start_2
                                                          for j_7 = phase_start_2:phase_stop_2
                                                              acc_lvl_q = (1 - 1) * acc_lvl_stop + j_7
                                                              tens_lvl_q = (1 - 1) * acc_lvl_stop + j_7
                                                              acc_lvl_2_q = acc_lvl_2_ptr_3[acc_lvl_q]
                                                              acc_lvl_2_q_stop = acc_lvl_2_ptr_3[acc_lvl_q + 1]
                                                              if acc_lvl_2_q < acc_lvl_2_q_stop
                                                                  acc_lvl_2_i1 = acc_lvl_2_idx_3[acc_lvl_2_q_stop - 1]
                                                              else
                                                                  acc_lvl_2_i1 = 0
                                                              end
                                                              tens_lvl_3_qos = tens_lvl_3_qos_fill + 1
                                                              tens_lvl_3_prev_pos < tens_lvl_q || throw((Finch.FinchProtocolError)("SparseListLevels cannot be updated multiple times"))
                                                              phase_stop_3 = min(acc_lvl_2_i1, acc_lvl_2_stop)
                                                              if phase_stop_3 >= 1
                                                                  if acc_lvl_2_idx_3[acc_lvl_2_q] < 1
                                                                      acc_lvl_2_q = Finch.scansearch(acc_lvl_2_idx_3, 1, acc_lvl_2_q, acc_lvl_2_q_stop - 1)
                                                                  end
                                                                  while true
                                                                      acc_lvl_2_i = acc_lvl_2_idx_3[acc_lvl_2_q]
                                                                      if acc_lvl_2_i < phase_stop_3
                                                                          acc_lvl_3_val_4 = acc_lvl_3_val_3[acc_lvl_2_q]
                                                                          if tens_lvl_3_qos > tens_lvl_3_qos_stop
                                                                              tens_lvl_3_qos_stop = max(tens_lvl_3_qos_stop << 1, 1)
                                                                              Finch.resize_if_smaller!(tens_lvl_3_idx_8, tens_lvl_3_qos_stop)
                                                                              Finch.resize_if_smaller!(tens_lvl_4_val_8, tens_lvl_3_qos_stop)
                                                                              Finch.fill_range!(tens_lvl_4_val_8, 0.0, tens_lvl_3_qos, tens_lvl_3_qos_stop)
                                                                          end
                                                                          tens_lvl_4_val_8[tens_lvl_3_qos] = acc_lvl_3_val_4 + acc_lvl_3_val_4
                                                                          tens_lvl_3_idx_8[tens_lvl_3_qos] = acc_lvl_2_i
                                                                          tens_lvl_3_qos += 1
                                                                          tens_lvl_3_prev_pos = tens_lvl_q
                                                                          acc_lvl_2_q += 1
                                                                      else
                                                                          phase_stop_5 = min(phase_stop_3, acc_lvl_2_i)
                                                                          if acc_lvl_2_i == phase_stop_5
                                                                              acc_lvl_3_val_4 = acc_lvl_3_val_3[acc_lvl_2_q]
                                                                              if tens_lvl_3_qos > tens_lvl_3_qos_stop
                                                                                  tens_lvl_3_qos_stop = max(tens_lvl_3_qos_stop << 1, 1)
                                                                                  Finch.resize_if_smaller!(tens_lvl_3_idx_8, tens_lvl_3_qos_stop)
                                                                                  Finch.resize_if_smaller!(tens_lvl_4_val_8, tens_lvl_3_qos_stop)
                                                                                  Finch.fill_range!(tens_lvl_4_val_8, 0.0, tens_lvl_3_qos, tens_lvl_3_qos_stop)
                                                                              end
                                                                              tens_lvl_4_val_8[tens_lvl_3_qos] = acc_lvl_3_val_4 + acc_lvl_3_val_4
                                                                              tens_lvl_3_idx_8[tens_lvl_3_qos] = phase_stop_5
                                                                              tens_lvl_3_qos += 1
                                                                              tens_lvl_3_prev_pos = tens_lvl_q
                                                                              acc_lvl_2_q += 1
                                                                          end
                                                                          break
                                                                      end
                                                                  end
                                                              end
                                                              tens_lvl_3_ptr_8[tens_lvl_q + 1] += (tens_lvl_3_qos - tens_lvl_3_qos_fill) - 1
                                                              tens_lvl_3_qos_fill = tens_lvl_3_qos - 1
                                                          end
                                                      end
                                                      phase_start_6 = max(1, 1 + fld(acc_lvl_stop * tid_3, n_2))
                                                      if acc_lvl_stop >= phase_start_6
                                                          acc_lvl_stop + 1
                                                      end
                                                  end
                                              resize!(tens_lvl_3_ptr_7, acc_lvl_stop + 1)
                                              for p_6 = 1:acc_lvl_stop
                                                  tens_lvl_3_ptr_7[p_6 + 1] += tens_lvl_3_ptr_7[p_6]
                                              end
                                              qos_stop_6 = tens_lvl_3_ptr_7[acc_lvl_stop + 1] - 1
                                              resize!(tens_lvl_3_idx_7, qos_stop_6)
                                              resize!(tens_lvl_4_val_7, qos_stop_6)
                                              res_12
                                          end)
                                  nothing
                              end
                      end
                  result = ()
                  tm = collect(1:n)
                  gfm = ones(Int, n)
                  lfm = ones(Int, n)
                  Finch.coalesce_level!((CoalesceLevel)(Finch.CPU{:t}(n), (SparseListLevel){Int64}(ElementLevel{0.0, Float64, Int64}(tens_lvl_4_val_6), acc_lvl_2_stop, tens_lvl_3_ptr_6, tens_lvl_3_idx_6), (SparseListLevel){Int64}(ElementLevel{0.0, Float64, Int64}(tens_lvl_6_val), tens_lvl_5_stop, tens_lvl_5_ptr, tens_lvl_5_idx), tens_lvl_2.schedule), gfm, lfm, tm, 1, n, (SparseListLevel){Int64}(ElementLevel{0.0, Float64, Int64}(tens_lvl_6_val), tens_lvl_5_stop, tens_lvl_5_ptr, tens_lvl_5_idx))
                  result
              end)
  end)

g = :(function run(tens::Tensor{DenseLevel{Int64, ShardLevel{CPU{:t}, SparseListLevel{Int64, Finch.MultiChannelBuffer{Vector{Int64}}, Finch.MultiChannelBuffer{Vector{Int64}}, ElementLevel{0.0, Float64, Int64, Finch.MultiChannelBuffer{Vector{Float64}}}}, Vector{Int64}, Vector{Int64}, Vector{Int64}, Vector{Int64}, Finch.FinchStaticSchedule{:dynamic}}}}, acc::Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}, dev::CPU{:t})
      @inbounds @fastmath(begin
                  tens_lvl = tens.lvl
                  tens_lvl_2 = tens_lvl.lvl
                  tens_lvl_2_ptr = tens_lvl_2.ptr
                  tens_lvl_2_task = tens_lvl_2.task
                  tens_lvl_2_qos_fill = tens_lvl_2.used
                  tens_lvl_2_qos_stop = tens_lvl_2.alloc
                  n = tens_lvl_2.device.n
                  tens_lvl_3 = tens_lvl_2.lvl
                  tens_lvl_3_ptr = tens_lvl_3.ptr
                  tens_lvl_3_idx = tens_lvl_3.idx
                  tens_lvl_4 = tens_lvl_3.lvl
                  tens_lvl_4_val = tens_lvl_4.val
                  acc_lvl = acc.lvl
                  acc_lvl_stop = acc_lvl.shape
                  acc_lvl_2 = acc_lvl.lvl
                  acc_lvl_2_ptr = acc_lvl_2.ptr
                  acc_lvl_2_idx = acc_lvl_2.idx
                  acc_lvl_2_stop = acc_lvl_2.shape
                  acc_lvl_3 = acc_lvl_2.lvl
                  acc_lvl_3_val = acc_lvl_3.val
                  n_2 = dev.n
                  tens_lvl_4_val_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_4_val)
                  tens_lvl_3_ptr_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_3_ptr)
                  tens_lvl_3_idx_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_3_idx)
                  tens_lvl_2_qos_fill_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_2_qos_fill)
                  tens_lvl_2_qos_stop_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_2_qos_stop)
                  Threads.@threads :dynamic for tid = 1:n
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              alloced_pos = tens_lvl_2_qos_stop_2[tid]
                                              tens_lvl_4_val_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_4_val)
                                              tens_lvl_3_ptr_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_3_ptr)
                                              tens_lvl_3_idx_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_3_idx)
                                              tens_lvl_2_qos_fill_3 = (Finch).transfer((Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)()), tens_lvl_2_qos_fill)
                                              tens_lvl_2_qos_stop_3 = (Finch).transfer((Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)()), tens_lvl_2_qos_stop)
                                              resize!(tens_lvl_3_ptr_3, alloced_pos + 1)
                                              for p = 1:alloced_pos
                                                  tens_lvl_3_ptr_3[p + 1] += tens_lvl_3_ptr_3[p]
                                              end
                                              qos_stop = tens_lvl_3_ptr_3[alloced_pos + 1] - 1
                                              resize!(tens_lvl_3_idx_3, qos_stop)
                                              resize!(tens_lvl_4_val_3, qos_stop)
                                              tens_lvl_2_qos_fill_3[tid] = 0
                                              tens_lvl_2_qos_stop_3[tid] = alloced_pos
                                          end)
                                  nothing
                              end
                      end
                  tens_lvl_4_val = (Finch).transfer(tens_lvl_4_val, tens_lvl_4_val_2)
                  tens_lvl_3_ptr = (Finch).transfer(tens_lvl_3_ptr, tens_lvl_3_ptr_2)
                  tens_lvl_3_idx = (Finch).transfer(tens_lvl_3_idx, tens_lvl_3_idx_2)
                  tens_lvl_2_qos_fill = (Finch).transfer(tens_lvl_2_qos_fill, tens_lvl_2_qos_fill_2)
                  tens_lvl_2_qos_stop = (Finch).transfer(tens_lvl_2_qos_stop, tens_lvl_2_qos_stop_2)
                  Finch.resize_if_smaller!(tens_lvl_2_task, acc_lvl_stop)
                  Finch.resize_if_smaller!(tens_lvl_2_ptr, acc_lvl_stop)
                  Finch.fill_range!(tens_lvl_2_ptr, 0, 1, acc_lvl_stop)
                  acc_lvl_3_val_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), acc_lvl_3_val)
                  acc_lvl_2_ptr_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), acc_lvl_2_ptr)
                  acc_lvl_2_idx_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), acc_lvl_2_idx)
                  tens_lvl_4_val_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), tens_lvl_4_val)
                  tens_lvl_3_ptr_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), tens_lvl_3_ptr)
                  tens_lvl_3_idx_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), tens_lvl_3_idx)
                  tens_lvl_2_ptr_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), tens_lvl_2_ptr)
                  tens_lvl_2_task_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), tens_lvl_2_task)
                  tens_lvl_2_qos_fill_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), tens_lvl_2_qos_fill)
                  tens_lvl_2_qos_stop_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), tens_lvl_2_qos_stop)
                  Threads.@threads :dynamic for tid_2 = 1:n_2
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              acc_lvl_3_val_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_2), (Finch.SerialTask)()), acc_lvl_3_val_2)
                                              acc_lvl_2_ptr_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_2), (Finch.SerialTask)()), acc_lvl_2_ptr_2)
                                              acc_lvl_2_idx_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_2), (Finch.SerialTask)()), acc_lvl_2_idx_2)
                                              tens_lvl_2_qos_fill_5 = tens_lvl_2_qos_fill_4[tid_2]
                                              tens_lvl_2_qos_stop_5 = tens_lvl_2_qos_stop_4[tid_2]
                                              tens_lvl_4_val_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n_2), n_2), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_2), (Finch.SerialTask)())), tens_lvl_4_val_4)
                                              tens_lvl_3_ptr_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n_2), n_2), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_2), (Finch.SerialTask)())), tens_lvl_3_ptr_4)
                                              tens_lvl_3_idx_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n_2), n_2), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_2), (Finch.SerialTask)())), tens_lvl_3_idx_4)
                                              tens_lvl_3_qos_fill = tens_lvl_3_ptr_5[tens_lvl_2_qos_stop_5 + 1] - 1
                                              tens_lvl_3_qos_stop = tens_lvl_3_qos_fill
                                              tens_lvl_3_prev_pos = Finch.scansearch(tens_lvl_3_ptr_5, tens_lvl_3_qos_fill + 1, 1, tens_lvl_2_qos_stop_5) - 1
                                              for p_2 = tens_lvl_2_qos_stop_5:-1:1
                                                  tens_lvl_3_ptr_5[p_2 + 1] = tens_lvl_3_ptr_5[p_2 + 1] - tens_lvl_3_ptr_5[p_2]
                                              end
                                              tens_lvl_2_ptr_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_2), (Finch.SerialTask)()), tens_lvl_2_ptr_2)
                                              println(tens_lvl_2_ptr)
                                              tens_lvl_2_task_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_2), (Finch.SerialTask)()), tens_lvl_2_task_2)
                                              (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_2), (Finch.SerialTask)())
                                              (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_2), (Finch.SerialTask)())
                                              res_11 = begin
                                                      phase_start_2 = max(1, 1 + fld(acc_lvl_stop * (tid_2 + -1), n_2))
                                                      phase_stop_2 = min(acc_lvl_stop, fld(acc_lvl_stop * tid_2, n_2))
                                                      if phase_stop_2 >= phase_start_2
                                                          for j_7 = phase_start_2:phase_stop_2
                                                              acc_lvl_q = (1 - 1) * acc_lvl_stop + j_7
                                                              tens_lvl_q = (1 - 1) * acc_lvl_stop + j_7
                                                              qos = tens_lvl_2_ptr_3[tens_lvl_q]
                                                              if qos == 0
                                                                  qos = (tens_lvl_2_qos_fill_5 += 1)
                                                                  tens_lvl_2_task_3[tens_lvl_q] = tid_2
                                                                  tens_lvl_2_ptr_3[tens_lvl_q] = tens_lvl_2_qos_fill_5
                                                                  if tens_lvl_2_qos_fill_5 > tens_lvl_2_qos_stop_5
                                                                      tens_lvl_2_qos_stop_5 = max(tens_lvl_2_qos_stop_5 << 1, 1)
                                                                      Finch.resize_if_smaller!(tens_lvl_3_ptr_5, tens_lvl_2_qos_stop_5 + 1)
                                                                      Finch.fill_range!(tens_lvl_3_ptr_5, 0, tens_lvl_2_qos_fill_5 + 1, tens_lvl_2_qos_stop_5 + 1)
                                                                  end
                                                              end
                                                              acc_lvl_2_q = acc_lvl_2_ptr_3[acc_lvl_q]
                                                              acc_lvl_2_q_stop = acc_lvl_2_ptr_3[acc_lvl_q + 1]
                                                              if acc_lvl_2_q < acc_lvl_2_q_stop
                                                                  acc_lvl_2_i1 = acc_lvl_2_idx_3[acc_lvl_2_q_stop - 1]
                                                              else
                                                                  acc_lvl_2_i1 = 0
                                                              end
                                                              tens_lvl_3_qos = tens_lvl_3_qos_fill + 1
                                                              tens_lvl_3_prev_pos < qos || throw((Finch.FinchProtocolError)("SparseListLevels cannot be updated multiple times"))
                                                              phase_stop_3 = min(acc_lvl_2_i1, acc_lvl_2_stop)
                                                              if phase_stop_3 >= 1
                                                                  if acc_lvl_2_idx_3[acc_lvl_2_q] < 1
                                                                      acc_lvl_2_q = Finch.scansearch(acc_lvl_2_idx_3, 1, acc_lvl_2_q, acc_lvl_2_q_stop - 1)
                                                                  end
                                                                  while true
                                                                      acc_lvl_2_i = acc_lvl_2_idx_3[acc_lvl_2_q]
                                                                      if acc_lvl_2_i < phase_stop_3
                                                                          acc_lvl_3_val_4 = acc_lvl_3_val_3[acc_lvl_2_q]
                                                                          if tens_lvl_3_qos > tens_lvl_3_qos_stop
                                                                              tens_lvl_3_qos_stop = max(tens_lvl_3_qos_stop << 1, 1)
                                                                              Finch.resize_if_smaller!(tens_lvl_3_idx_5, tens_lvl_3_qos_stop)
                                                                              Finch.resize_if_smaller!(tens_lvl_4_val_5, tens_lvl_3_qos_stop)
                                                                              Finch.fill_range!(tens_lvl_4_val_5, 0.0, tens_lvl_3_qos, tens_lvl_3_qos_stop)
                                                                          end
                                                                          tens_lvl_4_val_5[tens_lvl_3_qos] = acc_lvl_3_val_4 + acc_lvl_3_val_4
                                                                          tens_lvl_3_idx_5[tens_lvl_3_qos] = acc_lvl_2_i
                                                                          tens_lvl_3_qos += 1
                                                                          tens_lvl_3_prev_pos = qos
                                                                          acc_lvl_2_q += 1
                                                                      else
                                                                          phase_stop_5 = min(phase_stop_3, acc_lvl_2_i)
                                                                          if acc_lvl_2_i == phase_stop_5
                                                                              acc_lvl_3_val_4 = acc_lvl_3_val_3[acc_lvl_2_q]
                                                                              if tens_lvl_3_qos > tens_lvl_3_qos_stop
                                                                                  tens_lvl_3_qos_stop = max(tens_lvl_3_qos_stop << 1, 1)
                                                                                  Finch.resize_if_smaller!(tens_lvl_3_idx_5, tens_lvl_3_qos_stop)
                                                                                  Finch.resize_if_smaller!(tens_lvl_4_val_5, tens_lvl_3_qos_stop)
                                                                                  Finch.fill_range!(tens_lvl_4_val_5, 0.0, tens_lvl_3_qos, tens_lvl_3_qos_stop)
                                                                              end
                                                                              tens_lvl_4_val_5[tens_lvl_3_qos] = acc_lvl_3_val_4 + acc_lvl_3_val_4
                                                                              tens_lvl_3_idx_5[tens_lvl_3_qos] = phase_stop_5
                                                                              tens_lvl_3_qos += 1
                                                                              tens_lvl_3_prev_pos = qos
                                                                              acc_lvl_2_q += 1
                                                                          end
                                                                          break
                                                                      end
                                                                  end
                                                              end
                                                              tens_lvl_3_ptr_5[qos + 1] += (tens_lvl_3_qos - tens_lvl_3_qos_fill) - 1
                                                              tens_lvl_3_qos_fill = tens_lvl_3_qos - 1
                                                          end
                                                      end
                                                      phase_start_6 = max(1, 1 + fld(acc_lvl_stop * tid_2, n_2))
                                                      if acc_lvl_stop >= phase_start_6
                                                          acc_lvl_stop + 1
                                                      end
                                                  end
                                              resize!(tens_lvl_3_ptr_5, tens_lvl_2_qos_stop_5 + 1)
                                              for p_3 = 1:tens_lvl_2_qos_stop_5
                                                  tens_lvl_3_ptr_5[p_3 + 1] += tens_lvl_3_ptr_5[p_3]
                                              end
                                              qos_stop_3 = tens_lvl_3_ptr_5[tens_lvl_2_qos_stop_5 + 1] - 1
                                              resize!(tens_lvl_3_idx_5, qos_stop_3)
                                              resize!(tens_lvl_4_val_5, qos_stop_3)
                                              res_11
                                          end)
                                  nothing
                              end
                      end
                  ()
              end)
  end)

eval(f)
out = run(tens, acc, dev)
display(tens)