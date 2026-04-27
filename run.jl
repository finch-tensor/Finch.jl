using Finch

dev = cpu(:t, 2)
dev2 = cpu(:q, 2)
tens = Tensor(Dense(Shard(dev, Coalesce(dev2, SparseList(Element(0.0))))))
mtx = Tensor(Dense(SparseList(Element(0))), [ 0  0  0  0  0  0  0  0  0  0; 0  0  0  0  0  0  0  0  0  0; 0  0  0  0  0  0  0  1  0  0; 0  0  0  1  1  0  0  0  0  0; 0  0  0  0  1  0  0  0  0  0; 0  0  1  0  0  0  0  0  0  0; 0  0  0  0  0  0  0  0  0  1; 0  0  0  0  0  0  0  0  0  0; 0  0  0  0  0  0  0  0  0  0; 0  0  0  0  0  0  0  0  0  0])


code = :(function run(tens::Tensor{DenseLevel{Int64, ShardLevel{CPU{:t}, CoalesceLevel{CPU{:q}, SparseListLevel{Int64, Finch.MultiChannelBuffer{Finch.MultiChannelBuffer{Vector{Int64}}}, Finch.MultiChannelBuffer{Finch.MultiChannelBuffer{Vector{Int64}}}, ElementLevel{0.0, Float64, Int64, Finch.MultiChannelBuffer{Finch.MultiChannelBuffer{Vector{Float64}}}}}, SparseListLevel{Int64, Finch.MultiChannelBuffer{Vector{Int64}}, Finch.MultiChannelBuffer{Vector{Int64}}, ElementLevel{0.0, Float64, Int64, Finch.MultiChannelBuffer{Vector{Float64}}}}, Finch.FinchStaticSchedule{:dynamic}}, Vector{Int64}, Vector{Int64}, Vector{Int64}, Vector{Int64}, Finch.FinchStaticSchedule{:dynamic}}}}, dev::CPU{:t}, dev2::CPU{:q}, mtx::Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0, Int64, Int64, Vector{Int64}}}}})
      @inbounds @fastmath(begin
                  tens_lvl = tens.lvl
                  tens_lvl_2 = tens_lvl.lvl
                  tens_lvl_2_ptr = tens_lvl_2.ptr
                  tens_lvl_2_task = tens_lvl_2.task
                  tens_lvl_2_qos_fill = tens_lvl_2.used
                  tens_lvl_2_qos_stop = tens_lvl_2.alloc
                  n = tens_lvl_2.device.n
                  tens_lvl_3 = tens_lvl_2.lvl
                  n_2 = tens_lvl_3.device.n
                  tens_lvl_4 = tens_lvl_3.lvl
                  tens_lvl_4_ptr = tens_lvl_4.ptr
                  tens_lvl_4_idx = tens_lvl_4.idx
                  tens_lvl_5 = tens_lvl_4.lvl
                  tens_lvl_5_val = tens_lvl_5.val
                  tens_lvl_6 = tens_lvl_3.coalescent
                  tens_lvl_6_ptr = tens_lvl_6.ptr
                  tens_lvl_6_idx = tens_lvl_6.idx
                  tens_lvl_7 = tens_lvl_6.lvl
                  tens_lvl_7_val = tens_lvl_7.val
                  n_3 = dev.n
                  n_4 = dev2.n
                  mtx_lvl = mtx.lvl
                  mtx_lvl_stop = mtx_lvl.shape
                  mtx_lvl_2 = mtx_lvl.lvl
                  mtx_lvl_2_ptr = mtx_lvl_2.ptr
                  mtx_lvl_2_idx = mtx_lvl_2.idx
                  mtx_lvl_2_stop = mtx_lvl_2.shape
                  mtx_lvl_3 = mtx_lvl_2.lvl
                  mtx_lvl_3_val = mtx_lvl_3.val
                  tens_lvl_5_val_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_5_val)
                  tens_lvl_4_ptr_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_4_ptr)
                  tens_lvl_4_idx_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_4_idx)
                  tens_lvl_2_qos_fill_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_2_qos_fill)
                  tens_lvl_2_qos_stop_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_2_qos_stop)
                  Threads.@threads :dynamic for tid = 1:n
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              tens_lvl_7_val_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_7_val)
                                              tens_lvl_6_ptr_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_6_ptr)
                                              tens_lvl_6_idx_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_6_idx)
                                              tens_lvl_5_val_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_5_val)
                                              tens_lvl_4_ptr_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_4_ptr)
                                              tens_lvl_4_idx_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_4_idx)
                                              tens_lvl_7_val_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_7_val)
                                              tens_lvl_6_ptr_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_6_ptr)
                                              tens_lvl_6_idx_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_6_idx)
                                              tens_lvl_2_qos_fill_3 = (Finch).transfer((Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)()), tens_lvl_2_qos_fill)
                                              tens_lvl_2_qos_stop_3 = (Finch).transfer((Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)()), tens_lvl_2_qos_stop)
                                              tens_lvl_5_val_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_2)), tens_lvl_5_val_3)
                                              tens_lvl_4_ptr_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_2)), tens_lvl_4_ptr_3)
                                              tens_lvl_4_idx_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_2)), tens_lvl_4_idx_3)
                                              Threads.@threads :dynamic for tid_2 = 1:n_2
                                                      Finch.@barrier begin
                                                              @inbounds @fastmath(begin
                                                                          tens_lvl_5_val_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_2), n_2), (Finch.CPUThread)(tid_2, Finch.CPU{:q}(n_2), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)()))), tens_lvl_5_val_3)
                                                                          tens_lvl_4_ptr_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_2), n_2), (Finch.CPUThread)(tid_2, Finch.CPU{:q}(n_2), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)()))), tens_lvl_4_ptr_3)
                                                                          tens_lvl_4_idx_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_2), n_2), (Finch.CPUThread)(tid_2, Finch.CPU{:q}(n_2), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)()))), tens_lvl_4_idx_3)
                                                                          resize!(tens_lvl_4_ptr_5, 0 + 1)
                                                                          for p = 1:0
                                                                              tens_lvl_4_ptr_5[p + 1] += tens_lvl_4_ptr_5[p]
                                                                          end
                                                                          qos_stop = tens_lvl_4_ptr_5[0 + 1] - 1
                                                                          resize!(tens_lvl_4_idx_5, qos_stop)
                                                                          resize!(tens_lvl_5_val_5, qos_stop)
                                                                          Finch.VirtualSparseListLevel(:tens_lvl_4, Finch.VirtualElementLevel(:tens_lvl_5, 0.0, Float64, Int64, :tens_lvl_5_val_5), Int64, :tens_lvl_4_ptr_5, :tens_lvl_4_idx_5, value(mtx_lvl_stop, Int64), :tens_lvl_4_qos_fill, :tens_lvl_4_qos_stop, :tens_lvl_4_prev_pos)
                                                                      end)
                                                              nothing
                                                          end
                                                  end
                                              resize!(tens_lvl_6_ptr_3, 0 + 1)
                                              for p_2 = 1:0
                                                  tens_lvl_6_ptr_3[p_2 + 1] += tens_lvl_6_ptr_3[p_2]
                                              end
                                              qos_stop_2 = tens_lvl_6_ptr_3[0 + 1] - 1
                                              resize!(tens_lvl_6_idx_3, qos_stop_2)
                                              resize!(tens_lvl_7_val_3, qos_stop_2)
                                              tens_lvl_5_val_3 = (Finch).transfer(tens_lvl_5_val_3, tens_lvl_5_val_4)
                                              tens_lvl_4_ptr_3 = (Finch).transfer(tens_lvl_4_ptr_3, tens_lvl_4_ptr_4)
                                              tens_lvl_4_idx_3 = (Finch).transfer(tens_lvl_4_idx_3, tens_lvl_4_idx_4)
                                              tm = collect(1:n_2)
                                              gfm = ones(Int, n_2)
                                              lfm = ones(Int, n_2)
                                              Finch.coalesce_level!((SparseListLevel){Int64}(ElementLevel{0.0, Float64, Int64}(tens_lvl_5_val_3), mtx_lvl_stop, tens_lvl_4_ptr_3, tens_lvl_4_idx_3), gfm, lfm, tm, 0, n_2, tens_lvl_3.coalescent, tid)
                                              res_2 = begin
                                                      tens_lvl_2_qos_fill_3[tid] = 0
                                                      tens_lvl_2_qos_stop_3[tid] = 0
                                                  end
                                              tens_lvl_7_val = (Finch).transfer(tens_lvl_7_val, tens_lvl_7_val_2)
                                              tens_lvl_6_ptr = (Finch).transfer(tens_lvl_6_ptr, tens_lvl_6_ptr_2)
                                              tens_lvl_6_idx = (Finch).transfer(tens_lvl_6_idx, tens_lvl_6_idx_2)
                                              res_2
                                          end)
                                  nothing
                              end
                      end
                  tens_lvl_5_val = (Finch).transfer(tens_lvl_5_val, tens_lvl_5_val_2)
                  tens_lvl_4_ptr = (Finch).transfer(tens_lvl_4_ptr, tens_lvl_4_ptr_2)
                  tens_lvl_4_idx = (Finch).transfer(tens_lvl_4_idx, tens_lvl_4_idx_2)
                  tens_lvl_2_qos_fill = (Finch).transfer(tens_lvl_2_qos_fill, tens_lvl_2_qos_fill_2)
                  tens_lvl_2_qos_stop = (Finch).transfer(tens_lvl_2_qos_stop, tens_lvl_2_qos_stop_2)
                  Finch.resize_if_smaller!(tens_lvl_2_task, mtx_lvl_stop)
                  Finch.resize_if_smaller!(tens_lvl_2_ptr, mtx_lvl_stop)
                  Finch.fill_range!(tens_lvl_2_ptr, 0, 1, mtx_lvl_stop)
                  tens_lvl_5_val_6 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), tens_lvl_5_val)
                  tens_lvl_4_ptr_6 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), tens_lvl_4_ptr)
                  tens_lvl_4_idx_6 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), tens_lvl_4_idx)
                  tens_lvl_2_ptr_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), tens_lvl_2_ptr)
                  tens_lvl_2_task_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), tens_lvl_2_task)
                  tens_lvl_2_qos_fill_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), tens_lvl_2_qos_fill)
                  tens_lvl_2_qos_stop_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), tens_lvl_2_qos_stop)
                  mtx_lvl_3_val_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), mtx_lvl_3_val)
                  mtx_lvl_2_ptr_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), mtx_lvl_2_ptr)
                  mtx_lvl_2_idx_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), mtx_lvl_2_idx)
                  Threads.@threads :dynamic for tid_3 = 1:n_3
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              tens_lvl_2_ptr_3 = (Finch).transfer((Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), tens_lvl_2_ptr_2)
                                              tens_lvl_2_task_3 = (Finch).transfer((Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), tens_lvl_2_task_2)
                                              tens_lvl_2_qos_fill_6 = (Finch).transfer((Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), tens_lvl_2_qos_fill_4)
                                              tens_lvl_2_qos_stop_6 = (Finch).transfer((Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), tens_lvl_2_qos_stop_4)
                                              tens_lvl_2_qos_fill_5 = tens_lvl_2_qos_fill_6[tid_3]
                                              tens_lvl_2_qos_stop_5 = tens_lvl_2_qos_stop_6[tid_3]
                                              tens_lvl_7_val_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_7_val)
                                              tens_lvl_6_ptr_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_6_ptr)
                                              tens_lvl_6_idx_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_6_idx)
                                              tens_lvl_5_val_7 = (Finch).transfer((Finch.MemoryChannel)(tid_3, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), tens_lvl_5_val_6)
                                              tens_lvl_4_ptr_7 = (Finch).transfer((Finch.MemoryChannel)(tid_3, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), tens_lvl_4_ptr_6)
                                              tens_lvl_4_idx_7 = (Finch).transfer((Finch.MemoryChannel)(tid_3, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), tens_lvl_4_idx_6)
                                              tens_lvl_7_val_5 = (Finch).transfer((Finch.MemoryChannel)(tid_3, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), tens_lvl_7_val)
                                              tens_lvl_6_ptr_5 = (Finch).transfer((Finch.MemoryChannel)(tid_3, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), tens_lvl_6_ptr)
                                              tens_lvl_6_idx_5 = (Finch).transfer((Finch.MemoryChannel)(tid_3, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), tens_lvl_6_idx)
                                              mtx_lvl_3_val_3 = (Finch).transfer((Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), mtx_lvl_3_val_2)
                                              mtx_lvl_2_ptr_3 = (Finch).transfer((Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), mtx_lvl_2_ptr_2)
                                              mtx_lvl_2_idx_3 = (Finch).transfer((Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), mtx_lvl_2_idx_2)
                                              res_18 = begin
                                                      phase_start_2 = max(1, 1 + fld(mtx_lvl_stop * (tid_3 + -1), n_3))
                                                      phase_stop_2 = min(mtx_lvl_stop, fld(mtx_lvl_stop * tid_3, n_3))
                                                      if phase_stop_2 >= phase_start_2
                                                          for j_6 = phase_start_2:phase_stop_2
                                                              tens_lvl_q = (1 - 1) * mtx_lvl_stop + j_6
                                                              mtx_lvl_q = (1 - 1) * mtx_lvl_stop + j_6
                                                              qos = tens_lvl_2_ptr_3[tens_lvl_q]
                                                              if qos == 0
                                                                  qos = (tens_lvl_2_qos_fill_5 += 1)
                                                                  tens_lvl_2_task_3[tens_lvl_q] = tid_3
                                                                  tens_lvl_2_ptr_3[tens_lvl_q] = tens_lvl_2_qos_fill_5
                                                                  if tens_lvl_2_qos_fill_5 > tens_lvl_2_qos_stop_5
                                                                      tens_lvl_2_qos_stop_5 = max(tens_lvl_2_qos_stop_5 << 1, 1)
                                                                      tens_lvl_5_val_8 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_2)), tens_lvl_5_val_7)
                                                                      tens_lvl_4_ptr_8 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_2)), tens_lvl_4_ptr_7)
                                                                      tens_lvl_4_idx_8 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_2)), tens_lvl_4_idx_7)
                                                                      Threads.@threads :dynamic for tid_4 = 1:n_2
                                                                              Finch.@barrier begin
                                                                                      @inbounds @fastmath(begin
                                                                                                  tens_lvl_5_val_9 = (Finch).transfer((Finch.MemoryChannel)(tid_4, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_2), n_2), (Finch.CPUThread)(tid_4, Finch.CPU{:q}(n_2), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))), tens_lvl_5_val_7)
                                                                                                  tens_lvl_4_ptr_9 = (Finch).transfer((Finch.MemoryChannel)(tid_4, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_2), n_2), (Finch.CPUThread)(tid_4, Finch.CPU{:q}(n_2), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))), tens_lvl_4_ptr_7)
                                                                                                  tens_lvl_4_idx_9 = (Finch).transfer((Finch.MemoryChannel)(tid_4, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_2), n_2), (Finch.CPUThread)(tid_4, Finch.CPU{:q}(n_2), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))), tens_lvl_4_idx_7)
                                                                                                  pos_stop = -1 + tens_lvl_2_qos_fill_5
                                                                                                  for p_3 = pos_stop:-1:1
                                                                                                      tens_lvl_4_ptr_9[p_3 + 1] = tens_lvl_4_ptr_9[p_3 + 1] - tens_lvl_4_ptr_9[p_3]
                                                                                                  end
                                                                                                  Finch.resize_if_smaller!(tens_lvl_4_ptr_9, tens_lvl_2_qos_stop_5 + 1)
                                                                                                  Finch.fill_range!(tens_lvl_4_ptr_9, 0, tens_lvl_2_qos_fill_5 + 1, tens_lvl_2_qos_stop_5 + 1)
                                                                                                  resize!(tens_lvl_4_ptr_9, tens_lvl_2_qos_stop_5 + 1)
                                                                                                  for p_4 = 1:tens_lvl_2_qos_stop_5
                                                                                                      tens_lvl_4_ptr_9[p_4 + 1] += tens_lvl_4_ptr_9[p_4]
                                                                                                  end
                                                                                                  qos_stop_4 = tens_lvl_4_ptr_9[tens_lvl_2_qos_stop_5 + 1] - 1
                                                                                                  resize!(tens_lvl_4_idx_9, qos_stop_4)
                                                                                                  resize!(tens_lvl_5_val_9, qos_stop_4)
                                                                                                  Finch.VirtualSparseListLevel(:tens_lvl_4, Finch.VirtualElementLevel(:tens_lvl_5, 0.0, Float64, Int64, :tens_lvl_5_val_9), Int64, :tens_lvl_4_ptr_9, :tens_lvl_4_idx_9, value(mtx_lvl_stop, Int64), :tens_lvl_4_qos_fill, :tens_lvl_4_qos_stop, :tens_lvl_4_prev_pos)
                                                                                              end)
                                                                                      nothing
                                                                                  end
                                                                          end
                                                                      pos_stop_2 = -1 + tens_lvl_2_qos_fill_5
                                                                      for p_5 = pos_stop_2:-1:1
                                                                          tens_lvl_6_ptr_5[p_5 + 1] = tens_lvl_6_ptr_5[p_5 + 1] - tens_lvl_6_ptr_5[p_5]
                                                                      end
                                                                      Finch.resize_if_smaller!(tens_lvl_6_ptr_5, tens_lvl_2_qos_stop_5 + 1)
                                                                      Finch.fill_range!(tens_lvl_6_ptr_5, 0, tens_lvl_2_qos_fill_5 + 1, tens_lvl_2_qos_stop_5 + 1)
                                                                      resize!(tens_lvl_6_ptr_5, tens_lvl_2_qos_stop_5 + 1)
                                                                      for p_6 = 1:tens_lvl_2_qos_stop_5
                                                                          tens_lvl_6_ptr_5[p_6 + 1] += tens_lvl_6_ptr_5[p_6]
                                                                      end
                                                                      qos_stop_6 = tens_lvl_6_ptr_5[tens_lvl_2_qos_stop_5 + 1] - 1
                                                                      resize!(tens_lvl_6_idx_5, qos_stop_6)
                                                                      resize!(tens_lvl_7_val_5, qos_stop_6)
                                                                      tens_lvl_5_val_7 = (Finch).transfer(tens_lvl_5_val_7, tens_lvl_5_val_8)
                                                                      tens_lvl_4_ptr_7 = (Finch).transfer(tens_lvl_4_ptr_7, tens_lvl_4_ptr_8)
                                                                      tens_lvl_4_idx_7 = (Finch).transfer(tens_lvl_4_idx_7, tens_lvl_4_idx_8)
                                                                  end
                                                              else
                                                                  @assert tens_lvl_2_task_3[tens_lvl_q] == tid_3 "Task mismatch in ShardLevel"
                                                              end
                                                              tens_lvl_5_val_10 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), tens_lvl_5_val_7)
                                                              tens_lvl_4_ptr_10 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), tens_lvl_4_ptr_7)
                                                              tens_lvl_4_idx_10 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), tens_lvl_4_idx_7)
                                                              tens_lvl_2_ptr_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), tens_lvl_2_ptr_3)
                                                              tens_lvl_2_task_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), tens_lvl_2_task_3)
                                                              tens_lvl_2_qos_fill_7 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), tens_lvl_2_qos_fill_6)
                                                              tens_lvl_2_qos_stop_7 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), tens_lvl_2_qos_stop_6)
                                                              mtx_lvl_3_val_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), mtx_lvl_3_val_3)
                                                              mtx_lvl_2_ptr_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), mtx_lvl_2_ptr_3)
                                                              mtx_lvl_2_idx_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), mtx_lvl_2_idx_3)
                                                              Threads.@threads :dynamic for tid_5 = 1:n_4
                                                                      Finch.@barrier begin
                                                                              @inbounds @fastmath(begin
                                                                                          tens_lvl_5_val_11 = (Finch).transfer((Finch.MemoryChannel)(tid_5, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_4), n_4), (Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))), tens_lvl_5_val_10)
                                                                                          tens_lvl_4_ptr_11 = (Finch).transfer((Finch.MemoryChannel)(tid_5, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_4), n_4), (Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))), tens_lvl_4_ptr_10)
                                                                                          tens_lvl_4_idx_11 = (Finch).transfer((Finch.MemoryChannel)(tid_5, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_4), n_4), (Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))), tens_lvl_4_idx_10)
                                                                                          tens_lvl_4_qos_fill = tens_lvl_4_ptr_11[tens_lvl_2_qos_stop_5 + 1] - 1
                                                                                          tens_lvl_4_qos_stop = tens_lvl_4_qos_fill
                                                                                          tens_lvl_4_prev_pos = Finch.scansearch(tens_lvl_4_ptr_11, tens_lvl_4_qos_fill + 1, 1, tens_lvl_2_qos_stop_5) - 1
                                                                                          for p_7 = tens_lvl_2_qos_stop_5:-1:1
                                                                                              tens_lvl_4_ptr_11[p_7 + 1] = tens_lvl_4_ptr_11[p_7 + 1] - tens_lvl_4_ptr_11[p_7]
                                                                                          end
                                                                                          tens_lvl_5_val_12 = (Finch).transfer((Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), tens_lvl_5_val_10)
                                                                                          tens_lvl_4_ptr_12 = (Finch).transfer((Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), tens_lvl_4_ptr_10)
                                                                                          tens_lvl_4_idx_12 = (Finch).transfer((Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), tens_lvl_4_idx_10)
                                                                                          (Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))
                                                                                          (Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))
                                                                                          (Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))
                                                                                          (Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))
                                                                                          mtx_lvl_3_val_5 = (Finch).transfer((Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), mtx_lvl_3_val_4)
                                                                                          mtx_lvl_2_ptr_5 = (Finch).transfer((Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), mtx_lvl_2_ptr_4)
                                                                                          mtx_lvl_2_idx_5 = (Finch).transfer((Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_3, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), mtx_lvl_2_idx_4)
                                                                                          res_14 = begin
                                                                                                  tens_lvl_4_qos = tens_lvl_4_qos_fill + 1
                                                                                                  tens_lvl_4_prev_pos < qos || throw((Finch.FinchProtocolError)("SparseListLevels cannot be updated multiple times"))
                                                                                                  res_12 = begin
                                                                                                          phase_start_4 = max(1, 1 + fld(mtx_lvl_stop * (-1 + tid_5), n_4))
                                                                                                          phase_stop_4 = min(mtx_lvl_stop, fld(mtx_lvl_stop * tid_5, n_4))
                                                                                                          if phase_stop_4 >= phase_start_4
                                                                                                              for i_6 = phase_start_4:phase_stop_4
                                                                                                                  if tens_lvl_4_qos > tens_lvl_4_qos_stop
                                                                                                                      tens_lvl_4_qos_stop = max(tens_lvl_4_qos_stop << 1, 1)
                                                                                                                      Finch.resize_if_smaller!(tens_lvl_4_idx_12, tens_lvl_4_qos_stop)
                                                                                                                      Finch.resize_if_smaller!(tens_lvl_5_val_12, tens_lvl_4_qos_stop)
                                                                                                                      Finch.fill_range!(tens_lvl_5_val_12, 0.0, tens_lvl_4_qos, tens_lvl_4_qos_stop)
                                                                                                                  end
                                                                                                                  tens_lvl_4dirty = false
                                                                                                                  mtx_lvl_q_2 = (1 - 1) * mtx_lvl_stop + i_6
                                                                                                                  mtx_lvl_2_q = mtx_lvl_2_ptr_5[mtx_lvl_q]
                                                                                                                  mtx_lvl_2_q_stop = mtx_lvl_2_ptr_5[mtx_lvl_q + 1]
                                                                                                                  if mtx_lvl_2_q < mtx_lvl_2_q_stop
                                                                                                                      mtx_lvl_2_i1 = mtx_lvl_2_idx_5[mtx_lvl_2_q_stop - 1]
                                                                                                                  else
                                                                                                                      mtx_lvl_2_i1 = 0
                                                                                                                  end
                                                                                                                  mtx_lvl_2_q_2 = mtx_lvl_2_ptr_5[mtx_lvl_q_2]
                                                                                                                  mtx_lvl_2_q_stop_2 = mtx_lvl_2_ptr_5[mtx_lvl_q_2 + 1]
                                                                                                                  if mtx_lvl_2_q_2 < mtx_lvl_2_q_stop_2
                                                                                                                      mtx_lvl_2_i1_2 = mtx_lvl_2_idx_5[mtx_lvl_2_q_stop_2 - 1]
                                                                                                                  else
                                                                                                                      mtx_lvl_2_i1_2 = 0
                                                                                                                  end
                                                                                                                  phase_stop_5 = min(mtx_lvl_2_stop, mtx_lvl_2_i1, mtx_lvl_2_i1_2)
                                                                                                                  if phase_stop_5 >= 1
                                                                                                                      k = 1
                                                                                                                      if mtx_lvl_2_idx_5[mtx_lvl_2_q] < 1
                                                                                                                          mtx_lvl_2_q = Finch.scansearch(mtx_lvl_2_idx_5, 1, mtx_lvl_2_q, mtx_lvl_2_q_stop - 1)
                                                                                                                      end
                                                                                                                      if mtx_lvl_2_idx_5[mtx_lvl_2_q_2] < 1
                                                                                                                          mtx_lvl_2_q_2 = Finch.scansearch(mtx_lvl_2_idx_5, 1, mtx_lvl_2_q_2, mtx_lvl_2_q_stop_2 - 1)
                                                                                                                      end
                                                                                                                      while k <= phase_stop_5
                                                                                                                          mtx_lvl_2_i = mtx_lvl_2_idx_5[mtx_lvl_2_q]
                                                                                                                          mtx_lvl_2_i_2 = mtx_lvl_2_idx_5[mtx_lvl_2_q_2]
                                                                                                                          phase_stop_6 = min(mtx_lvl_2_i_2, phase_stop_5, mtx_lvl_2_i)
                                                                                                                          if mtx_lvl_2_i == phase_stop_6 && mtx_lvl_2_i_2 == phase_stop_6
                                                                                                                              mtx_lvl_3_val_6 = mtx_lvl_3_val_5[mtx_lvl_2_q]
                                                                                                                              mtx_lvl_3_val_7 = mtx_lvl_3_val_5[mtx_lvl_2_q_2]
                                                                                                                              tens_lvl_4dirty = true
                                                                                                                              tens_lvl_5_val_12[tens_lvl_4_qos] = mtx_lvl_3_val_7 * mtx_lvl_3_val_6 + tens_lvl_5_val_12[tens_lvl_4_qos]
                                                                                                                              mtx_lvl_2_q += 1
                                                                                                                              mtx_lvl_2_q_2 += 1
                                                                                                                          elseif mtx_lvl_2_i_2 == phase_stop_6
                                                                                                                              mtx_lvl_2_q_2 += 1
                                                                                                                          elseif mtx_lvl_2_i == phase_stop_6
                                                                                                                              mtx_lvl_2_q += 1
                                                                                                                          end
                                                                                                                          k = phase_stop_6 + 1
                                                                                                                      end
                                                                                                                  end
                                                                                                                  if tens_lvl_4dirty
                                                                                                                      tens_lvl_4_idx_12[tens_lvl_4_qos] = i_6
                                                                                                                      tens_lvl_4_qos += 1
                                                                                                                  end
                                                                                                              end
                                                                                                          end
                                                                                                          phase_start_10 = max(1, 1 + fld(mtx_lvl_stop * tid_5, n_4))
                                                                                                          if mtx_lvl_stop >= phase_start_10
                                                                                                              mtx_lvl_stop + 1
                                                                                                          end
                                                                                                      end
                                                                                                  tens_lvl_4_ptr_12[qos + 1] += (tens_lvl_4_qos - tens_lvl_4_qos_fill) - 1
                                                                                                  res_12
                                                                                              end
                                                                                          println(length(tens_lvl_4_ptr_11))
                                                                                          resize!(tens_lvl_4_ptr_11, tens_lvl_2_qos_stop_5 + 1)
                                                                                          for p_8 = 1:tens_lvl_2_qos_stop_5
                                                                                              tens_lvl_4_ptr_11[p_8 + 1] += tens_lvl_4_ptr_11[p_8]
                                                                                          end
                                                                                          qos_stop_8 = tens_lvl_4_ptr_11[tens_lvl_2_qos_stop_5 + 1] - 1
                                                                                          resize!(tens_lvl_4_idx_11, qos_stop_8)
                                                                                          resize!(tens_lvl_5_val_11, qos_stop_8)
                                                                                          res_14
                                                                                      end)
                                                                              nothing
                                                                          end
                                                                  end
                                                              tens_lvl_5_val_7 = (Finch).transfer(tens_lvl_5_val_7, tens_lvl_5_val_10)
                                                              tens_lvl_4_ptr_7 = (Finch).transfer(tens_lvl_4_ptr_7, tens_lvl_4_ptr_10)
                                                              tens_lvl_4_idx_7 = (Finch).transfer(tens_lvl_4_idx_7, tens_lvl_4_idx_10)
                                                              tens_lvl_2_ptr_3 = (Finch).transfer(tens_lvl_2_ptr_3, tens_lvl_2_ptr_4)
                                                              tens_lvl_2_task_3 = (Finch).transfer(tens_lvl_2_task_3, tens_lvl_2_task_4)
                                                              tens_lvl_2_qos_fill_6 = (Finch).transfer(tens_lvl_2_qos_fill_6, tens_lvl_2_qos_fill_7)
                                                              tens_lvl_2_qos_stop_6 = (Finch).transfer(tens_lvl_2_qos_stop_6, tens_lvl_2_qos_stop_7)
                                                          end
                                                      end
                                                      phase_start_11 = max(1, 1 + fld(mtx_lvl_stop * tid_3, n_3))
                                                      if mtx_lvl_stop >= phase_start_11
                                                          mtx_lvl_stop + 1
                                                      end
                                                  end
                                              tens_lvl_2_qos_fill_6[tid_3] = tens_lvl_2_qos_fill_5
                                              tens_lvl_2_qos_stop_6[tid_3] = tens_lvl_2_qos_stop_5
                                              tens_lvl_7_val = (Finch).transfer(tens_lvl_7_val, tens_lvl_7_val_4)
                                              tens_lvl_6_ptr = (Finch).transfer(tens_lvl_6_ptr, tens_lvl_6_ptr_4)
                                              tens_lvl_6_idx = (Finch).transfer(tens_lvl_6_idx, tens_lvl_6_idx_4)
                                              tm_2 = collect(1:n_2)
                                              gfm_2 = ones(Int, n_2)
                                              lfm_2 = ones(Int, n_2)
                                              Finch.coalesce_level!((SparseListLevel){Int64}(ElementLevel{0.0, Float64, Int64}(tens_lvl_5_val_7), mtx_lvl_stop, tens_lvl_4_ptr_7, tens_lvl_4_idx_7), gfm_2, lfm_2, tm_2, tens_lvl_2_qos_stop_5, n_2, tens_lvl_3.coalescent, tid_3)
                                              res_18
                                          end)
                                  nothing
                              end
                      end
                  ()
              end)
  end)


eval(code)

run(tens, dev, dev2, mtx)