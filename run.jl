using Finch

dev = cpu(:t, 2)
dev2 = cpu(:q, 2)

tens = Tensor(Dense(Shard(dev, SparseList(Element(0.0)))))
w = Tensor(Coalesce(dev2, SparseByteMap(Element(0.0))))
mtx = Tensor(Dense(SparseList(Element(0))), [ 0  0  0  0  0  0  0  0  0  0; 0  0  0  0  0  0  0  0  0  0; 0  0  0  0  0  0  0  1  0  0; 0  0  0  1  1  0  0  0  0  0; 0  0  0  0  1  0  0  0  0  0; 0  0  1  0  0  0  0  0  0  0; 0  0  0  0  0  0  0  0  0  1; 0  0  0  0  0  0  0  0  0  0; 0  0  0  0  0  0  0  0  0  0; 0  0  0  0  0  0  0  0  0  0])


code = :(function run(tens::Tensor{DenseLevel{Int64, ShardLevel{CPU{:t}, SparseListLevel{Int64, Finch.MultiChannelBuffer{Vector{Int64}}, Finch.MultiChannelBuffer{Vector{Int64}}, ElementLevel{0.0, Float64, Int64, Finch.MultiChannelBuffer{Vector{Float64}}}}, Vector{Int64}, Vector{Int64}, Vector{Int64}, Vector{Int64}, Finch.FinchStaticSchedule{:dynamic}}}}, w::Tensor{CoalesceLevel{CPU{:q}, SparseByteMapLevel{Int64, Finch.MultiChannelBuffer{Vector{Int64}}, Finch.MultiChannelBuffer{Vector{Bool}}, Finch.MultiChannelBuffer{Vector{Tuple{Int64, Int64}}}, ElementLevel{0.0, Float64, Int64, Finch.MultiChannelBuffer{Vector{Float64}}}}, SparseByteMapLevel{Int64, Vector{Int64}, Vector{Bool}, Vector{Tuple{Int64, Int64}}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}, Finch.FinchStaticSchedule{:dynamic}}}, dev::CPU{:t}, dev2::CPU{:q}, mtx::Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0, Int64, Int64, Vector{Int64}}}}})
      @inbounds @fastmath(begin
                  tens_lvl = tens.lvl
                  tens_lvl_stop = tens_lvl.shape
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
                  w_lvl = w.lvl
                  n_2 = w_lvl.device.n
                  w_lvl_2 = w_lvl.lvl
                  w_lvl_2_ptr = w_lvl_2.ptr
                  w_lvl_2_tbl = w_lvl_2.tbl
                  w_lvl_2_srt = w_lvl_2.srt
                  w_lvl_2_qos_stop = (w_lvl_2_qos_fill = length(w_lvl_2.srt))
                  w_lvl_3 = w_lvl_2.lvl
                  w_lvl_3_val = w_lvl_3.val
                  w_lvl_4 = w_lvl.coalescent
                  w_lvl_4_ptr = w_lvl_4.ptr
                  w_lvl_4_tbl = w_lvl_4.tbl
                  w_lvl_4_srt = w_lvl_4.srt
                  w_lvl_4_qos_fill = length(w_lvl_4.srt)
                  w_lvl_5 = w_lvl_4.lvl
                  w_lvl_5_val = w_lvl_5.val
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
                  mtx_lvl_stop == mtx_lvl_2_stop || throw(DimensionMismatch("mismatched dimension limits ($(mtx_lvl_stop) != $(mtx_lvl_2_stop))"))
                  (Finch.CPUSharedMemory)(Finch.CPU{:t}(n))
                  (Finch.CPUSharedMemory)(Finch.CPU{:t}(n))
                  (Finch.CPUSharedMemory)(Finch.CPU{:t}(n))
                  (Finch.CPUSharedMemory)(Finch.CPU{:t}(n))
                  (Finch.CPUSharedMemory)(Finch.CPU{:t}(n))
                  Threads.@threads :dynamic for tid = 1:n
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              tens_lvl_4_val_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_4_val)
                                              tens_lvl_3_ptr_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_3_ptr)
                                              tens_lvl_3_idx_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_3_idx)
                                              tens_lvl_2_qos_fill_3 = (Finch).transfer((Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)()), tens_lvl_2_qos_fill)
                                              tens_lvl_2_qos_stop_3 = (Finch).transfer((Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)()), tens_lvl_2_qos_stop)
                                              resize!(tens_lvl_3_ptr_3, 0 + 1)
                                              for p = 1:0
                                                  tens_lvl_3_ptr_3[p + 1] += tens_lvl_3_ptr_3[p]
                                              end
                                              qos_stop = tens_lvl_3_ptr_3[0 + 1] - 1
                                              resize!(tens_lvl_3_idx_3, qos_stop)
                                              resize!(tens_lvl_4_val_3, qos_stop)
                                              tens_lvl_2_qos_fill_3[tid] = 0
                                              tens_lvl_2_qos_stop_3[tid] = 0
                                          end)
                                  nothing
                              end
                      end
                  Finch.resize_if_smaller!(tens_lvl_2_task, tens_lvl_stop)
                  Finch.resize_if_smaller!(tens_lvl_2_ptr, tens_lvl_stop)
                  Finch.fill_range!(tens_lvl_2_ptr, 0, 1, tens_lvl_stop)
                  w_lvl_3_val_2 = (Finch).transfer((CPULocalMemory)(Finch.CPU{:t}(n_3)), w_lvl_3_val)
                  w_lvl_2_ptr_2 = (Finch).transfer((CPULocalMemory)(Finch.CPU{:t}(n_3)), w_lvl_2_ptr)
                  w_lvl_2_tbl_2 = (Finch).transfer((CPULocalMemory)(Finch.CPU{:t}(n_3)), w_lvl_2_tbl)
                  w_lvl_2_srt_2 = (Finch).transfer((CPULocalMemory)(Finch.CPU{:t}(n_3)), w_lvl_2_srt)
                  w_lvl_5_val_2 = (Finch).transfer((CPULocalMemory)(Finch.CPU{:t}(n_3)), w_lvl_5_val)
                  w_lvl_4_ptr_2 = (Finch).transfer((CPULocalMemory)(Finch.CPU{:t}(n_3)), w_lvl_4_ptr)
                  w_lvl_4_tbl_2 = (Finch).transfer((CPULocalMemory)(Finch.CPU{:t}(n_3)), w_lvl_4_tbl)
                  w_lvl_4_srt_2 = (Finch).transfer((CPULocalMemory)(Finch.CPU{:t}(n_3)), w_lvl_4_srt)
                  mtx_lvl_3_val_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), mtx_lvl_3_val)
                  mtx_lvl_2_ptr_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), mtx_lvl_2_ptr)
                  mtx_lvl_2_idx_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), mtx_lvl_2_idx)
                  Threads.@threads :dynamic for tid_2 = 1:n_3
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              w_lvl_3_val_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), w_lvl_3_val_2)
                                              w_lvl_2_ptr_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), w_lvl_2_ptr_2)
                                              w_lvl_2_tbl_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), w_lvl_2_tbl_2)
                                              w_lvl_2_srt_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), w_lvl_2_srt_2)
                                              println( typeof(w_lvl_2_srt_2.data))
                                              println(w_lvl_2_srt_3)
                                              w_lvl_5_val_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), w_lvl_5_val_2)
                                              w_lvl_4_ptr_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), w_lvl_4_ptr_2)
                                              w_lvl_4_tbl_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), w_lvl_4_tbl_2)
                                              w_lvl_4_srt_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), w_lvl_4_srt_2)
                                              mtx_lvl_3_val_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), mtx_lvl_3_val_2)
                                              mtx_lvl_2_ptr_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), mtx_lvl_2_ptr_2)
                                              mtx_lvl_2_idx_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), mtx_lvl_2_idx_2)
                                              phase_start_2 = max(1, 1 + fld(mtx_lvl_stop * (tid_2 + -1), n_3))
                                              phase_stop_2 = min(mtx_lvl_stop, fld(mtx_lvl_stop * tid_2, n_3))
                                              if phase_stop_2 >= phase_start_2
                                                  for j_5 = phase_start_2:phase_stop_2
                                                      mtx_lvl_q = (1 - 1) * mtx_lvl_stop + j_5
                                                      w_lvl_3_val_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_2)), w_lvl_3_val_3)
                                                      w_lvl_2_ptr_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_2)), w_lvl_2_ptr_3)
                                                      w_lvl_2_tbl_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_2)), w_lvl_2_tbl_3)
                                                      w_lvl_2_srt_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_2)), w_lvl_2_srt_3)
                                                      Threads.@threads :dynamic for tid_3 = 1:n_2
                                                              Finch.@barrier begin
                                                                      @inbounds @fastmath(begin
                                                                                  w_lvl_3_val_5 = (Finch).transfer((Finch.MemoryChannel)(tid_3, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_2), n_2), (Finch.CPUThread)(tid_3, Finch.CPU{:q}(n_2), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))), w_lvl_3_val_3)
                                                                                  w_lvl_2_ptr_5 = (Finch).transfer((Finch.MemoryChannel)(tid_3, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_2), n_2), (Finch.CPUThread)(tid_3, Finch.CPU{:q}(n_2), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))), w_lvl_2_ptr_3)
                                                                                  w_lvl_2_tbl_5 = (Finch).transfer((Finch.MemoryChannel)(tid_3, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_2), n_2), (Finch.CPUThread)(tid_3, Finch.CPU{:q}(n_2), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))), w_lvl_2_tbl_3)
                                                                                  w_lvl_2_srt_5 = (Finch).transfer((Finch.MemoryChannel)(tid_3, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_2), n_2), (Finch.CPUThread)(tid_3, Finch.CPU{:q}(n_2), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))), w_lvl_2_srt_3)
                                                                                  for w_lvl_2_r = 1:w_lvl_2_qos_fill
                                                                                      w_lvl_2_p = first(w_lvl_2_srt_5[w_lvl_2_r])
                                                                                      w_lvl_2_ptr_5[w_lvl_2_p] = 0
                                                                                      w_lvl_2_ptr_5[w_lvl_2_p + 1] = 0
                                                                                      w_lvl_2_i = last(w_lvl_2_srt_5[w_lvl_2_r])
                                                                                      w_lvl_2_q = (w_lvl_2_p - 1) * mtx_lvl_2_stop + w_lvl_2_i
                                                                                      w_lvl_2_tbl_5[w_lvl_2_q] = false
                                                                                      Finch.resize_if_smaller!(w_lvl_3_val_5, w_lvl_2_q)
                                                                                      Finch.fill_range!(w_lvl_3_val_5, 0.0, w_lvl_2_q, w_lvl_2_q)
                                                                                  end
                                                                                  w_lvl_2_qos_fill = 0
                                                                                  w_lvl_2_ptr_5[1] = 1
                                                                                  resize!(w_lvl_2_ptr_5, 0 + 1)
                                                                                  resize!(w_lvl_2_tbl_5, 0mtx_lvl_2_stop)
                                                                                  resize!(w_lvl_2_srt_5, 0)
                                                                                  sort!(w_lvl_2_srt_5)
                                                                                  w_lvl_2_p_prev = 0
                                                                                  for w_lvl_2_r_2 = 1:0
                                                                                      w_lvl_2_p_2 = first(w_lvl_2_srt_5[w_lvl_2_r_2])
                                                                                      if w_lvl_2_p_2 != w_lvl_2_p_prev
                                                                                          w_lvl_2_ptr_5[w_lvl_2_p_prev + 1] = w_lvl_2_r_2
                                                                                          w_lvl_2_ptr_5[w_lvl_2_p_2] = w_lvl_2_r_2
                                                                                      end
                                                                                      w_lvl_2_p_prev = w_lvl_2_p_2
                                                                                  end
                                                                                  w_lvl_2_ptr_5[w_lvl_2_p_prev + 1] = 0 + 1
                                                                                  w_lvl_2_qos_stop = 0
                                                                                  resize!(w_lvl_3_val_5, 0)
                                                                                  nothing
                                                                              end)
                                                                      nothing
                                                                  end
                                                          end
                                                      w_lvl_3_val_3 = (Finch).transfer(w_lvl_3_val_3, w_lvl_3_val_4)
                                                      w_lvl_2_ptr_3 = (Finch).transfer(w_lvl_2_ptr_3, w_lvl_2_ptr_4)
                                                      w_lvl_2_tbl_3 = (Finch).transfer(w_lvl_2_tbl_3, w_lvl_2_tbl_4)
                                                      w_lvl_2_srt_3 = (Finch).transfer(w_lvl_2_srt_3, w_lvl_2_srt_4)
                                                      for w_lvl_4_r = 1:w_lvl_4_qos_fill
                                                          w_lvl_4_p = first(w_lvl_4_srt_3[w_lvl_4_r])
                                                          w_lvl_4_ptr_3[w_lvl_4_p] = 0
                                                          w_lvl_4_ptr_3[w_lvl_4_p + 1] = 0
                                                          w_lvl_4_i = last(w_lvl_4_srt_3[w_lvl_4_r])
                                                          w_lvl_4_q = (w_lvl_4_p - 1) * mtx_lvl_2_stop + w_lvl_4_i
                                                          w_lvl_4_tbl_3[w_lvl_4_q] = false
                                                          Finch.resize_if_smaller!(w_lvl_5_val_3, w_lvl_4_q)
                                                          Finch.fill_range!(w_lvl_5_val_3, 0.0, w_lvl_4_q, w_lvl_4_q)
                                                      end
                                                      w_lvl_4_qos_fill = 0
                                                      w_lvl_4_ptr_3[1] = 1
                                                      resize!(w_lvl_4_ptr_3, 0 + 1)
                                                      resize!(w_lvl_4_tbl_3, 0mtx_lvl_2_stop)
                                                      resize!(w_lvl_4_srt_3, 0)
                                                      sort!(w_lvl_4_srt_3)
                                                      w_lvl_4_p_prev = 0
                                                      for w_lvl_4_r_2 = 1:0
                                                          w_lvl_4_p_2 = first(w_lvl_4_srt_3[w_lvl_4_r_2])
                                                          if w_lvl_4_p_2 != w_lvl_4_p_prev
                                                              w_lvl_4_ptr_3[w_lvl_4_p_prev + 1] = w_lvl_4_r_2
                                                              w_lvl_4_ptr_3[w_lvl_4_p_2] = w_lvl_4_r_2
                                                          end
                                                          w_lvl_4_p_prev = w_lvl_4_p_2
                                                      end
                                                      w_lvl_4_ptr_3[w_lvl_4_p_prev + 1] = 0 + 1
                                                      resize!(w_lvl_5_val_3, 0)
                                                      w_lvl_3_val_6 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_2)), w_lvl_3_val_3)
                                                      w_lvl_2_ptr_6 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_2)), w_lvl_2_ptr_3)
                                                      w_lvl_2_tbl_6 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_2)), w_lvl_2_tbl_3)
                                                      w_lvl_2_srt_6 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_2)), w_lvl_2_srt_3)
                                                      Threads.@threads :dynamic for tid_4 = 1:n_2
                                                              Finch.@barrier begin
                                                                      @inbounds @fastmath(begin
                                                                                  w_lvl_3_val_7 = (Finch).transfer((Finch.MemoryChannel)(tid_4, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_2), n_2), (Finch.CPUThread)(tid_4, Finch.CPU{:q}(n_2), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))), w_lvl_3_val_3)
                                                                                  w_lvl_2_ptr_7 = (Finch).transfer((Finch.MemoryChannel)(tid_4, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_2), n_2), (Finch.CPUThread)(tid_4, Finch.CPU{:q}(n_2), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))), w_lvl_2_ptr_3)
                                                                                  w_lvl_2_tbl_7 = (Finch).transfer((Finch.MemoryChannel)(tid_4, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_2), n_2), (Finch.CPUThread)(tid_4, Finch.CPU{:q}(n_2), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))), w_lvl_2_tbl_3)
                                                                                  w_lvl_2_srt_7 = (Finch).transfer((Finch.MemoryChannel)(tid_4, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_2), n_2), (Finch.CPUThread)(tid_4, Finch.CPU{:q}(n_2), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))), w_lvl_2_srt_3)
                                                                                  w_lvl_2q_stop = 1mtx_lvl_2_stop
                                                                                  Finch.resize_if_smaller!(w_lvl_2_ptr_7, 1 + 1)
                                                                                  Finch.fill_range!(w_lvl_2_ptr_7, 0, 1 + 1, 1 + 1)
                                                                                  w_lvl_2old = length(w_lvl_2_tbl_7) + 1
                                                                                  Finch.resize_if_smaller!(w_lvl_2_tbl_7, w_lvl_2q_stop)
                                                                                  Finch.fill_range!(w_lvl_2_tbl_7, false, w_lvl_2old, w_lvl_2q_stop)
                                                                                  Finch.resize_if_smaller!(w_lvl_3_val_7, w_lvl_2q_stop)
                                                                                  Finch.fill_range!(w_lvl_3_val_7, 0.0, w_lvl_2old, w_lvl_2q_stop)
                                                                                  resize!(w_lvl_2_ptr_7, 1 + 1)
                                                                                  resize!(w_lvl_2_tbl_7, 1mtx_lvl_2_stop)
                                                                                  resize!(w_lvl_2_srt_7, w_lvl_2_qos_fill)
                                                                                  sort!(w_lvl_2_srt_7)
                                                                                  w_lvl_2_p_prev_2 = 0
                                                                                  for w_lvl_2_r_3 = 1:w_lvl_2_qos_fill
                                                                                      w_lvl_2_p_4 = first(w_lvl_2_srt_7[w_lvl_2_r_3])
                                                                                      if w_lvl_2_p_4 != w_lvl_2_p_prev_2
                                                                                          w_lvl_2_ptr_7[w_lvl_2_p_prev_2 + 1] = w_lvl_2_r_3
                                                                                          w_lvl_2_ptr_7[w_lvl_2_p_4] = w_lvl_2_r_3
                                                                                      end
                                                                                      w_lvl_2_p_prev_2 = w_lvl_2_p_4
                                                                                  end
                                                                                  w_lvl_2_ptr_7[w_lvl_2_p_prev_2 + 1] = w_lvl_2_qos_fill + 1
                                                                                  w_lvl_2_qos_stop = w_lvl_2_qos_fill
                                                                                  resize!(w_lvl_3_val_7, mtx_lvl_2_stop)
                                                                                  nothing
                                                                              end)
                                                                      nothing
                                                                  end
                                                          end
                                                      w_lvl_4q_stop = 1mtx_lvl_2_stop
                                                      Finch.resize_if_smaller!(w_lvl_4_ptr_3, 1 + 1)
                                                      Finch.fill_range!(w_lvl_4_ptr_3, 0, 1 + 1, 1 + 1)
                                                      w_lvl_4old = length(w_lvl_4_tbl_3) + 1
                                                      Finch.resize_if_smaller!(w_lvl_4_tbl_3, w_lvl_4q_stop)
                                                      Finch.fill_range!(w_lvl_4_tbl_3, false, w_lvl_4old, w_lvl_4q_stop)
                                                      Finch.resize_if_smaller!(w_lvl_5_val_3, w_lvl_4q_stop)
                                                      Finch.fill_range!(w_lvl_5_val_3, 0.0, w_lvl_4old, w_lvl_4q_stop)
                                                      resize!(w_lvl_4_ptr_3, 1 + 1)
                                                      resize!(w_lvl_4_tbl_3, 1mtx_lvl_2_stop)
                                                      resize!(w_lvl_4_srt_3, 0)
                                                      sort!(w_lvl_4_srt_3)
                                                      w_lvl_4_p_prev_2 = 0
                                                      for w_lvl_4_r_3 = 1:0
                                                          w_lvl_4_p_4 = first(w_lvl_4_srt_3[w_lvl_4_r_3])
                                                          if w_lvl_4_p_4 != w_lvl_4_p_prev_2
                                                              w_lvl_4_ptr_3[w_lvl_4_p_prev_2 + 1] = w_lvl_4_r_3
                                                              w_lvl_4_ptr_3[w_lvl_4_p_4] = w_lvl_4_r_3
                                                          end
                                                          w_lvl_4_p_prev_2 = w_lvl_4_p_4
                                                      end
                                                      w_lvl_4_ptr_3[w_lvl_4_p_prev_2 + 1] = 0 + 1
                                                      resize!(w_lvl_5_val_3, mtx_lvl_2_stop)
                                                      w_lvl_3_val_3 = (Finch).transfer(w_lvl_3_val_3, w_lvl_3_val_6)
                                                      w_lvl_2_ptr_3 = (Finch).transfer(w_lvl_2_ptr_3, w_lvl_2_ptr_6)
                                                      w_lvl_2_tbl_3 = (Finch).transfer(w_lvl_2_tbl_3, w_lvl_2_tbl_6)
                                                      w_lvl_2_srt_3 = (Finch).transfer(w_lvl_2_srt_3, w_lvl_2_srt_6)
                                                      w_lvl_3_val_8 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), w_lvl_3_val_3)
                                                      w_lvl_2_ptr_8 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), w_lvl_2_ptr_3)
                                                      w_lvl_2_tbl_8 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), w_lvl_2_tbl_3)
                                                      w_lvl_2_srt_8 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), w_lvl_2_srt_3)
                                                      mtx_lvl_3_val_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), mtx_lvl_3_val_3)
                                                      mtx_lvl_2_ptr_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), mtx_lvl_2_ptr_3)
                                                      mtx_lvl_2_idx_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), mtx_lvl_2_idx_3)
                                                      Threads.@threads :dynamic for tid_5 = 1:n_4
                                                              Finch.@barrier begin
                                                                      @inbounds @fastmath(begin
                                                                                  w_lvl_3_val_9 = (Finch).transfer((Finch.MemoryChannel)(tid_5, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_4), n_4), (Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))), w_lvl_3_val_8)
                                                                                  w_lvl_2_ptr_9 = (Finch).transfer((Finch.MemoryChannel)(tid_5, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_4), n_4), (Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))), w_lvl_2_ptr_8)
                                                                                  w_lvl_2_tbl_9 = (Finch).transfer((Finch.MemoryChannel)(tid_5, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_4), n_4), (Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))), w_lvl_2_tbl_8)
                                                                                  w_lvl_2_srt_9 = (Finch).transfer((Finch.MemoryChannel)(tid_5, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_4), n_4), (Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))), w_lvl_2_srt_8)
                                                                                  w_lvl_3_val_10 = (Finch).transfer((Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), w_lvl_3_val_8)
                                                                                  (Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))
                                                                                  w_lvl_2_tbl_10 = (Finch).transfer((Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), w_lvl_2_tbl_8)
                                                                                  w_lvl_2_srt_10 = (Finch).transfer((Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), w_lvl_2_srt_8)
                                                                                  mtx_lvl_3_val_5 = (Finch).transfer((Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), mtx_lvl_3_val_4)
                                                                                  mtx_lvl_2_ptr_5 = (Finch).transfer((Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), mtx_lvl_2_ptr_4)
                                                                                  mtx_lvl_2_idx_5 = (Finch).transfer((Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), mtx_lvl_2_idx_4)
                                                                                  res_25 = begin
                                                                                          mtx_lvl_2_q = mtx_lvl_2_ptr_5[mtx_lvl_q]
                                                                                          mtx_lvl_2_q_stop = mtx_lvl_2_ptr_5[mtx_lvl_q + 1]
                                                                                          if mtx_lvl_2_q < mtx_lvl_2_q_stop
                                                                                              mtx_lvl_2_i1 = mtx_lvl_2_idx_5[mtx_lvl_2_q_stop - 1]
                                                                                          else
                                                                                              mtx_lvl_2_i1 = 0
                                                                                          end
                                                                                          phase_start_4 = max(1, 1 + fld(mtx_lvl_stop * (-1 + tid_5), n_4))
                                                                                          phase_stop_4 = min(mtx_lvl_stop, mtx_lvl_2_i1, fld(mtx_lvl_stop * tid_5, n_4))
                                                                                          if phase_stop_4 >= phase_start_4
                                                                                              if mtx_lvl_2_idx_5[mtx_lvl_2_q] < phase_start_4
                                                                                                  mtx_lvl_2_q = Finch.scansearch(mtx_lvl_2_idx_5, phase_start_4, mtx_lvl_2_q, mtx_lvl_2_q_stop - 1)
                                                                                              end
                                                                                              while true
                                                                                                  mtx_lvl_2_i = mtx_lvl_2_idx_5[mtx_lvl_2_q]
                                                                                                  if mtx_lvl_2_i < phase_stop_4
                                                                                                      mtx_lvl_3_val_7 = mtx_lvl_3_val_5[mtx_lvl_2_q]
                                                                                                      mtx_lvl_q_2 = (1 - 1) * mtx_lvl_stop + mtx_lvl_2_i
                                                                                                      mtx_lvl_2_q_2 = mtx_lvl_2_ptr_5[mtx_lvl_q_2]
                                                                                                      mtx_lvl_2_q_stop_2 = mtx_lvl_2_ptr_5[mtx_lvl_q_2 + 1]
                                                                                                      if mtx_lvl_2_q_2 < mtx_lvl_2_q_stop_2
                                                                                                          mtx_lvl_2_i1_2 = mtx_lvl_2_idx_5[mtx_lvl_2_q_stop_2 - 1]
                                                                                                      else
                                                                                                          mtx_lvl_2_i1_2 = 0
                                                                                                      end
                                                                                                      phase_stop_6 = min(mtx_lvl_2_stop, mtx_lvl_2_i1_2)
                                                                                                      if phase_stop_6 >= 1
                                                                                                          if mtx_lvl_2_idx_5[mtx_lvl_2_q_2] < 1
                                                                                                              mtx_lvl_2_q_2 = Finch.scansearch(mtx_lvl_2_idx_5, 1, mtx_lvl_2_q_2, mtx_lvl_2_q_stop_2 - 1)
                                                                                                          end
                                                                                                          while true
                                                                                                              mtx_lvl_2_i_2 = mtx_lvl_2_idx_5[mtx_lvl_2_q_2]
                                                                                                              if mtx_lvl_2_i_2 < phase_stop_6
                                                                                                                  mtx_lvl_3_val_8 = mtx_lvl_3_val_5[mtx_lvl_2_q_2]
                                                                                                                  w_lvl_2_q_2 = (1 - 1) * mtx_lvl_2_stop + mtx_lvl_2_i_2
                                                                                                                  w_lvl_3_val_10[w_lvl_2_q_2] = mtx_lvl_3_val_7 * mtx_lvl_3_val_8 + w_lvl_3_val_10[w_lvl_2_q_2]
                                                                                                                  if !(w_lvl_2_tbl_10[w_lvl_2_q_2])
                                                                                                                      w_lvl_2_tbl_10[w_lvl_2_q_2] = true
                                                                                                                      w_lvl_2_qos_fill += 1
                                                                                                                      if w_lvl_2_qos_fill > w_lvl_2_qos_stop
                                                                                                                          w_lvl_2_qos_stop = max(w_lvl_2_qos_stop << 1, 1)
                                                                                                                          Finch.resize_if_smaller!(w_lvl_2_srt_10, w_lvl_2_qos_stop)
                                                                                                                      end
                                                                                                                      w_lvl_2_srt_10[w_lvl_2_qos_fill] = (1, mtx_lvl_2_i_2)
                                                                                                                  end
                                                                                                                  mtx_lvl_2_q_2 += 1
                                                                                                              else
                                                                                                                  phase_stop_8 = min(phase_stop_6, mtx_lvl_2_i_2)
                                                                                                                  if mtx_lvl_2_i_2 == phase_stop_8
                                                                                                                      mtx_lvl_3_val_8 = mtx_lvl_3_val_5[mtx_lvl_2_q_2]
                                                                                                                      w_lvl_2_q_2 = (1 - 1) * mtx_lvl_2_stop + phase_stop_8
                                                                                                                      w_lvl_3_val_10[w_lvl_2_q_2] += mtx_lvl_3_val_7 * mtx_lvl_3_val_8
                                                                                                                      if !(w_lvl_2_tbl_10[w_lvl_2_q_2])
                                                                                                                          w_lvl_2_tbl_10[w_lvl_2_q_2] = true
                                                                                                                          w_lvl_2_qos_fill += 1
                                                                                                                          if w_lvl_2_qos_fill > w_lvl_2_qos_stop
                                                                                                                              w_lvl_2_qos_stop = max(w_lvl_2_qos_stop << 1, 1)
                                                                                                                              Finch.resize_if_smaller!(w_lvl_2_srt_10, w_lvl_2_qos_stop)
                                                                                                                          end
                                                                                                                          w_lvl_2_srt_10[w_lvl_2_qos_fill] = (1, phase_stop_8)
                                                                                                                      end
                                                                                                                      mtx_lvl_2_q_2 += 1
                                                                                                                  end
                                                                                                                  break
                                                                                                              end
                                                                                                          end
                                                                                                      end
                                                                                                      mtx_lvl_2_q += 1
                                                                                                  else
                                                                                                      phase_stop_10 = min(phase_stop_4, mtx_lvl_2_i)
                                                                                                      if mtx_lvl_2_i == phase_stop_10
                                                                                                          mtx_lvl_3_val_7 = mtx_lvl_3_val_5[mtx_lvl_2_q]
                                                                                                          mtx_lvl_q_2 = (1 - 1) * mtx_lvl_stop + phase_stop_10
                                                                                                          mtx_lvl_2_q_3 = mtx_lvl_2_ptr_5[mtx_lvl_q_2]
                                                                                                          mtx_lvl_2_q_stop_3 = mtx_lvl_2_ptr_5[mtx_lvl_q_2 + 1]
                                                                                                          if mtx_lvl_2_q_3 < mtx_lvl_2_q_stop_3
                                                                                                              mtx_lvl_2_i1_3 = mtx_lvl_2_idx_5[mtx_lvl_2_q_stop_3 - 1]
                                                                                                          else
                                                                                                              mtx_lvl_2_i1_3 = 0
                                                                                                          end
                                                                                                          phase_stop_11 = min(mtx_lvl_2_stop, mtx_lvl_2_i1_3)
                                                                                                          if phase_stop_11 >= 1
                                                                                                              if mtx_lvl_2_idx_5[mtx_lvl_2_q_3] < 1
                                                                                                                  mtx_lvl_2_q_3 = Finch.scansearch(mtx_lvl_2_idx_5, 1, mtx_lvl_2_q_3, mtx_lvl_2_q_stop_3 - 1)
                                                                                                              end
                                                                                                              while true
                                                                                                                  mtx_lvl_2_i_3 = mtx_lvl_2_idx_5[mtx_lvl_2_q_3]
                                                                                                                  if mtx_lvl_2_i_3 < phase_stop_11
                                                                                                                      mtx_lvl_3_val_9 = mtx_lvl_3_val_5[mtx_lvl_2_q_3]
                                                                                                                      w_lvl_2_q_3 = (1 - 1) * mtx_lvl_2_stop + mtx_lvl_2_i_3
                                                                                                                      w_lvl_3_val_10[w_lvl_2_q_3] = mtx_lvl_3_val_7 * mtx_lvl_3_val_9 + w_lvl_3_val_10[w_lvl_2_q_3]
                                                                                                                      if !(w_lvl_2_tbl_10[w_lvl_2_q_3])
                                                                                                                          w_lvl_2_tbl_10[w_lvl_2_q_3] = true
                                                                                                                          w_lvl_2_qos_fill += 1
                                                                                                                          if w_lvl_2_qos_fill > w_lvl_2_qos_stop
                                                                                                                              w_lvl_2_qos_stop = max(w_lvl_2_qos_stop << 1, 1)
                                                                                                                              Finch.resize_if_smaller!(w_lvl_2_srt_10, w_lvl_2_qos_stop)
                                                                                                                          end
                                                                                                                          w_lvl_2_srt_10[w_lvl_2_qos_fill] = (1, mtx_lvl_2_i_3)
                                                                                                                      end
                                                                                                                      mtx_lvl_2_q_3 += 1
                                                                                                                  else
                                                                                                                      phase_stop_13 = min(phase_stop_11, mtx_lvl_2_i_3)
                                                                                                                      if mtx_lvl_2_i_3 == phase_stop_13
                                                                                                                          mtx_lvl_3_val_9 = mtx_lvl_3_val_5[mtx_lvl_2_q_3]
                                                                                                                          w_lvl_2_q_3 = (1 - 1) * mtx_lvl_2_stop + phase_stop_13
                                                                                                                          w_lvl_3_val_10[w_lvl_2_q_3] += mtx_lvl_3_val_7 * mtx_lvl_3_val_9
                                                                                                                          if !(w_lvl_2_tbl_10[w_lvl_2_q_3])
                                                                                                                              w_lvl_2_tbl_10[w_lvl_2_q_3] = true
                                                                                                                              w_lvl_2_qos_fill += 1
                                                                                                                              if w_lvl_2_qos_fill > w_lvl_2_qos_stop
                                                                                                                                  w_lvl_2_qos_stop = max(w_lvl_2_qos_stop << 1, 1)
                                                                                                                                  Finch.resize_if_smaller!(w_lvl_2_srt_10, w_lvl_2_qos_stop)
                                                                                                                              end
                                                                                                                              w_lvl_2_srt_10[w_lvl_2_qos_fill] = (1, phase_stop_13)
                                                                                                                          end
                                                                                                                          mtx_lvl_2_q_3 += 1
                                                                                                                      end
                                                                                                                      break
                                                                                                                  end
                                                                                                              end
                                                                                                          end
                                                                                                          mtx_lvl_2_q += 1
                                                                                                      end
                                                                                                      break
                                                                                                  end
                                                                                              end
                                                                                          end
                                                                                          phase_start_15 = max(1, 1 + fld(mtx_lvl_stop * tid_5, n_4), 1 + mtx_lvl_2_i1)
                                                                                          if mtx_lvl_stop >= phase_start_15
                                                                                              mtx_lvl_stop + 1
                                                                                          end
                                                                                      end
                                                                                  resize!(w_lvl_2_ptr_9, 1 + 1)
                                                                                  resize!(w_lvl_2_tbl_9, 1mtx_lvl_2_stop)
                                                                                  resize!(w_lvl_2_srt_9, w_lvl_2_qos_fill)
                                                                                  sort!(w_lvl_2_srt_9)
                                                                                  w_lvl_2_p_prev_3 = 0
                                                                                  for w_lvl_2_r_4 = 1:w_lvl_2_qos_fill
                                                                                      w_lvl_2_p_6 = first(w_lvl_2_srt_9[w_lvl_2_r_4])
                                                                                      if w_lvl_2_p_6 != w_lvl_2_p_prev_3
                                                                                          w_lvl_2_ptr_9[w_lvl_2_p_prev_3 + 1] = w_lvl_2_r_4
                                                                                          w_lvl_2_ptr_9[w_lvl_2_p_6] = w_lvl_2_r_4
                                                                                      end
                                                                                      w_lvl_2_p_prev_3 = w_lvl_2_p_6
                                                                                  end
                                                                                  w_lvl_2_ptr_9[w_lvl_2_p_prev_3 + 1] = w_lvl_2_qos_fill + 1
                                                                                  w_lvl_2_qos_stop = w_lvl_2_qos_fill
                                                                                  resize!(w_lvl_3_val_9, mtx_lvl_2_stop * 1)
                                                                                  res_25
                                                                              end)
                                                                      nothing
                                                                  end
                                                          end
                                                      w_lvl_3_val_3 = (Finch).transfer(w_lvl_3_val_3, w_lvl_3_val_8)
                                                      w_lvl_2_ptr_3 = (Finch).transfer(w_lvl_2_ptr_3, w_lvl_2_ptr_8)
                                                      w_lvl_2_tbl_3 = (Finch).transfer(w_lvl_2_tbl_3, w_lvl_2_tbl_8)
                                                      w_lvl_2_srt_3 = (Finch).transfer(w_lvl_2_srt_3, w_lvl_2_srt_8)
                                                      tm = collect(1:n_2)
                                                      gfm = ones(Int, n_2)
                                                      lfm = ones(Int, n_2)

                                                      Finch.coalesce_level!((SparseByteMapLevel){Int64}(ElementLevel{0.0, Float64, Int64}(w_lvl_3_val_8), mtx_lvl_2_stop, w_lvl_2_ptr_8, w_lvl_2_tbl_8, w_lvl_2_srt_8), gfm, lfm, tm, 1, n_2, (SparseByteMapLevel){Int64}(ElementLevel{0.0, Float64, Int64}(w_lvl_5_val_3), mtx_lvl_2_stop, w_lvl_4_ptr_3, w_lvl_4_tbl_3, w_lvl_4_srt_3))
                                                  end
                                              end
                                              phase_start_16 = max(1, 1 + fld(mtx_lvl_stop * tid_2, n_3))
                                              if mtx_lvl_stop >= phase_start_16
                                                  mtx_lvl_stop + 1
                                              end
                                          end)
                                  nothing
                              end
                      end
                  ()
              end)
  end)

  eval(code)
  run(tens, w, dev, dev2, mtx)