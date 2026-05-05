using Finch

code = :(function spgemm_finch_dynamic_kernel(C::Tensor{DenseLevel{Int64, ShardLevel{CPU{:t}, SparseListLevel{Int64, Finch.MultiChannelBuffer{Vector{Int64}}, Finch.MultiChannelBuffer{Vector{Int64}}, ElementLevel{0.0, Float64, Int64, Finch.MultiChannelBuffer{Vector{Float64}}}}, Vector{Int64}, Vector{Int64}, Vector{Int64}, Vector{Int64}, Finch.FinchStaticSchedule{:dynamic}}}}, w::Tensor{SparseByteMapLevel{Int64, Vector{Int64}, Vector{Bool}, Vector{Tuple{Int64, Int64}}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A::Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}, B::Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}, dev::CPU{:t}, sch::Finch.FinchGreedySchedule{:static})
      @inbounds @fastmath(begin
                  C_lvl = C.lvl
                  C_lvl_2 = C_lvl.lvl
                  C_lvl_2_ptr = C_lvl_2.ptr
                  C_lvl_2_task = C_lvl_2.task
                  C_lvl_2_qos_fill = C_lvl_2.used
                  C_lvl_2_qos_stop = C_lvl_2.alloc
                  n = C_lvl_2.device.n
                  C_lvl_3 = C_lvl_2.lvl
                  C_lvl_3_ptr = C_lvl_3.ptr
                  C_lvl_3_idx = C_lvl_3.idx
                  C_lvl_4 = C_lvl_3.lvl
                  C_lvl_4_val = C_lvl_4.val
                  w_lvl = w.lvl
                  w_lvl_ptr = w_lvl.ptr
                  w_lvl_tbl = w_lvl.tbl
                  w_lvl_srt = w_lvl.srt
                  w_lvl_qos_stop = (w_lvl_qos_fill = length(w_lvl.srt))
                  w_lvl_2 = w_lvl.lvl
                  w_lvl_2_val = w_lvl_2.val
                  A_lvl = A.lvl
                  A_lvl_stop = A_lvl.shape
                  A_lvl_2 = A_lvl.lvl
                  A_lvl_2_ptr = A_lvl_2.ptr
                  A_lvl_2_idx = A_lvl_2.idx
                  A_lvl_2_stop = A_lvl_2.shape
                  A_lvl_3 = A_lvl_2.lvl
                  A_lvl_3_val = A_lvl_3.val
                  B_lvl = B.lvl
                  B_lvl_stop = B_lvl.shape
                  B_lvl_2 = B_lvl.lvl
                  B_lvl_2_ptr = B_lvl_2.ptr
                  B_lvl_2_idx = B_lvl_2.idx
                  B_lvl_2_stop = B_lvl_2.shape
                  B_lvl_3 = B_lvl_2.lvl
                  B_lvl_3_val = B_lvl_3.val
                  n_2 = dev.n
                  chk = sch.chk
                  B_lvl_2_stop == A_lvl_stop || throw(DimensionMismatch("mismatched dimension limits ($(B_lvl_2_stop) != $(A_lvl_stop))"))
                  C_lvl_4_val_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), C_lvl_4_val)
                  C_lvl_3_ptr_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), C_lvl_3_ptr)
                  C_lvl_3_idx_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), C_lvl_3_idx)
                  C_lvl_2_qos_fill_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), C_lvl_2_qos_fill)
                  C_lvl_2_qos_stop_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), C_lvl_2_qos_stop)
                  Threads.@threads :dynamic for tid = 1:n
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              C_lvl_4_val_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), C_lvl_4_val)
                                              C_lvl_3_ptr_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), C_lvl_3_ptr)
                                              C_lvl_3_idx_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), C_lvl_3_idx)
                                              C_lvl_2_qos_fill_3 = (Finch).transfer((Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)()), C_lvl_2_qos_fill)
                                              C_lvl_2_qos_stop_3 = (Finch).transfer((Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)()), C_lvl_2_qos_stop)
                                              resize!(C_lvl_3_ptr_3, 0 + 1)
                                              for p = 1:0
                                                  C_lvl_3_ptr_3[p + 1] += C_lvl_3_ptr_3[p]
                                              end
                                              qos_stop = C_lvl_3_ptr_3[0 + 1] - 1
                                              resize!(C_lvl_3_idx_3, qos_stop)
                                              resize!(C_lvl_4_val_3, qos_stop)
                                              C_lvl_2_qos_fill_3[tid] = 0
                                              C_lvl_2_qos_stop_3[tid] = 0
                                          end)
                                  nothing
                              end
                      end
                  C_lvl_4_val = (Finch).transfer(C_lvl_4_val, C_lvl_4_val_2)
                  C_lvl_3_ptr = (Finch).transfer(C_lvl_3_ptr, C_lvl_3_ptr_2)
                  C_lvl_3_idx = (Finch).transfer(C_lvl_3_idx, C_lvl_3_idx_2)
                  C_lvl_2_qos_fill = (Finch).transfer(C_lvl_2_qos_fill, C_lvl_2_qos_fill_2)
                  C_lvl_2_qos_stop = (Finch).transfer(C_lvl_2_qos_stop, C_lvl_2_qos_stop_2)
                  Finch.resize_if_smaller!(C_lvl_2_task, B_lvl_stop)
                  Finch.resize_if_smaller!(C_lvl_2_ptr, B_lvl_stop)
                  Finch.fill_range!(C_lvl_2_ptr, 0, 1, B_lvl_stop)
                  w_lvl_2_val_2 = (Finch).transfer((CPULocalMemory)(Finch.CPU{:t}(n_2)), w_lvl_2_val)
                  w_lvl_ptr_2 = (Finch).transfer((CPULocalMemory)(Finch.CPU{:t}(n_2)), w_lvl_ptr)
                  w_lvl_tbl_2 = (Finch).transfer((CPULocalMemory)(Finch.CPU{:t}(n_2)), w_lvl_tbl)
                  w_lvl_srt_2 = (Finch).transfer((CPULocalMemory)(Finch.CPU{:t}(n_2)), w_lvl_srt)
                  B_lvl_3_val_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), B_lvl_3_val)
                  B_lvl_2_ptr_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), B_lvl_2_ptr)
                  B_lvl_2_idx_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), B_lvl_2_idx)
                  A_lvl_3_val_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), A_lvl_3_val)
                  A_lvl_2_ptr_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), A_lvl_2_ptr)
                  A_lvl_2_idx_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), A_lvl_2_idx)
                  C_lvl_4_val_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), C_lvl_4_val)
                  C_lvl_3_ptr_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), C_lvl_3_ptr)
                  C_lvl_3_idx_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), C_lvl_3_idx)
                  C_lvl_2_ptr_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), C_lvl_2_ptr)
                  C_lvl_2_task_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), C_lvl_2_task)
                  C_lvl_2_qos_fill_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), C_lvl_2_qos_fill)
                  C_lvl_2_qos_stop_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_2)), C_lvl_2_qos_stop)
                  chk_ctr = Threads.Atomic{Int}(0)
                  num_chks = cld(B_lvl_stop, chk)
                  Threads.@threads :static for tid_2 = 1:n_2
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              w_lvl_2_val_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_2), (Finch.SerialTask)()), w_lvl_2_val_2)
                                              w_lvl_ptr_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_2), (Finch.SerialTask)()), w_lvl_ptr_2)
                                              w_lvl_tbl_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_2), (Finch.SerialTask)()), w_lvl_tbl_2)
                                              w_lvl_srt_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_2), (Finch.SerialTask)()), w_lvl_srt_2)
                                              B_lvl_3_val_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_2), (Finch.SerialTask)()), B_lvl_3_val_2)
                                              B_lvl_2_ptr_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_2), (Finch.SerialTask)()), B_lvl_2_ptr_2)
                                              B_lvl_2_idx_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_2), (Finch.SerialTask)()), B_lvl_2_idx_2)
                                              A_lvl_3_val_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_2), (Finch.SerialTask)()), A_lvl_3_val_2)
                                              A_lvl_2_ptr_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_2), (Finch.SerialTask)()), A_lvl_2_ptr_2)
                                              A_lvl_2_idx_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_2), (Finch.SerialTask)()), A_lvl_2_idx_2)
                                              C_lvl_2_ptr_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_2), (Finch.SerialTask)()), C_lvl_2_ptr_2)
                                              C_lvl_2_task_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_2), (Finch.SerialTask)()), C_lvl_2_task_2)
                                              C_lvl_2_qos_fill_6 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_2), (Finch.SerialTask)()), C_lvl_2_qos_fill_4)
                                              C_lvl_2_qos_stop_6 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_2), (Finch.SerialTask)()), C_lvl_2_qos_stop_4)
                                              C_lvl_2_qos_fill_5 = C_lvl_2_qos_fill_6[tid_2]
                                              C_lvl_2_qos_stop_5 = C_lvl_2_qos_stop_6[tid_2]
                                              C_lvl_4_val_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_2), (Finch.SerialTask)())), C_lvl_4_val_4)
                                              C_lvl_3_ptr_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_2), (Finch.SerialTask)())), C_lvl_3_ptr_4)
                                              C_lvl_3_idx_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_2), (Finch.SerialTask)())), C_lvl_3_idx_4)
                                              C_lvl_3_qos_fill = C_lvl_3_ptr_5[C_lvl_2_qos_stop_5 + 1] - 1
                                              C_lvl_3_qos_stop = C_lvl_3_qos_fill
                                              C_lvl_3_prev_pos = Finch.scansearch(C_lvl_3_ptr_5, C_lvl_3_qos_fill + 1, 1, C_lvl_2_qos_stop_5) - 1
                                              for p_2 = C_lvl_2_qos_stop_5:-1:1
                                                  C_lvl_3_ptr_5[p_2 + 1] = C_lvl_3_ptr_5[p_2 + 1] - C_lvl_3_ptr_5[p_2]
                                              end
                                              res_34 = while true
                                                      chk_id = Threads.atomic_add!(chk_ctr, 1)
                                                      if chk_id > num_chks
                                                          break
                                                      end
                                                      phase_start_2 = max(1, 1 + chk * (-1 + chk_id))
                                                      phase_stop_2 = min(B_lvl_stop, chk * chk_id)
                                                      if phase_stop_2 >= phase_start_2
                                                          for j_6 = phase_start_2:phase_stop_2
                                                              B_lvl_q = (1 - 1) * B_lvl_stop + j_6
                                                              C_lvl_q = (1 - 1) * B_lvl_stop + j_6
                                                              qos = C_lvl_2_ptr_3[C_lvl_q]
                                                              if qos == 0
                                                                  qos = (C_lvl_2_qos_fill_5 += 1)
                                                                  C_lvl_2_task_3[C_lvl_q] = tid_2
                                                                  C_lvl_2_ptr_3[C_lvl_q] = C_lvl_2_qos_fill_5
                                                                  if C_lvl_2_qos_fill_5 > C_lvl_2_qos_stop_5
                                                                      C_lvl_2_qos_stop_5 = max(C_lvl_2_qos_stop_5 << 1, 1)
                                                                      Finch.resize_if_smaller!(C_lvl_3_ptr_5, C_lvl_2_qos_stop_5 + 1)
                                                                      Finch.fill_range!(C_lvl_3_ptr_5, 0, C_lvl_2_qos_fill_5 + 1, C_lvl_2_qos_stop_5 + 1)
                                                                  end
                                                              else
                                                                  @assert C_lvl_2_task_3[C_lvl_q] == tid_2 "Task mismatch in ShardLevel"
                                                              end
                                                              for w_lvl_r = 1:w_lvl_qos_fill
                                                                  w_lvl_p = first(w_lvl_srt_3[w_lvl_r])
                                                                  w_lvl_ptr_3[w_lvl_p] = 0
                                                                  w_lvl_ptr_3[w_lvl_p + 1] = 0
                                                                  w_lvl_i = last(w_lvl_srt_3[w_lvl_r])
                                                                  w_lvl_q = (w_lvl_p - 1) * A_lvl_2_stop + w_lvl_i
                                                                  w_lvl_tbl_3[w_lvl_q] = false
                                                                  Finch.resize_if_smaller!(w_lvl_2_val_3, w_lvl_q)
                                                                  Finch.fill_range!(w_lvl_2_val_3, 0.0, w_lvl_q, w_lvl_q)
                                                              end
                                                              w_lvl_qos_fill = 0
                                                              w_lvl_ptr_3[1] = 1
                                                              w_lvlq_stop = 1A_lvl_2_stop
                                                              Finch.resize_if_smaller!(w_lvl_ptr_3, 1 + 1)
                                                              Finch.fill_range!(w_lvl_ptr_3, 0, 1 + 1, 1 + 1)
                                                              w_lvlold = length(w_lvl_tbl_3) + 1
                                                              Finch.resize_if_smaller!(w_lvl_tbl_3, w_lvlq_stop)
                                                              Finch.fill_range!(w_lvl_tbl_3, false, w_lvlold, w_lvlq_stop)
                                                              Finch.resize_if_smaller!(w_lvl_2_val_3, w_lvlq_stop)
                                                              Finch.fill_range!(w_lvl_2_val_3, 0.0, w_lvlold, w_lvlq_stop)
                                                              B_lvl_2_q = B_lvl_2_ptr_3[B_lvl_q]
                                                              B_lvl_2_q_stop = B_lvl_2_ptr_3[B_lvl_q + 1]
                                                              if B_lvl_2_q < B_lvl_2_q_stop
                                                                  B_lvl_2_i1 = B_lvl_2_idx_3[B_lvl_2_q_stop - 1]
                                                              else
                                                                  B_lvl_2_i1 = 0
                                                              end
                                                              phase_stop_3 = min(B_lvl_2_stop, B_lvl_2_i1)
                                                              if phase_stop_3 >= 1
                                                                  if B_lvl_2_idx_3[B_lvl_2_q] < 1
                                                                      B_lvl_2_q = Finch.scansearch(B_lvl_2_idx_3, 1, B_lvl_2_q, B_lvl_2_q_stop - 1)
                                                                  end
                                                                  while true
                                                                      B_lvl_2_i = B_lvl_2_idx_3[B_lvl_2_q]
                                                                      if B_lvl_2_i < phase_stop_3
                                                                          B_lvl_3_val_4 = B_lvl_3_val_3[B_lvl_2_q]
                                                                          A_lvl_q = (1 - 1) * A_lvl_stop + B_lvl_2_i
                                                                          A_lvl_2_q = A_lvl_2_ptr_3[A_lvl_q]
                                                                          A_lvl_2_q_stop = A_lvl_2_ptr_3[A_lvl_q + 1]
                                                                          if A_lvl_2_q < A_lvl_2_q_stop
                                                                              A_lvl_2_i1 = A_lvl_2_idx_3[A_lvl_2_q_stop - 1]
                                                                          else
                                                                              A_lvl_2_i1 = 0
                                                                          end
                                                                          phase_stop_5 = min(A_lvl_2_i1, A_lvl_2_stop)
                                                                          if phase_stop_5 >= 1
                                                                              if A_lvl_2_idx_3[A_lvl_2_q] < 1
                                                                                  A_lvl_2_q = Finch.scansearch(A_lvl_2_idx_3, 1, A_lvl_2_q, A_lvl_2_q_stop - 1)
                                                                              end
                                                                              while true
                                                                                  A_lvl_2_i = A_lvl_2_idx_3[A_lvl_2_q]
                                                                                  if A_lvl_2_i < phase_stop_5
                                                                                      A_lvl_3_val_4 = A_lvl_3_val_3[A_lvl_2_q]
                                                                                      w_lvl_q_2 = (1 - 1) * A_lvl_2_stop + A_lvl_2_i
                                                                                      w_lvl_2_val_3[w_lvl_q_2] = B_lvl_3_val_4 * A_lvl_3_val_4 + w_lvl_2_val_3[w_lvl_q_2]
                                                                                      if !(w_lvl_tbl_3[w_lvl_q_2])
                                                                                          w_lvl_tbl_3[w_lvl_q_2] = true
                                                                                          w_lvl_qos_fill += 1
                                                                                          if w_lvl_qos_fill > w_lvl_qos_stop
                                                                                              w_lvl_qos_stop = max(w_lvl_qos_stop << 1, 1)
                                                                                              Finch.resize_if_smaller!(w_lvl_srt_3, w_lvl_qos_stop)
                                                                                          end
                                                                                          w_lvl_srt_3[w_lvl_qos_fill] = (1, A_lvl_2_i)
                                                                                      end
                                                                                      A_lvl_2_q += 1
                                                                                  else
                                                                                      phase_stop_7 = min(phase_stop_5, A_lvl_2_i)
                                                                                      if A_lvl_2_i == phase_stop_7
                                                                                          A_lvl_3_val_4 = A_lvl_3_val_3[A_lvl_2_q]
                                                                                          w_lvl_q_2 = (1 - 1) * A_lvl_2_stop + phase_stop_7
                                                                                          w_lvl_2_val_3[w_lvl_q_2] += B_lvl_3_val_4 * A_lvl_3_val_4
                                                                                          if !(w_lvl_tbl_3[w_lvl_q_2])
                                                                                              w_lvl_tbl_3[w_lvl_q_2] = true
                                                                                              w_lvl_qos_fill += 1
                                                                                              if w_lvl_qos_fill > w_lvl_qos_stop
                                                                                                  w_lvl_qos_stop = max(w_lvl_qos_stop << 1, 1)
                                                                                                  Finch.resize_if_smaller!(w_lvl_srt_3, w_lvl_qos_stop)
                                                                                              end
                                                                                              w_lvl_srt_3[w_lvl_qos_fill] = (1, phase_stop_7)
                                                                                          end
                                                                                          A_lvl_2_q += 1
                                                                                      end
                                                                                      break
                                                                                  end
                                                                              end
                                                                          end
                                                                          B_lvl_2_q += 1
                                                                      else
                                                                          phase_stop_9 = min(phase_stop_3, B_lvl_2_i)
                                                                          if B_lvl_2_i == phase_stop_9
                                                                              B_lvl_3_val_4 = B_lvl_3_val_3[B_lvl_2_q]
                                                                              A_lvl_q = (1 - 1) * A_lvl_stop + phase_stop_9
                                                                              A_lvl_2_q_2 = A_lvl_2_ptr_3[A_lvl_q]
                                                                              A_lvl_2_q_stop_2 = A_lvl_2_ptr_3[A_lvl_q + 1]
                                                                              if A_lvl_2_q_2 < A_lvl_2_q_stop_2
                                                                                  A_lvl_2_i1_2 = A_lvl_2_idx_3[A_lvl_2_q_stop_2 - 1]
                                                                              else
                                                                                  A_lvl_2_i1_2 = 0
                                                                              end
                                                                              phase_stop_10 = min(A_lvl_2_stop, A_lvl_2_i1_2)
                                                                              if phase_stop_10 >= 1
                                                                                  if A_lvl_2_idx_3[A_lvl_2_q_2] < 1
                                                                                      A_lvl_2_q_2 = Finch.scansearch(A_lvl_2_idx_3, 1, A_lvl_2_q_2, A_lvl_2_q_stop_2 - 1)
                                                                                  end
                                                                                  while true
                                                                                      A_lvl_2_i_2 = A_lvl_2_idx_3[A_lvl_2_q_2]
                                                                                      if A_lvl_2_i_2 < phase_stop_10
                                                                                          A_lvl_3_val_5 = A_lvl_3_val_3[A_lvl_2_q_2]
                                                                                          w_lvl_q_3 = (1 - 1) * A_lvl_2_stop + A_lvl_2_i_2
                                                                                          w_lvl_2_val_3[w_lvl_q_3] = B_lvl_3_val_4 * A_lvl_3_val_5 + w_lvl_2_val_3[w_lvl_q_3]
                                                                                          if !(w_lvl_tbl_3[w_lvl_q_3])
                                                                                              w_lvl_tbl_3[w_lvl_q_3] = true
                                                                                              w_lvl_qos_fill += 1
                                                                                              if w_lvl_qos_fill > w_lvl_qos_stop
                                                                                                  w_lvl_qos_stop = max(w_lvl_qos_stop << 1, 1)
                                                                                                  Finch.resize_if_smaller!(w_lvl_srt_3, w_lvl_qos_stop)
                                                                                              end
                                                                                              w_lvl_srt_3[w_lvl_qos_fill] = (1, A_lvl_2_i_2)
                                                                                          end
                                                                                          A_lvl_2_q_2 += 1
                                                                                      else
                                                                                          phase_stop_12 = min(phase_stop_10, A_lvl_2_i_2)
                                                                                          if A_lvl_2_i_2 == phase_stop_12
                                                                                              A_lvl_3_val_5 = A_lvl_3_val_3[A_lvl_2_q_2]
                                                                                              w_lvl_q_3 = (1 - 1) * A_lvl_2_stop + phase_stop_12
                                                                                              w_lvl_2_val_3[w_lvl_q_3] += B_lvl_3_val_4 * A_lvl_3_val_5
                                                                                              if !(w_lvl_tbl_3[w_lvl_q_3])
                                                                                                  w_lvl_tbl_3[w_lvl_q_3] = true
                                                                                                  w_lvl_qos_fill += 1
                                                                                                  if w_lvl_qos_fill > w_lvl_qos_stop
                                                                                                      w_lvl_qos_stop = max(w_lvl_qos_stop << 1, 1)
                                                                                                      Finch.resize_if_smaller!(w_lvl_srt_3, w_lvl_qos_stop)
                                                                                                  end
                                                                                                  w_lvl_srt_3[w_lvl_qos_fill] = (1, phase_stop_12)
                                                                                              end
                                                                                              A_lvl_2_q_2 += 1
                                                                                          end
                                                                                          break
                                                                                      end
                                                                                  end
                                                                              end
                                                                              B_lvl_2_q += 1
                                                                          end
                                                                          break
                                                                      end
                                                                  end
                                                              end
                                                              resize!(w_lvl_ptr_3, 1 + 1)
                                                              resize!(w_lvl_tbl_3, 1A_lvl_2_stop)
                                                              resize!(w_lvl_srt_3, w_lvl_qos_fill)
                                                              sort!(w_lvl_srt_3)
                                                              w_lvl_p_prev = 0
                                                              for w_lvl_r_2 = 1:w_lvl_qos_fill
                                                                  w_lvl_p_2 = first(w_lvl_srt_3[w_lvl_r_2])
                                                                  if w_lvl_p_2 != w_lvl_p_prev
                                                                      w_lvl_ptr_3[w_lvl_p_prev + 1] = w_lvl_r_2
                                                                      w_lvl_ptr_3[w_lvl_p_2] = w_lvl_r_2
                                                                  end
                                                                  w_lvl_p_prev = w_lvl_p_2
                                                              end
                                                              w_lvl_ptr_3[w_lvl_p_prev + 1] = w_lvl_qos_fill + 1
                                                              w_lvl_qos_stop = w_lvl_qos_fill
                                                              resize!(w_lvl_2_val_3, A_lvl_2_stop)
                                                              C_lvl_3_qos = C_lvl_3_qos_fill + 1
                                                              C_lvl_3_prev_pos < qos || throw((Finch.FinchProtocolError)("SparseListLevels cannot be updated multiple times"))
                                                              w_lvl_r_3 = w_lvl_ptr_3[1]
                                                              w_lvl_r_stop = w_lvl_ptr_3[1 + 1]
                                                              if w_lvl_r_3 != 0 && w_lvl_r_3 < w_lvl_r_stop
                                                                  w_lvl_i_stop = last(w_lvl_srt_3[w_lvl_r_stop - 1])
                                                              else
                                                                  w_lvl_i_stop = 0
                                                              end
                                                              phase_stop_15 = min(A_lvl_2_stop, w_lvl_i_stop)
                                                              if phase_stop_15 >= 1
                                                                  while w_lvl_r_3 + 1 < w_lvl_r_stop && last(w_lvl_srt_3[w_lvl_r_3]) < 1
                                                                      w_lvl_r_3 += 1
                                                                  end
                                                                  while true
                                                                      w_lvl_i_2 = last(w_lvl_srt_3[w_lvl_r_3])
                                                                      if w_lvl_i_2 < phase_stop_15
                                                                          w_lvl_q_4 = (1 - 1) * A_lvl_2_stop + w_lvl_i_2
                                                                          w_lvl_2_val_4 = w_lvl_2_val_3[w_lvl_q_4]
                                                                          if C_lvl_3_qos > C_lvl_3_qos_stop
                                                                              C_lvl_3_qos_stop = max(C_lvl_3_qos_stop << 1, 1)
                                                                              Finch.resize_if_smaller!(C_lvl_3_idx_5, C_lvl_3_qos_stop)
                                                                              Finch.resize_if_smaller!(C_lvl_4_val_5, C_lvl_3_qos_stop)
                                                                              Finch.fill_range!(C_lvl_4_val_5, 0.0, C_lvl_3_qos, C_lvl_3_qos_stop)
                                                                          end
                                                                          C_lvl_4_val_5[C_lvl_3_qos] = w_lvl_2_val_4
                                                                          C_lvl_3_idx_5[C_lvl_3_qos] = w_lvl_i_2
                                                                          C_lvl_3_qos += 1
                                                                          C_lvl_3_prev_pos = qos
                                                                          w_lvl_r_3 += 1
                                                                      else
                                                                          phase_stop_17 = min(phase_stop_15, w_lvl_i_2)
                                                                          if w_lvl_i_2 == phase_stop_17
                                                                              w_lvl_q_4 = (1 - 1) * A_lvl_2_stop + w_lvl_i_2
                                                                              w_lvl_2_val_5 = w_lvl_2_val_3[w_lvl_q_4]
                                                                              if C_lvl_3_qos > C_lvl_3_qos_stop
                                                                                  C_lvl_3_qos_stop = max(C_lvl_3_qos_stop << 1, 1)
                                                                                  Finch.resize_if_smaller!(C_lvl_3_idx_5, C_lvl_3_qos_stop)
                                                                                  Finch.resize_if_smaller!(C_lvl_4_val_5, C_lvl_3_qos_stop)
                                                                                  Finch.fill_range!(C_lvl_4_val_5, 0.0, C_lvl_3_qos, C_lvl_3_qos_stop)
                                                                              end
                                                                              C_lvl_4_val_5[C_lvl_3_qos] = w_lvl_2_val_5
                                                                              C_lvl_3_idx_5[C_lvl_3_qos] = phase_stop_17
                                                                              C_lvl_3_qos += 1
                                                                              C_lvl_3_prev_pos = qos
                                                                              w_lvl_r_3 += 1
                                                                          end
                                                                          break
                                                                      end
                                                                  end
                                                              end
                                                              C_lvl_3_ptr_5[qos + 1] += (C_lvl_3_qos - C_lvl_3_qos_fill) - 1
                                                              C_lvl_3_qos_fill = C_lvl_3_qos - 1
                                                          end
                                                      end
                                                  end
                                              C_lvl_2_qos_fill_6[tid_2] = C_lvl_2_qos_fill_5
                                              C_lvl_2_qos_stop_6[tid_2] = C_lvl_2_qos_stop_5
                                              resize!(C_lvl_3_ptr_5, C_lvl_2_qos_stop_5 + 1)
                                              for p_3 = 1:C_lvl_2_qos_stop_5
                                                  C_lvl_3_ptr_5[p_3 + 1] += C_lvl_3_ptr_5[p_3]
                                              end
                                              qos_stop_3 = C_lvl_3_ptr_5[C_lvl_2_qos_stop_5 + 1] - 1
                                              resize!(C_lvl_3_idx_5, qos_stop_3)
                                              resize!(C_lvl_4_val_5, qos_stop_3)
                                              res_34
                                          end)
                                  nothing
                              end
                      end
                  (C = Tensor((DenseLevel){Int64}((ShardLevel)(Finch.CPU{:t}(n), (SparseListLevel){Int64}(ElementLevel{0.0, Float64, Int64}(C_lvl_4_val_4), A_lvl_2_stop, C_lvl_3_ptr_4, C_lvl_3_idx_4), C_lvl_2_ptr_2, C_lvl_2_task_2, C_lvl_2_qos_fill_4, C_lvl_2_qos_stop_4, C_lvl_2.schedule), B_lvl_stop)),)
              end)
  end)