using Finch

dev = cpu(1, 4)

tens = Tensor(Dense(Shard(dev, Sparse(Element(0.0)))), 4, 4)

f = :(function run(tens)
      @inbounds @fastmath(begin
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
                  tens_lvl_3_val = tens_lvl_3.val
                  tens_lvl_3_tbl = tens_lvl_3.tbl
                  tens_lvl_3_pool = tens_lvl_3.pool
                  tens_lvl_3_stop = tens_lvl_3.shape
                  tens_lvl_4 = tens_lvl_3.lvl
                  tens_lvl_4_val = tens_lvl_4.val

                  n_2 = 2

                  tens_lvl_3_stop == 4 || throw(DimensionMismatch("mismatched dimension limits ($(tens_lvl_3_stop) != $(4))"))
                  4 == tens_lvl_stop || throw(DimensionMismatch("mismatched dimension limits ($(4) != $(tens_lvl_stop))"))
                  tens_lvl_4_val_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU{1}(n_2)), tens_lvl_4_val)
                  tens_lvl_3_ptr_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU{1}(n_2)), tens_lvl_3_ptr)
                  tens_lvl_3_idx_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU{1}(n_2)), tens_lvl_3_idx)
                  tens_lvl_3_tbl_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU{1}(n_2)), tens_lvl_3_tbl)
                  tens_lvl_2_ptr_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU{1}(n_2)), tens_lvl_2_ptr)
                  tens_lvl_2_task_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU{1}(n_2)), tens_lvl_2_task)
                  tens_lvl_2_qos_fill_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU{1}(n_2)), tens_lvl_2_qos_fill)
                  tens_lvl_2_qos_stop_2 = (Finch).transfer(Finch.CPUSharedMemory(Finch.CPU{1}(n_2)), tens_lvl_2_qos_stop)
                  Threads.@threads :dynamic for tid = 1:n_2
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              tens_lvl_2_qos_fill_3 = tens_lvl_2_qos_fill_2[tid]
                                              tens_lvl_2_qos_stop_3 = tens_lvl_2_qos_stop_2[tid]
                                              tens_lvl_4_val_3 = (Finch).transfer(Finch.MemoryChannel(tid, Finch.MultiChannelMemory(Finch.CPU{1}(n_2), n_2), Finch.CPUThread(tid, Finch.CPU{1}(n_2), Finch.SerialTask())), tens_lvl_4_val_2)
                                              tens_lvl_3_ptr_3 = (Finch).transfer(Finch.MemoryChannel(tid, Finch.MultiChannelMemory(Finch.CPU{1}(n_2), n_2), Finch.CPUThread(tid, Finch.CPU{1}(n_2), Finch.SerialTask())), tens_lvl_3_ptr_2)
                                              tens_lvl_3_idx_3 = (Finch).transfer(Finch.MemoryChannel(tid, Finch.MultiChannelMemory(Finch.CPU{1}(n_2), n_2), Finch.CPUThread(tid, Finch.CPU{1}(n_2), Finch.SerialTask())), tens_lvl_3_idx_2)
                                              tens_lvl_3_tbl_3 = (Finch).transfer(Finch.MemoryChannel(tid, Finch.MultiChannelMemory(Finch.CPU{1}(n_2), n_2), Finch.CPUThread(tid, Finch.CPU{1}(n_2), Finch.SerialTask())), tens_lvl_3_tbl_2)
                                              tens_lvl_3_qos_stop = tens_lvl_3_ptr_3[tens_lvl_2_qos_stop_3 + 1] - 1
                                              tens_lvl_2_ptr_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU{1}(n_2), Finch.SerialTask()), tens_lvl_2_ptr_2)
                                              tens_lvl_2_task_3 = (Finch).transfer(Finch.CPUThread(tid, Finch.CPU{1}(n_2), Finch.SerialTask()), tens_lvl_2_task_2)
                                              Finch.CPU{1}(n_2)
                                              Finch.CPU{1}(n_2)
                                              res_5 = begin
                                                      phase_start_2 = max(1, 1 + fld(4 * (tid + -1), n_2))
                                                      phase_stop_2 = min(4, fld(4tid, n_2))
                                                      if phase_stop_2 >= phase_start_2
                                                          for j_6 = phase_start_2:phase_stop_2
                                                              tens_lvl_q = (1 - 1) * tens_lvl_stop + j_6
                                                              qos = tens_lvl_2_ptr_3[tens_lvl_q]
                                                              if qos == 0
                                                                  qos = (tens_lvl_2_qos_fill_3 += 1)
                                                                  tens_lvl_2_task_3[tens_lvl_q] = tid
                                                                  tens_lvl_2_ptr_3[tens_lvl_q] = tens_lvl_2_qos_fill_3
                                                                  if tens_lvl_2_qos_fill_3 > tens_lvl_2_qos_stop_3
                                                                      tens_lvl_2_qos_stop_3 = max(tens_lvl_2_qos_stop_3 << 1, 1)
                                                                  end
                                                              else
                                                                  @assert tens_lvl_2_task_3[tens_lvl_q] == tid "Task mismatch in ShardLevel"
                                                              end
                                                              for i_4 = 1:tens_lvl_3_stop
                                                                  tens_lvl_3_qos = get(tens_lvl_3_tbl_3, (qos, i_4), 0)
                                                                  if tens_lvl_3_qos == 0
                                                                      if !(isempty(tens_lvl_3_pool))
                                                                          tens_lvl_3_qos = pop!(tens_lvl_3_pool)
                                                                      else
                                                                          tens_lvl_3_qos = length(tens_lvl_3_tbl_3) + 1
                                                                          if tens_lvl_3_qos > tens_lvl_3_qos_stop
                                                                              tens_lvl_3_qos_stop = max(tens_lvl_3_qos_stop << 1, 1)
                                                                              Finch.resize_if_smaller!(tens_lvl_4_val_3, tens_lvl_3_qos_stop)
                                                                              Finch.fill_range!(tens_lvl_4_val_3, 0.0, tens_lvl_3_qos, tens_lvl_3_qos_stop)
                                                                              Finch.resize_if_smaller!(tens_lvl_3_val, tens_lvl_3_qos_stop)
                                                                              Finch.fill_range!(tens_lvl_3_val, 0, tens_lvl_3_qos, tens_lvl_3_qos_stop)
                                                                          end
                                                                      end
                                                                      tens_lvl_3_tbl_3[(qos, i_4)] = tens_lvl_3_qos
                                                                  end
                                                                  tens_lvl_4_val_3[tens_lvl_3_qos] = j_6
                                                                  tens_lvl_3_val[tens_lvl_3_qos] = tens_lvl_3_qos
                                                              end
                                                          end
                                                      end
                                                      phase_start_3 = max(1, 1 + fld(4tid, n_2))
                                                      if 4 >= phase_start_3
                                                          4 + 1
                                                      end
                                                  end
                                              resize!(tens_lvl_3_ptr_3, tens_lvl_2_qos_stop_3 + 1)
                                              tens_lvl_3_ptr_3[1] = 1
                                              Finch.fill_range!(tens_lvl_3_ptr_3, 0, 2, tens_lvl_2_qos_stop_3 + 1)
                                              pdx_tmp = Vector{Int64}(undef, length(tens_lvl_3_tbl_3))
                                              resize!(tens_lvl_3_idx_3, length(tens_lvl_3_tbl_3))
                                              resize!(tens_lvl_3_val, length(tens_lvl_3_tbl_3))
                                              idx_tmp = Vector{Int64}(undef, length(tens_lvl_3_tbl_3))
                                              val_tmp = Vector{Int64}(undef, length(tens_lvl_3_tbl_3))
                                              q = 0
                                              for entry = pairs(tens_lvl_3_tbl_3)
                                                  sugar_2 = entry[1]
                                                  p_3 = sugar_2[1]
                                                  i_3 = sugar_2[2]
                                                  v = entry[2]
                                                  q += 1
                                                  idx_tmp[q] = i_3
                                                  val_tmp[q] = v
                                                  pdx_tmp[q] = p_3
                                                  tens_lvl_3_ptr_3[p_3 + 1] += 1
                                              end
                                              for p_3 = 2:tens_lvl_2_qos_stop_3 + 1
                                                  tens_lvl_3_ptr_3[p_3] += tens_lvl_3_ptr_3[p_3 - 1]
                                              end
                                              perm = sortperm(idx_tmp)
                                              ptr_2 = copy(tens_lvl_3_ptr_3)
                                              for q = perm
                                                  p_3 = pdx_tmp[q]
                                                  r = ptr_2[p_3]
                                                  tens_lvl_3_idx_3[r] = idx_tmp[q]
                                                  tens_lvl_3_val[r] = val_tmp[q]
                                                  ptr_2[p_3] += 1
                                              end
                                              qos_stop = tens_lvl_3_ptr_3[tens_lvl_2_qos_stop_3 + 1] - 1
                                              resize!(tens_lvl_4_val_3, qos_stop)
                                              res_5
                                          end)
                                  nothing
                              end
                      end
                  ()
              end)
  end)

eval(f)
run(tens)
println(tens.lvl.lvl.lvl.tbl)
display(tens)