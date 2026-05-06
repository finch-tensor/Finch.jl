using Finch
using MatrixDepot
using SparseArrays
using InteractiveUtils

dev1 = cpu(:t, 2)
dev2 = cpu(:q, 2)

tens = Tensor(Dense(Shard(dev1, SparseList(Element(0.0)))))
w = Tensor(Coalesce(dev2, SparseByteMapLevel(Element(0.0))))
w1 = Tensor(SparseByteMapLevel(Element(0.0)))
mtx = matrixdepot("HB/arc130")

code = :(function run(tens::Tensor{DenseLevel{Int64, ShardLevel{CPU{:t}, SparseListLevel{Int64, Finch.MultiChannelBuffer{Vector{Int64}}, Finch.MultiChannelBuffer{Vector{Int64}}, ElementLevel{0.0, Float64, Int64, Finch.MultiChannelBuffer{Vector{Float64}}}}, Vector{Int64}, Vector{Int64}, Vector{Int64}, Vector{Int64}, Finch.FinchStaticSchedule{:dynamic}}}}, w::Tensor{CoalesceLevel{CPU{:q}, SparseByteMapLevel{Int64, Finch.MultiChannelBuffer{Vector{Int64}}, Finch.MultiChannelBuffer{Vector{Bool}}, Finch.MultiChannelBuffer{Vector{Tuple{Int64, Int64}}}, ElementLevel{0.0, Float64, Int64, Finch.MultiChannelBuffer{Vector{Float64}}}}, SparseByteMapLevel{Int64, Vector{Int64}, Vector{Bool}, Vector{Tuple{Int64, Int64}}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}, Finch.FinchStaticSchedule{:dynamic}}}, mtx::SparseArrays.SparseMatrixCSC{Float64, Int64}, dev1::CPU{:t}, dev2::CPU{:q})
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
                  mtx_m = mtx.m
                  mtx_n = mtx.n
                  mtx_ptr = mtx.colptr
                  mtx_idx = mtx.rowval
                  mtx_val = mtx.nzval
                  n_3 = dev1.n
                  n_4 = dev2.n
                  mtx_n == mtx_m || throw(DimensionMismatch("mismatched dimension limits ($(mtx_n) != $(mtx_m))"))
                  tens_lvl_4_val_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_4_val)
                  tens_lvl_3_ptr_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_3_ptr)
                  tens_lvl_3_idx_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_3_idx)
                  tens_lvl_2_qos_fill_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_2_qos_fill)
                  tens_lvl_2_qos_stop_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_2_qos_stop)
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
                  tens_lvl_4_val = (Finch).transfer(tens_lvl_4_val, tens_lvl_4_val_2)
                  tens_lvl_3_ptr = (Finch).transfer(tens_lvl_3_ptr, tens_lvl_3_ptr_2)
                  tens_lvl_3_idx = (Finch).transfer(tens_lvl_3_idx, tens_lvl_3_idx_2)
                  tens_lvl_2_qos_fill = (Finch).transfer(tens_lvl_2_qos_fill, tens_lvl_2_qos_fill_2)
                  tens_lvl_2_qos_stop = (Finch).transfer(tens_lvl_2_qos_stop, tens_lvl_2_qos_stop_2)
                  Finch.resize_if_smaller!(tens_lvl_2_task, mtx_n)
                  Finch.resize_if_smaller!(tens_lvl_2_ptr, mtx_n)
                  Finch.fill_range!(tens_lvl_2_ptr, 0, 1, mtx_n)
                  w_lvl_2_qos_fill_2 = w_lvl_2_qos_fill
                  w_lvl_2_qos_stop_2 = w_lvl_2_qos_stop
                  w_lvl_3_val_2 = (Finch).transfer((CPULocalMemory)(Finch.CPU{:t}(n_3)), w_lvl_3_val)
                  w_lvl_2_ptr_2 = (Finch).transfer((CPULocalMemory)(Finch.CPU{:t}(n_3)), w_lvl_2_ptr)
                  w_lvl_2_tbl_2 = (Finch).transfer((CPULocalMemory)(Finch.CPU{:t}(n_3)), w_lvl_2_tbl)
                  w_lvl_2_srt_2 = (Finch).transfer((CPULocalMemory)(Finch.CPU{:t}(n_3)), w_lvl_2_srt)
                  w_lvl_4_qos_fill_2 = w_lvl_4_qos_fill
                  w_lvl_5_val_2 = (Finch).transfer((CPULocalMemory)(Finch.CPU{:t}(n_3)), w_lvl_5_val)
                  w_lvl_4_ptr_2 = (Finch).transfer((CPULocalMemory)(Finch.CPU{:t}(n_3)), w_lvl_4_ptr)
                  w_lvl_4_tbl_2 = (Finch).transfer((CPULocalMemory)(Finch.CPU{:t}(n_3)), w_lvl_4_tbl)
                  w_lvl_4_srt_2 = (Finch).transfer((CPULocalMemory)(Finch.CPU{:t}(n_3)), w_lvl_4_srt)
                  tens_lvl_4_val_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), tens_lvl_4_val)
                  tens_lvl_3_ptr_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), tens_lvl_3_ptr)
                  tens_lvl_3_idx_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), tens_lvl_3_idx)
                  tens_lvl_2_ptr_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), tens_lvl_2_ptr)
                  tens_lvl_2_task_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), tens_lvl_2_task)
                  tens_lvl_2_qos_fill_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), tens_lvl_2_qos_fill)
                  tens_lvl_2_qos_stop_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n_3)), tens_lvl_2_qos_stop)
                  Threads.@threads :dynamic for tid_2 = 1:n_3
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              w_lvl_2_qos_fill_3 = w_lvl_2_qos_fill_2
                                              w_lvl_2_qos_stop_3 = w_lvl_2_qos_stop_2
                                              w_lvl_3_val_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), w_lvl_3_val_2)
                                              w_lvl_2_ptr_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), w_lvl_2_ptr_2)
                                              w_lvl_2_tbl_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), w_lvl_2_tbl_2)
                                              w_lvl_2_srt_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), w_lvl_2_srt_2)
                                              w_lvl_4_qos_fill_3 = w_lvl_4_qos_fill_2
                                              w_lvl_5_val_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), w_lvl_5_val_2)
                                              w_lvl_4_ptr_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), w_lvl_4_ptr_2)
                                              w_lvl_4_tbl_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), w_lvl_4_tbl_2)
                                              w_lvl_4_srt_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), w_lvl_4_srt_2)
                                              tens_lvl_2_ptr_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), tens_lvl_2_ptr_2)
                                              tens_lvl_2_task_3 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), tens_lvl_2_task_2)
                                              tens_lvl_2_qos_fill_6 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), tens_lvl_2_qos_fill_4)
                                              tens_lvl_2_qos_stop_6 = (Finch).transfer((Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()), tens_lvl_2_qos_stop_4)
                                              tens_lvl_2_qos_fill_5 = tens_lvl_2_qos_fill_6[tid_2]
                                              tens_lvl_2_qos_stop_5 = tens_lvl_2_qos_stop_6[tid_2]
                                              tens_lvl_4_val_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), tens_lvl_4_val_4)
                                              tens_lvl_3_ptr_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), tens_lvl_3_ptr_4)
                                              tens_lvl_3_idx_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), tens_lvl_3_idx_4)
                                              tens_lvl_3_qos_fill = tens_lvl_3_ptr_5[tens_lvl_2_qos_stop_5 + 1] - 1
                                              tens_lvl_3_qos_stop = tens_lvl_3_qos_fill
                                              tens_lvl_3_prev_pos = Finch.scansearch(tens_lvl_3_ptr_5, tens_lvl_3_qos_fill + 1, 1, tens_lvl_2_qos_stop_5) - 1
                                              for p_2 = tens_lvl_2_qos_stop_5:-1:1
                                                  tens_lvl_3_ptr_5[p_2 + 1] = tens_lvl_3_ptr_5[p_2 + 1] - tens_lvl_3_ptr_5[p_2]
                                              end
                                              res_40 = begin
                                                      phase_start_2 = max(1, 1 + fld(mtx_n * (tid_2 + -1), n_3))
                                                      phase_stop_2 = min(mtx_n, fld(mtx_n * tid_2, n_3))
                                                      if phase_stop_2 >= phase_start_2
                                                          for j_6 = phase_start_2:phase_stop_2
                                                              tens_lvl_q = (1 - 1) * mtx_n + j_6
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
                                                              else
                                                                  @assert tens_lvl_2_task_3[tens_lvl_q] == tid_2 "Task mismatch in ShardLevel"
                                                              end
                                                              w_lvl_2_qos_fill_4 = w_lvl_2_qos_fill_3
                                                              w_lvl_2_qos_stop_4 = w_lvl_2_qos_stop_3
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
                                                                                          for w_lvl_2_r = 1:w_lvl_2_qos_fill_3
                                                                                              w_lvl_2_p = first(w_lvl_2_srt_5[w_lvl_2_r])
                                                                                              w_lvl_2_ptr_5[w_lvl_2_p] = 0
                                                                                              w_lvl_2_ptr_5[w_lvl_2_p + 1] = 0
                                                                                              w_lvl_2_i = last(w_lvl_2_srt_5[w_lvl_2_r])
                                                                                              w_lvl_2_q = (w_lvl_2_p - 1) * mtx_m + w_lvl_2_i
                                                                                              w_lvl_2_tbl_5[w_lvl_2_q] = false
                                                                                              Finch.resize_if_smaller!(w_lvl_3_val_5, w_lvl_2_q)
                                                                                              Finch.fill_range!(w_lvl_3_val_5, 0.0, w_lvl_2_q, w_lvl_2_q)
                                                                                          end
                                                                                          w_lvl_2_ptr_5[1] = 1
                                                                                          resize!(w_lvl_2_ptr_5, 0 + 1)
                                                                                          resize!(w_lvl_2_tbl_5, 0mtx_m)
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
                                                                                          resize!(w_lvl_3_val_5, 0)
                                                                                          w_lvl_2_qos_fill_3 = 0
                                                                                          nothing
                                                                                      end)
                                                                              nothing
                                                                          end
                                                                  end
                                                              w_lvl_3_val_3 = (Finch).transfer(w_lvl_3_val_3, w_lvl_3_val_4)
                                                              w_lvl_2_ptr_3 = (Finch).transfer(w_lvl_2_ptr_3, w_lvl_2_ptr_4)
                                                              w_lvl_2_tbl_3 = (Finch).transfer(w_lvl_2_tbl_3, w_lvl_2_tbl_4)
                                                              w_lvl_2_srt_3 = (Finch).transfer(w_lvl_2_srt_3, w_lvl_2_srt_4)
                                                              for w_lvl_4_r = 1:w_lvl_4_qos_fill_3
                                                                  w_lvl_4_p = first(w_lvl_4_srt_3[w_lvl_4_r])
                                                                  w_lvl_4_ptr_3[w_lvl_4_p] = 0
                                                                  w_lvl_4_ptr_3[w_lvl_4_p + 1] = 0
                                                                  w_lvl_4_i = last(w_lvl_4_srt_3[w_lvl_4_r])
                                                                  w_lvl_4_q = (w_lvl_4_p - 1) * mtx_m + w_lvl_4_i
                                                                  w_lvl_4_tbl_3[w_lvl_4_q] = false
                                                                  Finch.resize_if_smaller!(w_lvl_5_val_3, w_lvl_4_q)
                                                                  Finch.fill_range!(w_lvl_5_val_3, 0.0, w_lvl_4_q, w_lvl_4_q)
                                                              end
                                                              w_lvl_4_qos_fill_3 = 0
                                                              w_lvl_4_ptr_3[1] = 1
                                                              resize!(w_lvl_4_ptr_3, 0 + 1)
                                                              resize!(w_lvl_4_tbl_3, 0mtx_m)
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
                                                                                          w_lvl_2q_stop = 1mtx_m
                                                                                          Finch.resize_if_smaller!(w_lvl_2_ptr_7, 1 + 1)
                                                                                          Finch.fill_range!(w_lvl_2_ptr_7, 0, 1 + 1, 1 + 1)
                                                                                          w_lvl_2old = length(w_lvl_2_tbl_7) + 1
                                                                                          Finch.resize_if_smaller!(w_lvl_2_tbl_7, w_lvl_2q_stop)
                                                                                          Finch.fill_range!(w_lvl_2_tbl_7, false, w_lvl_2old, w_lvl_2q_stop)
                                                                                          Finch.resize_if_smaller!(w_lvl_3_val_7, w_lvl_2q_stop)
                                                                                          Finch.fill_range!(w_lvl_3_val_7, 0.0, w_lvl_2old, w_lvl_2q_stop)
                                                                                          resize!(w_lvl_2_ptr_7, 1 + 1)
                                                                                          resize!(w_lvl_2_tbl_7, 1mtx_m)
                                                                                          resize!(w_lvl_2_srt_7, w_lvl_2_qos_fill_4)
                                                                                          sort!(w_lvl_2_srt_7)
                                                                                          w_lvl_2_p_prev_2 = 0
                                                                                          for w_lvl_2_r_3 = 1:w_lvl_2_qos_fill_4
                                                                                              w_lvl_2_p_4 = first(w_lvl_2_srt_7[w_lvl_2_r_3])
                                                                                              if w_lvl_2_p_4 != w_lvl_2_p_prev_2
                                                                                                  w_lvl_2_ptr_7[w_lvl_2_p_prev_2 + 1] = w_lvl_2_r_3
                                                                                                  w_lvl_2_ptr_7[w_lvl_2_p_4] = w_lvl_2_r_3
                                                                                              end
                                                                                              w_lvl_2_p_prev_2 = w_lvl_2_p_4
                                                                                          end
                                                                                          w_lvl_2_ptr_7[w_lvl_2_p_prev_2 + 1] = w_lvl_2_qos_fill_4 + 1
                                                                                          resize!(w_lvl_3_val_7, mtx_m)
                                                                                          nothing
                                                                                      end)
                                                                              nothing
                                                                          end
                                                                  end
                                                              w_lvl_4q_stop = 1mtx_m
                                                              Finch.resize_if_smaller!(w_lvl_4_ptr_3, 1 + 1)
                                                              Finch.fill_range!(w_lvl_4_ptr_3, 0, 1 + 1, 1 + 1)
                                                              w_lvl_4old = length(w_lvl_4_tbl_3) + 1
                                                              Finch.resize_if_smaller!(w_lvl_4_tbl_3, w_lvl_4q_stop)
                                                              Finch.fill_range!(w_lvl_4_tbl_3, false, w_lvl_4old, w_lvl_4q_stop)
                                                              Finch.resize_if_smaller!(w_lvl_5_val_3, w_lvl_4q_stop)
                                                              Finch.fill_range!(w_lvl_5_val_3, 0.0, w_lvl_4old, w_lvl_4q_stop)
                                                              resize!(w_lvl_4_ptr_3, 1 + 1)
                                                              resize!(w_lvl_4_tbl_3, 1mtx_m)
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
                                                              resize!(w_lvl_5_val_3, mtx_m)
                                                              w_lvl_3_val_3 = (Finch).transfer(w_lvl_3_val_3, w_lvl_3_val_6)
                                                              w_lvl_2_ptr_3 = (Finch).transfer(w_lvl_2_ptr_3, w_lvl_2_ptr_6)
                                                              w_lvl_2_tbl_3 = (Finch).transfer(w_lvl_2_tbl_3, w_lvl_2_tbl_6)
                                                              w_lvl_2_srt_3 = (Finch).transfer(w_lvl_2_srt_3, w_lvl_2_srt_6)
                                                              w_lvl_2_qos_fill_8 = w_lvl_2_qos_fill_4
                                                              w_lvl_2_qos_stop_8 = w_lvl_2_qos_stop_4
                                                              w_lvl_3_val_8 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), w_lvl_3_val_3)
                                                              w_lvl_2_ptr_8 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), w_lvl_2_ptr_3)
                                                              w_lvl_2_tbl_8 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), w_lvl_2_tbl_3)
                                                              w_lvl_2_srt_8 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_4)), w_lvl_2_srt_3)
                                                              Threads.@threads :dynamic for tid_5 = 1:n_4
                                                                      Finch.@barrier begin
                                                                              @inbounds @fastmath(begin
                                                                                          w_lvl_3_val_9 = (Finch).transfer((Finch.MemoryChannel)(tid_5, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_4), n_4), (Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))), w_lvl_3_val_8)
                                                                                          w_lvl_2_ptr_9 = (Finch).transfer((Finch.MemoryChannel)(tid_5, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_4), n_4), (Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))), w_lvl_2_ptr_8)
                                                                                          w_lvl_2_tbl_9 = (Finch).transfer((Finch.MemoryChannel)(tid_5, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_4), n_4), (Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))), w_lvl_2_tbl_8)
                                                                                          w_lvl_2_srt_9 = (Finch).transfer((Finch.MemoryChannel)(tid_5, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_4), n_4), (Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))), w_lvl_2_srt_8)
                                                                                          w_lvl_2_qos_fill_10 = w_lvl_2_qos_fill_8
                                                                                          w_lvl_2_qos_stop_10 = w_lvl_2_qos_stop_8
                                                                                          w_lvl_3_val_10 = (Finch).transfer((Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), w_lvl_3_val_8)
                                                                                          (Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)()))
                                                                                          w_lvl_2_tbl_10 = (Finch).transfer((Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), w_lvl_2_tbl_8)
                                                                                          w_lvl_2_srt_10 = (Finch).transfer((Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_4), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n_3), (Finch.SerialTask)())), w_lvl_2_srt_8)
                                                                                          res_25 = begin
                                                                                                  mtx_q = mtx_ptr[j_6]
                                                                                                  mtx_q_stop = mtx_ptr[j_6 + 1]
                                                                                                  if mtx_q < mtx_q_stop
                                                                                                      mtx_i1 = mtx_idx[mtx_q_stop - 1]
                                                                                                  else
                                                                                                      mtx_i1 = 0
                                                                                                  end
                                                                                                  phase_start_4 = max(1, 1 + fld(mtx_n * (-1 + tid_5), n_4))
                                                                                                  phase_stop_4 = min(mtx_n, mtx_i1, fld(mtx_n * tid_5, n_4))
                                                                                                  if phase_stop_4 >= phase_start_4
                                                                                                      if mtx_idx[mtx_q] < phase_start_4
                                                                                                          mtx_q = Finch.scansearch(mtx_idx, phase_start_4, mtx_q, mtx_q_stop - 1)
                                                                                                      end
                                                                                                      while true
                                                                                                          mtx_i = mtx_idx[mtx_q]
                                                                                                          if mtx_i < phase_stop_4
                                                                                                              mtx_val_2 = mtx_val[mtx_q]
                                                                                                              mtx_q_2 = mtx_ptr[mtx_i]
                                                                                                              mtx_q_stop_2 = mtx_ptr[mtx_i + 1]
                                                                                                              if mtx_q_2 < mtx_q_stop_2
                                                                                                                  mtx_i1_2 = mtx_idx[mtx_q_stop_2 - 1]
                                                                                                              else
                                                                                                                  mtx_i1_2 = 0
                                                                                                              end
                                                                                                              phase_stop_6 = min(mtx_m, mtx_i1_2)
                                                                                                              if phase_stop_6 >= 1
                                                                                                                  if mtx_idx[mtx_q_2] < 1
                                                                                                                      mtx_q_2 = Finch.scansearch(mtx_idx, 1, mtx_q_2, mtx_q_stop_2 - 1)
                                                                                                                  end
                                                                                                                  while true
                                                                                                                      mtx_i_2 = mtx_idx[mtx_q_2]
                                                                                                                      if mtx_i_2 < phase_stop_6
                                                                                                                          mtx_val_3 = mtx_val[mtx_q_2]
                                                                                                                          w_lvl_2_q_2 = (1 - 1) * mtx_m + mtx_i_2
                                                                                                                          w_lvl_3_val_10[w_lvl_2_q_2] = mtx_val_2 * mtx_val_3 + w_lvl_3_val_10[w_lvl_2_q_2]
                                                                                                                          if !(w_lvl_2_tbl_10[w_lvl_2_q_2])
                                                                                                                              w_lvl_2_tbl_10[w_lvl_2_q_2] = true
                                                                                                                              w_lvl_2_qos_fill_10 += 1
                                                                                                                              if w_lvl_2_qos_fill_10 > w_lvl_2_qos_stop_10
                                                                                                                                  w_lvl_2_qos_stop_10 = max(w_lvl_2_qos_stop_10 << 1, 1)
                                                                                                                                  Finch.resize_if_smaller!(w_lvl_2_srt_10, w_lvl_2_qos_stop_10)
                                                                                                                              end
                                                                                                                              w_lvl_2_srt_10[w_lvl_2_qos_fill_10] = (1, mtx_i_2)
                                                                                                                          end
                                                                                                                          mtx_q_2 += 1
                                                                                                                      else
                                                                                                                          phase_stop_8 = min(phase_stop_6, mtx_i_2)
                                                                                                                          if mtx_i_2 == phase_stop_8
                                                                                                                              mtx_val_3 = mtx_val[mtx_q_2]
                                                                                                                              w_lvl_2_q_2 = (1 - 1) * mtx_m + phase_stop_8
                                                                                                                              w_lvl_3_val_10[w_lvl_2_q_2] = mtx_val_2 * mtx_val_3 + w_lvl_3_val_10[w_lvl_2_q_2]
                                                                                                                              if !(w_lvl_2_tbl_10[w_lvl_2_q_2])
                                                                                                                                  w_lvl_2_tbl_10[w_lvl_2_q_2] = true
                                                                                                                                  w_lvl_2_qos_fill_10 += 1
                                                                                                                                  if w_lvl_2_qos_fill_10 > w_lvl_2_qos_stop_10
                                                                                                                                      w_lvl_2_qos_stop_10 = max(w_lvl_2_qos_stop_10 << 1, 1)
                                                                                                                                      Finch.resize_if_smaller!(w_lvl_2_srt_10, w_lvl_2_qos_stop_10)
                                                                                                                                  end
                                                                                                                                  w_lvl_2_srt_10[w_lvl_2_qos_fill_10] = (1, phase_stop_8)
                                                                                                                              end
                                                                                                                              mtx_q_2 += 1
                                                                                                                          end
                                                                                                                          break
                                                                                                                      end
                                                                                                                  end
                                                                                                              end
                                                                                                              mtx_q += 1
                                                                                                          else
                                                                                                              phase_stop_10 = min(phase_stop_4, mtx_i)
                                                                                                              if mtx_i == phase_stop_10
                                                                                                                  mtx_val_2 = mtx_val[mtx_q]
                                                                                                                  mtx_q_3 = mtx_ptr[phase_stop_10]
                                                                                                                  mtx_q_stop_3 = mtx_ptr[phase_stop_10 + 1]
                                                                                                                  if mtx_q_3 < mtx_q_stop_3
                                                                                                                      mtx_i1_3 = mtx_idx[mtx_q_stop_3 - 1]
                                                                                                                  else
                                                                                                                      mtx_i1_3 = 0
                                                                                                                  end
                                                                                                                  phase_stop_11 = min(mtx_m, mtx_i1_3)
                                                                                                                  if phase_stop_11 >= 1
                                                                                                                      if mtx_idx[mtx_q_3] < 1
                                                                                                                          mtx_q_3 = Finch.scansearch(mtx_idx, 1, mtx_q_3, mtx_q_stop_3 - 1)
                                                                                                                      end
                                                                                                                      while true
                                                                                                                          mtx_i_3 = mtx_idx[mtx_q_3]
                                                                                                                          if mtx_i_3 < phase_stop_11
                                                                                                                              mtx_val_4 = mtx_val[mtx_q_3]
                                                                                                                              w_lvl_2_q_3 = (1 - 1) * mtx_m + mtx_i_3
                                                                                                                              w_lvl_3_val_10[w_lvl_2_q_3] = mtx_val_2 * mtx_val_4 + w_lvl_3_val_10[w_lvl_2_q_3]
                                                                                                                              if !(w_lvl_2_tbl_10[w_lvl_2_q_3])
                                                                                                                                  w_lvl_2_tbl_10[w_lvl_2_q_3] = true
                                                                                                                                  w_lvl_2_qos_fill_10 += 1
                                                                                                                                  if w_lvl_2_qos_fill_10 > w_lvl_2_qos_stop_10
                                                                                                                                      w_lvl_2_qos_stop_10 = max(w_lvl_2_qos_stop_10 << 1, 1)
                                                                                                                                      Finch.resize_if_smaller!(w_lvl_2_srt_10, w_lvl_2_qos_stop_10)
                                                                                                                                  end
                                                                                                                                  w_lvl_2_srt_10[w_lvl_2_qos_fill_10] = (1, mtx_i_3)
                                                                                                                              end
                                                                                                                              mtx_q_3 += 1
                                                                                                                          else
                                                                                                                              phase_stop_13 = min(phase_stop_11, mtx_i_3)
                                                                                                                              if mtx_i_3 == phase_stop_13
                                                                                                                                  mtx_val_4 = mtx_val[mtx_q_3]
                                                                                                                                  w_lvl_2_q_3 = (1 - 1) * mtx_m + phase_stop_13
                                                                                                                                  w_lvl_3_val_10[w_lvl_2_q_3] = mtx_val_2 * mtx_val_4 + w_lvl_3_val_10[w_lvl_2_q_3]
                                                                                                                                  if !(w_lvl_2_tbl_10[w_lvl_2_q_3])
                                                                                                                                      w_lvl_2_tbl_10[w_lvl_2_q_3] = true
                                                                                                                                      w_lvl_2_qos_fill_10 += 1
                                                                                                                                      if w_lvl_2_qos_fill_10 > w_lvl_2_qos_stop_10
                                                                                                                                          w_lvl_2_qos_stop_10 = max(w_lvl_2_qos_stop_10 << 1, 1)
                                                                                                                                          Finch.resize_if_smaller!(w_lvl_2_srt_10, w_lvl_2_qos_stop_10)
                                                                                                                                      end
                                                                                                                                      w_lvl_2_srt_10[w_lvl_2_qos_fill_10] = (1, phase_stop_13)
                                                                                                                                  end
                                                                                                                                  mtx_q_3 += 1
                                                                                                                              end
                                                                                                                              break
                                                                                                                          end
                                                                                                                      end
                                                                                                                  end
                                                                                                                  mtx_q += 1
                                                                                                              end
                                                                                                              break
                                                                                                          end
                                                                                                      end
                                                                                                  end
                                                                                                  phase_start_15 = max(1, 1 + fld(mtx_n * tid_5, n_4), 1 + mtx_i1)
                                                                                                  if mtx_n >= phase_start_15
                                                                                                      mtx_n + 1
                                                                                                  end
                                                                                              end
                                                                                          resize!(w_lvl_2_ptr_9, 1 + 1)
                                                                                          resize!(w_lvl_2_tbl_9, 1mtx_m)
                                                                                          resize!(w_lvl_2_srt_9, w_lvl_2_qos_fill_8)
                                                                                          sort!(w_lvl_2_srt_9)
                                                                                          w_lvl_2_p_prev_3 = 0
                                                                                          for w_lvl_2_r_4 = 1:w_lvl_2_qos_fill_8
                                                                                              w_lvl_2_p_6 = first(w_lvl_2_srt_9[w_lvl_2_r_4])
                                                                                              if w_lvl_2_p_6 != w_lvl_2_p_prev_3
                                                                                                  w_lvl_2_ptr_9[w_lvl_2_p_prev_3 + 1] = w_lvl_2_r_4
                                                                                                  w_lvl_2_ptr_9[w_lvl_2_p_6] = w_lvl_2_r_4
                                                                                              end
                                                                                              w_lvl_2_p_prev_3 = w_lvl_2_p_6
                                                                                          end
                                                                                          w_lvl_2_ptr_9[w_lvl_2_p_prev_3 + 1] = w_lvl_2_qos_fill_8 + 1
                                                                                          resize!(w_lvl_3_val_9, mtx_m * 1)
                                                                                          w_lvl_2_qos_fill_8 = w_lvl_2_qos_fill_10
                                                                                          w_lvl_2_qos_stop_8 = w_lvl_2_qos_stop_10
                                                                                          println(tid_5)
                                                                                          println(w_lvl_2_qos_stop_10)
                                                                                          println(w_lvl_2_qos_stop_8)
                                                                                          println("--------------------")
                                                                                          res_25
                                                                                      end)
                                                                              nothing
                                                                          end
                                                                  end
                                                              println(w_lvl_2_qos_fill_8)
                                                              w_lvl_2_qos_fill_3 = w_lvl_2_qos_fill_8
                                                              w_lvl_2_qos_stop_3 = w_lvl_2_qos_stop_8
                                                              w_lvl_3_val_3 = (Finch).transfer(w_lvl_3_val_3, w_lvl_3_val_8)
                                                              w_lvl_2_ptr_3 = (Finch).transfer(w_lvl_2_ptr_3, w_lvl_2_ptr_8)
                                                              w_lvl_2_tbl_3 = (Finch).transfer(w_lvl_2_tbl_3, w_lvl_2_tbl_8)
                                                              w_lvl_2_srt_3 = (Finch).transfer(w_lvl_2_srt_3, w_lvl_2_srt_8)
                                                              tm = collect(1:n_2)
                                                              gfm = ones(Int, n_2)
                                                              lfm = ones(Int, n_2)
                                                              Finch.coalesce_level!((SparseByteMapLevel){Int64}(ElementLevel{0.0, Float64, Int64}(w_lvl_3_val_8), mtx_m, w_lvl_2_ptr_8, w_lvl_2_tbl_8, w_lvl_2_srt_8), gfm, lfm, tm, 1, n_2, (SparseByteMapLevel){Int64}(ElementLevel{0.0, Float64, Int64}(w_lvl_5_val_3), mtx_m, w_lvl_4_ptr_3, w_lvl_4_tbl_3, w_lvl_4_srt_3))
                                                              tens_lvl_3_qos = tens_lvl_3_qos_fill + 1
                                                              tens_lvl_3_prev_pos < qos || throw((Finch.FinchProtocolError)("SparseListLevels cannot be updated multiple times"))
                                                              w_lvl_4_r_4 = w_lvl_4_ptr_3[1]
                                                              w_lvl_4_r_stop = w_lvl_4_ptr_3[1 + 1]
                                                              if w_lvl_4_r_4 != 0 && w_lvl_4_r_4 < w_lvl_4_r_stop
                                                                  w_lvl_4_i_stop = last(w_lvl_4_srt_3[w_lvl_4_r_stop - 1])
                                                              else
                                                                  w_lvl_4_i_stop = 0
                                                              end
                                                              phase_stop_19 = min(mtx_m, w_lvl_4_i_stop)
                                                              if phase_stop_19 >= 1
                                                                  while w_lvl_4_r_4 + 1 < w_lvl_4_r_stop && last(w_lvl_4_srt_3[w_lvl_4_r_4]) < 1
                                                                      w_lvl_4_r_4 += 1
                                                                  end
                                                                  while true
                                                                      w_lvl_4_i_2 = last(w_lvl_4_srt_3[w_lvl_4_r_4])
                                                                      if w_lvl_4_i_2 < phase_stop_19
                                                                          w_lvl_4_q_2 = (1 - 1) * mtx_m + w_lvl_4_i_2
                                                                          w_lvl_5_val_4 = w_lvl_5_val_3[w_lvl_4_q_2]
                                                                          if tens_lvl_3_qos > tens_lvl_3_qos_stop
                                                                              tens_lvl_3_qos_stop = max(tens_lvl_3_qos_stop << 1, 1)
                                                                              Finch.resize_if_smaller!(tens_lvl_3_idx_5, tens_lvl_3_qos_stop)
                                                                              Finch.resize_if_smaller!(tens_lvl_4_val_5, tens_lvl_3_qos_stop)
                                                                              Finch.fill_range!(tens_lvl_4_val_5, 0.0, tens_lvl_3_qos, tens_lvl_3_qos_stop)
                                                                          end
                                                                          tens_lvl_4_val_5[tens_lvl_3_qos] = w_lvl_5_val_4
                                                                          tens_lvl_3_idx_5[tens_lvl_3_qos] = w_lvl_4_i_2
                                                                          tens_lvl_3_qos += 1
                                                                          tens_lvl_3_prev_pos = qos
                                                                          w_lvl_4_r_4 += 1
                                                                      else
                                                                          phase_stop_21 = min(phase_stop_19, w_lvl_4_i_2)
                                                                          if w_lvl_4_i_2 == phase_stop_21
                                                                              w_lvl_4_q_2 = (1 - 1) * mtx_m + w_lvl_4_i_2
                                                                              w_lvl_5_val_5 = w_lvl_5_val_3[w_lvl_4_q_2]
                                                                              if tens_lvl_3_qos > tens_lvl_3_qos_stop
                                                                                  tens_lvl_3_qos_stop = max(tens_lvl_3_qos_stop << 1, 1)
                                                                                  Finch.resize_if_smaller!(tens_lvl_3_idx_5, tens_lvl_3_qos_stop)
                                                                                  Finch.resize_if_smaller!(tens_lvl_4_val_5, tens_lvl_3_qos_stop)
                                                                                  Finch.fill_range!(tens_lvl_4_val_5, 0.0, tens_lvl_3_qos, tens_lvl_3_qos_stop)
                                                                              end
                                                                              tens_lvl_4_val_5[tens_lvl_3_qos] = w_lvl_5_val_5
                                                                              tens_lvl_3_idx_5[tens_lvl_3_qos] = phase_stop_21
                                                                              tens_lvl_3_qos += 1
                                                                              tens_lvl_3_prev_pos = qos
                                                                              w_lvl_4_r_4 += 1
                                                                          end
                                                                          break
                                                                      end
                                                                  end
                                                              end
                                                              tens_lvl_3_ptr_5[qos + 1] += (tens_lvl_3_qos - tens_lvl_3_qos_fill) - 1
                                                              tens_lvl_3_qos_fill = tens_lvl_3_qos - 1
                                                          end
                                                      end
                                                      phase_start_19 = max(1, 1 + fld(mtx_n * tid_2, n_3))
                                                      if mtx_n >= phase_start_19
                                                          mtx_n + 1
                                                      end
                                                  end
                                              w_lvl_2_qos_fill_2 = w_lvl_2_qos_fill_3
                                              w_lvl_2_qos_stop_2 = w_lvl_2_qos_stop_3
                                              w_lvl_4_qos_fill_2 = w_lvl_4_qos_fill_3
                                              tens_lvl_2_qos_fill_6[tid_2] = tens_lvl_2_qos_fill_5
                                              tens_lvl_2_qos_stop_6[tid_2] = tens_lvl_2_qos_stop_5
                                              resize!(tens_lvl_3_ptr_5, tens_lvl_2_qos_stop_5 + 1)
                                              for p_3 = 1:tens_lvl_2_qos_stop_5
                                                  tens_lvl_3_ptr_5[p_3 + 1] += tens_lvl_3_ptr_5[p_3]
                                              end
                                              qos_stop_3 = tens_lvl_3_ptr_5[tens_lvl_2_qos_stop_5 + 1] - 1
                                              resize!(tens_lvl_3_idx_5, qos_stop_3)
                                              resize!(tens_lvl_4_val_5, qos_stop_3)
                                              res_40
                                          end)
                                  nothing
                              end
                      end
                  (tens = Tensor((DenseLevel){Int64}((ShardLevel)(Finch.CPU{:t}(n), (SparseListLevel){Int64}(ElementLevel{0.0, Float64, Int64}(tens_lvl_4_val_4), mtx_m, tens_lvl_3_ptr_4, tens_lvl_3_idx_4), tens_lvl_2_ptr_2, tens_lvl_2_task_2, tens_lvl_2_qos_fill_4, tens_lvl_2_qos_stop_4, tens_lvl_2.schedule), mtx_n)),)
              end)
  end)

  eval(code)

  out = run(tens, w, mtx, dev1, dev2)