using Finch
using InteractiveUtils

dev1 = cpu(:t, 2)
dev2 = cpu(:q, 2)

tens = Tensor(Dense(Shard(dev1, Coalesce(dev2, SparseList(Element(0.0))))))

code = :(function run(test2::Tensor{DenseLevel{Int64, ShardLevel{CPU{:t}, CoalesceLevel{CPU{:q}, SparseListLevel{Int64, Finch.MultiChannelBuffer{Finch.MultiChannelBuffer{Vector{Int64}}}, Finch.MultiChannelBuffer{Finch.MultiChannelBuffer{Vector{Int64}}}, ElementLevel{0.0, Float64, Int64, Finch.MultiChannelBuffer{Finch.MultiChannelBuffer{Vector{Float64}}}}}, SparseListLevel{Int64, Finch.MultiChannelBuffer{Vector{Int64}}, Finch.MultiChannelBuffer{Vector{Int64}}, ElementLevel{0.0, Float64, Int64, Finch.MultiChannelBuffer{Vector{Float64}}}}, Finch.FinchStaticSchedule{:dynamic}}, Vector{Int64}, Vector{Int64}, Vector{Int64}, Vector{Int64}, Finch.FinchStaticSchedule{:dynamic}}}})
      @inbounds @fastmath(begin
                  test2_lvl = test2.lvl
                  test2_lvl_stop = test2_lvl.shape
                  test2_lvl_2 = test2_lvl.lvl
                  test2_lvl_2_ptr = test2_lvl_2.ptr
                  test2_lvl_2_task = test2_lvl_2.task
                  test2_lvl_2_qos_fill = test2_lvl_2.used
                  test2_lvl_2_qos_stop = test2_lvl_2.alloc
                  n = test2_lvl_2.device.n
                  test2_lvl_3 = test2_lvl_2.lvl
                  n_2 = test2_lvl_3.device.n
                  test2_lvl_4 = test2_lvl_3.lvl
                  test2_lvl_4_ptr = test2_lvl_4.ptr
                  test2_lvl_4_idx = test2_lvl_4.idx
                  test2_lvl_4_stop = test2_lvl_4.shape
                  test2_lvl_5 = test2_lvl_4.lvl
                  test2_lvl_5_val = test2_lvl_5.val
                  test2_lvl_6 = test2_lvl_3.coalescent
                  test2_lvl_6_ptr = test2_lvl_6.ptr
                  test2_lvl_6_idx = test2_lvl_6.idx
                  test2_lvl_6_stop = test2_lvl_6.shape
                  test2_lvl_7 = test2_lvl_6.lvl
                  test2_lvl_7_val = test2_lvl_7.val
                  (Finch.CPUSharedMemory)(Finch.CPU{:t}(n))
                  (Finch.CPUSharedMemory)(Finch.CPU{:t}(n))
                  (Finch.CPUSharedMemory)(Finch.CPU{:t}(n))
                  (Finch.CPUSharedMemory)(Finch.CPU{:t}(n))
                  (Finch.CPUSharedMemory)(Finch.CPU{:t}(n))
                  Threads.@threads :dynamic for tid = 1:n
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              test2_lvl_5_val_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), test2_lvl_5_val)
                                              test2_lvl_4_ptr_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), test2_lvl_4_ptr)
                                              test2_lvl_4_idx_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), test2_lvl_4_idx)
                                              test2_lvl_2_qos_fill_3 = (Finch).transfer((Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)()), test2_lvl_2_qos_fill)
                                              test2_lvl_2_qos_stop_3 = (Finch).transfer((Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)()), test2_lvl_2_qos_stop)
                                              test2_lvl_5_val_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_2)), test2_lvl_5_val_3)
                                              test2_lvl_4_ptr_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_2)), test2_lvl_4_ptr_3)
                                              test2_lvl_4_idx_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_2)), test2_lvl_4_idx_3)
                                              test2_lvl_7_val_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_2)), test2_lvl_7_val)
                                              test2_lvl_6_ptr_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_2)), test2_lvl_6_ptr)
                                              test2_lvl_6_idx_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_2)), test2_lvl_6_idx)
                                              Threads.@threads :dynamic for tid_2 = 1:n_2
                                                      Finch.@barrier begin
                                                              @inbounds @fastmath(begin
                                                                          test2_lvl_5_val_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_2), n_2), (Finch.CPUThread)(tid_2, Finch.CPU{:q}(n_2), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)()))), test2_lvl_5_val_3)
                                                                          test2_lvl_4_ptr_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_2), n_2), (Finch.CPUThread)(tid_2, Finch.CPU{:q}(n_2), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)()))), test2_lvl_4_ptr_3)
                                                                          test2_lvl_4_idx_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_2), n_2), (Finch.CPUThread)(tid_2, Finch.CPU{:q}(n_2), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)()))), test2_lvl_4_idx_3)
                                                                          resize!(test2_lvl_4_ptr_5, 0 + 1)
                                                                          for p = 1:0
                                                                              test2_lvl_4_ptr_5[p + 1] += test2_lvl_4_ptr_5[p]
                                                                          end
                                                                          qos_stop = test2_lvl_4_ptr_5[0 + 1] - 1
                                                                          resize!(test2_lvl_4_idx_5, qos_stop)
                                                                          resize!(test2_lvl_5_val_5, qos_stop)
                                                                          test2_lvl_7_val_3 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_2), n_2), (Finch.CPUThread)(tid_2, Finch.CPU{:q}(n_2), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)()))), test2_lvl_7_val_2)
                                                                          println(@which (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_2), n_2), (Finch.CPUThread)(tid_2, Finch.CPU{:q}(n_2), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)()))), test2_lvl_6_ptr_2))
                                                                          test2_lvl_6_ptr_3 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_2), n_2), (Finch.CPUThread)(tid_2, Finch.CPU{:q}(n_2), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)()))), test2_lvl_6_ptr_2)
                                                                          test2_lvl_6_idx_3 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_2), n_2), (Finch.CPUThread)(tid_2, Finch.CPU{:q}(n_2), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)()))), test2_lvl_6_idx_2)
                                                                          resize!(test2_lvl_6_ptr_3, 0 + 1)
                                                                          for p_2 = 1:0
                                                                              test2_lvl_6_ptr_3[p_2 + 1] += test2_lvl_6_ptr_3[p_2]
                                                                          end
                                                                          qos_stop_2 = test2_lvl_6_ptr_3[0 + 1] - 1
                                                                          resize!(test2_lvl_6_idx_3, qos_stop_2)
                                                                          resize!(test2_lvl_7_val_3, qos_stop_2)
                                                                          Finch.VirtualSparseListLevel(:test2_lvl_6, Finch.VirtualElementLevel(:test2_lvl_7, 0.0, Float64, Int64, :test2_lvl_7_val_3), Int64, :test2_lvl_6_ptr_3, :test2_lvl_6_idx_3, value(test2_lvl_6_stop, Int64), :test2_lvl_6_qos_fill, :test2_lvl_6_qos_stop, :test2_lvl_6_prev_pos)
                                                                      end)
                                                              nothing
                                                          end
                                                  end
                                              test2_lvl_5_val_3 = (Finch).transfer(test2_lvl_5_val_3, test2_lvl_5_val_4)
                                              test2_lvl_4_ptr_3 = (Finch).transfer(test2_lvl_4_ptr_3, test2_lvl_4_ptr_4)
                                              test2_lvl_4_idx_3 = (Finch).transfer(test2_lvl_4_idx_3, test2_lvl_4_idx_4)
                                              test2_lvl_7_val = (Finch).transfer(test2_lvl_7_val, test2_lvl_7_val_2)
                                              test2_lvl_6_ptr = (Finch).transfer(test2_lvl_6_ptr, test2_lvl_6_ptr_2)
                                              test2_lvl_6_idx = (Finch).transfer(test2_lvl_6_idx, test2_lvl_6_idx_2)
                                              tm = collect(1:n_2)
                                              gfm = ones(Int, n_2)
                                              lfm = ones(Int, n_2)
                                              Finch.coalesce_level!((SparseListLevel){Int64}(ElementLevel{0.0, Float64, Int64}(test2_lvl_5_val_3), test2_lvl_4_stop, test2_lvl_4_ptr_3, test2_lvl_4_idx_3), gfm, lfm, tm, 0, n_2, (SparseListLevel){Int64}(ElementLevel{0.0, Float64, Int64}(test2_lvl_7_val), test2_lvl_6_stop, test2_lvl_6_ptr, test2_lvl_6_idx))
                                              test2_lvl_2_qos_fill_3[tid] = 0
                                              test2_lvl_2_qos_stop_3[tid] = 0
                                          end)
                                  nothing
                              end
                      end
                  Finch.resize_if_smaller!(test2_lvl_2_task, test2_lvl_stop)
                  Finch.resize_if_smaller!(test2_lvl_2_ptr, test2_lvl_stop)
                  Finch.fill_range!(test2_lvl_2_ptr, 0, 1, test2_lvl_stop)
                  ()
              end)
  end)

eval(code)
run(tens)