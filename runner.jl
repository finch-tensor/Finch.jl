using Finch

dev = cpu(:t, 2)
tens = Tensor(Coalesce(dev, Dense(SparseList(Element(0.0)))))

func = :(function run(tens::Tensor{CoalesceLevel{CPU{:t}, DenseLevel{Int64, SparseListLevel{Int64, Finch.MultiChannelBuffer{Vector{Int64}}, Finch.MultiChannelBuffer{Vector{Int64}}, ElementLevel{0.0, Float64, Int64, Finch.MultiChannelBuffer{Vector{Float64}}}}}, DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, Finch.FinchStaticSchedule{:dynamic}}})
      @inbounds @fastmath(begin
                  tens_lvl = tens.lvl
                  n = tens_lvl.device.n
                  tens_lvl_2 = tens_lvl.lvl
                  tens_lvl_3 = tens_lvl_2.lvl
                  tens_lvl_3_ptr = tens_lvl_3.ptr
                  println(tens_lvl_3.ptr)
                  tens_lvl_3_idx = tens_lvl_3.idx
                  tens_lvl_4 = tens_lvl_3.lvl
                  tens_lvl_4_val = tens_lvl_4.val
                  tens_lvl_5 = tens_lvl.coalescent
                  tens_lvl_5_stop = tens_lvl_5.shape
                  tens_lvl_6 = tens_lvl_5.lvl
                  tens_lvl_6_ptr = tens_lvl_6.ptr
                  tens_lvl_6_idx = tens_lvl_6.idx
                  tens_lvl_7 = tens_lvl_6.lvl
                  tens_lvl_7_val = tens_lvl_7.val
                  tens_lvl_4_val_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_4_val)
                  tens_lvl_3_ptr_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_3_ptr)
                  tens_lvl_3_idx_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_3_idx)
                  Threads.@threads :dynamic for tid = 1:n
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              tens_lvl_4_val_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_4_val)
                                              tens_lvl_3_ptr_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_3_ptr)
                                              tens_lvl_3_idx_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_3_idx)
                                              println(tens_lvl_3.ptr)
                                              resize!(tens_lvl_3_ptr_3, 4 + 1)
                                              for p = 1:4
                                                  tens_lvl_3_ptr_3[p + 1] += tens_lvl_3_ptr_3[p]
                                              end
                                              println(tens_lvl_3.ptr)
                                              qos_stop = tens_lvl_3_ptr_3[4 + 1] - 1
                                              resize!(tens_lvl_3_idx_3, qos_stop)
                                              resize!(tens_lvl_4_val_3, qos_stop)
                                              Finch.VirtualDenseLevel(:tens_lvl_2, Finch.VirtualSparseListLevel(:tens_lvl_3, Finch.VirtualElementLevel(:tens_lvl_4, 0.0, Float64, Int64, :tens_lvl_4_val_3), Int64, :tens_lvl_3_ptr_3, :tens_lvl_3_idx_3, Finch.literal(4), :tens_lvl_3_qos_fill, :tens_lvl_3_qos_stop, :tens_lvl_3_prev_pos), Int64, Finch.literal(4))
                                          end)
                                  nothing
                              end
                      end
                  tens_lvl_4_val = (Finch).transfer(tens_lvl_4_val, tens_lvl_4_val_2)
                  tens_lvl_3_ptr = (Finch).transfer(tens_lvl_3_ptr, tens_lvl_3_ptr_2)
                  tens_lvl_3_idx = (Finch).transfer(tens_lvl_3_idx, tens_lvl_3_idx_2)
                  resize!(tens_lvl_6_ptr, tens_lvl_5_stop + 1)
                  for p_2 = 1:tens_lvl_5_stop
                      tens_lvl_6_ptr[p_2 + 1] += tens_lvl_6_ptr[p_2]
                  end
                  qos_stop_2 = tens_lvl_6_ptr[tens_lvl_5_stop + 1] - 1
                  resize!(tens_lvl_6_idx, qos_stop_2)
                  resize!(tens_lvl_7_val, qos_stop_2)
                  (Finch.CPUSharedMemory)(Finch.CPU{:t}(n))
                  (Finch.CPUSharedMemory)(Finch.CPU{:t}(n))
                  (Finch.CPUSharedMemory)(Finch.CPU{:t}(n))
                  Threads.@threads :dynamic for tid_2 = 1:n
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              tens_lvl_4_val_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_4_val)
                                              tens_lvl_3_ptr_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_3_ptr)
                                              tens_lvl_3_idx_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_3_idx)
                                              for p_3 = 4:-1:1
                                                  tens_lvl_3_ptr_5[p_3 + 1] = tens_lvl_3_ptr_5[p_3 + 1] - tens_lvl_3_ptr_5[p_3]
                                              end
                                              resize!(tens_lvl_3_ptr_5, 4 + 1)
                                              for p_4 = 1:4
                                                  tens_lvl_3_ptr_5[p_4 + 1] += tens_lvl_3_ptr_5[p_4]
                                              end
                                              qos_stop_4 = tens_lvl_3_ptr_5[4 + 1] - 1
                                              resize!(tens_lvl_3_idx_5, qos_stop_4)
                                              resize!(tens_lvl_4_val_5, qos_stop_4)
                                              Finch.VirtualDenseLevel(:tens_lvl_2, Finch.VirtualSparseListLevel(:tens_lvl_3, Finch.VirtualElementLevel(:tens_lvl_4, 0.0, Float64, Int64, :tens_lvl_4_val_5), Int64, :tens_lvl_3_ptr_5, :tens_lvl_3_idx_5, Finch.literal(4), :tens_lvl_3_qos_fill, :tens_lvl_3_qos_stop, :tens_lvl_3_prev_pos), Int64, Finch.literal(4))
                                          end)
                                  nothing
                              end
                      end
                  ()
              end)
  end)

eval(func)
run(tens)
display(tens)