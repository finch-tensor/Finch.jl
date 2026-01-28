using Finch

dev = cpu(:t, 2)
tens = Tensor(Dense(Coalesce(dev, SparseList(Element(0.0)))),2,2)

f = :(function run(tens::Tensor{DenseLevel{Int64, CoalesceLevel{CPU{:t}, SparseListLevel{Int64, Finch.MultiChannelBuffer{Vector{Int64}}, Finch.MultiChannelBuffer{Vector{Int64}}, ElementLevel{0.0, Float64, Int64, Finch.MultiChannelBuffer{Vector{Float64}}}}, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}, Finch.FinchStaticSchedule{:dynamic}}}})
      @inbounds @fastmath(begin
                  tens_lvl = tens.lvl
                  tens_lvl_stop = tens_lvl.shape
                  tens_lvl_2 = tens_lvl.lvl
                  n = tens_lvl_2.device.n
                  tens_lvl_3 = tens_lvl_2.lvl
                  tens_lvl_3_ptr = tens_lvl_3.ptr
                  tens_lvl_3_idx = tens_lvl_3.idx
                  tens_lvl_3_stop = tens_lvl_3.shape
                  tens_lvl_4 = tens_lvl_3.lvl
                  tens_lvl_4_val = tens_lvl_4.val
                  tens_lvl_5 = tens_lvl_2.coalescent
                  tens_lvl_5_ptr = tens_lvl_5.ptr
                  tens_lvl_5_idx = tens_lvl_5.idx
                  tens_lvl_5_stop = tens_lvl_5.shape
                  tens_lvl_6 = tens_lvl_5.lvl
                  tens_lvl_6_val = tens_lvl_6.val

                  tens_lvl_4_val_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_4_val)
                  tens_lvl_3_ptr_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_3_ptr)
                  tens_lvl_3_idx_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_3_idx)
                  println(tens_lvl_3_ptr_2)
                  Threads.@threads :dynamic for tid = 1:n
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              tens_lvl_4_val_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_4_val)
                                              tens_lvl_3_ptr_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_3_ptr)
                                              tens_lvl_3_idx_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_3_idx)
                                              println(tens_lvl_3_ptr_3)
                                              resize!(tens_lvl_3_ptr_3, tens_lvl_stop + 1)
                                              println(tens_lvl_3_ptr_3)
                                              for p = 1:tens_lvl_stop
                                                  tens_lvl_3_ptr_3[p + 1] += tens_lvl_3_ptr_3[p]
                                              end
                                              qos_stop = tens_lvl_3_ptr_3[tens_lvl_stop + 1] - 1
                                              resize!(tens_lvl_3_idx_3, qos_stop)
                                              resize!(tens_lvl_4_val_3, qos_stop)
                                              Finch.VirtualSparseListLevel(:tens_lvl_3, Finch.VirtualElementLevel(:tens_lvl_4, 0.0, Float64, Int64, :tens_lvl_4_val_3), Int64, :tens_lvl_3_ptr_3, :tens_lvl_3_idx_3, value(tens_lvl_3_stop, Int64), :tens_lvl_3_qos_fill, :tens_lvl_3_qos_stop, :tens_lvl_3_prev_pos)
                                          end)
                                  nothing
                              end
                      end
                  tens_lvl_4_val = (Finch).transfer(tens_lvl_4_val, tens_lvl_4_val_2)
                  tens_lvl_3_ptr = (Finch).transfer(tens_lvl_3_ptr, tens_lvl_3_ptr_2)
                  tens_lvl_3_idx = (Finch).transfer(tens_lvl_3_idx, tens_lvl_3_idx_2)
                  resize!(tens_lvl_5_ptr, tens_lvl_stop + 1)
                  for p_2 = 1:tens_lvl_stop
                      tens_lvl_5_ptr[p_2 + 1] += tens_lvl_5_ptr[p_2]
                  end
                  qos_stop_2 = tens_lvl_5_ptr[tens_lvl_stop + 1] - 1
                  resize!(tens_lvl_5_idx, qos_stop_2)
                  resize!(tens_lvl_6_val, qos_stop_2)
                  tens_lvl_4_val_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_4_val)
                  tens_lvl_3_ptr_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_3_ptr)
                  tens_lvl_3_idx_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_3_idx)
                  Finch.resize_if_smaller!(tens_lvl_5_ptr, tens_lvl_stop + 1)
                  Finch.fill_range!(tens_lvl_5_ptr, 0, 1 + 1, tens_lvl_stop + 1)
                  resize!(tens_lvl_5_ptr, tens_lvl_stop + 1)
                  for p_4 = 1:tens_lvl_stop
                      tens_lvl_5_ptr[p_4 + 1] += tens_lvl_5_ptr[p_4]
                  end
                  qos_stop_4 = tens_lvl_5_ptr[tens_lvl_stop + 1] - 1
                  resize!(tens_lvl_5_idx, qos_stop_4)
                  resize!(tens_lvl_6_val, qos_stop_4)
                  tens_lvl_4_val = (Finch).transfer(tens_lvl_4_val, tens_lvl_4_val_4)
                  tens_lvl_3_ptr = (Finch).transfer(tens_lvl_3_ptr, tens_lvl_3_ptr_4)
                  tens_lvl_3_idx = (Finch).transfer(tens_lvl_3_idx, tens_lvl_3_idx_4)
                  result = ()
                  tm = collect(1:n)
                  gfm = ones(Int, n)
                  lfm = ones(Int, n)
                  Finch.coalesce_level!((CoalesceLevel)(Finch.CPU{:t}(n), (SparseListLevel){Int64}(ElementLevel{0.0, Float64, Int64}(tens_lvl_4_val), tens_lvl_3_stop, tens_lvl_3_ptr, tens_lvl_3_idx), (SparseListLevel){Int64}(ElementLevel{0.0, Float64, Int64}(tens_lvl_6_val), tens_lvl_5_stop, tens_lvl_5_ptr, tens_lvl_5_idx), tens_lvl_2.schedule), gfm, lfm, tm, 1, n, (SparseListLevel){Int64}(ElementLevel{0.0, Float64, Int64}(tens_lvl_6_val), tens_lvl_5_stop, tens_lvl_5_ptr, tens_lvl_5_idx))
                  result
              end)
  end)

eval(f)
run(tens)
display(tens)