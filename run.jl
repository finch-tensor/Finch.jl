using Finch

dev1 = cpu(:t, 2)
dev2 = cpu(:q, 2)
tens = Tensor(Coalesce(dev1, Coalesce(dev2, Dense(Element(0.0)))))

code = :(function run(tens::Tensor{CoalesceLevel{CPU{:t}, CoalesceLevel{CPU{:q}, DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Finch.MultiChannelBuffer{Finch.MultiChannelBuffer{Vector{Float64}}}}}, DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Finch.MultiChannelBuffer{Vector{Float64}}}}, Finch.FinchStaticSchedule{:dynamic}}, CoalesceLevel{CPU{:q}, DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}, DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}, Finch.FinchStaticSchedule{:dynamic}}, Finch.FinchStaticSchedule{:dynamic}}})
      @inbounds @fastmath(begin
                  tens_lvl = tens.lvl
                  n = tens_lvl.device.n
                  tens_lvl_2 = tens_lvl.lvl
                  tens_lvl_2_coalref = tens_lvl_2.coalescent
                  n_2 = tens_lvl_2.device.n
                  tens_lvl_3 = tens_lvl_2.lvl
                  tens_lvl_3_stop = tens_lvl_3.shape
                  tens_lvl_4 = tens_lvl_3.lvl
                  tens_lvl_4_val = tens_lvl_4.val
                  tens_lvl_5 = tens_lvl_2.coalescent
                  tens_lvl_5_stop = tens_lvl_5.shape
                  tens_lvl_6 = tens_lvl_5.lvl
                  tens_lvl_6_val = tens_lvl_6.val
                  tens_lvl_7 = tens_lvl.coalescent
                  n_3 = tens_lvl_7.device.n
                  tens_lvl_8 = tens_lvl_7.lvl
                  tens_lvl_8_stop = tens_lvl_8.shape
                  tens_lvl_9 = tens_lvl_8.lvl
                  tens_lvl_9_val = tens_lvl_9.val
                  tens_lvl_10 = tens_lvl_7.coalescent
                  tens_lvl_10_stop = tens_lvl_10.shape
                  tens_lvl_11 = tens_lvl_10.lvl
                  tens_lvl_11_val = tens_lvl_11.val
                  tens_lvl_4_val_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_4_val)
                  Threads.@threads :dynamic for tid = 1:n
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              tens_lvl_4_val_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_4_val)
                                              tens_lvl_4_val_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_2)), tens_lvl_4_val_3)
                                              Threads.@threads :dynamic for tid_2 = 1:n_2
                                                      Finch.@barrier begin
                                                              @inbounds @fastmath(begin
                                                                          tens_lvl_4_val_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_2), n_2), (Finch.CPUThread)(tid_2, Finch.CPU{:q}(n_2), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)()))), tens_lvl_4_val_3)
                                                                          resize!(tens_lvl_4_val_5, tens_lvl_3_stop)
                                                                          Finch.VirtualDenseLevel(:tens_lvl_3, Finch.VirtualElementLevel(:tens_lvl_4, 0.0, Float64, Int64, :tens_lvl_4_val_5), Int64, value(tens_lvl_3_stop, Int64))
                                                                      end)
                                                              nothing
                                                          end
                                                  end
                                              tens_lvl_4_val_3 = (Finch).transfer(tens_lvl_4_val_3, tens_lvl_4_val_4)
                                              resize!(tens_lvl_6_val, tens_lvl_5_stop)
                                              tm = collect(1:n_2)
                                              gfm = ones(Int, n_2)
                                              lfm = ones(Int, n_2)
                                              tens_lvl_2_coalref = Finch.coalesce_level!((DenseLevel){Int64}(ElementLevel{0.0, Float64, Int64}(tens_lvl_4_val_3), tens_lvl_3_stop), gfm, lfm, tm, 1, n_2, (DenseLevel){Int64}(ElementLevel{0.0, Float64, Int64}(tens_lvl_6_val), tens_lvl_5_stop))
                                              Finch.VirtualCoalesceLevel(:tens_lvl_2, Finch.VirtualCPU(value(n_2, Int64), :q), Finch.VirtualDenseLevel(:tens_lvl_3, Finch.VirtualElementLevel(:tens_lvl_4, 0.0, Float64, Int64, :tens_lvl_4_val_3), Int64, value(tens_lvl_3_stop, Int64)), Finch.VirtualDenseLevel(:tens_lvl_5, Finch.VirtualElementLevel(:tens_lvl_6, 0.0, Float64, Int64, :tens_lvl_6_val), Int64, value(tens_lvl_5_stop, Int64)), Finch.VirtualFinchStaticSchedule(:dynamic), Float64, CPU{:q}, DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Finch.MultiChannelBuffer{Finch.MultiChannelBuffer{Vector{Float64}}}}}, DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Finch.MultiChannelBuffer{Vector{Float64}}}}, Finch.FinchStaticSchedule{:dynamic}, :tens_lvl_2_qos_stop, :tens_lvl_2_coalref)
                                          end)
                                  nothing
                              end
                      end
                  tens_lvl_4_val = (Finch).transfer(tens_lvl_4_val, tens_lvl_4_val_2)
                  tens_lvl_9_val_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_3)), tens_lvl_9_val)
                  Threads.@threads :dynamic for tid_3 = 1:n_3
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              tens_lvl_9_val_3 = (Finch).transfer((Finch.MemoryChannel)(tid_3, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_3), n_3), (Finch.CPUThread)(tid_3, Finch.CPU{:q}(n_3), (Finch.SerialTask)())), tens_lvl_9_val)
                                              resize!(tens_lvl_9_val_3, tens_lvl_8_stop)
                                              Finch.VirtualDenseLevel(:tens_lvl_8, Finch.VirtualElementLevel(:tens_lvl_9, 0.0, Float64, Int64, :tens_lvl_9_val_3), Int64, value(tens_lvl_8_stop, Int64))
                                          end)
                                  nothing
                              end
                      end
                  tens_lvl_9_val = (Finch).transfer(tens_lvl_9_val, tens_lvl_9_val_2)
                  resize!(tens_lvl_11_val, tens_lvl_10_stop)
                  tm_2 = collect(1:n_3)
                  gfm_2 = ones(Int, n_3)
                  lfm_2 = ones(Int, n_3)
                  Finch.coalesce_level!((DenseLevel){Int64}(ElementLevel{0.0, Float64, Int64}(tens_lvl_9_val), tens_lvl_8_stop), gfm_2, lfm_2, tm_2, 1, n_3, (DenseLevel){Int64}(ElementLevel{0.0, Float64, Int64}(tens_lvl_11_val), tens_lvl_10_stop))
                  tens_lvl_4_val_6 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens_lvl_4_val)
                  Threads.@threads :dynamic for tid_4 = 1:n
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              tens_lvl_4_val_7 = (Finch).transfer((Finch.MemoryChannel)(tid_4, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid_4, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens_lvl_4_val)
                                              tens_lvl_4_val_8 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_2)), tens_lvl_4_val_7)
                                              Threads.@threads :dynamic for tid_5 = 1:n_2
                                                      Finch.@barrier begin
                                                              @inbounds @fastmath(begin
                                                                          tens_lvl_4_val_9 = (Finch).transfer((Finch.MemoryChannel)(tid_5, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_2), n_2), (Finch.CPUThread)(tid_5, Finch.CPU{:q}(n_2), (Finch.CPUThread)(tid_4, Finch.CPU{:t}(n), (Finch.SerialTask)()))), tens_lvl_4_val_7)
                                                                          resize!(tens_lvl_4_val_9, 0)
                                                                          Finch.VirtualDenseLevel(:tens_lvl_3, Finch.VirtualElementLevel(:tens_lvl_4, 0.0, Float64, Int64, :tens_lvl_4_val_9), Int64, value(tens_lvl_3_stop, Int64))
                                                                      end)
                                                              nothing
                                                          end
                                                  end
                                              tens_lvl_4_val_7 = (Finch).transfer(tens_lvl_4_val_7, tens_lvl_4_val_8)
                                              tens_lvl_4_val_10 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_2)), tens_lvl_4_val_7)
                                              Threads.@threads :dynamic for tid_6 = 1:n_2
                                                      Finch.@barrier begin
                                                              @inbounds @fastmath(begin
                                                                          tens_lvl_4_val_11 = (Finch).transfer((Finch.MemoryChannel)(tid_6, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_2), n_2), (Finch.CPUThread)(tid_6, Finch.CPU{:q}(n_2), (Finch.CPUThread)(tid_4, Finch.CPU{:t}(n), (Finch.SerialTask)()))), tens_lvl_4_val_7)
                                                                          Finch.resize_if_smaller!(tens_lvl_4_val_11, tens_lvl_3_stop)
                                                                          Finch.fill_range!(tens_lvl_4_val_11, 0.0, 1, tens_lvl_3_stop)
                                                                          resize!(tens_lvl_4_val_11, tens_lvl_3_stop)
                                                                          Finch.VirtualDenseLevel(:tens_lvl_3, Finch.VirtualElementLevel(:tens_lvl_4, 0.0, Float64, Int64, :tens_lvl_4_val_11), Int64, value(tens_lvl_3_stop, Int64))
                                                                      end)
                                                              nothing
                                                          end
                                                  end
                                              Finch.resize_if_smaller!(tens_lvl_6_val, tens_lvl_5_stop)
                                              Finch.fill_range!(tens_lvl_6_val, 0.0, 1, tens_lvl_5_stop)
                                              resize!(tens_lvl_6_val, tens_lvl_5_stop)
                                              tens_lvl_4_val_7 = (Finch).transfer(tens_lvl_4_val_7, tens_lvl_4_val_10)
                                              tm_3 = collect(1:n_2)
                                              gfm_3 = ones(Int, n_2)
                                              lfm_3 = ones(Int, n_2)
                                              tens_lvl_2_coalref = Finch.coalesce_level!((DenseLevel){Int64}(ElementLevel{0.0, Float64, Int64}(tens_lvl_4_val_7), tens_lvl_3_stop), gfm_3, lfm_3, tm_3, 1, n_2, (DenseLevel){Int64}(ElementLevel{0.0, Float64, Int64}(tens_lvl_6_val), tens_lvl_5_stop))
                                              Finch.VirtualCoalesceLevel(:tens_lvl_2, Finch.VirtualCPU(value(n_2, Int64), :q), Finch.VirtualDenseLevel(:tens_lvl_3, Finch.VirtualElementLevel(:tens_lvl_4, 0.0, Float64, Int64, :tens_lvl_4_val_7), Int64, value(tens_lvl_3_stop, Int64)), Finch.VirtualDenseLevel(:tens_lvl_5, Finch.VirtualElementLevel(:tens_lvl_6, 0.0, Float64, Int64, :tens_lvl_6_val), Int64, value(tens_lvl_5_stop, Int64)), Finch.VirtualFinchStaticSchedule(:dynamic), Float64, CPU{:q}, DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Finch.MultiChannelBuffer{Finch.MultiChannelBuffer{Vector{Float64}}}}}, DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Finch.MultiChannelBuffer{Vector{Float64}}}}, Finch.FinchStaticSchedule{:dynamic}, :tens_lvl_2_qos_stop, :tens_lvl_2_coalref)
                                          end)
                                  nothing
                              end
                      end
                  tens_lvl_9_val_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_3)), tens_lvl_9_val)
                  Threads.@threads :dynamic for tid_7 = 1:n_3
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              tens_lvl_9_val_5 = (Finch).transfer((Finch.MemoryChannel)(tid_7, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_3), n_3), (Finch.CPUThread)(tid_7, Finch.CPU{:q}(n_3), (Finch.SerialTask)())), tens_lvl_9_val)
                                              resize!(tens_lvl_9_val_5, 0)
                                              Finch.VirtualDenseLevel(:tens_lvl_8, Finch.VirtualElementLevel(:tens_lvl_9, 0.0, Float64, Int64, :tens_lvl_9_val_5), Int64, value(tens_lvl_8_stop, Int64))
                                          end)
                                  nothing
                              end
                      end
                  tens_lvl_9_val = (Finch).transfer(tens_lvl_9_val, tens_lvl_9_val_4)
                  tens_lvl_9_val_6 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:q}(n_3)), tens_lvl_9_val)
                  Threads.@threads :dynamic for tid_8 = 1:n_3
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              tens_lvl_9_val_7 = (Finch).transfer((Finch.MemoryChannel)(tid_8, (Finch.MultiChannelMemory)(Finch.CPU{:q}(n_3), n_3), (Finch.CPUThread)(tid_8, Finch.CPU{:q}(n_3), (Finch.SerialTask)())), tens_lvl_9_val)
                                              Finch.resize_if_smaller!(tens_lvl_9_val_7, tens_lvl_8_stop)
                                              Finch.fill_range!(tens_lvl_9_val_7, 0.0, 1, tens_lvl_8_stop)
                                              resize!(tens_lvl_9_val_7, tens_lvl_8_stop)
                                              Finch.VirtualDenseLevel(:tens_lvl_8, Finch.VirtualElementLevel(:tens_lvl_9, 0.0, Float64, Int64, :tens_lvl_9_val_7), Int64, value(tens_lvl_8_stop, Int64))
                                          end)
                                  nothing
                              end
                      end
                  Finch.resize_if_smaller!(tens_lvl_11_val, tens_lvl_10_stop)
                  Finch.fill_range!(tens_lvl_11_val, 0.0, 1, tens_lvl_10_stop)
                  resize!(tens_lvl_11_val, tens_lvl_10_stop)
                  tens_lvl_9_val = (Finch).transfer(tens_lvl_9_val, tens_lvl_9_val_6)
                  tm_4 = collect(1:n_3)
                  gfm_4 = ones(Int, n_3)
                  lfm_4 = ones(Int, n_3)
                  tens_lvl_7_coalref = Finch.coalesce_level!((DenseLevel){Int64}(ElementLevel{0.0, Float64, Int64}(tens_lvl_9_val), tens_lvl_8_stop), gfm_4, lfm_4, tm_4, 1, n_3, (DenseLevel){Int64}(ElementLevel{0.0, Float64, Int64}(tens_lvl_11_val), tens_lvl_10_stop))
                  tens_lvl_4_val = (Finch).transfer(tens_lvl_4_val, tens_lvl_4_val_6)
                  result = ()
                  tm_5 = collect(1:n)
                  gfm_5 = ones(Int, n)
                  lfm_5 = ones(Int, n)
                  Finch.coalesce_level!((CoalesceLevel)(Finch.CPU{:q}(n_2), (DenseLevel){Int64}(ElementLevel{0.0, Float64, Int64}(tens_lvl_4_val), tens_lvl_3_stop), tens_lvl_2_coalref, tens_lvl_2.schedule), gfm_5, lfm_5, tm_5, 1, n, (CoalesceLevel)(Finch.CPU{:q}(n_3), (DenseLevel){Int64}(ElementLevel{0.0, Float64, Int64}(tens_lvl_9_val), tens_lvl_8_stop), tens_lvl_7_coalref, tens_lvl_7.schedule))
                  result
              end)
  end)

eval(code)
run(tens)