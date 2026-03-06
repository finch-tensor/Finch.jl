using Finch
dev1 = cpu(:t, 2)
tens2 = Tensor(Coalesce(dev1, Dense(Element(0.0))))

code = :(function run(tens2::Tensor{CoalesceLevel{CPU{:t}, DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Finch.MultiChannelBuffer{Vector{Float64}}}}, DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}, Finch.FinchStaticSchedule{:dynamic}}})
      @inbounds @fastmath(begin
                  tens2_lvl = tens2.lvl
                  n = tens2_lvl.device.n
                  tens2_lvl_2 = tens2_lvl.lvl
                  tens2_lvl_2_stop = tens2_lvl_2.shape
                  tens2_lvl_3 = tens2_lvl_2.lvl
                  tens2_lvl_3_val = tens2_lvl_3.val
                  tens2_lvl_4 = tens2_lvl.coalescent
                  tens2_lvl_4_stop = tens2_lvl_4.shape
                  tens2_lvl_5 = tens2_lvl_4.lvl
                  tens2_lvl_5_val = tens2_lvl_5.val
                  tens2_lvl_3_val_2 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens2_lvl_3_val)
                  tens2_lvl_coalref = (DenseLevel){Int64}(ElementLevel{0.0, Float64, Int64}(tens2_lvl_5_val), tens2_lvl_4_stop)
                  Threads.@threads :dynamic for tid = 1:n
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              tens2_lvl_3_val_3 = (Finch).transfer((Finch.MemoryChannel)(tid, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens2_lvl_3_val)
                                              resize!(tens2_lvl_3_val_3, tens2_lvl_2_stop)
                                              Finch.VirtualDenseLevel(:tens2_lvl_2, Finch.VirtualElementLevel(:tens2_lvl_3, 0.0, Float64, Int64, :tens2_lvl_3_val_3), Int64, value(tens2_lvl_2_stop, Int64))
                                          end)
                                  nothing
                              end
                      end
                  tens2_lvl_3_val = (Finch).transfer(tens2_lvl_3_val, tens2_lvl_3_val_2)
                  resize!(tens2_lvl_5_val, tens2_lvl_4_stop)
                  tens2_lvl_3_val_4 = (Finch).transfer((Finch.CPUSharedMemory)(Finch.CPU{:t}(n)), tens2_lvl_3_val)
                  Threads.@threads :dynamic for tid_2 = 1:n
                          Finch.@barrier begin
                                  @inbounds @fastmath(begin
                                              tens2_lvl_3_val_5 = (Finch).transfer((Finch.MemoryChannel)(tid_2, (Finch.MultiChannelMemory)(Finch.CPU{:t}(n), n), (Finch.CPUThread)(tid_2, Finch.CPU{:t}(n), (Finch.SerialTask)())), tens2_lvl_3_val)
                                              Finch.resize_if_smaller!(tens2_lvl_3_val_5, tens2_lvl_2_stop)
                                              Finch.fill_range!(tens2_lvl_3_val_5, 0.0, 1, tens2_lvl_2_stop)
                                              resize!(tens2_lvl_3_val_5, tens2_lvl_2_stop)
                                              Finch.VirtualDenseLevel(:tens2_lvl_2, Finch.VirtualElementLevel(:tens2_lvl_3, 0.0, Float64, Int64, :tens2_lvl_3_val_5), Int64, value(tens2_lvl_2_stop, Int64))
                                          end)
                                  nothing
                              end
                      end
                  Finch.resize_if_smaller!(tens2_lvl_5_val, tens2_lvl_4_stop)
                  Finch.fill_range!(tens2_lvl_5_val, 0.0, 1, tens2_lvl_4_stop)
                  resize!(tens2_lvl_5_val, tens2_lvl_4_stop)
                  tens2_lvl_3_val = (Finch).transfer(tens2_lvl_3_val, tens2_lvl_3_val_4)
                  result = ()
                  tm = collect(1:n)
                  gfm = ones(Int, n)
                  lfm = ones(Int, n)
                  Finch.coalesce_level!((CoalesceLevel)(Finch.CPU{:t}(n), (DenseLevel){Int64}(ElementLevel{0.0, Float64, Int64}(tens2_lvl_3_val), tens2_lvl_2_stop), tens2_lvl_coalref, tens2_lvl.schedule), gfm, lfm, tm, 1, n, (DenseLevel){Int64}(ElementLevel{0.0, Float64, Int64}(tens2_lvl_5_val), tens2_lvl_4_stop))
                  display(tens2)
                  result
              end)
  end)
eval(code)
run(tens2)