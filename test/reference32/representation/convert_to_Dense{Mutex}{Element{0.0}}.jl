quote
    tmp_lvl = ((ex.bodies[1]).bodies[1]).tns.bind.lvl
    tmp_lvl_locks = ((ex.bodies[1]).bodies[1]).tns.bind.lvl.locks
    tmp_lvl_2 = tmp_lvl.lvl
    tmp_lvl_3 = tmp_lvl_2.lvl
    tmp_lvl_2_val = tmp_lvl_2.lvl.val
    ref_lvl = ((ex.bodies[1]).bodies[2]).body.rhs.tns.bind.lvl
    ref_lvl_ptr = ref_lvl.ptr
    ref_lvl_idx = ref_lvl.idx
    ref_lvl_val = ref_lvl.lvl.val
    Finch.resize_if_smaller!(tmp_lvl_locks, 1)
    @inbounds for idx = 1:1
            tmp_lvl_locks[idx] = Finch.make_lock(eltype(Vector{Base.Threads.SpinLock}))
        end
    Finch.resize_if_smaller!(tmp_lvl_2_val, ref_lvl.shape)
    Finch.fill_range!(tmp_lvl_2_val, 0.0, 1, ref_lvl.shape)
    tmp_lvlatomicArraysAcc = Finch.get_lock(CPU(1), tmp_lvl_locks, 1, eltype(Vector{Base.Threads.SpinLock}))
    Finch.aquire_lock!(CPU(1), tmp_lvlatomicArraysAcc)
    ref_lvl_q = ref_lvl_ptr[1]
    ref_lvl_q_stop = ref_lvl_ptr[1 + 1]
    if ref_lvl_q < ref_lvl_q_stop
        ref_lvl_i1 = ref_lvl_idx[ref_lvl_q_stop - 1]
    else
        ref_lvl_i1 = 0
    end
    phase_stop = min(ref_lvl.shape, ref_lvl_i1)
    if phase_stop >= 1
        if ref_lvl_idx[ref_lvl_q] < 1
            ref_lvl_q = Finch.scansearch(ref_lvl_idx, 1, ref_lvl_q, ref_lvl_q_stop - 1)
        end
        while true
            ref_lvl_i = ref_lvl_idx[ref_lvl_q]
            if ref_lvl_i < phase_stop
                ref_lvl_2_val = ref_lvl_val[ref_lvl_q]
                tmp_lvl_2_q = (1 - 1) * ref_lvl.shape + ref_lvl_i
                tmp_lvl_2_val[tmp_lvl_2_q] = ref_lvl_2_val
                ref_lvl_q += 1
            else
                phase_stop_3 = min(phase_stop, ref_lvl_i)
                if ref_lvl_i == phase_stop_3
                    ref_lvl_2_val = ref_lvl_val[ref_lvl_q]
                    tmp_lvl_2_q = (1 - 1) * ref_lvl.shape + phase_stop_3
                    tmp_lvl_2_val[tmp_lvl_2_q] = ref_lvl_2_val
                    ref_lvl_q += 1
                end
                break
            end
        end
    end
    Finch.release_lock!(CPU(1), tmp_lvlatomicArraysAcc)
    resize!(tmp_lvl_locks, 1)
    resize!(tmp_lvl_2_val, ref_lvl.shape)
    (tmp = Tensor((MutexLevel){Vector{Base.Threads.SpinLock}, DenseLevel{Int32, ElementLevel{0.0, Float64, Int32, Vector{Float64}}}}((DenseLevel){Int32}(tmp_lvl_3, ref_lvl.shape), tmp_lvl_locks)),)
end
