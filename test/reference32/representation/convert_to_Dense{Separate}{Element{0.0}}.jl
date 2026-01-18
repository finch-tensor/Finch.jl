quote
    tmp_lvl = ((ex.bodies[1]).bodies[1]).tns.bind.lvl
    tmp_lvl_val = ((ex.bodies[1]).bodies[1]).tns.bind.lvl.val
    tmp_lvl_2 = ((ex.bodies[1]).bodies[1]).tns.bind.lvl.lvl
    tmp_lvl_3 = tmp_lvl_2.lvl
    ref_lvl = ((ex.bodies[1]).bodies[2]).body.rhs.tns.bind.lvl
    ref_lvl_ptr = ref_lvl.ptr
    ref_lvl_idx = ref_lvl.idx
    ref_lvl_val = ref_lvl.lvl.val
    Finch.resize_if_smaller!(tmp_lvl_val, 1)
    for pos = 1:1
        pointer_to_lvl = Finch.similar_level(tmp_lvl.lvl, Finch.level_fill_value(typeof(tmp_lvl.lvl)), Finch.level_eltype(typeof(tmp_lvl.lvl)), ref_lvl.shape)
        pointer_to_lvl_3 = pointer_to_lvl.lvl
        pointer_to_lvl_2_val = pointer_to_lvl.lvl.val
        Finch.resize_if_smaller!(pointer_to_lvl_2_val, ref_lvl.shape)
        Finch.fill_range!(pointer_to_lvl_2_val, 0.0, 1, ref_lvl.shape)
        resize!(pointer_to_lvl_2_val, ref_lvl.shape)
        tmp_lvl_val[pos] = (DenseLevel){Int32}(pointer_to_lvl_3, ref_lvl.shape)
    end
    pointer_to_lvl_5 = tmp_lvl_val[1]
    pointer_to_lvl_6 = pointer_to_lvl_5.lvl
    pointer_to_lvl_5_val = pointer_to_lvl_5.lvl.val
    Finch.resize_if_smaller!(pointer_to_lvl_5_val, pointer_to_lvl_5.shape)
    Finch.fill_range!(pointer_to_lvl_5_val, 0.0, 1, pointer_to_lvl_5.shape)
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
                pointer_to_lvl_5_q = (1 - 1) * pointer_to_lvl_5.shape + ref_lvl_i
                pointer_to_lvl_5_val[pointer_to_lvl_5_q] = ref_lvl_2_val
                ref_lvl_q += 1
            else
                phase_stop_3 = min(phase_stop, ref_lvl_i)
                if ref_lvl_i == phase_stop_3
                    ref_lvl_2_val = ref_lvl_val[ref_lvl_q]
                    pointer_to_lvl_5_q = (1 - 1) * pointer_to_lvl_5.shape + phase_stop_3
                    pointer_to_lvl_5_val[pointer_to_lvl_5_q] = ref_lvl_2_val
                    ref_lvl_q += 1
                end
                break
            end
        end
    end
    resize!(pointer_to_lvl_5_val, pointer_to_lvl_5.shape)
    tmp_lvl_val[1] = (DenseLevel){Int32}(pointer_to_lvl_6, pointer_to_lvl_5.shape)
    (tmp = Tensor((SeparateLevel){DenseLevel{Int32, ElementLevel{0.0, Float64, Int32, Vector{Float64}}}, Vector{DenseLevel{Int32, ElementLevel{0.0, Float64, Int32, Vector{Float64}}}}}((DenseLevel){Int32}(tmp_lvl_3, ref_lvl.shape), tmp_lvl_val)),)
end
