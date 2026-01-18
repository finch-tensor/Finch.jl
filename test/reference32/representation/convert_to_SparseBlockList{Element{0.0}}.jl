quote
    tmp_lvl = ((ex.bodies[1]).bodies[1]).tns.bind.lvl
    tmp_lvl_ptr = tmp_lvl.ptr
    tmp_lvl_idx = tmp_lvl.idx
    tmp_lvl_ofs = tmp_lvl.ofs
    tmp_lvl_2 = tmp_lvl.lvl
    tmp_lvl_val = tmp_lvl.lvl.val
    ref_lvl = ((ex.bodies[1]).bodies[2]).body.rhs.tns.bind.lvl
    ref_lvl_ptr = ref_lvl.ptr
    ref_lvl_idx = ref_lvl.idx
    ref_lvl_val = ref_lvl.lvl.val
    tmp_lvl_qos_stop = 0
    tmp_lvl_ros_stop = 0
    Finch.resize_if_smaller!(tmp_lvl_ofs, 1)
    tmp_lvl_ofs[1] = 1
    Finch.resize_if_smaller!(tmp_lvl_ptr, 1 + 1)
    Finch.fill_range!(tmp_lvl_ptr, 0, 1 + 1, 1 + 1)
    tmp_lvl_ros = 0
    tmp_lvl_qos = 0 + 1
    tmp_lvl_i_prev = -1
    0 < 1 || throw(FinchProtocolError("SparseBlockListLevels cannot be updated multiple times"))
    ref_lvl_q = ref_lvl_ptr[1]
    ref_lvl_q_stop = ref_lvl_ptr[1 + 1]
    if ref_lvl_q < ref_lvl_q_stop
        ref_lvl_i1 = ref_lvl_idx[ref_lvl_q_stop - 1]
    else
        ref_lvl_i1 = 0
    end
    phase_stop = min(ref_lvl_i1, ref_lvl.shape)
    if phase_stop >= 1
        if ref_lvl_idx[ref_lvl_q] < 1
            ref_lvl_q = Finch.scansearch(ref_lvl_idx, 1, ref_lvl_q, ref_lvl_q_stop - 1)
        end
        while true
            ref_lvl_i = ref_lvl_idx[ref_lvl_q]
            if ref_lvl_i < phase_stop
                ref_lvl_2_val = ref_lvl_val[ref_lvl_q]
                if tmp_lvl_qos > tmp_lvl_qos_stop
                    tmp_lvl_qos_stop = max(tmp_lvl_qos_stop << 1, 1)
                    Finch.resize_if_smaller!(tmp_lvl_val, tmp_lvl_qos_stop)
                    Finch.fill_range!(tmp_lvl_val, 0.0, tmp_lvl_qos, tmp_lvl_qos_stop)
                end
                tmp_lvl_val[tmp_lvl_qos] = ref_lvl_2_val
                if ref_lvl_i > tmp_lvl_i_prev + 1
                    tmp_lvl_ros += 1
                    if tmp_lvl_ros > tmp_lvl_ros_stop
                        tmp_lvl_ros_stop = max(tmp_lvl_ros_stop << 1, 1)
                        Finch.resize_if_smaller!(tmp_lvl_idx, tmp_lvl_ros_stop)
                        Finch.resize_if_smaller!(tmp_lvl_ofs, tmp_lvl_ros_stop + 1)
                    end
                end
                tmp_lvl_idx[tmp_lvl_ros] = (tmp_lvl_i_prev = ref_lvl_i)
                tmp_lvl_qos += 1
                tmp_lvl_ofs[tmp_lvl_ros + 1] = tmp_lvl_qos
                ref_lvl_q += 1
            else
                phase_stop_3 = min(phase_stop, ref_lvl_i)
                if ref_lvl_i == phase_stop_3
                    ref_lvl_2_val = ref_lvl_val[ref_lvl_q]
                    if tmp_lvl_qos > tmp_lvl_qos_stop
                        tmp_lvl_qos_stop = max(tmp_lvl_qos_stop << 1, 1)
                        Finch.resize_if_smaller!(tmp_lvl_val, tmp_lvl_qos_stop)
                        Finch.fill_range!(tmp_lvl_val, 0.0, tmp_lvl_qos, tmp_lvl_qos_stop)
                    end
                    tmp_lvl_val[tmp_lvl_qos] = ref_lvl_2_val
                    if phase_stop_3 > tmp_lvl_i_prev + 1
                        tmp_lvl_ros += 1
                        if tmp_lvl_ros > tmp_lvl_ros_stop
                            tmp_lvl_ros_stop = max(tmp_lvl_ros_stop << 1, 1)
                            Finch.resize_if_smaller!(tmp_lvl_idx, tmp_lvl_ros_stop)
                            Finch.resize_if_smaller!(tmp_lvl_ofs, tmp_lvl_ros_stop + 1)
                        end
                    end
                    tmp_lvl_idx[tmp_lvl_ros] = (tmp_lvl_i_prev = phase_stop_3)
                    tmp_lvl_qos += 1
                    tmp_lvl_ofs[tmp_lvl_ros + 1] = tmp_lvl_qos
                    ref_lvl_q += 1
                end
                break
            end
        end
    end
    tmp_lvl_ptr[1 + 1] = tmp_lvl_ros - 0
    resize!(tmp_lvl_ptr, 1 + 1)
    for p = 2:1 + 1
        tmp_lvl_ptr[p] += tmp_lvl_ptr[p - 1]
    end
    ros_stop = tmp_lvl_ptr[1 + 1] - 1
    resize!(tmp_lvl_idx, ros_stop)
    resize!(tmp_lvl_ofs, ros_stop + 1)
    qos_stop = tmp_lvl_ofs[ros_stop + 1] - 1
    resize!(tmp_lvl_val, qos_stop)
    (tmp = Tensor((SparseBlockListLevel){Int32}(tmp_lvl_2, ref_lvl.shape, tmp_lvl_ptr, tmp_lvl_idx, tmp_lvl_ofs)),)
end
