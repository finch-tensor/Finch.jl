begin
    B = ((ex.bodies[1]).bodies[1]).tns.bind
    B_idx = B.nzind
    B_val = B.nzval
    A = ((ex.bodies[1]).bodies[2]).body.rhs.tns.bind
    A_idx = A.nzind
    A_val = A.nzval
    B_qos_stop = 0
    B_qos = 0 + 1
    A_q = 1
    A_q_stop = length(A_idx) + 1
    if 1 < A_q_stop
        A_i1 = A_idx[A_q_stop - 1]
    else
        A_i1 = 0
    end
    phase_stop = min(A_i1, ((ex.bodies[1]).bodies[2]).body.rhs.tns.bind.n)
    if phase_stop >= 1
        if A_idx[1] < 1
            A_q = Finch.scansearch(A_idx, 1, 1, A_q_stop - 1)
        end
        while true
            A_i = A_idx[A_q]
            if A_i < phase_stop
                A_val_2 = A_val[A_q]
                if B_qos > B_qos_stop
                    B_qos_stop = max(B_qos_stop << 1, 1)
                    Finch.resize_if_smaller!(B_idx, B_qos_stop)
                    Finch.resize_if_smaller!(B_val, B_qos_stop)
                end
                B_val[B_qos] = A_val_2
                B_idx[B_qos] = A_i
                B_qos += 1
                A_q += 1
            else
                phase_stop_3 = min(phase_stop, A_i)
                if A_i == phase_stop_3
                    A_val_2 = A_val[A_q]
                    if B_qos > B_qos_stop
                        B_qos_stop = max(B_qos_stop << 1, 1)
                        Finch.resize_if_smaller!(B_idx, B_qos_stop)
                        Finch.resize_if_smaller!(B_val, B_qos_stop)
                    end
                    B_val[B_qos] = A_val_2
                    B_idx[B_qos] = phase_stop_3
                    B_qos += 1
                    A_q += 1
                end
                break
            end
        end
    end
    B_qos_fill = B_qos - 1
    resize!(B_idx, B_qos_fill)
    resize!(B_val, B_qos_fill)
    (B = (SparseVector)(((ex.bodies[1]).bodies[2]).body.rhs.tns.bind.n, B_idx, B_val),)
end
