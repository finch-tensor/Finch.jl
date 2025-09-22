import pytest
import time
import scipy.sparse
from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.wr_scanner import ValsWrScan
from sam.sim.src.joiner import Intersect2, Union2
from sam.sim.src.compute import Multiply2, Add2
from sam.sim.src.crd_manager import CrdDrop, CrdHold
from sam.sim.src.repeater import Repeat, RepeatSigGen
from sam.sim.src.accumulator import Reduce
from sam.sim.src.accumulator import SparseAccumulator1, SparseAccumulator2
from sam.sim.src.token import *
from sam.sim.test.test import *
from sam.sim.test.gold import *
import os
import csv

cwd = os.getcwd()
formatted_dir = os.getenv('SUITESPARSE_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))
other_dir = os.getenv('OTHER_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))

arr_dict1 = {"vi_seg": [0, 2],
             "vi_crd": [1, 2],
             "vi_vals": [1, 2],
             "vj_seg": [0, 2],
             "vj_crd": [1, 2],
             "vj_vals": [5, 6],
             "mi_seg": [0, 2],
             "mi_crd": [0, 2],
             "mj_seg": [0, 1, 2],
             "mj_crd": [1, 1],
             "m_vals": [3, 4],
             "gold_seg": [0, 3],
             "gold_crd": [0, 1, 2],
             "gold_vals": [30, 2, 44]}

arr_dict2 = {"vi_seg": [0, 5],
             "vi_crd": [1, 2, 4, 8, 15],
             "vi_vals": [1, 1, 1, 1, 1],
             "vj_seg": [0, 4],
             "vj_crd": [0, 3, 9, 15],
             "vj_vals": [2, 2, 2, 2],
             "mi_seg": [0, 16],
             "mi_crd": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             "mj_seg": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32],
             "mj_crd": [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
                        14, 14, 15, 15, 1],
             "m_vals": [-3] * 32}


# FIXME: Figureout formats
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.parametrize("arrs", [arr_dict1, arr_dict2])
def test_mat_mattransmul_direct(arrs, check_gold, debug_sim, backpressure, depth, fill=0):
    C_shape = (16, 16)

    C_seg1 = copy.deepcopy(arrs["mi_seg"])
    C_crd1 = copy.deepcopy(arrs["mi_crd"])
    C_seg0 = copy.deepcopy(arrs["mj_seg"])
    C_crd0 = copy.deepcopy(arrs["mj_crd"])
    C_vals = copy.deepcopy(arrs["m_vals"])

    f_shape = [C_shape[1]]
    f_seg0 = copy.deepcopy(arrs["vi_seg"])
    f_crd0 = copy.deepcopy(arrs["vi_crd"])
    f_vals = copy.deepcopy(arrs["vi_vals"])

    d_shape = [C_shape[0]]
    d_seg0 = copy.deepcopy(arrs["vj_seg"])
    d_crd0 = copy.deepcopy(arrs["vj_crd"])
    d_vals = copy.deepcopy(arrs["vj_vals"])

    e_vals = [2]
    e_shape = [0]

    b_shape = [0]
    b_vals = [2]

    fiberlookup_Ci_27 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1,
                                            debug=debug_sim, back_en=backpressure, depth=int(depth))
    fiberlookup_fi_28 = CompressedCrdRdScan(crd_arr=f_crd0, seg_arr=f_seg0, debug=debug_sim,
                                            back_en=backpressure, depth=int(depth))
    unioni_26 = Union2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    fiberlookup_Cj_18 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim,
                                            back_en=backpressure, depth=int(depth))
    fiberwrite_x0_1 = CompressWrScan(seg_size=2, size=C_shape[1], fill=fill, debug=debug_sim,
                                     back_en=backpressure, depth=int(depth))
    repsiggen_i_24 = RepeatSigGen(debug=debug_sim, back_en=backpressure, depth=int(depth))
    repeat_bi_20 = Repeat(debug=debug_sim, back_en=backpressure, depth=int(depth))
    repeat_di_21 = Repeat(debug=debug_sim, back_en=backpressure, depth=int(depth))
    repeat_ei_22 = Repeat(debug=debug_sim, back_en=backpressure, depth=int(depth))
    fiberlookup_dj_19 = CompressedCrdRdScan(crd_arr=d_crd0, seg_arr=d_seg0, debug=debug_sim,
                                            back_en=backpressure, depth=int(depth))
    intersectj_17 = Intersect2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    repsiggen_j_16 = RepeatSigGen(debug=debug_sim, back_en=backpressure, depth=int(depth))
    arrayvals_C_7 = Array(init_arr=C_vals, debug=debug_sim, back_en=backpressure, depth=int(depth))
    arrayvals_d_8 = Array(init_arr=d_vals, debug=debug_sim, back_en=backpressure, depth=int(depth))
    repeat_bj_12 = Repeat(debug=debug_sim, back_en=backpressure, depth=int(depth))
    repeat_ej_13 = Repeat(debug=debug_sim, back_en=backpressure, depth=int(depth))
    repeat_fj_14 = Repeat(debug=debug_sim, back_en=backpressure, depth=int(depth))
    arrayvals_b_6 = Array(init_arr=b_vals, debug=debug_sim, back_en=backpressure, depth=int(depth))
    arrayvals_e_10 = Array(init_arr=e_vals, debug=debug_sim, back_en=backpressure, depth=int(depth))
    arrayvals_f_11 = Array(init_arr=f_vals, debug=debug_sim, back_en=backpressure, depth=int(depth))
    mul_5 = Multiply2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    mul_9 = Multiply2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    mul_4 = Multiply2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    add_3 = Add2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    reduce_2 = Reduce(debug=debug_sim, back_en=backpressure, depth=int(depth))
    fiberwrite_xvals_0 = ValsWrScan(size=1 * C_shape[1], fill=fill, debug=debug_sim, back_en=backpressure,
                                    depth=int(depth))
    in_ref_C = [0, 'D']
    in_ref_f = [0, 'D']
    in_ref_b = [0, 'D']
    in_ref_d = [0, 'D']
    in_ref_e = [0, 'D']
    done = False
    time_cnt = 0

    temp = []
    temp1 = []
    temp2 = []
    temp3 = []
    temp4 = []
    temp5 = []
    temp6 = []
    temp7 = []
    temp8 = []
    temp9 = []
    temp10 = []
    temp11 = []
    temp12 = []
    temp13 = []
    temp14 = []
    temp_add1 = []
    temp_add2 = []
    while not done and time_cnt < TIMEOUT:
        if len(in_ref_C) > 0:
            fiberlookup_Ci_27.set_in_ref(in_ref_C.pop(0), "")

        if len(in_ref_f) > 0:
            fiberlookup_fi_28.set_in_ref(in_ref_f.pop(0), "")

        unioni_26.set_in1(fiberlookup_Ci_27.out_ref(), fiberlookup_Ci_27.out_crd(), fiberlookup_Ci_27)
        unioni_26.set_in2(fiberlookup_fi_28.out_ref(), fiberlookup_fi_28.out_crd(), fiberlookup_fi_28)

        fiberlookup_Cj_18.set_in_ref(unioni_26.out_ref1(), unioni_26)

        fiberwrite_x0_1.set_input(unioni_26.out_crd(), unioni_26)

        repsiggen_i_24.set_istream(unioni_26.out_crd(), unioni_26)

        if len(in_ref_b) > 0:
            repeat_bi_20.set_in_ref(in_ref_b.pop(0), "")
        repeat_bi_20.set_in_repsig(repsiggen_i_24.out_repsig(), repsiggen_i_24)

        if len(in_ref_d) > 0:
            repeat_di_21.set_in_ref(in_ref_d.pop(0), "")
        repeat_di_21.set_in_repsig(repsiggen_i_24.out_repsig(), repsiggen_i_24)

        if len(in_ref_e) > 0:
            repeat_ei_22.set_in_ref(in_ref_e.pop(0), "")
        repeat_ei_22.set_in_repsig(repsiggen_i_24.out_repsig(), repsiggen_i_24)

        fiberlookup_dj_19.set_in_ref(repeat_di_21.out_ref(), repeat_di_21)

        intersectj_17.set_in1(fiberlookup_dj_19.out_ref(), fiberlookup_dj_19.out_crd(), fiberlookup_dj_19)
        intersectj_17.set_in2(fiberlookup_Cj_18.out_ref(), fiberlookup_Cj_18.out_crd(), fiberlookup_Cj_18)

        repsiggen_j_16.set_istream(intersectj_17.out_crd(), intersectj_17)

        arrayvals_C_7.set_load(intersectj_17.out_ref2(), intersectj_17)

        arrayvals_d_8.set_load(intersectj_17.out_ref1(), intersectj_17)

        repeat_bj_12.set_in_ref(repeat_bi_20.out_ref(), repeat_bi_20)
        repeat_bj_12.set_in_repsig(repsiggen_j_16.out_repsig(), repsiggen_j_16)

        arrayvals_e_10.set_load(repeat_ei_22.out_ref(), repeat_ei_22)

        arrayvals_f_11.set_load(unioni_26.out_ref2(), unioni_26)

        mul_9.set_in1(arrayvals_e_10.out_val(), arrayvals_e_10)
        mul_9.set_in2(arrayvals_f_11.out_val(), arrayvals_f_11)

        arrayvals_b_6.set_load(repeat_bj_12.out_ref(), repeat_bj_12)

        mul_5.set_in1(arrayvals_b_6.out_val(), arrayvals_b_6)
        mul_5.set_in2(arrayvals_C_7.out_val(), arrayvals_C_7)

        mul_4.set_in1(mul_5.out_val(), mul_5)
        mul_4.set_in2(arrayvals_d_8.out_val(), arrayvals_d_8)

        reduce_2.set_in_val(mul_4.out_val(), mul_4)

        add_3.set_in1(reduce_2.out_val(), reduce_2)
        add_3.set_in2(mul_9.out_val(), mul_9)

        fiberwrite_xvals_0.set_input(add_3.out_val(), add_3)

        fiberlookup_Ci_27.update()
        fiberlookup_fi_28.update()
        unioni_26.update()
        fiberlookup_Cj_18.update()
        fiberwrite_x0_1.update()
        repsiggen_i_24.update()
        repeat_bi_20.update()
        repeat_di_21.update()
        repeat_ei_22.update()
        fiberlookup_dj_19.update()
        intersectj_17.update()
        repsiggen_j_16.update()
        arrayvals_C_7.update()
        arrayvals_d_8.update()
        repeat_bj_12.update()
        arrayvals_e_10.update()
        arrayvals_f_11.update()
        mul_9.update()
        arrayvals_b_6.update()
        mul_5.update()
        mul_4.update()
        reduce_2.update()
        add_3.update()
        fiberwrite_xvals_0.update()

        temp7.append(unioni_26.out_crd())
        temp6.append(unioni_26.out_ref1())
        temp8.append(unioni_26.out_ref2())
        temp12.append(repeat_di_21.out_ref())
        temp9.append(repeat_bi_20.out_ref())
        temp10.append(repeat_ei_22.out_ref())

        temp.append(fiberlookup_Cj_18.out_crd())
        temp1.append(fiberlookup_dj_19.out_crd())
        temp2.append(fiberlookup_Cj_18.out_ref())
        temp3.append(fiberlookup_dj_19.out_ref())

        temp13.append(intersectj_17.out_crd())
        temp4.append(repeat_bj_12.out_ref())
        temp11.append(repeat_ej_13.out_ref())
        temp5.append(repeat_fj_14.out_ref())

        temp_add1.append(mul_9.out_val())
        temp_add2.append(mul_4.out_val())
        done = fiberwrite_x0_1.out_done() and fiberwrite_xvals_0.out_done()
        time_cnt += 1

    fiberwrite_x0_1.autosize()
    fiberwrite_xvals_0.autosize()

    out_crds = [fiberwrite_x0_1.get_arr()]
    out_segs = [fiberwrite_x0_1.get_seg_arr()]
    out_vals = fiberwrite_xvals_0.get_arr()

    if debug_sim:
        print("unioni", remove_emptystr(temp7))
        print(remove_emptystr(temp6))
        print(remove_emptystr(temp8))
        print("repeat di", remove_emptystr(temp12))
        print("rep bi", remove_emptystr(temp9))
        print("rep ei", remove_emptystr(temp10))

        print("Cj", remove_emptystr(temp))
        print(remove_emptystr(temp2))
        print("dj", remove_emptystr(temp1))
        print(remove_emptystr(temp3))

        print("intj", remove_emptystr(temp13))
        print("rep bj", remove_emptystr(temp4))
        print("rep ej", remove_emptystr(temp11))
        print("rep fj", remove_emptystr(temp5))
        print()

        print(out_crds[0])
        print(out_segs[0])
        print(out_vals)
        print()

    if "gold_seg" in arrs.keys():
        gold_seg = copy.deepcopy(arrs["gold_seg"])
        gold_crd = copy.deepcopy(arrs["gold_crd"])
        gold_vals = copy.deepcopy(arrs["gold_vals"])

        assert out_crds[0] == gold_crd
        assert out_segs[0] == gold_seg
        assert out_vals == gold_vals

    else:
        B_scipy = scipy.sparse.csr_matrix((C_vals, C_crd0, C_seg0), shape=C_shape)
        b_nd = np.zeros(f_shape)
        d_nd = np.zeros(d_shape)

        for i in range(len(f_crd0)):
            val = f_vals[i]
            crd = f_crd0[i]
            b_nd[crd] = val

        for i in range(len(d_crd0)):
            val = d_vals[i]
            crd = d_crd0[i]
            d_nd[crd] = val

        gold_nd = 2 * B_scipy @ d_nd + 2 * b_nd

        gold_tup = convert_ndarr_point_tuple(gold_nd)
        if debug_sim:
            print("Out segs:", out_segs)
            print("Out crds:", out_crds)
            print("Out vals:", out_vals)
            print("Dense Vec1:\n", b_nd)
            print("Dense Mat1:\n", B_scipy.toarray())
            print("Dense Vec2:\n", d_nd)
            print("Dense Gold:", gold_nd)
            print("Gold:", gold_tup)

        if not out_vals:
            assert out_vals == gold_tup
        elif not gold_tup:
            assert all([v == 0 for v in out_vals])
        else:
            out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_vals))
            out_tup = remove_zeros(out_tup)
            assert (check_point_tuple(out_tup, gold_tup))


@pytest.mark.parametrize("dim", [4, 16, 32, 64])
@pytest.mark.parametrize("nnz", [0.2, 0.5, 0.8])
def test_unit_mat_mattransmul_random(dim, nnz, debug_sim, backpressure, depth, max_val=10, fill=0):
    C_shape = [dim, dim]
    C_crds, C_segs = gen_n_comp_arrs(2, dim)
    C_vals = gen_val_arr(len(C_crds[-1]), max_val, -max_val)
    C_crd0 = C_crds[1]
    C_seg0 = C_segs[1]
    C_crd1 = C_crds[0]
    C_seg1 = C_segs[0]

    f_shape = [dim]
    f_crd0 = [random.randint(0, dim - 1) for _ in range(1 + int(nnz * dim))]
    f_crd0 = sorted(set(f_crd0))
    f_seg0 = [0, len(f_crd0)]
    f_vals = [random.randint(0, max_val) for _ in range(len(f_crd0))]

    d_shape = [dim]
    d_crd0 = [random.randint(0, dim - 1) for _ in range(1 + int(nnz * dim))]
    d_crd0 = sorted(set(d_crd0))
    d_seg0 = [0, len(d_crd0)]
    d_vals = [random.randint(0, max_val) for _ in range(len(d_crd0))]

    e_vals = [2]
    e_shape = [0]

    b_shape = [0]
    b_vals = [2]

    C_tup = convert_point_tuple(get_point_list(C_crds, C_segs, C_vals))
    C_nd = convert_point_tuple_ndarr(C_tup, dim)

    if debug_sim:
        print("b:", f_seg0, f_crd0, b_vals)
        print("C0:", C_seg0, C_crd0)
        print("C1:", C_seg1, C_crd1, C_vals)
        print("d:", d_seg0, d_crd0, d_vals)

    fiberlookup_Ci_27 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim,
                                            back_en=backpressure, depth=int(depth))
    fiberlookup_fi_28 = CompressedCrdRdScan(crd_arr=f_crd0, seg_arr=f_seg0, debug=debug_sim,
                                            back_en=backpressure, depth=int(depth))
    unioni_26 = Union2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    fiberlookup_Cj_18 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim,
                                            back_en=backpressure, depth=int(depth))
    fiberwrite_x0_1 = CompressWrScan(seg_size=2, size=C_shape[1], fill=fill, debug=debug_sim,
                                     back_en=backpressure, depth=int(depth))
    repsiggen_i_24 = RepeatSigGen(debug=debug_sim, back_en=backpressure, depth=int(depth))
    repeat_bi_20 = Repeat(debug=debug_sim, back_en=backpressure, depth=int(depth))
    repeat_di_21 = Repeat(debug=debug_sim, back_en=backpressure, depth=int(depth))
    repeat_ei_22 = Repeat(debug=debug_sim, back_en=backpressure, depth=int(depth))
    fiberlookup_dj_19 = CompressedCrdRdScan(crd_arr=d_crd0, seg_arr=d_seg0, debug=debug_sim,
                                            back_en=backpressure, depth=int(depth))
    intersectj_17 = Intersect2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    repsiggen_j_16 = RepeatSigGen(debug=debug_sim, back_en=backpressure, depth=int(depth))
    arrayvals_C_7 = Array(init_arr=C_vals, debug=debug_sim, back_en=backpressure, depth=int(depth))
    arrayvals_d_8 = Array(init_arr=d_vals, debug=debug_sim, back_en=backpressure, depth=int(depth))
    repeat_bj_12 = Repeat(debug=debug_sim, back_en=backpressure, depth=int(depth))
    # repeat_ej_13 = Repeat(debug=debug_sim, union=True)
    # repeat_fj_14 = Repeat(debug=debug_sim, union=True)
    arrayvals_b_6 = Array(init_arr=b_vals, debug=debug_sim, back_en=backpressure, depth=int(depth))
    arrayvals_e_10 = Array(init_arr=e_vals, debug=debug_sim, back_en=backpressure, depth=int(depth))
    arrayvals_f_11 = Array(init_arr=f_vals, debug=debug_sim, back_en=backpressure, depth=int(depth))
    mul_5 = Multiply2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    mul_9 = Multiply2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    mul_4 = Multiply2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    add_3 = Add2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    reduce_2 = Reduce(debug=debug_sim, back_en=backpressure, depth=int(depth))
    fiberwrite_xvals_0 = ValsWrScan(size=1 * C_shape[1], fill=fill, debug=debug_sim,
                                    back_en=backpressure, depth=int(depth))
    in_ref_C = [0, 'D']
    in_ref_f = [0, 'D']
    in_ref_b = [0, 'D']
    in_ref_d = [0, 'D']
    in_ref_e = [0, 'D']
    done = False
    time_cnt = 0

    temp = []
    temp1 = []
    temp2 = []
    temp3 = []
    temp4 = []
    temp5 = []
    temp6 = []
    temp7 = []
    temp8 = []
    temp9 = []
    temp10 = []
    temp11 = []
    temp12 = []
    temp13 = []
    temp14 = []
    while not done and time_cnt < TIMEOUT:
        if len(in_ref_C) > 0:
            fiberlookup_Ci_27.set_in_ref(in_ref_C.pop(0), "")

        if len(in_ref_f) > 0:
            fiberlookup_fi_28.set_in_ref(in_ref_f.pop(0), "")

        unioni_26.set_in1(fiberlookup_Ci_27.out_ref(), fiberlookup_Ci_27.out_crd(), fiberlookup_Ci_27)
        unioni_26.set_in2(fiberlookup_fi_28.out_ref(), fiberlookup_fi_28.out_crd(), fiberlookup_fi_28)

        fiberlookup_Cj_18.set_in_ref(unioni_26.out_ref1(), unioni_26)

        fiberwrite_x0_1.set_input(unioni_26.out_crd(), unioni_26)

        repsiggen_i_24.set_istream(unioni_26.out_crd(), unioni_26)

        if len(in_ref_b) > 0:
            repeat_bi_20.set_in_ref(in_ref_b.pop(0), "")
        repeat_bi_20.set_in_repsig(repsiggen_i_24.out_repsig(), repsiggen_i_24)

        if len(in_ref_d) > 0:
            repeat_di_21.set_in_ref(in_ref_d.pop(0), "")
        repeat_di_21.set_in_repsig(repsiggen_i_24.out_repsig(), repsiggen_i_24)

        if len(in_ref_e) > 0:
            repeat_ei_22.set_in_ref(in_ref_e.pop(0), "")
        repeat_ei_22.set_in_repsig(repsiggen_i_24.out_repsig(), repsiggen_i_24)

        fiberlookup_dj_19.set_in_ref(repeat_di_21.out_ref(), repeat_di_21)

        intersectj_17.set_in1(fiberlookup_dj_19.out_ref(), fiberlookup_dj_19.out_crd(), fiberlookup_dj_19)
        intersectj_17.set_in2(fiberlookup_Cj_18.out_ref(), fiberlookup_Cj_18.out_crd(), fiberlookup_Cj_18)

        repsiggen_j_16.set_istream(intersectj_17.out_crd(), intersectj_17)

        arrayvals_C_7.set_load(intersectj_17.out_ref2(), intersectj_17)

        arrayvals_d_8.set_load(intersectj_17.out_ref1(), intersectj_17)

        repeat_bj_12.set_in_ref(repeat_bi_20.out_ref(), repeat_bi_20)
        repeat_bj_12.set_in_repsig(repsiggen_j_16.out_repsig(), repsiggen_j_16)

        temp7.append(unioni_26.out_crd())
        temp6.append(unioni_26.out_ref1())
        temp8.append(unioni_26.out_ref2())
        temp12.append(repeat_di_21.out_ref())
        temp9.append(repeat_bi_20.out_ref())
        temp10.append(repeat_ei_22.out_ref())

        temp.append(fiberlookup_Cj_18.out_crd())
        temp1.append(fiberlookup_dj_19.out_crd())
        temp2.append(fiberlookup_Cj_18.out_ref())
        temp3.append(fiberlookup_dj_19.out_ref())

        temp13.append(intersectj_17.out_crd())
        temp4.append(repeat_bj_12.out_ref())
        # temp11.append(repeat_ej_13.out_ref())
        # temp5.append(repeat_fj_14.out_ref())

        arrayvals_e_10.set_load(repeat_ei_22.out_ref(), repeat_ei_22)

        arrayvals_f_11.set_load(unioni_26.out_ref2(), unioni_26)

        mul_9.set_in1(arrayvals_e_10.out_val(), arrayvals_e_10)
        mul_9.set_in2(arrayvals_f_11.out_val(), arrayvals_f_11)

        arrayvals_b_6.set_load(repeat_bj_12.out_ref(), repeat_bj_12)

        mul_5.set_in1(arrayvals_b_6.out_val(), arrayvals_b_6)
        mul_5.set_in2(arrayvals_C_7.out_val(), arrayvals_C_7)

        mul_4.set_in1(mul_5.out_val(), mul_5)
        mul_4.set_in2(arrayvals_d_8.out_val(), arrayvals_d_8)

        reduce_2.set_in_val(mul_4.out_val(), mul_4)

        add_3.set_in1(reduce_2.out_val(), reduce_2)
        add_3.set_in2(mul_9.out_val(), mul_9)

        fiberwrite_xvals_0.set_input(add_3.out_val(), add_3)

        fiberlookup_Ci_27.update()
        fiberlookup_fi_28.update()
        unioni_26.update()
        fiberlookup_Cj_18.update()
        fiberwrite_x0_1.update()
        repsiggen_i_24.update()
        repeat_bi_20.update()
        repeat_di_21.update()
        repeat_ei_22.update()
        fiberlookup_dj_19.update()
        intersectj_17.update()
        repsiggen_j_16.update()
        arrayvals_C_7.update()
        arrayvals_d_8.update()
        repeat_bj_12.update()
        arrayvals_e_10.update()
        arrayvals_f_11.update()
        mul_9.update()
        arrayvals_b_6.update()
        mul_5.update()
        mul_4.update()
        reduce_2.update()
        add_3.update()
        fiberwrite_xvals_0.update()

        done = fiberwrite_x0_1.out_done() and fiberwrite_xvals_0.out_done()
        time_cnt += 1

    fiberwrite_x0_1.autosize()
    fiberwrite_xvals_0.autosize()

    out_crds = [fiberwrite_x0_1.get_arr()]
    out_segs = [fiberwrite_x0_1.get_seg_arr()]
    out_vals = fiberwrite_xvals_0.get_arr()

    if debug_sim:
        print("unioni", remove_emptystr(temp7))
        print(remove_emptystr(temp6))
        print(remove_emptystr(temp8))
        print("repeat di", remove_emptystr(temp12))
        print("rep bi", remove_emptystr(temp9))
        print("rep ei", remove_emptystr(temp10))

        print("Cj", remove_emptystr(temp))
        print(remove_emptystr(temp2))
        print("dj", remove_emptystr(temp1))
        print(remove_emptystr(temp3))

        print("intj", remove_emptystr(temp13))
        print("rep bj", remove_emptystr(temp4))
        print("rep ej", remove_emptystr(temp11))
        print("rep fj", remove_emptystr(temp5))
        print()

        print(out_crds[0])
        print(out_segs[0])
        print(out_vals)
        print()

        b_nd = np.zeros(b_shape)
        d_nd = np.zeros(d_shape)

        for i in range(len(f_crd0)):
            val = f_vals[i]
            crd = f_crd0[i]
            b_nd[crd] = val

        for i in range(len(d_crd0)):
            val = d_vals[i]
            crd = d_crd0[i]
            d_nd[crd] = val

        gold_nd = 2 * C_nd @ d_nd + 2 * b_nd

        gold_tup = convert_ndarr_point_tuple(gold_nd)
        if debug_sim:
            print("Out segs:", out_segs)
            print("Out crds:", out_crds)
            print("Out vals:", out_vals)
            print("Dense Vec1:\n", b_nd)
            print("Dense Mat1:\n", C_nd)
            print("Dense Vec2:\n", d_nd)
            print("Dense Gold:", gold_nd)
            print("Gold:", gold_tup)

        if not out_vals:
            assert out_vals == gold_tup
        elif not gold_tup:
            assert all([v == 0 for v in out_vals])
        else:
            out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_vals))
            out_tup = remove_zeros(out_tup)
            assert (check_point_tuple(out_tup, gold_tup))
