import copy

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

arr_dict1 = {
    "shape": [16, 16], "vi_seg": [0, 2],
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
    "gold_vals": [-15, 1, -18]}

arr_dict2 = {
    "shape": [16, 16],
    "vi_seg": [0, 5],
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

arr_dict3 = {"shape": [4, 4],
             "vi_seg": [0, 1], "vi_crd": [3], "vi_vals": [3],
             "mi_seg": [0, 1], "mi_crd": [0],
             "mj_seg": [0, 2], "mj_crd": [0, 2], "m_vals": [-7, 2],
             "vj_seg": [0, 1], "vj_crd": [3], "vj_vals": [5]
             }

arr_dict4 = {"shape": [4, 4],
             "vi_seg": [0, 3], "vi_crd": [0, 1, 2], "vi_vals": [6, 1, 2],
             "mi_seg": [0, 2], "mi_crd": [0, 1],
             "mj_seg": [0, 3, 4], "mj_crd": [0, 2, 3, 1], "m_vals": [-2, -1, 6, 7],
             "vj_seg": [0, 4], "vj_crd": [0, 1, 2, 3], "vj_vals": [9, 4, 5, 3]
             }

arr_dict5 = {"shape": [16, 16],
             "vi_seg": [0, 3], "vi_crd": [2, 4, 14], "vi_vals": [4, 9, 7],
             "mi_seg": [0, 6], "mi_crd": [0, 1, 4, 9, 10, 11],
             "mj_seg": [0, 6, 17, 27, 31, 38, 45],
             "mj_crd": [1, 3, 5, 7, 9, 14, 0, 2, 3, 4, 7, 9, 10, 12, 13, 14, 15, 0, 1, 3, 5, 6, 7, 10, 12, 13, 15, 0,
                        2, 3, 13, 3, 7, 8, 10, 13, 14, 15, 0, 1, 2, 3, 8, 10, 13],
             "m_vals": [9, -6, 1, -9, 9, -10, 7, 7, 10, -6, -9, -7, 1, -5, -2, -6, -7, -10, 5, -1, 1, -9, 7, 5, 6, -1,
                        -2, -2, 3, -4, -8, 10, -5, -4, 4, -3, -1, -9, 5, -9, 10, 1, -8, -6, -3],
             "vj_seg": [0, 3], "vj_crd": [0, 6, 8], "vj_vals": [6, 0, 10]
             }


@pytest.mark.parametrize("arrs", [arr_dict1, arr_dict2, arr_dict3, arr_dict4, arr_dict5])
def test_unit_mat_residual_direct(arrs, check_gold, debug_sim, backpressure, depth, fill=0):
    C_shape = copy.deepcopy(arrs["shape"])

    C_seg0 = copy.deepcopy(arrs["mi_seg"])
    C_crd0 = copy.deepcopy(arrs["mi_crd"])
    C_seg1 = copy.deepcopy(arrs["mj_seg"])
    C_crd1 = copy.deepcopy(arrs["mj_crd"])
    C_vals = copy.deepcopy(arrs["m_vals"])

    b_shape = [C_shape[0]]
    b_seg0 = copy.deepcopy(arrs["vi_seg"])
    b_crd0 = copy.deepcopy(arrs["vi_crd"])
    b_vals = copy.deepcopy(arrs["vi_vals"])

    d_shape = [C_shape[1]]
    d_seg0 = copy.deepcopy(arrs["vj_seg"])
    d_crd0 = copy.deepcopy(arrs["vj_crd"])
    d_vals = copy.deepcopy(arrs["vj_vals"])

    fiberlookup_bi_17 = CompressedCrdRdScan(crd_arr=b_crd0, seg_arr=b_seg0, debug=debug_sim,
                                            back_en=backpressure, depth=int(depth))
    fiberlookup_Ci_18 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim,
                                            back_en=backpressure, depth=int(depth))
    unioni_16 = Union2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    fiberlookup_Cj_11 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim,
                                            back_en=backpressure, depth=int(depth))
    fiberwrite_x0_1 = CompressWrScan(seg_size=2, size=b_shape[0], fill=fill, debug=debug_sim,
                                     back_en=backpressure, depth=int(depth))
    repsiggen_i_14 = RepeatSigGen(debug=False, back_en=backpressure, depth=int(depth))
    repeat_di_13 = Repeat(debug=False, back_en=backpressure, depth=int(depth))
    fiberlookup_dj_12 = CompressedCrdRdScan(crd_arr=d_crd0, seg_arr=d_seg0, debug=debug_sim,
                                            back_en=backpressure, depth=int(depth))
    intersectj_10 = Intersect2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    unionj1 = Union2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    unionj2 = Union2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    repsiggen_j_9 = RepeatSigGen(debug=debug_sim, back_en=backpressure, depth=int(depth))
    arrayvals_C_6 = Array(init_arr=C_vals, debug=debug_sim, back_en=backpressure, depth=int(depth))
    arrayvals_d_7 = Array(init_arr=d_vals, debug=debug_sim, back_en=backpressure, depth=int(depth))
    repeat_bj_8 = Repeat(debug=debug_sim, back_en=backpressure, depth=int(depth))
    mul_5 = Multiply2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    arrayvals_b_4 = Array(init_arr=b_vals, debug=debug_sim, back_en=backpressure, depth=int(depth))
    add_3 = Add2(debug=debug_sim, neg2=True, back_en=backpressure, depth=int(depth))
    reduce_2 = Reduce(debug=debug_sim, back_en=backpressure, depth=int(depth))
    fiberwrite_xvals_0 = ValsWrScan(size=1 * b_shape[0], fill=fill, debug=debug_sim,
                                    back_en=backpressure, depth=int(depth))
    in_ref_b = [0, 'D']
    in_ref_C = [0, 'D']
    in_ref_d = [0, 'D']
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

    temp02 = []
    temp21 = []
    temp22 = []
    temp23 = []
    temp24 = []
    temp25 = []
    temp26 = []

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_b) > 0:
            fiberlookup_bi_17.set_in_ref(in_ref_b.pop(0), "")

        if len(in_ref_C) > 0:
            fiberlookup_Ci_18.set_in_ref(in_ref_C.pop(0), "")

        unioni_16.set_in1(fiberlookup_bi_17.out_ref(), fiberlookup_bi_17.out_crd(), fiberlookup_bi_17)
        unioni_16.set_in2(fiberlookup_Ci_18.out_ref(), fiberlookup_Ci_18.out_crd(), fiberlookup_Ci_18)

        fiberlookup_Cj_11.set_in_ref(unioni_16.out_ref2(), unioni_16)

        fiberwrite_x0_1.set_input(unioni_16.out_crd(), unioni_16)

        repsiggen_i_14.set_istream(unioni_16.out_crd(), unioni_16)

        if len(in_ref_d) > 0:
            repeat_di_13.set_in_ref(in_ref_d.pop(0), "")
        repeat_di_13.set_in_repsig(repsiggen_i_14.out_repsig(), repsiggen_i_14)

        fiberlookup_dj_12.set_in_ref(repeat_di_13.out_ref(), repeat_di_13)

        temp.append(fiberlookup_Cj_11.out_crd())
        temp1.append(fiberlookup_dj_12.out_crd())
        temp2.append(fiberlookup_Cj_11.out_ref())
        temp3.append(fiberlookup_dj_12.out_ref())
        intersectj_10.set_in1(fiberlookup_dj_12.out_ref(), fiberlookup_dj_12.out_crd(), fiberlookup_dj_12)
        intersectj_10.set_in2(fiberlookup_Cj_11.out_ref(), fiberlookup_Cj_11.out_crd(), fiberlookup_Cj_11)

        # repsiggen_j_9.set_istream(intersectj_10.out_crd())

        arrayvals_C_6.set_load(intersectj_10.out_ref2(), intersectj_10)

        arrayvals_d_7.set_load(intersectj_10.out_ref1(), intersectj_10)

        # repeat_bj_8.set_in_ref(unioni_16.out_ref1())
        # repeat_bj_8.set_in_repsig(repsiggen_j_9.out_repsig())

        arrayvals_b_4.set_load(unioni_16.out_ref1(), unioni_16)

        mul_5.set_in1(arrayvals_C_6.out_val(), arrayvals_C_6)
        mul_5.set_in2(arrayvals_d_7.out_val(), arrayvals_d_7)

        reduce_2.set_in_val(mul_5.out_val(), mul_5)

        add_3.set_in1(arrayvals_b_4.out_val(), arrayvals_b_4)
        add_3.set_in2(reduce_2.out_val(), reduce_2)

        fiberwrite_xvals_0.set_input(add_3.out_val(), add_3)

        fiberlookup_bi_17.update()
        fiberlookup_Ci_18.update()
        unioni_16.update()
        fiberlookup_Cj_11.update()
        fiberwrite_x0_1.update()
        repsiggen_i_14.update()
        repeat_di_13.update()
        fiberlookup_dj_12.update()
        intersectj_10.update()
        # repsiggen_j_9.update()
        arrayvals_C_6.update()
        arrayvals_d_7.update()
        # repeat_bj_8.update()
        arrayvals_b_4.update()
        mul_5.update()
        reduce_2.update()
        add_3.update()
        fiberwrite_xvals_0.update()

        temp7.append(unioni_16.out_crd())
        temp6.append(unioni_16.out_ref1())
        temp8.append(unioni_16.out_ref2())
        temp5.append(repeat_bj_8.out_ref())
        temp4.append(arrayvals_b_4.out_val())
        temp9.append(repsiggen_j_9.out_repsig())
        print("union crd", remove_emptystr(temp7))
        print("Ci ref", remove_emptystr(temp8))
        print("bi ref", remove_emptystr(temp6))
        print("Cj", remove_emptystr(temp))
        print(remove_emptystr(temp2))
        print("dj", remove_emptystr(temp1))
        print(remove_emptystr(temp3))
        print("bj rep", remove_emptystr(temp5))
        print("rsg", remove_emptystr(temp9))

        temp02.append(repeat_bj_8.out_ref())
        temp21.append(arrayvals_b_4.out_val())
        temp22.append(intersectj_10.out_ref2())
        temp23.append(arrayvals_C_6.out_val())
        temp24.append(intersectj_10.out_ref1())
        temp25.append(arrayvals_d_7.out_val())
        temp26.append(mul_5.out_val())

        print("intj 2", remove_emptystr(temp22))
        print("C_vals", remove_emptystr(temp23))
        print("intj 1", remove_emptystr(temp24))
        print("d_vals", remove_emptystr(temp25))
        print("mul out", remove_emptystr(temp26))
        print("rep bj", remove_emptystr(temp02))
        print("b_vals", remove_emptystr(temp21))
        print()

        done = fiberwrite_x0_1.out_done() and fiberwrite_xvals_0.out_done()
        time_cnt += 1

    fiberwrite_x0_1.autosize()
    fiberwrite_xvals_0.autosize()

    out_crds = [fiberwrite_x0_1.get_arr()]
    out_segs = [fiberwrite_x0_1.get_seg_arr()]
    out_vals = fiberwrite_xvals_0.get_arr()

    if "gold_seg" in arrs.keys():
        gold_seg = copy.deepcopy(arrs["gold_seg"])
        gold_crd = copy.deepcopy(arrs["gold_crd"])
        gold_vals = copy.deepcopy(arrs["gold_vals"])

        assert out_crds[0] == gold_crd
        assert out_segs[0] == gold_seg
        assert out_vals == gold_vals

    else:
        C_tup = convert_point_tuple(get_point_list([C_crd0, C_crd1], [C_seg0, C_seg1], C_vals))

        C_nd = convert_point_tuple_ndarr(C_tup, C_shape[0])

        b_nd = np.zeros(b_shape)
        d_nd = np.zeros(d_shape)

        for i in range(len(b_crd0)):
            val = b_vals[i]
            crd = b_crd0[i]
            b_nd[crd] = val

        for i in range(len(d_crd0)):
            val = d_vals[i]
            crd = d_crd0[i]
            d_nd[crd] = val

        gold_nd = b_nd - C_nd @ d_nd

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


@pytest.mark.parametrize("dim", [4, 16, 32, 64])
@pytest.mark.parametrize("nnz", [0.2, 0.5, 0.8])
def test_unit_mat_residual_random(dim, nnz, debug_sim, backpressure, depth, max_val=10, fill=0):
    C_shape = [dim, dim]
    C_crds, C_segs = gen_n_comp_arrs(2, dim)
    C_vals = gen_val_arr(len(C_crds[-1]), max_val, -max_val)
    C_crd0 = C_crds[0]
    C_seg0 = C_segs[0]
    C_crd1 = C_crds[1]
    C_seg1 = C_segs[1]

    b_shape = [dim]
    b_crd0 = [random.randint(0, dim - 1) for _ in range(1 + int(nnz * dim))]
    b_crd0 = sorted(set(b_crd0))
    b_seg0 = [0, len(b_crd0)]
    b_vals = [random.randint(0, max_val) for _ in range(len(b_crd0))]

    d_shape = [dim]
    d_crd0 = [random.randint(0, dim - 1) for _ in range(1 + int(nnz * dim))]
    d_crd0 = sorted(set(d_crd0))
    d_seg0 = [0, len(d_crd0)]
    d_vals = [random.randint(0, max_val) for _ in range(len(d_crd0))]

    C_tup = convert_point_tuple(get_point_list(C_crds, C_segs, C_vals))

    C_nd = convert_point_tuple_ndarr(C_tup, dim)

    if debug_sim:
        print("b:", b_seg0, b_crd0, b_vals)
        print("C0:", C_seg0, C_crd0)
        print("C1:", C_seg1, C_crd1, C_vals)
        print("d:", d_seg0, d_crd0, d_vals)

    fiberlookup_bi_17 = CompressedCrdRdScan(crd_arr=b_crd0, seg_arr=b_seg0, debug=debug_sim,
                                            back_en=backpressure, depth=int(depth))
    fiberlookup_Ci_18 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim,
                                            back_en=backpressure, depth=int(depth))
    unioni_16 = Union2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    fiberlookup_Cj_11 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim,
                                            back_en=backpressure, depth=int(depth))
    fiberwrite_x0_1 = CompressWrScan(seg_size=2, size=b_shape[0], fill=fill, debug=debug_sim,
                                     back_en=backpressure, depth=int(depth))
    repsiggen_i_14 = RepeatSigGen(debug=debug_sim, back_en=backpressure, depth=int(depth))
    repeat_di_13 = Repeat(debug=debug_sim, back_en=backpressure, depth=int(depth))
    fiberlookup_dj_12 = CompressedCrdRdScan(crd_arr=d_crd0, seg_arr=d_seg0, debug=debug_sim,
                                            back_en=backpressure, depth=int(depth))
    intersectj_10 = Intersect2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    # unionj1 = Union2(debug=debug_sim)
    # unionj2 = Union2(debug=debug_sim)
    repsiggen_j_9 = RepeatSigGen(debug=debug_sim, back_en=backpressure, depth=int(depth))
    arrayvals_C_6 = Array(init_arr=C_vals, debug=debug_sim, back_en=backpressure, depth=int(depth))
    arrayvals_d_7 = Array(init_arr=d_vals, debug=debug_sim, back_en=backpressure, depth=int(depth))
    repeat_bj_8 = Repeat(debug=debug_sim, back_en=backpressure, depth=int(depth))
    mul_5 = Multiply2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    arrayvals_b_4 = Array(init_arr=b_vals, debug=debug_sim, back_en=backpressure, depth=int(depth))
    add_3 = Add2(debug=debug_sim, neg2=True, back_en=backpressure, depth=int(depth))
    reduce_2 = Reduce(debug=debug_sim, back_en=backpressure, depth=int(depth))
    fiberwrite_xvals_0 = ValsWrScan(size=1 * b_shape[0], fill=fill, debug=debug_sim,
                                    back_en=backpressure, depth=int(depth))
    in_ref_b = [0, 'D']
    in_ref_C = [0, 'D']
    in_ref_d = [0, 'D']
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
    while not done and time_cnt < TIMEOUT:
        if len(in_ref_b) > 0:
            fiberlookup_bi_17.set_in_ref(in_ref_b.pop(0), "")

        if len(in_ref_C) > 0:
            fiberlookup_Ci_18.set_in_ref(in_ref_C.pop(0), "")

        unioni_16.set_in1(fiberlookup_bi_17.out_ref(), fiberlookup_bi_17.out_crd(), fiberlookup_bi_17)
        unioni_16.set_in2(fiberlookup_Ci_18.out_ref(), fiberlookup_Ci_18.out_crd(), fiberlookup_Ci_18)

        fiberlookup_Cj_11.set_in_ref(unioni_16.out_ref2(), unioni_16)

        fiberwrite_x0_1.set_input(unioni_16.out_crd(), unioni_16)

        repsiggen_i_14.set_istream(unioni_16.out_crd(), unioni_16)

        if len(in_ref_d) > 0:
            repeat_di_13.set_in_ref(in_ref_d.pop(0), "")
        repeat_di_13.set_in_repsig(repsiggen_i_14.out_repsig(), repsiggen_i_14)

        fiberlookup_dj_12.set_in_ref(repeat_di_13.out_ref(), repeat_di_13)

        temp.append(fiberlookup_Cj_11.out_crd())
        temp1.append(fiberlookup_dj_12.out_crd())
        temp2.append(fiberlookup_Cj_11.out_ref())
        temp3.append(fiberlookup_dj_12.out_ref())
        intersectj_10.set_in1(fiberlookup_dj_12.out_ref(), fiberlookup_dj_12.out_crd(), fiberlookup_dj_12)
        intersectj_10.set_in2(fiberlookup_Cj_11.out_ref(), fiberlookup_Cj_11.out_crd(), fiberlookup_Cj_11)

        # repsiggen_j_9.set_istream(intersectj_10.out_crd())

        arrayvals_C_6.set_load(intersectj_10.out_ref2(), intersectj_10)

        arrayvals_d_7.set_load(intersectj_10.out_ref1(), intersectj_10)

        # repeat_bj_8.set_in_ref(unioni_16.out_ref1())
        # repeat_bj_8.set_in_repsig(repsiggen_j_9.out_repsig())

        # unionj1.set_in1(repeat_bj_8.out_ref())
        # unionj1.set_in2(intersectj_10.out_crd(), intersectj_10.out_ref1())
        #
        # unionj2.set_in1(repeat_bj_8.out_ref())
        # unionj2.set_in2(intersectj_10.out_crd(), intersectj_10.out_ref2())

        arrayvals_b_4.set_load(unioni_16.out_ref1(), unioni_16)

        mul_5.set_in1(arrayvals_C_6.out_val(), arrayvals_C_6)
        mul_5.set_in2(arrayvals_d_7.out_val(), arrayvals_d_7)

        reduce_2.set_in_val(mul_5.out_val(), mul_5)

        add_3.set_in1(arrayvals_b_4.out_val(), arrayvals_b_4)
        add_3.set_in2(reduce_2.out_val(), reduce_2)

        fiberwrite_xvals_0.set_input(add_3.out_val(), add_3)

        fiberlookup_bi_17.update()
        fiberlookup_Ci_18.update()
        unioni_16.update()
        fiberlookup_Cj_11.update()
        fiberwrite_x0_1.update()
        repsiggen_i_14.update()
        repeat_di_13.update()
        fiberlookup_dj_12.update()
        intersectj_10.update()
        # repsiggen_j_9.update()
        arrayvals_C_6.update()
        arrayvals_d_7.update()
        # repeat_bj_8.update()
        # unionj1.update()
        # unionj2.update()
        arrayvals_b_4.update()
        mul_5.update()
        reduce_2.update()
        add_3.update()
        fiberwrite_xvals_0.update()

        # temp7.append(unioni_16.out_crd())
        # temp6.append(unioni_16.out_ref1())
        # temp8.append(unioni_16.out_ref2())
        # temp5.append(repeat_bj_8.out_ref())
        # temp4.append(arrayvals_b_4.out_val())
        # temp9.append(repsiggen_j_9.out_repsig())
        # print("union crd", remove_emptystr(temp7))
        # print("Ci ref", remove_emptystr(temp8))
        # print("bi ref", remove_emptystr(temp6))
        # print("Cj", remove_emptystr(temp))
        # print(remove_emptystr(temp2))
        # print("dj", remove_emptystr(temp1))
        # print(remove_emptystr(temp3))
        # print("bj rep", remove_emptystr(temp5))
        # print("rsg", remove_emptystr(temp9))
        # print()

        # temp.append(repeat_bj_8.out_ref())
        # temp1.append(arrayvals_b_4.out_val())
        # temp2.append(intersectj_10.out_ref2())
        # temp3.append(arrayvals_C_6.out_val())
        # temp4.append(intersectj_10.out_ref1())
        # temp5.append(arrayvals_d_7.out_val())
        # temp6.append(mul_5.out_val())
        #
        # print("intj 2", remove_emptystr(temp2))
        # print("C_vals", remove_emptystr(temp3))
        # print("intj 1", remove_emptystr(temp4))
        # print("d_vals", remove_emptystr(temp5))
        # print("mul out", remove_emptystr(temp6))
        # print("rep bj", remove_emptystr(temp))
        # print("b_vals", remove_emptystr(temp1))

        done = fiberwrite_x0_1.out_done() and fiberwrite_xvals_0.out_done()
        time_cnt += 1

    fiberwrite_x0_1.autosize()
    fiberwrite_xvals_0.autosize()

    out_crds = [fiberwrite_x0_1.get_arr()]
    out_segs = [fiberwrite_x0_1.get_seg_arr()]
    out_vals = fiberwrite_xvals_0.get_arr()

    b_nd = np.zeros(b_shape)
    d_nd = np.zeros(d_shape)

    for i in range(len(b_crd0)):
        val = b_vals[i]
        crd = b_crd0[i]
        b_nd[crd] = val

    for i in range(len(d_crd0)):
        val = d_vals[i]
        crd = d_crd0[i]
        d_nd[crd] = val

    gold_nd = b_nd - C_nd @ d_nd

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
