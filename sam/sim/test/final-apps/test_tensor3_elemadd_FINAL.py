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
formatted_dir = os.getenv('FROSTT_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))


# FIXME: Figureout formats
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.frostt
def test_tensor3_elemadd_FINAL(samBench, frosttname, check_gold, report_stats, debug_sim, backpressure, depth, fill=0):
    B_dirname = os.path.join(cwd, "tmp_mat")
    B_shape_filename = os.path.join(B_dirname, "tensor_B_mode_shape")
    B_shape = read_inputs(B_shape_filename)

    B0_seg_filename = os.path.join(B_dirname, "tensor_B_mode_0_seg")
    B_seg0 = read_inputs(B0_seg_filename)
    B0_crd_filename = os.path.join(B_dirname, "tensor_B_mode_0_crd")
    B_crd0 = read_inputs(B0_crd_filename)

    B1_seg_filename = os.path.join(B_dirname, "tensor_B_mode_1_seg")
    B_seg1 = read_inputs(B1_seg_filename)
    B1_crd_filename = os.path.join(B_dirname, "tensor_B_mode_1_crd")
    B_crd1 = read_inputs(B1_crd_filename)

    B2_seg_filename = os.path.join(B_dirname, "tensor_B_mode_2_seg")
    B_seg2 = read_inputs(B2_seg_filename)
    B2_crd_filename = os.path.join(B_dirname, "tensor_B_mode_2_crd")
    B_crd2 = read_inputs(B2_crd_filename)

    B_vals_filename = os.path.join(B_dirname, "tensor_B_mode_vals")
    B_vals = read_inputs(B_vals_filename, float)

    C_dirname = os.path.join(cwd, "tmp_mat")
    C_shape_filename = os.path.join(C_dirname, "tensor_C_mode_shape")
    C_shape = read_inputs(C_shape_filename)

    C0_seg_filename = os.path.join(C_dirname, "tensor_C_mode_0_seg")
    C_seg0 = read_inputs(C0_seg_filename)
    C0_crd_filename = os.path.join(C_dirname, "tensor_C_mode_0_crd")
    C_crd0 = read_inputs(C0_crd_filename)

    C1_seg_filename = os.path.join(C_dirname, "tensor_C_mode_1_seg")
    C_seg1 = read_inputs(C1_seg_filename)
    C1_crd_filename = os.path.join(C_dirname, "tensor_C_mode_1_crd")
    C_crd1 = read_inputs(C1_crd_filename)

    C2_seg_filename = os.path.join(C_dirname, "tensor_C_mode_2_seg")
    C_seg2 = read_inputs(C1_seg_filename)
    C2_crd_filename = os.path.join(C_dirname, "tensor_C_mode_2_crd")
    C_crd2 = read_inputs(C1_crd_filename)

    C_vals_filename = os.path.join(C_dirname, "tensor_C_mode_vals")
    C_vals = read_inputs(C_vals_filename, float)

    fiberlookup_Bi_14 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    fiberlookup_Ci_15 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    unioni_13 = Union2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_X0_3 = CompressWrScan(seg_size=2, size=2 * len(B_crd0), fill=fill, debug=debug_sim, statistics=report_stats,
                                     back_en=backpressure, depth=int(depth))
    fiberlookup_Bj_11 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    fiberlookup_Cj_12 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    unionj_10 = Union2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_X1_2 = CompressWrScan(seg_size=2 * len(B_crd0) + 1, size=2 * len(B_crd1), fill=fill,
                                     debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Bk_8 = CompressedCrdRdScan(crd_arr=B_crd2, seg_arr=B_seg2, debug=debug_sim, statistics=report_stats,
                                           back_en=backpressure, depth=int(depth))
    fiberlookup_Ck_9 = CompressedCrdRdScan(crd_arr=C_crd2, seg_arr=C_seg2, debug=debug_sim, statistics=report_stats,
                                           back_en=backpressure, depth=int(depth))
    unionk_7 = Union2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_X2_1 = CompressWrScan(seg_size=2 * len(B_crd1) + 1, size=len(B_vals) * 2, fill=fill,
                                     debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_B_5 = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_C_6 = Array(init_arr=C_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    add_4 = Add2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_Xvals_0 = ValsWrScan(size=5804660 * 2, fill=fill, debug=debug_sim, statistics=report_stats,
                                    back_en=backpressure, depth=int(depth))
    in_ref_B = [0, 'D']
    in_ref_C = [0, 'D']
    done = False
    time_cnt = 0
    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_14.set_in_ref(in_ref_B.pop(0), "")

        if len(in_ref_C) > 0:
            fiberlookup_Ci_15.set_in_ref(in_ref_C.pop(0), "")

        unioni_13.set_in1(fiberlookup_Bi_14.out_ref(), fiberlookup_Bi_14.out_crd(), fiberlookup_Bi_14)
        unioni_13.set_in2(fiberlookup_Ci_15.out_ref(), fiberlookup_Ci_15.out_crd(), fiberlookup_Ci_15)

        fiberwrite_X0_3.set_input(unioni_13.out_crd(), unioni_13)

        fiberlookup_Bj_11.set_in_ref(unioni_13.out_ref1(), unioni_13)

        fiberlookup_Cj_12.set_in_ref(unioni_13.out_ref2(), unioni_13)

        unionj_10.set_in1(fiberlookup_Bj_11.out_ref(), fiberlookup_Bj_11.out_crd(), fiberlookup_Bj_11)
        unionj_10.set_in2(fiberlookup_Cj_12.out_ref(), fiberlookup_Cj_12.out_crd(), fiberlookup_Cj_12)
        fiberwrite_X1_2.set_input(unionj_10.out_crd(), unionj_10)
        fiberlookup_Bk_8.set_in_ref(unionj_10.out_ref1(), unionj_10)
        fiberlookup_Ck_9.set_in_ref(unionj_10.out_ref2(), unionj_10)
        unionk_7.set_in1(fiberlookup_Bk_8.out_ref(), fiberlookup_Bk_8.out_crd(), fiberlookup_Bk_8)
        unionk_7.set_in2(fiberlookup_Ck_9.out_ref(), fiberlookup_Ck_9.out_crd(), fiberlookup_Ck_9)
        fiberwrite_X2_1.set_input(unionk_7.out_crd(), unionk_7)
        arrayvals_B_5.set_load(unionk_7.out_ref1(), unionk_7)
        arrayvals_C_6.set_load(unionk_7.out_ref2(), unionk_7)
        add_4.set_in1(arrayvals_B_5.out_val(), arrayvals_B_5)
        add_4.set_in2(arrayvals_C_6.out_val(), arrayvals_C_6)
        fiberwrite_Xvals_0.set_input(add_4.out_val(), add_4)

        fiberlookup_Bi_14.update()
        fiberlookup_Ci_15.update()
        unioni_13.update()
        fiberwrite_X0_3.update()
        fiberlookup_Bj_11.update()
        fiberlookup_Cj_12.update()
        unionj_10.update()
        fiberwrite_X1_2.update()
        fiberlookup_Bk_8.update()
        fiberlookup_Ck_9.update()
        unionk_7.update()
        fiberwrite_X2_1.update()
        arrayvals_B_5.update()
        arrayvals_C_6.update()
        add_4.update()
        fiberwrite_Xvals_0.update()

        done = fiberwrite_X0_3.out_done() and fiberwrite_X1_2.out_done() and fiberwrite_X2_1.out_done() and \
            fiberwrite_Xvals_0.out_done()

    fiberwrite_X0_3.autosize()
    fiberwrite_X1_2.autosize()
    fiberwrite_X2_1.autosize()
    fiberwrite_Xvals_0.autosize()

    out_crds = [fiberwrite_X0_3.get_arr(), fiberwrite_X1_2.get_arr(), fiberwrite_X2_1.get_arr()]
    out_segs = [fiberwrite_X0_3.get_seg_arr(), fiberwrite_X1_2.get_seg_arr(), fiberwrite_X2_1.get_seg_arr()]
    out_vals = fiberwrite_Xvals_0.get_arr()

    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = frosttname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_C_shape"] = C_shape
    extra_info["tensor_B/nnz"] = len(B_vals)
    extra_info["tensor_C/nnz"] = len(C_vals)

    extra_info["result/vals_size"] = len(out_vals)
    extra_info["result/nnz"] = len([x for x in out_vals if x != 0])

    sample_dict = unioni_13.return_statistics()
    for k in sample_dict.keys():
        extra_info["unioni_13" + "/" + k] = sample_dict[k]

    sample_dict = unionj_10.return_statistics()
    for k in sample_dict.keys():
        extra_info["unionj_10" + "/" + k] = sample_dict[k]

    sample_dict = unionk_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["unionk_7" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_3" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_2" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_X2_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X2_1" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_B_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_5" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_C_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_6" + "/" + k] = sample_dict[k]

    sample_dict = add_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["add_4" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bi_14.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bi_14" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Ci_15.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ci_15" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bj_11.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bj_11" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Cj_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Cj_12" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bk_8.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bk_8" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Ck_9.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ck_9" + "/" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_tensor3_elemadd(frosttname, debug_sim, out_crds, out_segs, out_vals, "sss012")
    samBench(bench, extra_info)
