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
from sam.sim.test.gen_gantt import gen_gantt

cwd = os.getcwd()
formatted_dir = os.getenv('SUITESPARSE_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))


# csv file path

@pytest.mark.suitesparse
def test_mat_elemadd_FINAL(samBench, ssname, cast, positive_only, check_gold, report_stats, debug_sim, backpressure,
                           depth, fill=0):
    B_dirname = os.path.join(formatted_dir, ssname, "mat_elemadd")
    B_shape_filename = os.path.join(B_dirname, "tensor_B_mode_shape")
    B_shape = read_inputs(B_shape_filename, positive_only=positive_only)

    B0_seg_filename = os.path.join(B_dirname, "tensor_B_mode_0_seg")
    B_seg0 = read_inputs(B0_seg_filename, positive_only=positive_only)
    B0_crd_filename = os.path.join(B_dirname, "tensor_B_mode_0_crd")
    B_crd0 = read_inputs(B0_crd_filename, positive_only=positive_only)

    B1_seg_filename = os.path.join(B_dirname, "tensor_B_mode_1_seg")
    B_seg1 = read_inputs(B1_seg_filename, positive_only=positive_only)
    B1_crd_filename = os.path.join(B_dirname, "tensor_B_mode_1_crd")
    B_crd1 = read_inputs(B1_crd_filename, positive_only=positive_only)

    B_vals_filename = os.path.join(B_dirname, "tensor_B_mode_vals")
    B_vals = read_inputs(B_vals_filename, float, positive_only=positive_only)

    C_dirname = B_dirname
    C_shape_filename = os.path.join(C_dirname, "tensor_C_mode_shape")
    C_shape = read_inputs(C_shape_filename, positive_only=positive_only)

    C0_seg_filename = os.path.join(C_dirname, "tensor_C_mode_0_seg")
    C_seg0 = read_inputs(C0_seg_filename, positive_only=positive_only)
    C0_crd_filename = os.path.join(C_dirname, "tensor_C_mode_0_crd")
    C_crd0 = read_inputs(C0_crd_filename, positive_only=positive_only)

    C1_seg_filename = os.path.join(C_dirname, "tensor_C_mode_1_seg")
    C_seg1 = read_inputs(C1_seg_filename, positive_only=positive_only)
    C1_crd_filename = os.path.join(C_dirname, "tensor_C_mode_1_crd")
    C_crd1 = read_inputs(C1_crd_filename, positive_only=positive_only)

    C_vals_filename = os.path.join(C_dirname, "tensor_C_mode_vals")
    C_vals = read_inputs(C_vals_filename, float, positive_only=positive_only)

    fiberlookup_Bi_10 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    fiberlookup_Ci_11 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    unioni_9 = Union2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_X0_2 = CompressWrScan(seg_size=2, size=2 * len(B_crd0), fill=fill, debug=debug_sim,
                                     statistics=report_stats,
                                     back_en=backpressure, depth=int(depth))
    fiberlookup_Bj_7 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats,
                                           back_en=backpressure, depth=int(depth))
    fiberlookup_Cj_8 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim, statistics=report_stats,
                                           back_en=backpressure, depth=int(depth))
    unionj_6 = Union2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_X1_1 = CompressWrScan(seg_size=2 * len(B_crd0) + 1, size=2 * len(B_vals), fill=fill,
                                     debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_B_4 = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure,
                          depth=int(depth))
    arrayvals_C_5 = Array(init_arr=C_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure,
                          depth=int(depth))
    add_3 = Add2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_Xvals_0 = ValsWrScan(size=2 * len(B_vals), fill=fill, debug=debug_sim, statistics=report_stats,
                                    back_en=backpressure, depth=int(depth))
    in_ref_B = [0, 'D']
    in_ref_C = [0, 'D']
    done = False
    time_cnt = 0
    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_10.set_in_ref(in_ref_B.pop(0), "")

        if len(in_ref_C) > 0:
            fiberlookup_Ci_11.set_in_ref(in_ref_C.pop(0), "")

        unioni_9.set_in1(fiberlookup_Bi_10.out_ref(), fiberlookup_Bi_10.out_crd(), fiberlookup_Bi_10)
        unioni_9.set_in2(fiberlookup_Ci_11.out_ref(), fiberlookup_Ci_11.out_crd(), fiberlookup_Ci_11)

        fiberwrite_X0_2.set_input(unioni_9.out_crd(), unioni_9)

        fiberlookup_Bj_7.set_in_ref(unioni_9.out_ref1(), unioni_9)

        fiberlookup_Cj_8.set_in_ref(unioni_9.out_ref2(), unioni_9)

        unionj_6.set_in1(fiberlookup_Bj_7.out_ref(), fiberlookup_Bj_7.out_crd(), fiberlookup_Bj_7)
        unionj_6.set_in2(fiberlookup_Cj_8.out_ref(), fiberlookup_Cj_8.out_crd(), fiberlookup_Cj_8)

        fiberwrite_X1_1.set_input(unionj_6.out_crd(), unionj_6)

        arrayvals_B_4.set_load(unionj_6.out_ref1(), unionj_6)

        arrayvals_C_5.set_load(unionj_6.out_ref2(), unionj_6)

        add_3.set_in1(arrayvals_B_4.out_val(), arrayvals_B_4)
        add_3.set_in2(arrayvals_C_5.out_val(), arrayvals_C_5)

        fiberwrite_Xvals_0.set_input(add_3.out_val(), add_3)

        fiberlookup_Bi_10.update()
        fiberlookup_Ci_11.update()
        unioni_9.update()
        fiberwrite_X0_2.update()
        fiberlookup_Bj_7.update()
        fiberlookup_Cj_8.update()
        unionj_6.update()
        fiberwrite_X1_1.update()
        arrayvals_B_4.update()
        arrayvals_C_5.update()
        add_3.update()
        fiberwrite_Xvals_0.update()

        done = fiberwrite_X0_2.out_done() and fiberwrite_X1_1.out_done() and fiberwrite_Xvals_0.out_done()
        time_cnt += 1

    fiberwrite_X0_2.autosize()
    fiberwrite_X1_1.autosize()
    fiberwrite_Xvals_0.autosize()

    out_crds = [fiberwrite_X0_2.get_arr(), fiberwrite_X1_1.get_arr()]
    out_segs = [fiberwrite_X0_2.get_seg_arr(), fiberwrite_X1_1.get_seg_arr()]
    out_vals = fiberwrite_Xvals_0.get_arr()

    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = ssname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_C_shape"] = C_shape
    extra_info["tensor_B/nnz"] = len(B_vals)
    extra_info["tensor_C/nnz"] = len(C_vals)

    extra_info["result/vals_size"] = len(out_vals)
    extra_info["result/nnz"] = len([x for x in out_vals if x != 0])

    sample_dict = fiberlookup_Bi_10.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bi_10" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Ci_11.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ci_11" + "/" + k] = sample_dict[k]

    sample_dict = unioni_9.return_statistics()
    for k in sample_dict.keys():
        extra_info["unioni_9" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_2" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bj_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bj_7" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Cj_8.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Cj_8" + "/" + k] = sample_dict[k]

    sample_dict = unionj_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["unionj_6" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_1" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_B_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_4" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_C_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_5" + "/" + k] = sample_dict[k]

    sample_dict = add_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["add_3" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "/" + k] = sample_dict[k]

    # code for generating csv, gantt chart, txt file
    extra_info["backpressure"] = backpressure
    extra_info["depth"] = depth
    gen_gantt(extra_info, "mat_elemadd")

    if check_gold:
        print("Checking gold...")
        check_gold_mat_elemadd(ssname, debug_sim, cast, positive_only, out_crds, out_segs, out_vals, "ss01")
    samBench(bench, extra_info)
