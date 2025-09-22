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

other_dir = os.getenv('OTHER_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))


# FIXME: Figureout formats
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.frostt
def test_tensor3_ttv_FINAL(samBench, frosttname, check_gold, report_stats, debug_sim, backpressure, depth, fill=0):
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

    c_dirname = os.path.join(cwd, "tmp_mat")
    c_fname = [f for f in os.listdir(c_dirname) if frosttname + "-vec_mode2" in f]
    assert len(c_fname) == 1, "Should only have one 'other' folder that matches"
    c_fname = c_fname[0]
    c_dirname = os.path.join(c_dirname, c_fname)

    c_shape_filename = os.path.join(c_dirname, "tensor_c_mode_shape")
    c_shape = read_inputs(c_shape_filename)

    c0_seg_filename = os.path.join(c_dirname, "tensor_c_mode_0_seg")
    c_seg0 = read_inputs(c0_seg_filename)
    c0_crd_filename = os.path.join(c_dirname, "tensor_c_mode_0_crd")
    c_crd0 = read_inputs(c0_crd_filename)

    c_vals_filename = os.path.join(c_dirname, "tensor_c_mode_vals")
    c_vals = read_inputs(c_vals_filename, float)

    fiberlookup_Bi_17 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    fiberlookup_Bj_13 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    fiberwrite_X0_2 = CompressWrScan(seg_size=2, size=len(B_crd0), fill=fill, debug=debug_sim, statistics=report_stats,
                                     back_en=backpressure, depth=int(depth))
    repsiggen_i_15 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Bk_8 = CompressedCrdRdScan(crd_arr=B_crd2, seg_arr=B_seg2, debug=debug_sim, statistics=report_stats,
                                           back_en=backpressure, depth=int(depth))
    fiberwrite_X1_1 = CompressWrScan(seg_size=len(B_crd0) + 1, size=len(B_crd1), fill=fill,
                                     debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repsiggen_j_11 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_ci_14 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_cj_10 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_ck_9 = CompressedCrdRdScan(crd_arr=c_crd0, seg_arr=c_seg0, debug=debug_sim, statistics=report_stats,
                                           back_en=backpressure, depth=int(depth))
    intersectk_7 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_B_5 = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_c_6 = Array(init_arr=c_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    mul_4 = Multiply2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    reduce_3 = Reduce(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * len(B_crd1), fill=fill, debug=debug_sim, statistics=report_stats,
                                    back_en=backpressure, depth=int(depth))
    in_ref_B = [0, 'D']
    in_ref_c = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_17.set_in_ref(in_ref_B.pop(0), "")
        fiberlookup_Bj_13.set_in_ref(fiberlookup_Bi_17.out_ref(), fiberlookup_Bi_17)

        fiberwrite_X0_2.set_input(fiberlookup_Bi_17.out_crd(), fiberlookup_Bi_17)

        repsiggen_i_15.set_istream(fiberlookup_Bi_17.out_crd(), fiberlookup_Bi_17)

        if len(in_ref_c) > 0:
            repeat_ci_14.set_in_ref(in_ref_c.pop(0), "")
        repeat_ci_14.set_in_repsig(repsiggen_i_15.out_repsig(), repsiggen_i_15)

        fiberlookup_Bk_8.set_in_ref(fiberlookup_Bj_13.out_ref(), fiberlookup_Bj_13)

        fiberwrite_X1_1.set_input(fiberlookup_Bj_13.out_crd(), fiberlookup_Bj_13)

        repsiggen_j_11.set_istream(fiberlookup_Bj_13.out_crd(), fiberlookup_Bj_13)

        repeat_cj_10.set_in_ref(repeat_ci_14.out_ref(), repeat_ci_14)
        repeat_cj_10.set_in_repsig(repsiggen_j_11.out_repsig(), repsiggen_j_11)

        fiberlookup_ck_9.set_in_ref(repeat_cj_10.out_ref(), repeat_cj_10)

        intersectk_7.set_in1(fiberlookup_ck_9.out_ref(), fiberlookup_ck_9.out_crd(), fiberlookup_ck_9)
        intersectk_7.set_in2(fiberlookup_Bk_8.out_ref(), fiberlookup_Bk_8.out_crd(), fiberlookup_Bk_8)

        arrayvals_B_5.set_load(intersectk_7.out_ref2(), intersectk_7)

        arrayvals_c_6.set_load(intersectk_7.out_ref1(), intersectk_7)

        mul_4.set_in1(arrayvals_B_5.out_val(), arrayvals_B_5)
        mul_4.set_in2(arrayvals_c_6.out_val(), arrayvals_c_6)

        reduce_3.set_in_val(mul_4.out_val(), mul_4)

        fiberwrite_Xvals_0.set_input(reduce_3.out_val(), reduce_3)

        fiberlookup_Bi_17.update()
        fiberlookup_Bj_13.update()
        fiberwrite_X0_2.update()
        repsiggen_i_15.update()
        repeat_ci_14.update()
        fiberlookup_Bk_8.update()
        fiberwrite_X1_1.update()
        repsiggen_j_11.update()
        repeat_cj_10.update()
        fiberlookup_ck_9.update()
        intersectk_7.update()
        arrayvals_B_5.update()
        arrayvals_c_6.update()
        mul_4.update()
        reduce_3.update()
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
    extra_info["dataset"] = frosttname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_c_shape"] = c_shape
    extra_info["tensor_B/nnz"] = len(B_vals)
    extra_info["tensor_C/nnz"] = len(c_vals)

    extra_info["result/vals_size"] = len(out_vals)
    extra_info["result/nnz"] = len([x for x in out_vals if x != 0])

    sample_dict = fiberwrite_X0_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_2" + "/" + k] = sample_dict[k]

    sample_dict = repeat_ci_14.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_ci_14" + "/" + k] = sample_dict[k]

    sample_dict = repeat_cj_10.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_cj_10" + "/" + k] = sample_dict[k]

    sample_dict = intersectk_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectk_7" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_B_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_5" + "/" + k] = sample_dict[k]

    sample_dict = reduce_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["reduce_3" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_c_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_c_6" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_1" + "/" + k] = sample_dict[k]

    sample_dict = mul_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["mul_4" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bi_17.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bi_17" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bj_13.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bj_13" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bj_8.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bj_8" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_ck_9.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_ck_9" + "/" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_tensor3_ttv(frosttname, debug_sim, out_crds, out_segs, out_vals, "ss01")
    samBench(bench, extra_info)
