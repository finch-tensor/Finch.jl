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
def test_tensor3_ttm_FINAL(samBench, frosttname, check_gold, report_stats, debug_sim, backpressure, depth, fill=0):
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
    C_fname = [f for f in os.listdir(C_dirname) if frosttname + "-mat_mode2_ttm" in f]
    assert len(C_fname) == 1, "Should only have one 'other' folder that matches"
    C_fname = C_fname[0]
    C_dirname = os.path.join(C_dirname, C_fname)

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

    C_vals_filename = os.path.join(C_dirname, "tensor_C_mode_vals")
    C_vals = read_inputs(C_vals_filename, float)

    fiberlookup_Bi_22 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    fiberlookup_Bj_18 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    fiberwrite_X0_3 = CompressWrScan(seg_size=2, size=len(B_crd0), fill=fill, debug=debug_sim, statistics=report_stats,
                                     back_en=backpressure, depth=int(depth))
    repsiggen_i_20 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_X1_2 = CompressWrScan(seg_size=len(B_crd0) + 1, size=len(B_crd1), fill=fill,
                                     debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repsiggen_j_16 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_Ci_19 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_Cj_15 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Ck_14 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    fiberlookup_Cl_10 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    fiberwrite_X2_1 = CompressWrScan(seg_size=len(B_crd1) + 1, size=len(B_crd1) * len(C_crd0),
                                     fill=fill, debug=debug_sim, statistics=report_stats,
                                     back_en=backpressure, depth=int(depth))
    repsiggen_k_12 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_Bk_11 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Bl_9 = CompressedCrdRdScan(crd_arr=B_crd2, seg_arr=B_seg2, debug=debug_sim, statistics=report_stats,
                                           back_en=backpressure, depth=int(depth))
    intersectl_8 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_B_6 = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_C_7 = Array(init_arr=C_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    mul_5 = Multiply2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    reduce_4 = Reduce(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * len(B_crd1) * len(C_crd0), fill=fill, debug=debug_sim, statistics=report_stats,
                                    back_en=backpressure, depth=int(depth))
    in_ref_B = [0, 'D']
    in_ref_C = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_22.set_in_ref(in_ref_B.pop(0), "")

        fiberlookup_Bj_18.set_in_ref(fiberlookup_Bi_22.out_ref(), fiberlookup_Bi_22)
        fiberwrite_X0_3.set_input(fiberlookup_Bi_22.out_crd(), fiberlookup_Bi_22)
        repsiggen_i_20.set_istream(fiberlookup_Bi_22.out_crd(), fiberlookup_Bi_22)
        if len(in_ref_C) > 0:
            repeat_Ci_19.set_in_ref(in_ref_C.pop(0), "")
        repeat_Ci_19.set_in_repsig(repsiggen_i_20.out_repsig(), repsiggen_i_20)
        fiberwrite_X1_2.set_input(fiberlookup_Bj_18.out_crd(), fiberlookup_Bj_18)
        repsiggen_j_16.set_istream(fiberlookup_Bj_18.out_crd(), fiberlookup_Bj_18)
        repeat_Cj_15.set_in_ref(repeat_Ci_19.out_ref(), repeat_Ci_19)
        repeat_Cj_15.set_in_repsig(repsiggen_j_16.out_repsig(), repsiggen_j_16)
        fiberlookup_Ck_14.set_in_ref(repeat_Cj_15.out_ref(), repeat_Cj_15)
        fiberlookup_Cl_10.set_in_ref(fiberlookup_Ck_14.out_ref(), fiberlookup_Ck_14)
        fiberwrite_X2_1.set_input(fiberlookup_Ck_14.out_crd(), fiberlookup_Ck_14)
        repsiggen_k_12.set_istream(fiberlookup_Ck_14.out_crd(), fiberlookup_Ck_14)
        repeat_Bk_11.set_in_repsig(repsiggen_k_12.out_repsig(), repsiggen_k_12)
        repeat_Bk_11.set_in_ref(fiberlookup_Bj_18.out_ref(), fiberlookup_Bj_18)
        fiberlookup_Bl_9.set_in_ref(repeat_Bk_11.out_ref(), repeat_Bk_11)
        intersectl_8.set_in1(fiberlookup_Bl_9.out_ref(), fiberlookup_Bl_9.out_crd(), fiberlookup_Bl_9)
        intersectl_8.set_in2(fiberlookup_Cl_10.out_ref(), fiberlookup_Cl_10.out_crd(), fiberlookup_Cl_10)
        arrayvals_B_6.set_load(intersectl_8.out_ref1(), intersectl_8)
        arrayvals_C_7.set_load(intersectl_8.out_ref2(), intersectl_8)
        mul_5.set_in1(arrayvals_B_6.out_val(), arrayvals_B_6)
        mul_5.set_in2(arrayvals_C_7.out_val(), arrayvals_C_7)
        reduce_4.set_in_val(mul_5.out_val(), mul_5)
        fiberwrite_Xvals_0.set_input(reduce_4.out_val(), reduce_4)

        fiberlookup_Bi_22.update()
        fiberlookup_Bj_18.update()
        fiberwrite_X0_3.update()
        repsiggen_i_20.update()
        repeat_Ci_19.update()
        fiberwrite_X1_2.update()
        repsiggen_j_16.update()
        repeat_Cj_15.update()
        fiberlookup_Ck_14.update()
        fiberlookup_Cl_10.update()
        fiberwrite_X2_1.update()
        repsiggen_k_12.update()
        repeat_Bk_11.update()
        fiberlookup_Bl_9.update()
        intersectl_8.update()
        arrayvals_B_6.update()
        arrayvals_C_7.update()
        mul_5.update()
        reduce_4.update()
        fiberwrite_Xvals_0.update()
        done = fiberwrite_X0_3.out_done() and fiberwrite_X1_2.out_done() and fiberwrite_X2_1.out_done() and \
            fiberwrite_Xvals_0.out_done()
        time_cnt += 1

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

    sample_dict = fiberwrite_X0_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_3" + "/" + k] = sample_dict[k]

    sample_dict = repeat_Ci_19.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Ci_19" + "/" + k] = sample_dict[k]

    sample_dict = repeat_Cj_15.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Cj_15" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_X2_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X2_1" + "/" + k] = sample_dict[k]

    sample_dict = repeat_Bk_11.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Bk_11" + "/" + k] = sample_dict[k]

    sample_dict = intersectl_8.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectl_8" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_B_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_6" + "/" + k] = sample_dict[k]

    sample_dict = reduce_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["reduce_4" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_C_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_7" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_2" + "/" + k] = sample_dict[k]

    sample_dict = mul_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["mul_5" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bi_22.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bi_22" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bj_18.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bj_18" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Ck_14.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ck_14" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Cl_10.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Cl_10" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bl_9.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bl_9" + "/" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_tensor3_ttm(frosttname, debug_sim, out_crds, out_segs, out_vals, "sss012")
    samBench(bench, extra_info)
