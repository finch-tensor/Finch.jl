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
def test_tensor3_innerprod_FINAL(samBench, frosttname, check_gold, report_stats, debug_sim, backpressure, depth, fill=0):
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
    B2_crd_filename = os.path.join(B_dirname, "tensor_B_mode_2_seg")
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
    C1_crd_filename = os.path.join(C_dirname, "tensor_C_mode_1_seg")
    C_crd1 = read_inputs(C1_crd_filename)

    C2_seg_filename = os.path.join(C_dirname, "tensor_C_mode_2_seg")
    C_seg2 = read_inputs(C2_seg_filename)
    C2_crd_filename = os.path.join(C_dirname, "tensor_C_mode_2_seg")
    C_crd2 = read_inputs(C2_crd_filename)

    C_vals_filename = os.path.join(C_dirname, "tensor_C_mode_vals")
    C_vals = read_inputs(C_vals_filename, float)

    fiberlookup_Bi_14 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    fiberlookup_Ci_15 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    intersecti_13 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Bj_11 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    fiberlookup_Cj_12 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    intersectj_10 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Bk_8 = CompressedCrdRdScan(crd_arr=B_crd2, seg_arr=B_seg2, debug=debug_sim, statistics=report_stats,
                                           back_en=backpressure, depth=int(depth))
    fiberlookup_Ck_9 = CompressedCrdRdScan(crd_arr=C_crd2, seg_arr=C_seg2, debug=debug_sim, statistics=report_stats,
                                           back_en=backpressure, depth=int(depth))
    intersectk_7 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_B_5 = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_C_6 = Array(init_arr=C_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    mul_4 = Multiply2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    reduce_3 = Reduce(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    reduce_2 = Reduce(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    reduce_1 = Reduce(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_xvals_0 = ValsWrScan(size=1, fill=fill, debug=debug_sim, statistics=report_stats, back_en=backpressure,
                                    depth=int(depth))
    in_ref_B = [0, 'D']
    in_ref_C = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_14.set_in_ref(in_ref_B.pop(0), "")
        if len(in_ref_C) > 0:
            fiberlookup_Ci_15.set_in_ref(in_ref_C.pop(0), "")
        intersecti_13.set_in1(fiberlookup_Bi_14.out_ref(), fiberlookup_Bi_14.out_crd(), fiberlookup_Bi_14)
        intersecti_13.set_in2(fiberlookup_Ci_15.out_ref(), fiberlookup_Ci_15.out_crd(), fiberlookup_Ci_15)
        fiberlookup_Bj_11.set_in_ref(intersecti_13.out_ref1(), intersecti_13)
        fiberlookup_Cj_12.set_in_ref(intersecti_13.out_ref2(), intersecti_13)
        intersectj_10.set_in1(fiberlookup_Bj_11.out_ref(), fiberlookup_Bj_11.out_crd(), fiberlookup_Bj_11)
        intersectj_10.set_in2(fiberlookup_Cj_12.out_ref(), fiberlookup_Cj_12.out_crd(), fiberlookup_Cj_12)
        fiberlookup_Bk_8.set_in_ref(intersectj_10.out_ref1(), intersectj_10)
        fiberlookup_Ck_9.set_in_ref(intersectj_10.out_ref2(), intersectj_10)
        intersectk_7.set_in1(fiberlookup_Bk_8.out_ref(), fiberlookup_Bk_8.out_crd(), fiberlookup_Bk_8)
        intersectk_7.set_in2(fiberlookup_Ck_9.out_ref(), fiberlookup_Ck_9.out_crd(), fiberlookup_Ck_9)
        arrayvals_B_5.set_load(intersectk_7.out_ref1(), intersectk_7)
        arrayvals_C_6.set_load(intersectk_7.out_ref2(), intersectk_7)
        mul_4.set_in1(arrayvals_B_5.out_val(), arrayvals_B_5)
        mul_4.set_in2(arrayvals_C_6.out_val(), arrayvals_C_6)
        reduce_3.set_in_val(mul_4.out_val(), mul_4)
        reduce_2.set_in_val(reduce_3.out_val(), reduce_3)
        reduce_1.set_in_val(reduce_2.out_val(), reduce_2)
        fiberwrite_xvals_0.set_input(reduce_1.out_val(), reduce_1)

        fiberlookup_Bi_14.update()
        fiberlookup_Ci_15.update()
        intersecti_13.update()
        fiberlookup_Bj_11.update()
        fiberlookup_Cj_12.update()
        intersectj_10.update()
        fiberlookup_Bk_8.update()
        fiberlookup_Ck_9.update()
        intersectk_7.update()
        arrayvals_B_5.update()
        arrayvals_C_6.update()
        mul_4.update()
        reduce_3.update()
        reduce_2.update()
        reduce_1.update()
        fiberwrite_xvals_0.update()
        done = fiberwrite_xvals_0.out_done()
        time_cnt += 1

    fiberwrite_xvals_0.autosize()

    out_crds = []
    out_segs = []
    out_vals = fiberwrite_xvals_0.get_arr()

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

    sample_dict = intersecti_13.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersecti_13" + "/" + k] = sample_dict[k]

    sample_dict = intersectj_10.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectj_10" + "/" + k] = sample_dict[k]

    sample_dict = intersectk_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectk_7" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_B_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_5" + "/" + k] = sample_dict[k]

    sample_dict = reduce_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["reduce_3" + "/" + k] = sample_dict[k]

    sample_dict = reduce_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["reduce_2" + "/" + k] = sample_dict[k]

    sample_dict = reduce_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["reduce_1" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_xvals_0" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_C_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_6" + "/" + k] = sample_dict[k]

    sample_dict = mul_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["mul_4" + "/" + k] = sample_dict[k]

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
        check_gold_tensor3_innerprod(frosttname, debug_sim, out_crds, out_segs, out_vals, "none")
    samBench(bench, extra_info)
