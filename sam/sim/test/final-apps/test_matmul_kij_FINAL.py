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


@pytest.mark.suitesparse
def test_matmul_kij_FINAL(samBench, ssname, cast, positive_only, check_gold, report_stats, debug_sim, backpressure,
                          depth, fill=0):
    B_dirname = os.path.join(formatted_dir, ssname, "matmul_kij")
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

    # THIS IS FOR SIZE INFO
    Bs_dirname = B_dirname
    Bs_seg = read_inputs(os.path.join(Bs_dirname, "tensor_B_mode_0_seg"))

    fiberlookup_Bk_17 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    fiberlookup_Ck_18 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    intersectk_16 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Bi_15 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    repsiggen_i_13 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_Ci_12 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Cj_11 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    arrayvals_C_7 = Array(init_arr=C_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold_4 = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repsiggen_j_9 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_Bj_8 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_B_6 = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    mul_5 = Multiply2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    spaccumulator2_3 = SparseAccumulator2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    spaccumulator2_3_drop_crd_inner = StknDrop(debug=debug_sim, statistics=report_stats,
                                               back_en=backpressure, depth=int(depth))
    spaccumulator2_3_drop_crd_outer = StknDrop(debug=debug_sim, statistics=report_stats,
                                               back_en=backpressure, depth=int(depth))
    spaccumulator2_3_drop_val = StknDrop(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * Bs_seg[-1] * Bs_seg[-1], fill=fill, debug=debug_sim, statistics=report_stats,
                                    back_en=backpressure, depth=int(depth))
    fiberwrite_X1_1 = CompressWrScan(seg_size=Bs_seg[-1] + 1, size=Bs_seg[-1] * Bs_seg[-1], fill=fill,
                                     debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_X0_2 = CompressWrScan(seg_size=2, size=Bs_seg[-1], fill=fill, debug=debug_sim, statistics=report_stats,
                                     back_en=backpressure, depth=int(depth))
    in_ref_B = [0, 'D']
    in_ref_C = [0, 'D']
    done = False
    time_cnt = 0
    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bk_17.set_in_ref(in_ref_B.pop(0), "")

        if len(in_ref_C) > 0:
            fiberlookup_Ck_18.set_in_ref(in_ref_C.pop(0), "")

        intersectk_16.set_in1(fiberlookup_Bk_17.out_ref(), fiberlookup_Bk_17.out_crd(), fiberlookup_Bk_17)
        intersectk_16.set_in2(fiberlookup_Ck_18.out_ref(), fiberlookup_Ck_18.out_crd(), fiberlookup_Ck_18)

        fiberlookup_Bi_15.set_in_ref(intersectk_16.out_ref1(), intersectk_16)

        repsiggen_i_13.set_istream(fiberlookup_Bi_15.out_crd(), fiberlookup_Bi_15)

        repeat_Ci_12.set_in_ref(intersectk_16.out_ref2(), intersectk_16)
        repeat_Ci_12.set_in_repsig(repsiggen_i_13.out_repsig(), repsiggen_i_13)

        fiberlookup_Cj_11.set_in_ref(repeat_Ci_12.out_ref(), repeat_Ci_12)

        arrayvals_C_7.set_load(fiberlookup_Cj_11.out_ref(), fiberlookup_Cj_11)

        crdhold_4.set_outer_crd(fiberlookup_Bi_15.out_crd(), fiberlookup_Bi_15)
        crdhold_4.set_inner_crd(fiberlookup_Cj_11.out_crd(), fiberlookup_Cj_11)
        repsiggen_j_9.set_istream(fiberlookup_Cj_11.out_crd(), fiberlookup_Cj_11)

        repeat_Bj_8.set_in_ref(fiberlookup_Bi_15.out_ref(), fiberlookup_Bi_15)
        repeat_Bj_8.set_in_repsig(repsiggen_j_9.out_repsig(), repsiggen_j_9)

        arrayvals_B_6.set_load(repeat_Bj_8.out_ref(), repeat_Bj_8)

        mul_5.set_in1(arrayvals_B_6.out_val(), arrayvals_B_6)
        mul_5.set_in2(arrayvals_C_7.out_val(), arrayvals_C_7)

        spaccumulator2_3_drop_crd_outer.set_in_stream(crdhold_4.out_crd_outer(), crdhold_4)
        spaccumulator2_3_drop_crd_inner.set_in_stream(crdhold_4.out_crd_inner(), crdhold_4)
        spaccumulator2_3_drop_val.set_in_stream(mul_5.out_val(), mul_5)
        spaccumulator2_3.set_crd_outer(spaccumulator2_3_drop_crd_outer.out_val(), spaccumulator2_3_drop_crd_outer)
        spaccumulator2_3.set_crd_inner(spaccumulator2_3_drop_crd_inner.out_val(), spaccumulator2_3_drop_crd_inner)
        spaccumulator2_3.set_val(spaccumulator2_3_drop_val.out_val(), spaccumulator2_3_drop_val)

        fiberwrite_Xvals_0.set_input(spaccumulator2_3.out_val(), spaccumulator2_3)

        fiberwrite_X1_1.set_input(spaccumulator2_3.out_crd_inner(), spaccumulator2_3)

        fiberwrite_X0_2.set_input(spaccumulator2_3.out_crd_outer(), spaccumulator2_3)

        fiberlookup_Bk_17.update()
        fiberlookup_Ck_18.update()
        intersectk_16.update()
        fiberlookup_Bi_15.update()
        repsiggen_i_13.update()
        repeat_Ci_12.update()
        fiberlookup_Cj_11.update()
        arrayvals_C_7.update()
        crdhold_4.update()
        repsiggen_j_9.update()
        repeat_Bj_8.update()
        arrayvals_B_6.update()
        mul_5.update()
        spaccumulator2_3_drop_crd_outer.update()
        spaccumulator2_3_drop_crd_inner.update()
        spaccumulator2_3_drop_val.update()
        spaccumulator2_3.update()
        fiberwrite_Xvals_0.update()
        fiberwrite_X1_1.update()
        fiberwrite_X0_2.update()

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

    sample_dict = intersectk_16.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectk_16" + "/" + k] = sample_dict[k]

    sample_dict = spaccumulator2_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["spaccumulator2_3" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_1" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_2" + "/" + k] = sample_dict[k]

    sample_dict = repeat_Ci_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Ci_12" + "/" + k] = sample_dict[k]

    sample_dict = repeat_Bj_8.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Bj_8" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_B_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_6" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_C_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_7" + "/" + k] = sample_dict[k]

    sample_dict = mul_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["mul_5" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bk_17.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bk_17" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Ck_18.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ck_18" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bi_15.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bi_15" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Cj_11.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Cj_11" + "/" + k] = sample_dict[k]

    gen_gantt(extra_info, "matmul_kij")

    if check_gold:
        print("Checking gold...")
        check_gold_matmul(ssname, debug_sim, cast, positive_only, out_crds, out_segs, out_vals, "ss01")
    samBench(bench, extra_info)
