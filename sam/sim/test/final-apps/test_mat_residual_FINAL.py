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


# FIXME: Figureout formats
@pytest.mark.suitesparse
def test_mat_residual(samBench, ssname, cast, positive_only, check_gold, report_stats, debug_sim, backpressure, depth, fill=0):
    C_dirname = os.path.join(formatted_dir, ssname, "mat_residual")
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

    b_dirname = C_dirname
#    b_fname = [f for f in os.listdir(b_dirname) if ssname + "-vec_mode0" in f]
#    assert len(b_fname) == 1, "Should only have one 'other' folder that matches"
#    b_fname = b_fname[0]
#    b_dirname = os.path.join(b_dirname, b_fname)

    b_shape = [C_shape[0]]
    b0_seg_filename = os.path.join(b_dirname, "tensor_b_mode_0_seg")
    b_seg0 = read_inputs(b0_seg_filename, positive_only=positive_only)
    b0_crd_filename = os.path.join(b_dirname, "tensor_b_mode_0_crd")
    b_crd0 = read_inputs(b0_crd_filename, positive_only=positive_only)

    b_vals_filename = os.path.join(b_dirname, "tensor_b_mode_vals")
    b_vals = read_inputs(b_vals_filename, float)

    d_dirname = C_dirname
#    d_fname = [f for f in os.listdir(d_dirname) if ssname + "-vec_mode1" in f]
#    assert len(d_fname) == 1, "Should only have one 'other' folder that matches"
#    d_fname = d_fname[0]
#    d_dirname = os.path.join(d_dirname, d_fname)

    d_shape = [C_shape[1]]
    d0_seg_filename = os.path.join(d_dirname, "tensor_d_mode_0_seg")
    d_seg0 = read_inputs(d0_seg_filename, positive_only=positive_only)
    d0_crd_filename = os.path.join(d_dirname, "tensor_d_mode_0_crd")
    d_crd0 = read_inputs(d0_crd_filename, positive_only=positive_only)

    d_vals_filename = os.path.join(d_dirname, "tensor_d_mode_vals")
    d_vals = read_inputs(d_vals_filename, float, positive_only=positive_only)

    C_shape0_min = min(len(b_vals) + len(C_crd0), b_shape[0])
    fiberlookup_bi_17 = CompressedCrdRdScan(crd_arr=b_crd0, seg_arr=b_seg0, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    fiberlookup_Ci_18 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    unioni_16 = Union2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Cj_11 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    fiberwrite_x0_1 = CompressWrScan(seg_size=2, size=C_shape0_min, fill=fill, debug=debug_sim, statistics=report_stats,
                                     back_en=backpressure, depth=int(depth))
    repsiggen_i_14 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_di_13 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_dj_12 = CompressedCrdRdScan(crd_arr=d_crd0, seg_arr=d_seg0, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    intersectj_10 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_C_6 = Array(init_arr=C_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_d_7 = Array(init_arr=d_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    mul_5 = Multiply2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_b_4 = Array(init_arr=b_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    add_3 = Add2(debug=debug_sim, neg2=True, statistics=report_stats, back_en=backpressure, depth=int(depth))
    reduce_2 = Reduce(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_xvals_0 = ValsWrScan(size=1 * C_shape0_min, fill=fill, debug=debug_sim, statistics=report_stats,
                                    back_en=backpressure, depth=int(depth))
    in_ref_b = [0, 'D']
    in_ref_C = [0, 'D']
    in_ref_d = [0, 'D']
    done = False
    time_cnt = 0

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

        intersectj_10.set_in1(fiberlookup_dj_12.out_ref(), fiberlookup_dj_12.out_crd(), fiberlookup_dj_12)
        intersectj_10.set_in2(fiberlookup_Cj_11.out_ref(), fiberlookup_Cj_11.out_crd(), fiberlookup_Cj_11)

        arrayvals_C_6.set_load(intersectj_10.out_ref2(), intersectj_10)

        arrayvals_d_7.set_load(intersectj_10.out_ref1(), intersectj_10)

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
        arrayvals_C_6.update()
        arrayvals_d_7.update()
        arrayvals_b_4.update()
        mul_5.update()
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

    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = ssname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_b_shape"] = b_shape
    extra_info["tensor_C_shape"] = C_shape
    extra_info["tensor_d_shape"] = d_shape

    extra_info["tensor_b/nnz"] = len(b_vals)
    extra_info["tensor_C/nnz"] = len(C_vals)
    extra_info["tensor_d/nnz"] = len(d_vals)

    extra_info["result/vals_size"] = len(out_vals)
    extra_info["result/nnz"] = len([x for x in out_vals if x != 0])

    sample_dict = unioni_16.return_statistics()
    for k in sample_dict.keys():
        extra_info["unioni_16" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_x0_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_x0_1" + "/" + k] = sample_dict[k]

    sample_dict = repeat_di_13.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_di_13" + "/" + k] = sample_dict[k]

    sample_dict = intersectj_10.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectj_10" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_b_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_b_4" + "/" + k] = sample_dict[k]

    sample_dict = reduce_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["reduce_2" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_xvals_0" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_C_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_6" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_d_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_d_7" + "/" + k] = sample_dict[k]

    sample_dict = mul_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["mul_5" + "/" + k] = sample_dict[k]

    sample_dict = add_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["add_3" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_bi_17.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_bi_17" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Ci_18.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ci_18" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Cj_11.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Cj_11" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_dj_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_dj_12" + "/" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_mat_residual(ssname, debug_sim, cast, positive_only, out_crds, out_segs, out_vals, "s0")
    samBench(bench, extra_info)
