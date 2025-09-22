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
def test_mat_mattransmul_FINAL(samBench, ssname, cast, positive_only, check_gold, report_stats, backpressure, depth,
                               debug_sim, fill=0):
    C_dirname = os.path.join(formatted_dir, ssname, "mat_mattransmul")
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

    d_dirname = C_dirname
#    d_fname = [f for f in os.listdir(d_dirname) if ssname + "-vec_mode0" in f]
#    assert len(d_fname) == 1, "Should only have one 'other' folder that matches"
#    d_fname = d_fname[0]
#    d_dirname = os.path.join(d_dirname, d_fname)

    d_shape = [C_shape[0]]

    d0_seg_filename = os.path.join(d_dirname, "tensor_d_mode_0_seg")
    d_seg0 = read_inputs(d0_seg_filename)
    d0_crd_filename = os.path.join(d_dirname, "tensor_d_mode_0_crd")
    d_crd0 = read_inputs(d0_crd_filename)

    d_vals_filename = os.path.join(d_dirname, "tensor_d_mode_vals")
    d_vals = read_inputs(d_vals_filename, float)

    f_dirname = C_dirname
#    f_fname = [f for f in os.listdir(f_dirname) if ssname + "-vec_mode1" in f]
#    assert len(f_fname) == 1, "Should only have one 'other' folder that matches"
#    f_fname = f_fname[0]
#    f_dirname = os.path.join(f_dirname, f_fname)
    f_shape = [C_shape[1]]

    f0_seg_filename = os.path.join(f_dirname, "tensor_f_mode_0_seg")
    f_seg0 = read_inputs(f0_seg_filename)
    f0_crd_filename = os.path.join(f_dirname, "tensor_f_mode_0_crd")
    f_crd0 = read_inputs(f0_crd_filename)

    f_vals_filename = os.path.join(f_dirname, "tensor_f_mode_vals")
    f_vals = read_inputs(f_vals_filename, float)

    e_shape = [0]
#    e_vals_filename = os.path.join(f_dirname, "tensor_e_mode_vals")
#    e_vals = read_inputs(e_vals_filename, float)
    e_vals = [2]

    b_shape = [0]
#    b_vals_filename = os.path.join(f_dirname, "tensor_b_mode_vals")
#    b_vals = read_inputs(e_vals_filename, float)
    b_vals = [2]

    C_shape1_min = min(C_shape[1], len(f_crd0) + len(C_crd1))
    fiberlookup_Ci_27 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    fiberlookup_fi_28 = CompressedCrdRdScan(crd_arr=f_crd0, seg_arr=f_seg0, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    unioni_26 = Union2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Cj_18 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    fiberwrite_x0_1 = CompressWrScan(seg_size=2, size=C_shape1_min, fill=fill, debug=debug_sim, statistics=report_stats,
                                     back_en=backpressure, depth=int(depth))
    repsiggen_i_24 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_bi_20 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_di_21 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_ei_22 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_dj_19 = CompressedCrdRdScan(crd_arr=d_crd0, seg_arr=d_seg0, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    intersectj_17 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repsiggen_j_16 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_C_7 = Array(init_arr=C_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure,
                          depth=int(depth))
    arrayvals_d_8 = Array(init_arr=d_vals, debug=debug_sim, statistics=report_stats)
    repeat_bj_12 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_b_6 = Array(init_arr=b_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure,
                          depth=int(depth))
    arrayvals_e_10 = Array(init_arr=e_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure,
                           depth=int(depth))
    arrayvals_f_11 = Array(init_arr=f_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure,
                           depth=int(depth))
    mul_5 = Multiply2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    mul_9 = Multiply2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    mul_4 = Multiply2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    add_3 = Add2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    reduce_2 = Reduce(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_xvals_0 = ValsWrScan(size=1 * C_shape1_min, fill=fill, debug=debug_sim, statistics=report_stats,
                                    back_en=backpressure, depth=int(depth))
    in_ref_C = [0, 'D']
    in_ref_f = [0, 'D']
    in_ref_b = [0, 'D']
    in_ref_d = [0, 'D']
    in_ref_e = [0, 'D']
    done = False
    time_cnt = 0

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
    extra_info["tensor_e_shape"] = e_shape
    extra_info["tensor_f_shape"] = f_shape

    extra_info["tensor_b/nnz"] = len(b_vals)
    extra_info["tensor_C/nnz"] = len(C_vals)
    extra_info["tensor_D/nnz"] = len(d_vals)
    extra_info["tensor_e/nnz"] = len(e_vals)
    extra_info["tensor_f/nnz"] = len(f_vals)

    extra_info["result/vals_size"] = len(out_vals)
    extra_info["result/nnz"] = len([x for x in out_vals if x != 0])

    sample_dict = fiberlookup_Ci_27.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ci_27" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_fi_28.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_fi_28" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Cj_18.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Cj_18" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_dj_19.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_dj_19" + "/" + k] = sample_dict[k]

    sample_dict = unioni_26.return_statistics()
    for k in sample_dict.keys():
        extra_info["unioni_26" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_x0_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_x0_1" + "/" + k] = sample_dict[k]

    sample_dict = repeat_bi_20.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_bi_20" + "/" + k] = sample_dict[k]

    sample_dict = repeat_bj_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_bj_12" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_b_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_b_6" + "/" + k] = sample_dict[k]

    sample_dict = reduce_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["reduce_2" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_xvals_0" + "/" + k] = sample_dict[k]

    sample_dict = repeat_di_21.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_di_21" + "/" + k] = sample_dict[k]

    sample_dict = intersectj_17.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectj_17" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_e_10.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_e_10" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_f_11.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_f_11" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_C_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_7" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_d_8.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_d_8" + "/" + k] = sample_dict[k]

    sample_dict = repeat_ei_22.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_ei_22" + "_" + k] = sample_dict[k]

    sample_dict = mul_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["mul_5" + "_" + k] = sample_dict[k]

    sample_dict = mul_9.return_statistics()
    for k in sample_dict.keys():
        extra_info["mul_9" + "_" + k] = sample_dict[k]

    sample_dict = mul_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["mul_4" + "_" + k] = sample_dict[k]

    sample_dict = add_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["add_3" + "_" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_mat_mattransmul(ssname, debug_sim, cast, positive_only, out_crds, out_segs, out_vals, "s0")
    samBench(bench, extra_info)
