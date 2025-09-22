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


@pytest.mark.suitesparse
def test_mat_vecmul_FINAL(samBench, ssname, cast, positive_only, check_gold, report_stats, debug_sim, backpressure,
                          depth, fill=0):
    B_dirname = os.path.join(formatted_dir, ssname, "mat_vecmul")
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

    c_dirname = B_dirname
#    c_fname = [f for f in os.listdir(c_dirname) if ssname + "-vec_mode1" in f]
#    assert len(c_fname) == 1, "Should only have one 'other' folder that matches"
#    c_fname = c_fname[0]
#    c_dirname = os.path.join(c_dirname, c_fname)

    c_shape_filename = os.path.join(c_dirname, "tensor_c_mode_shape")
    c_shape = read_inputs(c_shape_filename, positive_only=positive_only)

    c0_seg_filename = os.path.join(c_dirname, "tensor_c_mode_0_seg")
    c_seg0 = read_inputs(c0_seg_filename, positive_only=positive_only)
    c0_crd_filename = os.path.join(c_dirname, "tensor_c_mode_0_crd")
    c_crd0 = read_inputs(c0_crd_filename, positive_only=positive_only)

    c_vals_filename = os.path.join(c_dirname, "tensor_c_mode_vals")
    c_vals = read_inputs(c_vals_filename, float, positive_only=positive_only)

    # THIS IS FOR SIZE INFO
    Bs_dirname = B_dirname
    Bs_seg = read_inputs(os.path.join(Bs_dirname, "tensor_B_mode_0_seg"))

    fiberlookup_Bj_11 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    fiberlookup_cj_12 = CompressedCrdRdScan(crd_arr=c_crd0, seg_arr=c_seg0, debug=debug_sim, statistics=report_stats,
                                            back_en=backpressure, depth=int(depth))
    intersectj_10 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Bi_9 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats,
                                           back_en=backpressure, depth=int(depth))
    arrayvals_B_4 = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repsiggen_i_7 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_ci_6 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_c_5 = Array(init_arr=c_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    mul_3 = Multiply2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    spaccumulator1_2 = SparseAccumulator1(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    spaccumulator1_2_drop_crd_outer = StknDrop(debug=debug_sim, statistics=report_stats,
                                               back_en=backpressure, depth=int(depth))
    spaccumulator1_2_drop_crd_inner = StknDrop(debug=debug_sim, statistics=report_stats,
                                               back_en=backpressure, depth=int(depth))
    spaccumulator1_2_drop_val = StknDrop(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_xvals_0 = ValsWrScan(size=1 * Bs_seg[-1], fill=fill, debug=debug_sim, statistics=report_stats,
                                    back_en=backpressure, depth=int(depth))
    fiberwrite_x0_1 = CompressWrScan(seg_size=2, size=Bs_seg[-1], fill=fill, debug=debug_sim, statistics=report_stats,
                                     back_en=backpressure, depth=int(depth))

    into_spacc_outer, into_spacc_inner, into_spacc_val = [], [], []
    tvals = []
    t0 = []
    tintj = []
    tBvals = []
    tcvals = []
    ti = []

    in_ref_B = [0, 'D']
    in_ref_c = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bj_11.set_in_ref(in_ref_B.pop(0), "")

        if len(in_ref_c) > 0:
            fiberlookup_cj_12.set_in_ref(in_ref_c.pop(0), "")

        intersectj_10.set_in1(fiberlookup_Bj_11.out_ref(), fiberlookup_Bj_11.out_crd(), fiberlookup_Bj_11)
        intersectj_10.set_in2(fiberlookup_cj_12.out_ref(), fiberlookup_cj_12.out_crd(), fiberlookup_cj_12)

        fiberlookup_Bi_9.set_in_ref(intersectj_10.out_ref1(), intersectj_10)

        arrayvals_B_4.set_load(fiberlookup_Bi_9.out_ref(), fiberlookup_Bi_9)

        repsiggen_i_7.set_istream(fiberlookup_Bi_9.out_crd(), fiberlookup_Bi_9)

        repeat_ci_6.set_in_ref(intersectj_10.out_ref2(), intersectj_10)
        repeat_ci_6.set_in_repsig(repsiggen_i_7.out_repsig(), repsiggen_i_7)

        arrayvals_c_5.set_load(repeat_ci_6.out_ref(), repeat_ci_6)

        mul_3.set_in1(arrayvals_c_5.out_val(), arrayvals_c_5)
        mul_3.set_in2(arrayvals_B_4.out_val(), arrayvals_B_4)

        crdhold.set_outer_crd(intersectj_10.out_crd(), intersectj_10)
        crdhold.set_inner_crd(fiberlookup_Bi_9.out_crd(), fiberlookup_Bi_9)

        spaccumulator1_2_drop_crd_outer.set_in_stream(crdhold.out_crd_outer(), crdhold)
        spaccumulator1_2_drop_crd_inner.set_in_stream(crdhold.out_crd_inner(), crdhold)
        spaccumulator1_2_drop_val.set_in_stream(mul_3.out_val(), mul_3)

        spacc_outer_val = 0 if isinstance(spaccumulator1_2_drop_crd_outer.out_val(), int) else \
            spaccumulator1_2_drop_crd_outer.out_val()

        spaccumulator1_2.set_crd_outer(spacc_outer_val, spaccumulator1_2_drop_crd_outer)
        spaccumulator1_2.set_crd_inner(spaccumulator1_2_drop_crd_inner.out_val(), spaccumulator1_2_drop_crd_inner)
        spaccumulator1_2.set_val(spaccumulator1_2_drop_val.out_val(), spaccumulator1_2_drop_val)

        fiberwrite_xvals_0.set_input(spaccumulator1_2.out_val(), spaccumulator1_2)

        out_crdi = spaccumulator1_2.out_crd_inner() if isinstance(spaccumulator1_2.out_crd_inner(),
                                                                  int) else decrement_stkn(
            spaccumulator1_2.out_crd_inner()) \
            if is_stkn(spaccumulator1_2.out_crd_inner()) else 'D' if spaccumulator1_2.out_crd_inner() == 'D' else ''

        fiberwrite_x0_1.set_input(out_crdi, spaccumulator1_2)

        into_spacc_outer.append(spaccumulator1_2_drop_crd_outer.out_val())
        into_spacc_inner.append(spaccumulator1_2_drop_crd_inner.out_val())
        into_spacc_val.append(spaccumulator1_2_drop_val.out_val())
        tvals.append(spaccumulator1_2.out_val())
        t0.append(out_crdi)
        tintj.append(intersectj_10.out_crd())
        ti.append(fiberlookup_Bi_9.out_crd())
        tBvals.append(arrayvals_B_4.out_val())
        tcvals.append(arrayvals_c_5.out_val())

        fiberlookup_Bj_11.update()
        fiberlookup_cj_12.update()
        intersectj_10.update()
        fiberlookup_Bi_9.update()
        arrayvals_B_4.update()
        repsiggen_i_7.update()
        repeat_ci_6.update()
        arrayvals_c_5.update()
        mul_3.update()
        crdhold.update()
        spaccumulator1_2_drop_crd_outer.update()
        spaccumulator1_2_drop_crd_inner.update()
        spaccumulator1_2_drop_val.update()
        spaccumulator1_2.update()
        fiberwrite_x0_1.update()
        fiberwrite_xvals_0.update()

        done = fiberwrite_x0_1.out_done() and fiberwrite_xvals_0.out_done()
        time_cnt += 1

    fiberwrite_x0_1.autosize()
    fiberwrite_xvals_0.autosize()

    if debug_sim:
        print("intj", remove_emptystr(tintj))
        print("i", remove_emptystr(ti))
        print("Bvals", remove_emptystr(tBvals))
        print("cvals", remove_emptystr(tcvals))
        print()
        print("into spacc outer", remove_emptystr(into_spacc_outer))
        print("into spacc inner", remove_emptystr(into_spacc_inner))
        print("into spacc val", remove_emptystr(into_spacc_val))

        print("out_vals", remove_emptystr(tvals))
        print("out_crdi", remove_emptystr(t0))

    out_crds = [fiberwrite_x0_1.get_arr()]
    out_segs = [fiberwrite_x0_1.get_seg_arr()]
    out_vals = fiberwrite_xvals_0.get_arr()

    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = ssname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_c_shape"] = c_shape
    extra_info["tensor_B/nnz"] = len(B_vals)
    extra_info["tensor_c/nnz"] = len(c_vals)

    extra_info["result/vals_size"] = len(out_vals)
    extra_info["result/nnz"] = len([x for x in out_vals if x != 0])

    sample_dict = intersectj_10.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectj_10" + "/" + k] = sample_dict[k]

    sample_dict = spaccumulator1_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["spaccumulator1_2" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_xvals_0" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_x0_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_x0_1" + "/" + k] = sample_dict[k]

    sample_dict = repeat_ci_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_ci_6" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_c_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_c_5" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_B_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_4" + "/" + k] = sample_dict[k]

    sample_dict = mul_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["mul_3" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bj_11.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bi_11" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_cj_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_ci_12" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bi_9.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bi_9" + "/" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_mat_vecmul_ji(ssname, debug_sim, cast, positive_only, out_crds, out_segs, out_vals, "s0")
    samBench(bench, extra_info)
