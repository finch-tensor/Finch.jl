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


# FIXME: Figureout formats
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.suitesparse
def test_mat_identity(samBench, ssname, cast, positive_only, check_gold, debug_sim, report_stats, fill=0):
    B_dirname = os.path.join(formatted_dir, ssname, "mat_identity")
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

    fiberlookup_Bi_5 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats)
    fiberwrite_X0_2 = CompressWrScan(seg_size=2, size=B_shape[0], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberlookup_Bj_4 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats)
    fiberwrite_X1_1 = CompressWrScan(seg_size=B_shape[0] + 1, size=B_shape[0] * B_shape[1], fill=fill, debug=debug_sim,
                                     statistics=report_stats)
    arrayvals_B_3 = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats)
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * B_shape[0] * B_shape[1], fill=fill, debug=debug_sim,
                                    statistics=report_stats)
    in_ref_B = [0, 'D']
    done = False
    time_cnt = 0

    B_lvl0 = []
    B_lvl1 = []
    B_v = []
    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_5.set_in_ref(in_ref_B.pop(0))
        fiberwrite_X0_2.set_input(fiberlookup_Bi_5.out_crd())
        fiberlookup_Bj_4.set_in_ref(fiberlookup_Bi_5.out_ref())
        fiberwrite_X1_1.set_input(fiberlookup_Bj_4.out_crd())
        arrayvals_B_3.set_load(fiberlookup_Bj_4.out_ref())
        fiberwrite_Xvals_0.set_input(arrayvals_B_3.out_val())

        fiberlookup_Bi_5.update()
        fiberwrite_X0_2.update()
        fiberlookup_Bj_4.update()
        fiberwrite_X1_1.update()
        arrayvals_B_3.update()
        fiberwrite_Xvals_0.update()

        B_lvl0.append(fiberlookup_Bi_5.out_crd())
        B_lvl1.append(fiberlookup_Bj_4.out_crd())
        B_vals.append(arrayvals_B_3.out_val())

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

    if debug_sim:
        print()
        print("Bi stream", B_lvl0)
        print("Bj stream", B_lvl1)
        print("Bvals stream", B_vals)

    extra_info["stream_Bi_noncontrol"] = sum([1 for x in B_lvl0 if isinstance(x, int)])
    extra_info["stream_Bi_stop"] = sum([1 for x in B_lvl0 if isinstance(x, str) and is_stkn(x)])
    extra_info["stream_Bi_empty"] = sum([1 for x in B_lvl0 if isinstance(x, str) and x == ''])
    extra_info["stream_Bj_noncontrol"] = sum([1 for x in B_lvl1 if isinstance(x, int)])
    extra_info["stream_Bj_stop"] = sum([1 for x in B_lvl1 if isinstance(x, str) and is_stkn(x)])
    extra_info["stream_Bj_empty"] = sum([1 for x in B_lvl1 if isinstance(x, str) and x == ''])
    extra_info["stream_Bvals_noncontrol"] = sum([1 for x in B_v if isinstance(x, int)])
    extra_info["stream_Bvals_stop"] = sum([1 for x in B_v if isinstance(x, str) and is_stkn(x)])
    extra_info["stream_Bvals_empty"] = sum([1 for x in B_v if isinstance(x, str) and x == ''])

    extra_info["dataset"] = ssname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    sample_dict = fiberlookup_Bi_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bi_5" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_2" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bj_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bj_4" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_1" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_B_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_3" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "_" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_mat_identity(ssname, debug_sim, cast, positive_only, out_crds, out_segs, out_vals, "ss01")
    samBench(bench, extra_info)
