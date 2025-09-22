import pytest
import time
from sam.onyx.generate_matrices import create_matrix_from_point_list, get_tensor_from_files
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
import numpy
from sam.sim.test.gen_gantt import gen_gantt

cwd = os.getcwd()
formatted_dir = os.getenv('SUITESPARSE_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))
# formatted_dir = os.getenv('FROSTT_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))

synthetic_dir = os.getenv('SYNTHETIC_PATH', default=os.path.join(cwd, 'synthetic'))


# FIXME: Figureout formats
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.synth
# @pytest.mark.parametrize("sparsity", [0.95])
def test_reorder_matmul_ikj(samBench, ssname, check_gold, report_stats, debug_sim, cast, backpressure, depth, fill=0):
    B_dirname = os.path.join(formatted_dir, ssname, "matmul_kij")
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

    B_vals_filename = os.path.join(B_dirname, "tensor_B_mode_vals")
    B_vals = read_inputs(B_vals_filename, float)

    # DCSR
    C_dirname = B_dirname
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

    # THIS IS FOR SIZE INFO
    Bs_dirname = B_dirname
    Bs_seg = read_inputs(os.path.join(Bs_dirname, "tensor_B_mode_0_seg"))

    # B_dirname = os.path.join(formatted_dir, ssname, "orig", "ss01")
    # B_shape_filename = os.path.join(B_dirname, "B_shape.txt")
    # B_shape = read_inputs(B_shape_filename)

    # B0_seg_filename = os.path.join(B_dirname, "B0_seg.txt")
    # B_seg0 = read_inputs(B0_seg_filename)
    # B0_crd_filename = os.path.join(B_dirname, "B0_crd.txt")
    # B_crd0 = read_inputs(B0_crd_filename)

    # B1_seg_filename = os.path.join(B_dirname, "B1_seg.txt")
    # B_seg1 = read_inputs(B1_seg_filename)
    # B1_crd_filename = os.path.join(B_dirname, "B1_crd.txt")
    # B_crd1 = read_inputs(B1_crd_filename)

    # B_vals_filename = os.path.join(B_dirname, "B_vals.txt")
    # B_vals = read_inputs(B_vals_filename, float)

    # C_dirname = os.path.join(formatted_dir, ssname, "shift-trans", "ss01")
    # C_shape_filename = os.path.join(C_dirname, "C_shape.txt")
    # C_shape = read_inputs(C_shape_filename)

    # C0_seg_filename = os.path.join(C_dirname, "C0_seg.txt")
    # C_seg0 = read_inputs(C0_seg_filename)
    # C0_crd_filename = os.path.join(C_dirname, "C0_crd.txt")
    # C_crd0 = read_inputs(C0_crd_filename)

    # C1_seg_filename = os.path.join(C_dirname, "C1_seg.txt")
    # C_seg1 = read_inputs(C1_seg_filename)
    # C1_crd_filename = os.path.join(C_dirname, "C1_crd.txt")
    # C_crd1 = read_inputs(C1_crd_filename)

    # C_vals_filename = os.path.join(C_dirname, "C_vals.txt")
    # C_vals = read_inputs(C_vals_filename, float)

    fiberlookup_Bi_19 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim,
                                            back_en=backpressure, depth=int(depth))
    fiberlookup_Bk_14 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim,
                                            back_en=backpressure, depth=int(depth))
    repsiggen_i_17 = RepeatSigGen(debug=debug_sim, back_en=backpressure, depth=int(depth))
    repeat_Ci_16 = Repeat(debug=debug_sim, back_en=backpressure, depth=int(depth))
    fiberlookup_Ck_15 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim,
                                            back_en=backpressure, depth=int(depth))
    intersectk_13 = Intersect2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    crdhold_5 = CrdHold(debug=debug_sim, back_en=backpressure, depth=int(depth))
    fiberlookup_Cj_12 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim,
                                            back_en=backpressure, depth=int(depth))
    arrayvals_C_8 = Array(init_arr=C_vals, debug=debug_sim, back_en=backpressure, depth=int(depth))
    crdhold_4 = CrdHold(debug=debug_sim, back_en=backpressure, depth=int(depth))
    repsiggen_j_10 = RepeatSigGen(debug=debug_sim, back_en=backpressure, depth=int(depth))
    repeat_Bj_9 = Repeat(debug=debug_sim, back_en=backpressure, depth=int(depth))
    arrayvals_B_7 = Array(init_arr=B_vals, debug=debug_sim, back_en=backpressure, depth=int(depth))
    mul_6 = Multiply2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    spaccumulator1_3 = SparseAccumulator1(debug=debug_sim, back_en=backpressure, depth=int(depth))
    spaccumulator1_3_drop_crd_inner = StknDrop(debug=debug_sim, back_en=backpressure, depth=int(depth))
    spaccumulator1_3_drop_crd_outer = StknDrop(debug=debug_sim, back_en=backpressure, depth=int(depth))
    spaccumulator1_3_drop_val = StknDrop(debug=debug_sim, back_en=backpressure, depth=int(depth))
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * B_shape[0] * C_shape[1], fill=fill, debug=debug_sim,
                                    back_en=backpressure, depth=int(depth))
    fiberwrite_X1_1 = CompressWrScan(seg_size=B_shape[0] + 1, size=B_shape[0] * C_shape[1], fill=fill, debug=debug_sim,
                                     back_en=backpressure, depth=int(depth))
    fiberwrite_X0_2 = CompressWrScan(seg_size=2, size=B_shape[0], fill=fill, debug=debug_sim,
                                     back_en=backpressure, depth=int(depth))
    in_ref_B = [0, 'D']
    in_ref_C = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_19.set_in_ref(in_ref_B.pop(0), "")

        fiberlookup_Bk_14.set_in_ref(fiberlookup_Bi_19.out_ref(), fiberlookup_Bi_19)

        repsiggen_i_17.set_istream(fiberlookup_Bi_19.out_crd(), fiberlookup_Bi_19)

        if len(in_ref_C) > 0:
            repeat_Ci_16.set_in_ref(in_ref_C.pop(0), "")
        repeat_Ci_16.set_in_repsig(repsiggen_i_17.out_repsig())

        fiberlookup_Ck_15.set_in_ref(repeat_Ci_16.out_ref(), repeat_Ci_16)

        intersectk_13.set_in1(fiberlookup_Ck_15.out_ref(), fiberlookup_Ck_15.out_crd(), fiberlookup_Ck_15)
        intersectk_13.set_in2(fiberlookup_Bk_14.out_ref(), fiberlookup_Bk_14.out_crd(), fiberlookup_Bk_14)

        crdhold_5.set_outer_crd(fiberlookup_Bi_19.out_crd(), fiberlookup_Bi_19)
        crdhold_5.set_inner_crd(intersectk_13.out_crd(), intersectk_13)
        fiberlookup_Cj_12.set_in_ref(intersectk_13.out_ref1(), intersectk_13)

        arrayvals_C_8.set_load(fiberlookup_Cj_12.out_ref(), fiberlookup_Cj_12)

        crdhold_4.set_outer_crd(crdhold_5.out_crd_outer(), crdhold_5)
        crdhold_4.set_inner_crd(fiberlookup_Cj_12.out_crd(), fiberlookup_Cj_12)
        repsiggen_j_10.set_istream(fiberlookup_Cj_12.out_crd(), fiberlookup_Cj_12)

        repeat_Bj_9.set_in_ref(intersectk_13.out_ref2(), intersectk_13)
        repeat_Bj_9.set_in_repsig(repsiggen_j_10.out_repsig(), repsiggen_j_10)

        arrayvals_B_7.set_load(repeat_Bj_9.out_ref(), repeat_Bj_9)

        mul_6.set_in1(arrayvals_B_7.out_val(), arrayvals_B_7)
        mul_6.set_in2(arrayvals_C_8.out_val(), arrayvals_C_8)

        spaccumulator1_3_drop_crd_outer.set_in_stream(crdhold_4.out_crd_outer(), crdhold_4)
        spaccumulator1_3_drop_crd_inner.set_in_stream(crdhold_4.out_crd_inner(), crdhold_4)
        spaccumulator1_3_drop_val.set_in_stream(mul_6.out_val(), mul_6)
        spaccumulator1_3.set_crd_outer(spaccumulator1_3_drop_crd_outer.out_val(), spaccumulator1_3_drop_crd_outer)
        spaccumulator1_3.set_crd_inner(spaccumulator1_3_drop_crd_inner.out_val(), spaccumulator1_3_drop_crd_inner)
        spaccumulator1_3.set_val(spaccumulator1_3_drop_val.out_val(), spaccumulator1_3_drop_val)

        fiberwrite_Xvals_0.set_input(spaccumulator1_3.out_val(), spaccumulator1_3)

        fiberwrite_X1_1.set_input(spaccumulator1_3.out_crd_inner(), spaccumulator1_3)

        fiberwrite_X0_2.set_input(spaccumulator1_3.out_crd_outer(), spaccumulator1_3)

        fiberlookup_Bi_19.update()
        fiberlookup_Bk_14.update()
        repsiggen_i_17.update()
        repeat_Ci_16.update()
        fiberlookup_Ck_15.update()
        intersectk_13.update()
        crdhold_5.update()
        fiberlookup_Cj_12.update()
        arrayvals_C_8.update()
        crdhold_4.update()
        repsiggen_j_10.update()
        repeat_Bj_9.update()
        arrayvals_B_7.update()
        mul_6.update()
        spaccumulator1_3_drop_crd_outer.update()
        spaccumulator1_3_drop_crd_inner.update()
        spaccumulator1_3_drop_val.update()
        spaccumulator1_3.update()
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
        # check_gold_matmul(ssname, debug_sim, out_crds, out_segs, out_vals, "ss10")

    extra_info = dict()
    # extra_info["dataset"] = ssname
    extra_info["dataset"] = "synthetic"
    extra_info["test_name"] = "ikj"
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_C_shape"] = C_shape
    sample_dict = spaccumulator1_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["spaccumulator1_3" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_1" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_2" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Ci_16.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Ci_16" + "_" + k] = sample_dict[k]

    sample_dict = intersectk_13.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectk_13" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Bj_9.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Bj_9" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_B_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_7" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_C_8.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_8" + "_" + k] = sample_dict[k]

    gen_gantt(extra_info, "matmul_ikj")

    if check_gold:
        print("Checking gold...")
        sim_pt_list = get_point_list(out_crds, out_segs, val_arr=out_vals)
        sim_mg = create_matrix_from_point_list(name="X", pt_list=sim_pt_list, shape=[B_shape[0], C_shape[1]])
        x_mat_sim = sim_mg.get_matrix()

        # GET NUMPY REPS OF INPUT MATS
        b_mg = get_tensor_from_files(name="B", files_dir=B_dirname, shape=B_shape)
        b_mat = b_mg.get_matrix()
        # print(b_mat)
        # C is stored in DCSC - need to transpose upon reading.
        c_mg = get_tensor_from_files(name="C", files_dir=C_dirname, shape=C_shape)
        c_mat = c_mg.get_matrix()
        # c_mat_transpose = numpy.transpose(c_mat)
        # print(c_mat_transpose)
        c_mat_transpose = c_mat

        x_mat_gold = numpy.matmul(b_mat, c_mat_transpose)
        print(x_mat_gold)
        print(x_mat_sim)

        assert numpy.array_equal(x_mat_gold, x_mat_sim)
    samBench(bench, extra_info)
