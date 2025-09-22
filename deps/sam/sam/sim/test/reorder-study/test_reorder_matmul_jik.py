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
from sam.onyx.generate_matrices import create_matrix_from_point_list, get_tensor_from_files
import numpy
from sam.sim.test.gen_gantt import gen_gantt

cwd = os.getcwd()
formatted_dir = os.getenv('SUITESPARSE_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))
formatted_dir = os.getenv('FROSTT_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))

synthetic_dir = os.getenv('SYNTHETIC_PATH', default=os.path.join(cwd, 'synthetic'))


# FIXME: Figureout formats
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.synth
@pytest.mark.parametrize("sparsity", [0.95])
def test_reorder_matmul_jik(samBench, sparsity, check_gold, debug_sim, backpressure, depth, fill=0):

    # DCSR
    B_dirname = os.path.join(synthetic_dir, f"matrix/DCSR/B_random_sp_{sparsity}/")
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

    # DCSC
    C_dirname = os.path.join(synthetic_dir, f"matrix/DCSC/C_random_sp_{sparsity}/")
    C_shape_filename = os.path.join(C_dirname, "tensor_C_mode_shape")
    C_shape = read_inputs(C_shape_filename)

    # C0_seg_filename = os.path.join(C_dirname, "tensor_C_mode_0_seg")
    C0_seg_filename = os.path.join(C_dirname, "tensor_C_mode_1_seg")
    C_seg0 = read_inputs(C0_seg_filename)
    # C0_crd_filename = os.path.join(C_dirname, "tensor_C_mode_0_crd")
    C0_crd_filename = os.path.join(C_dirname, "tensor_C_mode_1_crd")
    C_crd0 = read_inputs(C0_crd_filename)

    # C1_seg_filename = os.path.join(C_dirname, "tensor_C_mode_1_seg")
    C1_seg_filename = os.path.join(C_dirname, "tensor_C_mode_0_seg")
    C_seg1 = read_inputs(C1_seg_filename)
    # C1_crd_filename = os.path.join(C_dirname, "tensor_C_mode_1_crd")
    C1_crd_filename = os.path.join(C_dirname, "tensor_C_mode_0_crd")
    C_crd1 = read_inputs(C1_crd_filename)

    C_vals_filename = os.path.join(C_dirname, "tensor_C_mode_vals")
    C_vals = read_inputs(C_vals_filename, float)

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

    # C_dirname = os.path.join(formatted_dir, ssname, "shift-trans", "ss10")
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

    fiberlookup_Cj_17 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim,
                                            back_en=backpressure, depth=int(depth))
    fiberwrite_X1_2 = CompressWrScan(seg_size=2, size=C_shape[1], fill=fill, debug=debug_sim,
                                     back_en=backpressure, depth=int(depth))
    repsiggen_j_15 = RepeatSigGen(debug=debug_sim, back_en=backpressure, depth=int(depth))
    repeat_Bj_14 = Repeat(debug=debug_sim, back_en=backpressure, depth=int(depth))
    fiberlookup_Bi_13 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim,
                                            back_en=backpressure, depth=int(depth))
    fiberlookup_Bk_8 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim,
                                           back_en=backpressure, depth=int(depth))
    fiberwrite_X0_1 = CompressWrScan(seg_size=C_shape[1] + 1, size=C_shape[1] * B_shape[0],
                                     fill=fill, debug=debug_sim,
                                     back_en=backpressure, depth=int(depth))
    repsiggen_i_11 = RepeatSigGen(debug=debug_sim, back_en=backpressure, depth=int(depth))
    repeat_Ci_10 = Repeat(debug=debug_sim, back_en=backpressure, depth=int(depth))
    fiberlookup_Ck_9 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim,
                                           back_en=backpressure, depth=int(depth))
    intersectk_7 = Intersect2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    arrayvals_B_5 = Array(init_arr=B_vals, debug=debug_sim, back_en=backpressure, depth=int(depth))
    arrayvals_C_6 = Array(init_arr=C_vals, debug=debug_sim, back_en=backpressure, depth=int(depth))
    mul_4 = Multiply2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    reduce_3 = Reduce(debug=debug_sim, back_en=backpressure, depth=int(depth))
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * C_shape[1] * B_shape[0], fill=fill, debug=debug_sim,
                                    back_en=backpressure, depth=int(depth))
    in_ref_C = [0, 'D']
    in_ref_B = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_C) > 0:
            fiberlookup_Cj_17.set_in_ref(in_ref_C.pop(0), "")

        fiberwrite_X1_2.set_input(fiberlookup_Cj_17.out_crd(), fiberlookup_Cj_17)

        repsiggen_j_15.set_istream(fiberlookup_Cj_17.out_crd(), fiberlookup_Cj_17)

        if len(in_ref_B) > 0:
            repeat_Bj_14.set_in_ref(in_ref_B.pop(0), "")
        repeat_Bj_14.set_in_repsig(repsiggen_j_15.out_repsig(), repsiggen_j_15)

        fiberlookup_Bi_13.set_in_ref(repeat_Bj_14.out_ref(), repeat_Bj_14)

        fiberlookup_Bk_8.set_in_ref(fiberlookup_Bi_13.out_ref(), fiberlookup_Bi_13)

        fiberwrite_X0_1.set_input(fiberlookup_Bi_13.out_crd(), fiberlookup_Bi_13)

        repsiggen_i_11.set_istream(fiberlookup_Bi_13.out_crd(), fiberlookup_Bi_13)

        repeat_Ci_10.set_in_ref(fiberlookup_Cj_17.out_ref(), fiberlookup_Cj_17)
        repeat_Ci_10.set_in_repsig(repsiggen_i_11.out_repsig(), repsiggen_i_11)

        fiberlookup_Ck_9.set_in_ref(repeat_Ci_10.out_ref(), repeat_Ci_10)

        intersectk_7.set_in1(fiberlookup_Ck_9.out_ref(), fiberlookup_Ck_9.out_crd(), fiberlookup_Ck_9)
        intersectk_7.set_in2(fiberlookup_Bk_8.out_ref(), fiberlookup_Bk_8.out_crd(), fiberlookup_Bk_8)

        arrayvals_B_5.set_load(intersectk_7.out_ref2(), intersectk_7)

        arrayvals_C_6.set_load(intersectk_7.out_ref1(), intersectk_7)

        mul_4.set_in1(arrayvals_B_5.out_val(), arrayvals_B_5)
        mul_4.set_in2(arrayvals_C_6.out_val(), arrayvals_C_6)

        reduce_3.set_in_val(mul_4.out_val(), mul_4)

        fiberwrite_Xvals_0.set_input(reduce_3.out_val(), reduce_3)

        fiberlookup_Cj_17.update()
        fiberwrite_X1_2.update()
        repsiggen_j_15.update()
        repeat_Bj_14.update()
        fiberlookup_Bi_13.update()
        fiberlookup_Bk_8.update()
        fiberwrite_X0_1.update()
        repsiggen_i_11.update()
        repeat_Ci_10.update()
        fiberlookup_Ck_9.update()
        intersectk_7.update()
        arrayvals_B_5.update()
        arrayvals_C_6.update()
        mul_4.update()
        reduce_3.update()
        fiberwrite_Xvals_0.update()

        done = fiberwrite_X1_2.out_done() and fiberwrite_X0_1.out_done() and fiberwrite_Xvals_0.out_done()
        time_cnt += 1

    fiberwrite_X1_2.autosize()
    fiberwrite_X0_1.autosize()
    fiberwrite_Xvals_0.autosize()

    out_crds = [fiberwrite_X1_2.get_arr(), fiberwrite_X0_1.get_arr()]
    out_segs = [fiberwrite_X1_2.get_seg_arr(), fiberwrite_X0_1.get_seg_arr()]
    out_vals = fiberwrite_Xvals_0.get_arr()

    def bench():
        time.sleep(0.01)

    extra_info = dict()
    # extra_info["dataset"] = ssname
    extra_info["dataset"] = "synthetic"
    extra_info["test_name"] = "jik"
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_C_shape"] = C_shape
    sample_dict = fiberwrite_X1_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_2" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Bj_14.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Bj_14" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_1" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Ci_10.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Ci_10" + "_" + k] = sample_dict[k]

    sample_dict = intersectk_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectk_7" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_B_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_5" + "_" + k] = sample_dict[k]

    sample_dict = reduce_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["reduce_3" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_C_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_6" + "_" + k] = sample_dict[k]

    gen_gantt(extra_info, "matmul_jik")

    if check_gold:
        print("Checking gold...")
        sim_pt_list = get_point_list(out_crds, out_segs, val_arr=out_vals)
        sim_mg = create_matrix_from_point_list(name="X", pt_list=sim_pt_list, shape=[B_shape[0], C_shape[0]])
        x_mat_sim = sim_mg.get_matrix()

        # GET NUMPY REPS OF INPUT MATS
        b_mg = get_tensor_from_files(name="B", files_dir=B_dirname, shape=B_shape)
        b_mat = b_mg.get_matrix()
        # print(b_mat)
        # C is stored in DCSC - need to transpose upon reading.
        c_mg = get_tensor_from_files(name="C", files_dir=C_dirname, shape=C_shape)
        c_mat = c_mg.get_matrix()
        c_mat_transpose = numpy.transpose(c_mat)
        # print(c_mat_transpose)
        # c_mat_transpose = c_mat

        x_mat_gold = numpy.matmul(b_mat, c_mat_transpose)

        # sim stored DCSC
        x_mat_sim = numpy.transpose(x_mat_sim)

        print(x_mat_gold)
        print(x_mat_sim)

        assert numpy.array_equal(x_mat_gold, x_mat_sim)

    samBench(bench, extra_info)
