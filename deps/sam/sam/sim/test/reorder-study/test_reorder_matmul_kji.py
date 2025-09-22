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
def test_reorder_matmul_kji(samBench, sparsity, check_gold, debug_sim, backpressure, depth, fill=0):
    # DCSC
    B_dirname = os.path.join(synthetic_dir, f"matrix/DCSC/B_random_sp_{sparsity}/")
    B_shape_filename = os.path.join(B_dirname, "tensor_B_mode_shape")
    B_shape = read_inputs(B_shape_filename)

    B0_seg_filename = os.path.join(B_dirname, "tensor_B_mode_1_seg")
    # B0_seg_filename = os.path.join(B_dirname, "tensor_B_mode_0_seg")
    B_seg0 = read_inputs(B0_seg_filename)
    B0_crd_filename = os.path.join(B_dirname, "tensor_B_mode_1_crd")
    # B0_crd_filename = os.path.join(B_dirname, "tensor_B_mode_0_crd")
    B_crd0 = read_inputs(B0_crd_filename)

    B1_seg_filename = os.path.join(B_dirname, "tensor_B_mode_0_seg")
    # B1_seg_filename = os.path.join(B_dirname, "tensor_B_mode_1_seg")
    B_seg1 = read_inputs(B1_seg_filename)
    B1_crd_filename = os.path.join(B_dirname, "tensor_B_mode_0_crd")
    # B1_crd_filename = os.path.join(B_dirname, "tensor_B_mode_1_crd")
    B_crd1 = read_inputs(B1_crd_filename)

    B_vals_filename = os.path.join(B_dirname, "tensor_B_mode_vals")
    B_vals = read_inputs(B_vals_filename, float)

    # DCSR
    C_dirname = os.path.join(synthetic_dir, f"matrix/DCSR/C_random_sp_{sparsity}/")
    C_shape_filename = os.path.join(C_dirname, "tensor_C_mode_shape")
    C_shape = read_inputs(C_shape_filename)

    C0_seg_filename = os.path.join(C_dirname, "tensor_C_mode_0_seg")
    # C0_seg_filename = os.path.join(C_dirname, "tensor_C_mode_1_seg")
    C_seg0 = read_inputs(C0_seg_filename)
    C0_crd_filename = os.path.join(C_dirname, "tensor_C_mode_0_crd")
    # C0_crd_filename = os.path.join(C_dirname, "tensor_C_mode_1_crd")
    C_crd0 = read_inputs(C0_crd_filename)

    C1_seg_filename = os.path.join(C_dirname, "tensor_C_mode_1_seg")
    # C1_seg_filename = os.path.join(C_dirname, "tensor_C_mode_0_seg")
    C_seg1 = read_inputs(C1_seg_filename)
    C1_crd_filename = os.path.join(C_dirname, "tensor_C_mode_1_crd")
    # C1_crd_filename = os.path.join(C_dirname, "tensor_C_mode_0_crd")
    C_crd1 = read_inputs(C1_crd_filename)

    C_vals_filename = os.path.join(C_dirname, "tensor_C_mode_vals")
    C_vals = read_inputs(C_vals_filename, float)

    # B_dirname = os.path.join(formatted_dir, ssname, "orig", "ss10")
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

    fiberlookup_Bk_17 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim,
                                            back_en=backpressure, depth=int(depth))
    fiberlookup_Ck_18 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim,
                                            back_en=backpressure, depth=int(depth))
    intersectk_16 = Intersect2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    fiberlookup_Cj_15 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim,
                                            back_en=backpressure, depth=int(depth))
    repsiggen_j_13 = RepeatSigGen(debug=debug_sim, back_en=backpressure, depth=int(depth))
    repeat_Bj_12 = Repeat(debug=debug_sim, back_en=backpressure, depth=int(depth))
    fiberlookup_Bi_11 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim,
                                            back_en=backpressure, depth=int(depth))
    arrayvals_B_6 = Array(init_arr=B_vals, debug=debug_sim, back_en=backpressure, depth=int(depth))
    crdhold_4 = CrdHold(debug=debug_sim, back_en=backpressure, depth=int(depth))
    repsiggen_i_9 = RepeatSigGen(debug=debug_sim, back_en=backpressure, depth=int(depth))
    repeat_Ci_8 = Repeat(debug=debug_sim, back_en=backpressure, depth=int(depth))
    arrayvals_C_7 = Array(init_arr=C_vals, debug=debug_sim, back_en=backpressure, depth=int(depth))
    mul_5 = Multiply2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    spaccumulator2_3 = SparseAccumulator2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    spaccumulator2_3_drop_crd_inner = StknDrop(debug=debug_sim, back_en=backpressure, depth=int(depth))
    spaccumulator2_3_drop_crd_outer = StknDrop(debug=debug_sim, back_en=backpressure, depth=int(depth))
    spaccumulator2_3_drop_val = StknDrop(debug=debug_sim, back_en=backpressure, depth=int(depth))
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * C_shape[1] * B_shape[0], fill=fill, debug=debug_sim,
                                    back_en=backpressure, depth=int(depth))
    fiberwrite_X0_1 = CompressWrScan(seg_size=C_shape[1] + 1, size=C_shape[1] * B_shape[0], fill=fill, debug=debug_sim,
                                     back_en=backpressure, depth=int(depth))
    fiberwrite_X1_2 = CompressWrScan(seg_size=2, size=C_shape[1], fill=fill, debug=debug_sim,
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

        fiberlookup_Cj_15.set_in_ref(intersectk_16.out_ref2(), intersectk_16)

        repsiggen_j_13.set_istream(fiberlookup_Cj_15.out_crd(), fiberlookup_Cj_15)

        repeat_Bj_12.set_in_ref(intersectk_16.out_ref1(), intersectk_16)
        repeat_Bj_12.set_in_repsig(repsiggen_j_13.out_repsig(), repsiggen_j_13)

        fiberlookup_Bi_11.set_in_ref(repeat_Bj_12.out_ref(), repeat_Bj_12)

        arrayvals_B_6.set_load(fiberlookup_Bi_11.out_ref(), fiberlookup_Bi_11)

        crdhold_4.set_inner_crd(fiberlookup_Bi_11.out_crd(), fiberlookup_Bi_11)
        crdhold_4.set_outer_crd(fiberlookup_Cj_15.out_crd(), fiberlookup_Cj_15)
        repsiggen_i_9.set_istream(fiberlookup_Bi_11.out_crd(), fiberlookup_Bi_11)

        repeat_Ci_8.set_in_repsig(repsiggen_i_9.out_repsig(), repsiggen_i_9)
        repeat_Ci_8.set_in_ref(fiberlookup_Cj_15.out_ref(), fiberlookup_Cj_15)

        arrayvals_C_7.set_load(repeat_Ci_8.out_ref(), repeat_Ci_8)

        mul_5.set_in1(arrayvals_C_7.out_val(), arrayvals_C_7)
        mul_5.set_in2(arrayvals_B_6.out_val(), arrayvals_B_6)

        spaccumulator2_3_drop_crd_outer.set_in_stream(crdhold_4.out_crd_outer(), crdhold_4)
        spaccumulator2_3_drop_crd_inner.set_in_stream(crdhold_4.out_crd_inner(), crdhold_4)
        spaccumulator2_3_drop_val.set_in_stream(mul_5.out_val(), mul_5)
        spaccumulator2_3.set_crd_outer(spaccumulator2_3_drop_crd_outer.out_val(), spaccumulator2_3_drop_crd_outer)
        spaccumulator2_3.set_crd_inner(spaccumulator2_3_drop_crd_inner.out_val(), spaccumulator2_3_drop_crd_inner)
        spaccumulator2_3.set_val(spaccumulator2_3_drop_val.out_val(), spaccumulator2_3_drop_val)

        fiberwrite_Xvals_0.set_input(spaccumulator2_3.out_val(), spaccumulator2_3)

        fiberwrite_X0_1.set_input(spaccumulator2_3.out_crd_inner(), spaccumulator2_3)

        fiberwrite_X1_2.set_input(spaccumulator2_3.out_crd_outer(), spaccumulator2_3)

        fiberlookup_Bk_17.update()
        fiberlookup_Ck_18.update()
        intersectk_16.update()
        fiberlookup_Cj_15.update()
        repsiggen_j_13.update()
        repeat_Bj_12.update()
        fiberlookup_Bi_11.update()
        arrayvals_B_6.update()
        crdhold_4.update()
        repsiggen_i_9.update()
        repeat_Ci_8.update()
        arrayvals_C_7.update()
        mul_5.update()
        spaccumulator2_3_drop_crd_outer.update()
        spaccumulator2_3_drop_crd_inner.update()
        spaccumulator2_3_drop_val.update()
        spaccumulator2_3.update()
        fiberwrite_Xvals_0.update()
        fiberwrite_X0_1.update()
        fiberwrite_X1_2.update()

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
    extra_info["test_name"] = "kji"
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_C_shape"] = C_shape

    sample_dict = intersectk_16.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectk_16" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Bj_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Bj_12" + "_" + k] = sample_dict[k]

    sample_dict = spaccumulator2_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["spaccumulator2_3" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_1" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_2" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Ci_8.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Ci_8" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_C_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_7" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_B_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_6" + "_" + k] = sample_dict[k]

    gen_gantt(extra_info, "matmul_kji")

    if check_gold:
        print("Checking gold...")
        sim_pt_list = get_point_list(out_crds, out_segs, val_arr=out_vals)
        sim_mg = create_matrix_from_point_list(name="X", pt_list=sim_pt_list, shape=[B_shape[1], C_shape[1]])
        x_mat_sim = sim_mg.get_matrix()
        x_mat_sim = numpy.transpose(x_mat_sim)

        # GET NUMPY REPS OF INPUT MATS
        b_mg = get_tensor_from_files(name="B", files_dir=B_dirname, shape=B_shape)
        b_mat = b_mg.get_matrix()
        b_mat = numpy.transpose(b_mat)
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
        # check_gold_matmul(ssname, debug_sim, out_crds, out_segs, out_vals, "ss10")
    samBench(bench, extra_info)
