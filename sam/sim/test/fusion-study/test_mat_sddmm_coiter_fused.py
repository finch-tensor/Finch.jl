import enum
import pytest
import time
import scipy.sparse
import math

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

cwd = os.getcwd()
formatted_dir = os.getenv('SUITESPARSE_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))

other_dir = os.getenv('OTHER_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))

synthetic_dir = os.getenv('SYNTHETIC_PATH', default=os.path.join(cwd, 'synthetic'))


# FIXME: Figureout formats
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.synth
@pytest.mark.parametrize("sparsity", [0.95])
@pytest.mark.parametrize("KDIM", [1, 10, 100])
def test_mat_sddmm_coiter_fused(samBench, sparsity, check_gold, debug_sim, backpressure, depth, KDIM, fill=0):

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

    C_shape = (B_shape[0], KDIM)
    C_vals = np.arange(math.prod(C_shape)).tolist()
    for idx in range(len(C_vals)):
        C_vals[idx] = 1
    # C_vals = np.ones(math.prod(C_shape)).to_list()
    # C_vals = np.ones([C_shape[0] * C_shape[1]]).to_list()

    # print(C_vals)

    D_shape = (KDIM, B_shape[1])
    D_vals = np.arange(math.prod(D_shape)).tolist()
    for idx in range(len(D_vals)):
        D_vals[idx] = 1
    # D_vals = np.ones(D_shape).to_list()

    fiberlookup_Bi_25 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim,
                                            back_en=backpressure, depth=depth)
    fiberlookup_Ci_26 = UncompressCrdRdScan(dim=C_shape[0], debug=debug_sim, back_en=backpressure, depth=depth)
    intersecti_24 = Intersect2(debug=debug_sim, back_en=backpressure, depth=depth)
    fiberlookup_Bj_19 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim,
                                            back_en=backpressure, depth=depth)
    repsiggen_i_22 = RepeatSigGen(debug=debug_sim, back_en=backpressure, depth=depth)
    repeat_Di_21 = Repeat(debug=debug_sim, back_en=backpressure, depth=depth)
    fiberlookup_Dj_20 = UncompressCrdRdScan(dim=D_shape[1], debug=debug_sim, back_en=backpressure, depth=depth)
    intersectj_18 = Intersect2(debug=debug_sim, back_en=backpressure, depth=depth)
    fiberlookup_Dk_14 = UncompressCrdRdScan(dim=D_shape[0], debug=debug_sim, back_en=backpressure, depth=depth)
    crddrop_9 = CrdDrop(debug=debug_sim, back_en=backpressure, depth=depth)
    repsiggen_j_16 = RepeatSigGen(debug=debug_sim, back_en=backpressure, depth=depth)
    fiberwrite_X0_2 = CompressWrScan(seg_size=2, size=B_shape[0], fill=fill, debug=debug_sim,
                                     back_en=backpressure, depth=depth)
    fiberwrite_X1_1 = CompressWrScan(seg_size=B_shape[0] + 1, size=B_shape[0] * B_shape[1], fill=fill, debug=debug_sim,
                                     back_en=backpressure, depth=depth)
    repeat_Cj_15 = Repeat(debug=debug_sim, back_en=backpressure, depth=depth)
    fiberlookup_Ck_13 = UncompressCrdRdScan(dim=C_shape[1], debug=debug_sim, back_en=backpressure, depth=depth)
    intersectk_12 = Intersect2(debug=debug_sim, back_en=backpressure, depth=depth)
    repsiggen_k_11 = RepeatSigGen(debug=debug_sim, back_en=backpressure, depth=depth)
    arrayvals_C_7 = Array(init_arr=C_vals, debug=debug_sim, back_en=backpressure, depth=depth)
    arrayvals_D_8 = Array(init_arr=D_vals, debug=debug_sim, back_en=backpressure, depth=depth)
    repeat_Bk_10 = Repeat(debug=debug_sim, back_en=backpressure, depth=depth)
    arrayvals_B_6 = Array(init_arr=B_vals, debug=debug_sim, back_en=backpressure, depth=depth)
    mul_5 = Multiply2(debug=debug_sim, back_en=backpressure, depth=depth)
    mul_4 = Multiply2(debug=debug_sim, back_en=backpressure, depth=depth)
    reduce_3 = Reduce(debug=debug_sim, back_en=backpressure, depth=depth)
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * B_shape[0] * B_shape[1], fill=fill, debug=debug_sim,
                                    back_en=backpressure, depth=depth)
    in_ref_B = [0, 'D']
    in_ref_C = [0, 'D']
    in_ref_D = [0, 'D']
    done = False
    time_cnt = 0

    tempb = []
    tempc = []
    tempd = []
    tempci = []
    tempcj = []
    tempck = []
    tmp_bj_crd = []
    tmp_bj_ref = []
    tmp_intj = []
    tmp_bj_out = []
    tmp_dj_out = []
    tmp_repeat_di = []
    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_25.set_in_ref(in_ref_B.pop(0), "")

        if len(in_ref_C) > 0:
            fiberlookup_Ci_26.set_in_ref(in_ref_C.pop(0), "")

        intersecti_24.set_in1(fiberlookup_Bi_25.out_ref(), fiberlookup_Bi_25.out_crd(), fiberlookup_Bi_25)
        intersecti_24.set_in2(fiberlookup_Ci_26.out_ref(), fiberlookup_Ci_26.out_crd(), fiberlookup_Ci_26)

        fiberlookup_Bj_19.set_in_ref(intersecti_24.out_ref1(), intersecti_24)

        repsiggen_i_22.set_istream(intersecti_24.out_crd(), intersecti_24)

        if len(in_ref_D) > 0:
            repeat_Di_21.set_in_ref(in_ref_D.pop(0), "")
        repeat_Di_21.set_in_repsig(repsiggen_i_22.out_repsig(), repsiggen_i_22)

        fiberlookup_Dj_20.set_in_ref(repeat_Di_21.out_ref(), repeat_Di_21)

        intersectj_18.set_in1(fiberlookup_Dj_20.out_ref(), fiberlookup_Dj_20.out_crd(), fiberlookup_Dj_20)
        intersectj_18.set_in2(fiberlookup_Bj_19.out_ref(), fiberlookup_Bj_19.out_crd(), fiberlookup_Bj_19)

        fiberlookup_Dk_14.set_in_ref(intersectj_18.out_ref1(), intersectj_18)

        crddrop_9.set_outer_crd(intersecti_24.out_crd(), intersecti_24)
        crddrop_9.set_inner_crd(intersectj_18.out_crd(), intersectj_18)

        repsiggen_j_16.set_istream(intersectj_18.out_crd(), intersectj_18)

        repeat_Cj_15.set_in_ref(intersecti_24.out_ref2(), intersecti_24)
        repeat_Cj_15.set_in_repsig(repsiggen_j_16.out_repsig(), repsiggen_j_16)

        fiberlookup_Ck_13.set_in_ref(repeat_Cj_15.out_ref(), repeat_Cj_15)

        intersectk_12.set_in1(fiberlookup_Ck_13.out_ref(), fiberlookup_Ck_13.out_crd(), fiberlookup_Ck_13)
        intersectk_12.set_in2(fiberlookup_Dk_14.out_ref(), fiberlookup_Dk_14.out_crd(), fiberlookup_Dk_14)

        repsiggen_k_11.set_istream(intersectk_12.out_crd(), intersectk_12)

        arrayvals_C_7.set_load(intersectk_12.out_ref1(), intersectk_12)

        arrayvals_D_8.set_load(intersectk_12.out_ref2(), intersectk_12)

        repeat_Bk_10.set_in_ref(intersectj_18.out_ref2(), intersectj_18)
        repeat_Bk_10.set_in_repsig(repsiggen_k_11.out_repsig(), repsiggen_k_11)

        arrayvals_B_6.set_load(repeat_Bk_10.out_ref(), repeat_Bk_10)
        mul_5.set_in1(arrayvals_B_6.out_val(), arrayvals_B_6)
        mul_5.set_in2(arrayvals_C_7.out_val(), arrayvals_C_7)

        mul_4.set_in1(mul_5.out_val(), mul_5)
        mul_4.set_in2(arrayvals_D_8.out_val(), arrayvals_D_8)

        # tempb.append(arrayvals_B_6.out_val())
        # tempc.append(arrayvals_C_7.out_val())
        # tempd.append(arrayvals_D_8.out_val())
        # tempci.append(fiberlookup_Ci_26.out_ref())
        # tempcj.append(repeat_Cj_15.out_ref())
        # tempck.append(fiberlookup_Ck_13.out_ref())
        # tmp_bj_crd.append(fiberlookup_Bj_19.out_crd())
        # tmp_bj_ref.append(fiberlookup_Bj_19.out_ref())
        # tmp_intj.append(intersectj_18.out_crd())
        # tmp_bj_out.append(fiberlookup_Bj_19.out_crd())
        # tmp_dj_out.append(fiberlookup_Dj_20.out_crd())
        # tmp_repeat_di.append(repeat_Di_21.out_ref())
        # print("B", remove_emptystr(tempb))
        # print("C", remove_emptystr(tempc))
        # print("D", remove_emptystr(tempd))
        # print("Ci", remove_emptystr(tempci))
        # print("Cj", remove_emptystr(tempcj))
        # print("Ck", remove_emptystr(tempck))
        # print("bj crd", remove_emptystr(tmp_bj_crd))
        # print("bj ref", remove_emptystr(tmp_bj_ref))
        # print("bj crd", remove_emptystr(tmp_bj_out))
        # print("dj crd", remove_emptystr(tmp_dj_out))
        # print("repeat_di", remove_emptystr(tmp_repeat_di))

        reduce_3.set_in_val(mul_4.out_val(), mul_4)

        fiberwrite_Xvals_0.set_input(reduce_3.out_val(), reduce_3)

        fiberwrite_X0_2.set_input(crddrop_9.out_crd_outer(), crddrop_9)

        fiberwrite_X1_1.set_input(crddrop_9.out_crd_inner(), crddrop_9)

        fiberlookup_Bi_25.update()
        fiberlookup_Ci_26.update()
        intersecti_24.update()
        fiberlookup_Bj_19.update()
        repsiggen_i_22.update()
        repeat_Di_21.update()
        fiberlookup_Dj_20.update()
        intersectj_18.update()
        fiberlookup_Dk_14.update()
        crddrop_9.update()
        repsiggen_j_16.update()
        repeat_Cj_15.update()
        fiberlookup_Ck_13.update()
        intersectk_12.update()
        repsiggen_k_11.update()
        arrayvals_C_7.update()
        arrayvals_D_8.update()
        repeat_Bk_10.update()
        arrayvals_B_6.update()
        mul_5.update()
        mul_4.update()
        reduce_3.update()
        fiberwrite_Xvals_0.update()
        fiberwrite_X0_2.update()
        fiberwrite_X1_1.update()
        done = fiberwrite_X0_2.out_done() and fiberwrite_X1_1.out_done() and fiberwrite_Xvals_0.out_done()
        time_cnt += 1

    print("TOTAL TIME", time_cnt)

    fiberwrite_X0_2.autosize()
    fiberwrite_X1_1.autosize()
    fiberwrite_Xvals_0.autosize()

    out_crds = [fiberwrite_X0_2.get_arr(), fiberwrite_X1_1.get_arr()]
    out_segs = [fiberwrite_X0_2.get_seg_arr(), fiberwrite_X1_1.get_seg_arr()]
    out_vals = fiberwrite_Xvals_0.get_arr()

    def bench():
        time.sleep(0.01)

    extra_info = dict()
    # extra_info["dataset"] = ssname
    extra_info["dataset"] = "synthetic"
    extra_info["test_name"] = "sddmm_coiter_fused"
    extra_info["cycles"] = time_cnt
    extra_info["kdim"] = KDIM
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_C_shape"] = C_shape
    extra_info["tensor_D_shape"] = D_shape
    # sample_dict = intersecti_24.return_statistics()
    # for k in sample_dict.keys():
    #     extra_info["intersecti_24" + "_" + k] = sample_dict[k]

    # sample_dict = crddrop_9.return_statistics()
    # for k in sample_dict.keys():
    #     extra_info["crddrop_9" + "_" + k] = sample_dict[k]

    # sample_dict = fiberwrite_X0_2.return_statistics()
    # for k in sample_dict.keys():
    #     extra_info["fiberwrite_X0_2" + "_" + k] = sample_dict[k]

    # sample_dict = fiberwrite_X1_1.return_statistics()
    # for k in sample_dict.keys():
    #     extra_info["fiberwrite_X1_1" + "_" + k] = sample_dict[k]

    # sample_dict = repeat_Di_21.return_statistics()
    # for k in sample_dict.keys():
    #     extra_info["repeat_Di_21" + "_" + k] = sample_dict[k]

    # sample_dict = intersectj_18.return_statistics()
    # for k in sample_dict.keys():
    #     extra_info["intersectj_18" + "_" + k] = sample_dict[k]

    # sample_dict = repeat_Cj_15.return_statistics()
    # for k in sample_dict.keys():
    #     extra_info["repeat_Cj_15" + "_" + k] = sample_dict[k]

    # sample_dict = intersectk_12.return_statistics()
    # for k in sample_dict.keys():
    #     extra_info["intersectk_12" + "_" + k] = sample_dict[k]

    # sample_dict = repeat_Bk_10.return_statistics()
    # for k in sample_dict.keys():
    #     extra_info["repeat_Bk_10" + "_" + k] = sample_dict[k]

    # sample_dict = arrayvals_B_6.return_statistics()
    # for k in sample_dict.keys():
    #     extra_info["arrayvals_B_6" + "_" + k] = sample_dict[k]

    # sample_dict = reduce_3.return_statistics()
    # for k in sample_dict.keys():
    #     extra_info["reduce_3" + "_" + k] = sample_dict[k]

    # sample_dict = fiberwrite_Xvals_0.return_statistics()
    # for k in sample_dict.keys():
    #     extra_info["fiberwrite_Xvals_0" + "_" + k] = sample_dict[k]

    # sample_dict = arrayvals_C_7.return_statistics()
    # for k in sample_dict.keys():
    #     extra_info["arrayvals_C_7" + "_" + k] = sample_dict[k]

    # sample_dict = arrayvals_D_8.return_statistics()
    # for k in sample_dict.keys():
    #     extra_info["arrayvals_D_8" + "_" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")

        sim_pt_list = get_point_list(out_crds, out_segs, val_arr=out_vals)
        sim_mg = create_matrix_from_point_list(name="X", pt_list=sim_pt_list, shape=[B_shape[0], B_shape[1]])
        x_mat_sim = sim_mg.get_matrix()

        # GET NUMPY REPS OF INPUT MATS
        b_mg = get_tensor_from_files(name="B", files_dir=B_dirname, shape=B_shape)
        b_mat = b_mg.get_matrix()
        # print(b_mat)
        # C is stored in DCSC - need to transpose upon reading.
        c_mat = numpy.ones(C_shape)
        d_mat = numpy.ones(D_shape)

        x_mat_gold = numpy.multiply(b_mat, numpy.matmul(c_mat, d_mat))
        print(x_mat_gold)
        print(x_mat_sim)

        assert numpy.array_equal(x_mat_gold, x_mat_sim)
        # check_gold_mat_sddmm(ssname, debug_sim, out_crds, out_segs, out_vals, "ss01")
    samBench(bench, extra_info)
