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
from sam.sim.src.channel import memory_block, output_memory_block
from sam.sim.src.accumulator import SparseAccumulator1, SparseAccumulator2
from sam.sim.src.token import *
from sam.sim.test.test import *
from sam.sim.test.gold import *
import os
import csv
import pickle
import yaml


cwd = os.getcwd()
formatted_dir = os.getenv('TILED_SUITESPARSE_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))
sam_home = os.getenv('SAM_HOME', default=os.path.join(cwd, 'mode-formats'))


# FIXME: Figureout formats
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.suitesparse
def test_matmul_ikj_tiled_lp(samBench, ssname, check_gold, debug_sim, report_stats, skip_empty, yaml_name, nbuffer, fill=0):
    stats_dict = {"mul_6_ops": 0, "spacc1_3_rmw_ops": []}
    with open(os.path.join(sam_home, "tiles/matmul_ikj/tensor_sizes"), "rb") as ff:
        sizes_dict_level_full = pickle.load(ff)
    assert sizes_dict_level_full["B"][0] == sizes_dict_level_full["C"][1]
    with open(os.path.join(sam_home, "tiles/matmul_ikj/hw_level_0_sizes"), "rb") as ff:
        sizes_dict_level0 = pickle.load(ff)
    with open(os.path.join(sam_home, "tiles/matmul_ikj/hw_level_1_sizes"), "rb") as ff:
        sizes_dict_level1 = pickle.load(ff)
    full_size = 0
    for sizes in sizes_dict_level_full:
        full_size = sizes_dict_level_full[sizes]
    with open(os.path.join(sam_home, "sam/sim/src/tiling/" + yaml_name), "r") as stream:
        loop_config = yaml.safe_load(stream)
    with open(os.path.join(sam_home, "./sam/sim/src/tiling/" + yaml_name), "r") as stream:
        memory_config = yaml.safe_load(stream)
    struct = {"i00": 1 + int(sizes_dict_level_full["B"][0]) // (loop_config["Glb_tile_size"] * loop_config["Mem_tile_size"]),
              "k00": 1 + int(sizes_dict_level_full["B"][1]) // (loop_config["Glb_tile_size"] * loop_config["Mem_tile_size"]),
              "j00": 1 + int(sizes_dict_level_full["C"][1]) // (loop_config["Glb_tile_size"] * loop_config["Mem_tile_size"]),
              "i0": int(loop_config["Glb_tile_size"]), "k0": int(loop_config["Glb_tile_size"]),
              "j0": int(loop_config["Glb_tile_size"])}
    fiberlookup_Bi00 = UncompressCrdRdScan(dim=struct["i00"], debug=debug_sim)
    fiberlookup_Bk00 = UncompressCrdRdScan(dim=struct["k00"], debug=debug_sim)
    repsiggen_i00 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Ci00 = Repeat(debug=debug_sim, statistics=report_stats)
    fiberlookup_Ck00 = UncompressCrdRdScan(dim=struct["k00"], debug=debug_sim)
    fiberlookup_Cj00 = UncompressCrdRdScan(dim=struct["j00"], debug=debug_sim)
    repsiggen_j00 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Bj00 = Repeat(debug=debug_sim, statistics=report_stats)
    in_ref_B00 = [0, 'D']
    in_ref_C00 = [0, 'D']
    in_ref_B0 = []
    in_ref_C0 = []
    done = False
    time_cnt = 0
    glb_model_b = memory_block(name="GLB_B", skip_blocks=skip_empty, nbuffer=nbuffer,
                               element_size=memory_config["Bytes_per_element"], size=memory_config["Glb_memory"],
                               bandwidth=memory_config["Glb_tile_bandwidth"],
                               latency=memory_config["Global_Glb_latency"], debug=debug_sim)
    glb_model_c = memory_block(name="GLB_C", skip_blocks=skip_empty, nbuffer=nbuffer,
                               element_size=memory_config["Bytes_per_element"], size=memory_config["Glb_memory"],
                               bandwidth=memory_config["Glb_tile_bandwidth"], latency=memory_config["Global_Glb_latency"],
                               debug=debug_sim)
    mem_model_b = memory_block(name="B", skip_blocks=skip_empty, nbuffer=nbuffer,
                               element_size=memory_config["Bytes_per_element"], size=memory_config["Mem_memory"],
                               bandwidth=memory_config["Mem_tile_bandwidth"], latency=memory_config["Glb_Mem_latency"],
                               debug=debug_sim)
    mem_model_c = memory_block(name="C", skip_blocks=skip_empty, nbuffer=nbuffer,
                               element_size=memory_config["Bytes_per_element"],
                               size=memory_config["Mem_memory"], bandwidth=memory_config["Mem_tile_bandwidth"],
                               latency=memory_config["Glb_Mem_latency"], debug=debug_sim)
    mem_model_x = output_memory_block(name="X", element_size=memory_config["Bytes_per_element"],
                                      level="mem2glb", bandwidth=memory_config["Mem_tile_bandwidth"],
                                      latency=memory_config["Glb_Mem_latency"], debug=debug_sim,
                                      loop_order=[struct["i0"], struct["k0"], struct["j0"]])
    glb_model_x = output_memory_block(name="X", element_size=memory_config["Bytes_per_element"],
                                      level="glb2global", bandwidth=memory_config["Glb_tile_bandwidth"],
                                      latency=memory_config["Glb_Mem_latency"], debug=debug_sim,
                                      loop_order=[struct["i00"], struct["k00"], struct["j00"],
                                                  struct["i0"], struct["k0"], struct["j0"]])
    flag_glb = False
    flag = False
    tiled_done = False
    tiled_done_processed = False
    check_flag = True
    tiled_skip = False
    array = []
    array2 = []
    glb_b_fifo = []
    glb_c_fifo = []
    mem_b_fifo = []
    mem_c_fifo = []
    tile_signalled = False
    while not done and time_cnt < TIMEOUT:
        if debug_sim:
            print(time_cnt)
        if len(in_ref_B00) > 0:
            fiberlookup_Bi00.set_in_ref(in_ref_B00.pop(0))
        fiberlookup_Bk00.set_in_ref(fiberlookup_Bi00.out_ref())
        repsiggen_i00.set_istream(fiberlookup_Bi00.out_crd())
        if len(in_ref_C00) > 0:
            repeat_Ci00.set_in_ref(in_ref_C00.pop(0))
        repeat_Ci00.set_in_repsig(repsiggen_i00.out_repsig())
        fiberlookup_Ck00.set_in_ref(repeat_Ci00.out_ref())
        fiberlookup_Cj00.set_in_ref(fiberlookup_Ck00.out_ref())
        repsiggen_j00.set_istream(fiberlookup_Cj00.out_crd())
        repeat_Bj00.set_in_ref(fiberlookup_Bk00.out_ref())
        repeat_Bj00.set_in_repsig(repsiggen_j00.out_repsig())
        flag_print_once = False
        if glb_model_b.valid_tile() and glb_model_c.valid_tile():
            flag_glb = True
            flag_print_once = True
            in_ref_B0 = [0, 'D']
            in_ref_C0 = [0, 'D']
            fiberlookup_Bi0 = UncompressCrdRdScan(dim=struct["i0"], debug=debug_sim)
            fiberlookup_Bk0 = UncompressCrdRdScan(dim=struct["k0"], debug=debug_sim)
            repsiggen_i0 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
            repeat_Ci0 = Repeat(debug=debug_sim, statistics=report_stats)
            fiberlookup_Ck0 = UncompressCrdRdScan(dim=struct["k0"], debug=debug_sim)
            fiberlookup_Cj0 = UncompressCrdRdScan(dim=struct["j0"], debug=debug_sim)
            repsiggen_j0 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
            repeat_Bj0 = Repeat(debug=debug_sim, statistics=report_stats)
            # print("INITIALIZE Mem loop at ", time_cnt)
            glb_model_b.valid_tile_recieved()
            glb_model_c.valid_tile_recieved()

        if True:
            if isinstance(repeat_Bj00.out_ref(), int):
                glb_b_fifo.append(repeat_Bj00.out_ref())
            elif repeat_Bj00.out_ref() != "":
                glb_b_fifo.append(repeat_Bj00.out_ref())
            if isinstance(fiberlookup_Cj00.out_ref(), int):
                glb_c_fifo.append(fiberlookup_Cj00.out_ref())
            elif fiberlookup_Cj00.out_ref() != "":
                glb_c_fifo.append(fiberlookup_Cj00.out_ref())
            if len(glb_b_fifo) > 0 and len(glb_c_fifo) > 0 and isinstance(glb_b_fifo[0], int) and\
                    isinstance(glb_c_fifo[0], int) and (glb_b_fifo[0] // struct["k00"],
                                                        glb_b_fifo[0] % struct["k00"]) in sizes_dict_level0["B"] and\
                    (glb_c_fifo[0] // struct["j00"], glb_c_fifo[0] % struct["j00"]) in sizes_dict_level0["C"]:
                glb_model_b.add_tile(glb_b_fifo[0], sizes_dict_level0["B"][(glb_b_fifo[0] // struct["k00"],
                                     glb_b_fifo[0] % struct["k00"])])
                glb_model_c.add_tile(glb_c_fifo[0],
                                     sizes_dict_level0["C"][(glb_c_fifo[0] // struct["j00"],
                                                             glb_c_fifo[0] % struct["j00"])])
                glb_b_fifo.pop(0)
                glb_c_fifo.pop(0)
            elif len(glb_b_fifo) > 0 and len(glb_c_fifo) > 0 and\
                    (not isinstance(glb_b_fifo[0], int) or not isinstance(glb_c_fifo[0], int)):
                glb_model_b.add_tile(glb_b_fifo[0], 8)
                glb_model_c.add_tile(glb_c_fifo[0], 8)
                glb_b_fifo.pop(0)
                glb_c_fifo.pop(0)
            elif len(glb_b_fifo) > 0 and len(glb_c_fifo) > 0:
                glb_b_fifo.pop(0)
                glb_c_fifo.pop(0)
        glb_model_b.check_if_done(mem_model_b.out_done_in())
        glb_model_c.check_if_done(mem_model_c.out_done_in())
        if flag_glb:
            if len(in_ref_B0) > 0:
                fiberlookup_Bi0.set_in_ref(in_ref_B0.pop(0))
            fiberlookup_Bk0.set_in_ref(fiberlookup_Bi0.out_ref())
            repsiggen_i0.set_istream(fiberlookup_Bi0.out_crd())
            if len(in_ref_C0) > 0:
                repeat_Ci0.set_in_ref(in_ref_C0.pop(0))
            repeat_Ci0.set_in_repsig(repsiggen_i0.out_repsig())
            fiberlookup_Ck0.set_in_ref(repeat_Ci0.out_ref())
            fiberlookup_Cj0.set_in_ref(fiberlookup_Ck0.out_ref())
            repsiggen_j0.set_istream(fiberlookup_Cj0.out_crd())
            repeat_Bj0.set_in_ref(fiberlookup_Bk0.out_ref())
            repeat_Bj0.set_in_repsig(repsiggen_j0.out_repsig())
            if isinstance(glb_model_b.token(), int) and isinstance(glb_model_c.token(), int):
                B_k00_ = glb_model_b.token() % struct["k00"]
                B_i00_ = glb_model_b.token() // struct["k00"]
                C_k00_ = glb_model_c.token() // struct["j00"]
                C_j00_ = glb_model_c.token() % struct["j00"]
            else:
                B_k00_ = glb_model_b.token()
                B_i00_ = glb_model_b.token()
                C_k00_ = glb_model_c.token()
                C_j00_ = glb_model_c.token()
            if True:  # not skip_empty:
                if isinstance(repeat_Bj0.out_ref(), int):
                    mem_b_fifo.append(B_i00_ * 10 ** 8 + B_k00_ * 10 ** 4 + repeat_Bj0.out_ref())
                elif repeat_Bj0.out_ref() != "":
                    mem_b_fifo.append(str(glb_model_b.token()) + "_" + repeat_Bj0.out_ref())
                if isinstance(fiberlookup_Cj0.out_ref(), int):
                    mem_c_fifo.append(C_k00_ * 10 ** 8 + C_j00_ * 10 ** 4 + fiberlookup_Cj0.out_ref())
                elif fiberlookup_Cj0.out_ref() != "":
                    mem_c_fifo.append(str(glb_model_c.token()) + "_" + fiberlookup_Cj0.out_ref())

                if len(mem_b_fifo) > 0 and len(mem_c_fifo) > 0:
                    if isinstance(mem_b_fifo[0], int) and isinstance(mem_c_fifo[0], int):
                        B_i00_new = mem_b_fifo[0] // 10 ** 8
                        B_k00_new = (mem_b_fifo[0] % 10 ** 8) // 10 ** 4
                        B_i0_new = (mem_b_fifo[0] % 10 ** 4) // struct["k0"]
                        B_k0_new = (mem_b_fifo[0] % 10 ** 4) % struct["k0"]
                        glb_B_token = B_i00_new * struct["k00"] + B_k00_new
                        C_k00_new = mem_c_fifo[0] // 10 ** 8
                        C_j00_new = (mem_c_fifo[0] % 10 ** 8) // 10 ** 4
                        C_k0_new = (mem_c_fifo[0] % 10 ** 4) // struct["j0"]
                        C_j0_new = (mem_c_fifo[0] % 10 ** 4) % struct["j0"]
                        glb_C_token = C_k00_new * struct["j00"] + C_j00_new

                        if (B_i00_new, B_k00_new, B_i0_new, B_k0_new) in sizes_dict_level1["B"] and\
                                (C_k00_new, C_j00_new, C_k0_new, C_j0_new) in sizes_dict_level1["C"]:
                            mem_model_b.add_tile(mem_b_fifo[0] % 10 ** 4,
                                                 sizes_dict_level1["B"][(B_i00_new, B_k00_new, B_i0_new, B_k0_new)],
                                                 glb_B_token)
                            mem_model_c.add_tile(mem_c_fifo[0] % 10 ** 4,
                                                 sizes_dict_level1["C"][(C_k00_new, C_j00_new, C_k0_new, C_j0_new)],
                                                 glb_C_token)
                    else:
                        b_t = mem_b_fifo[0].split("_")
                        c_t = mem_c_fifo[0].split("_")
                        mem_model_b.add_tile(b_t[1], 8, int(b_t[0]))
                        mem_model_c.add_tile(c_t[1], 8, int(c_t[0]))
                    mem_b_fifo.pop(0)
                    mem_c_fifo.pop(0)

        flag_token_mem = False
        if mem_model_b.valid_tile() and mem_model_c.valid_tile():
            flag = True
            flag_token_mem = True
            tiled_done_processed = False
            in_ref_B = [0, 'D']
            in_ref_C = [0, 'D']
            if nbuffer:
                B_k00 = (mem_model_b.token() // 1000000) % struct["k00"]
                B_i00 = (mem_model_b.token() // 1000000) // struct["k00"]
                B_k0 = (mem_model_b.token() % 1000000) % struct["k0"]
                B_i0 = (mem_model_b.token() % 1000000) // struct["k0"]
            else:
                B_k00 = (glb_model_b.token()) % struct["k00"]
                B_i00 = (glb_model_b.token()) // struct["k00"]
                B_k0 = (mem_model_b.token()) % struct["k0"]
                B_i0 = (mem_model_b.token()) // struct["k0"]
            if nbuffer:
                C_j00 = (mem_model_c.token() // 1000000) % struct["j00"]
                C_k00 = (mem_model_c.token() // 1000000) // struct["j00"]
                C_j0 = (mem_model_c.token() % 1000000) % struct["j0"]
                C_k0 = (mem_model_c.token() % 1000000) // struct["j0"]
            else:
                C_j00 = (glb_model_c.token()) % struct["j00"]
                C_k00 = (glb_model_c.token()) // struct["j00"]
                C_j0 = mem_model_c.token() % struct["j0"]
                C_k0 = mem_model_c.token() // struct["j0"]

            B_dir = "tensor_B_tile_" + str(B_i00) + "_" + str(B_k00) + "_" + str(B_i0) + "_" + str(B_k0)
            B_dirname = os.path.join(formatted_dir, B_dir)
            C_dir = "tensor_C_tile_" + str(C_k00) + "_" + str(C_j00) + "_" + str(C_k0) + "_" + str(C_j0)
            C_dirname = os.path.join(formatted_dir, C_dir)
            B_shape_filename = os.path.join(B_dirname, "B_shape.txt")
            B0_seg_filename = os.path.join(B_dirname, "B0_seg.txt")
            B0_crd_filename = os.path.join(B_dirname, "B0_crd.txt")
            B1_seg_filename = os.path.join(B_dirname, "B1_seg.txt")
            B1_crd_filename = os.path.join(B_dirname, "B1_crd.txt")
            B_vals_filename = os.path.join(B_dirname, "B_vals.txt")
            C_shape_filename = os.path.join(C_dirname, "C_shape.txt")
            C0_seg_filename = os.path.join(C_dirname, "C0_seg.txt")
            C0_crd_filename = os.path.join(C_dirname, "C0_crd.txt")
            C1_seg_filename = os.path.join(C_dirname, "C1_seg.txt")
            C1_crd_filename = os.path.join(C_dirname, "C1_crd.txt")
            C_vals_filename = os.path.join(C_dirname, "C_vals.txt")
            if os.path.exists(B_shape_filename):
                B_shape = read_inputs(B_shape_filename)
                B_seg0 = read_inputs(B0_seg_filename)
                B_crd0 = read_inputs(B0_crd_filename)
                B_seg1 = read_inputs(B1_seg_filename)
                B_crd1 = read_inputs(B1_crd_filename)
                B_vals = read_inputs(B_vals_filename, float)
            else:
                B_shape = [8, 8]
                B_seg0 = [0, 1]
                B_crd0 = [0]
                B_seg1 = [0, 1]
                B_crd1 = [0]
                B_vals = [0]
            if os.path.exists(C_shape_filename):
                C_shape = read_inputs(C_shape_filename)
                C_seg0 = read_inputs(C0_seg_filename)
                C_crd0 = read_inputs(C0_crd_filename)
                C_seg1 = read_inputs(C1_seg_filename)
                C_crd1 = read_inputs(C1_crd_filename)
                C_vals = read_inputs(C_vals_filename, float)
            else:
                C_shape = [8, 8]
                C_seg0 = [0, 1]
                C_crd0 = [0]
                C_seg1 = [0, 1]
                C_crd1 = [0]
                C_vals = [0]

            if skip_empty and (not os.path.exists(B_shape_filename) or not os.path.exists(C_shape_filename)):
                B_seg0 = [0, 1]
                B_crd0 = [0]
                B_seg1 = [0, 1]
                B_crd1 = [0]
                B_vals = [0]
                C_seg0 = [0, 1]
                C_crd0 = [0]
                C_seg1 = [0, 1]
                C_crd1 = [0]
                C_vals = [0]
                tiled_skip = True
            B_shape = [loop_config["Mem_tile_size"], loop_config["Mem_tile_size"]]
            C_shape = [loop_config["Mem_tile_size"], loop_config["Mem_tile_size"]]
            fiberlookup_Bi_19 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0,
                                                    debug=debug_sim, statistics=report_stats)
            fiberlookup_Bk_14 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1,
                                                    debug=debug_sim, statistics=report_stats)
            repsiggen_i_17 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
            repeat_Ci_16 = Repeat(debug=debug_sim, statistics=report_stats)
            fiberlookup_Ck_15 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0,
                                                    debug=debug_sim, statistics=report_stats)
            intersectk_13 = Intersect2(debug=debug_sim, statistics=report_stats)
            crdhold_5 = CrdHold(debug=debug_sim, statistics=report_stats)
            fiberlookup_Cj_12 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1,
                                                    debug=debug_sim, statistics=report_stats)
            arrayvals_C_8 = Array(init_arr=C_vals, debug=debug_sim, statistics=report_stats)
            crdhold_4 = CrdHold(debug=debug_sim, statistics=report_stats)
            repsiggen_j_10 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
            repeat_Bj_9 = Repeat(debug=debug_sim, statistics=report_stats)
            arrayvals_B_7 = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats)
            mul_6 = Multiply2(debug=debug_sim, statistics=report_stats)
            spaccumulator1_3 = SparseAccumulator1(debug=debug_sim, statistics=report_stats)
            spaccumulator1_3_drop_crd_inner = StknDrop(debug=debug_sim, statistics=report_stats)
            spaccumulator1_3_drop_crd_outer = StknDrop(debug=debug_sim, statistics=report_stats)
            spaccumulator1_3_drop_val = StknDrop(debug=debug_sim, statistics=report_stats)
            fiberwrite_Xvals_0 = ValsWrScan(size=1 * B_shape[0] * C_shape[1], fill=fill,
                                            debug=debug_sim, statistics=report_stats)
            fiberwrite_X1_1 = CompressWrScan(seg_size=B_shape[0] + 1, size=B_shape[0] * C_shape[1],
                                             fill=fill, debug=debug_sim, statistics=report_stats)
            fiberwrite_X0_2 = CompressWrScan(seg_size=2, size=B_shape[0],
                                             fill=fill, debug=debug_sim,
                                             statistics=report_stats)
            initialize_cntr = time_cnt
            mem_model_b.valid_tile_recieved()
            mem_model_c.valid_tile_recieved()
            check_flag = True

        if mem_model_b.valid_tile() and mem_model_c.valid_tile() and mem_model_b.out_done() and mem_model_c.out_done():
            mem_model_x.add_upstream(tilecoord="D", data=[[0], [0], [0], [0], [0]], valid=True)
            mem_model_b.valid_tile_recieved()
            mem_model_c.valid_tile_recieved()

        if flag:
            if len(in_ref_B) > 0:
                fiberlookup_Bi_19.set_in_ref(in_ref_B.pop(0))
            fiberlookup_Bk_14.set_in_ref(fiberlookup_Bi_19.out_ref())
            repsiggen_i_17.set_istream(fiberlookup_Bi_19.out_crd())
            if len(in_ref_C) > 0:
                repeat_Ci_16.set_in_ref(in_ref_C.pop(0))
            repeat_Ci_16.set_in_repsig(repsiggen_i_17.out_repsig())
            fiberlookup_Ck_15.set_in_ref(repeat_Ci_16.out_ref())
            intersectk_13.set_in1(fiberlookup_Ck_15.out_ref(), fiberlookup_Ck_15.out_crd())
            intersectk_13.set_in2(fiberlookup_Bk_14.out_ref(), fiberlookup_Bk_14.out_crd())
            crdhold_5.set_outer_crd(fiberlookup_Bi_19.out_crd())
            crdhold_5.set_inner_crd(intersectk_13.out_crd())
            fiberlookup_Cj_12.set_in_ref(intersectk_13.out_ref1())
            arrayvals_C_8.set_load(fiberlookup_Cj_12.out_ref())
            crdhold_4.set_outer_crd(crdhold_5.out_crd_outer())
            crdhold_4.set_inner_crd(fiberlookup_Cj_12.out_crd())
            repsiggen_j_10.set_istream(fiberlookup_Cj_12.out_crd())
            repeat_Bj_9.set_in_ref(intersectk_13.out_ref2())
            repeat_Bj_9.set_in_repsig(repsiggen_j_10.out_repsig())
            arrayvals_B_7.set_load(repeat_Bj_9.out_ref())
            mul_6.set_in1(arrayvals_B_7.out_val())
            mul_6.set_in2(arrayvals_C_8.out_val())
            spaccumulator1_3_drop_crd_outer.set_in_stream(crdhold_4.out_crd_outer())
            spaccumulator1_3_drop_crd_inner.set_in_stream(crdhold_4.out_crd_inner())
            spaccumulator1_3_drop_val.set_in_stream(mul_6.out_val())
            spaccumulator1_3.set_crd_outer(spaccumulator1_3_drop_crd_outer.out_val())
            spaccumulator1_3.set_crd_inner(spaccumulator1_3_drop_crd_inner.out_val())
            spaccumulator1_3.set_val(spaccumulator1_3_drop_val.out_val())
            fiberwrite_Xvals_0.set_input(spaccumulator1_3.out_val())
            fiberwrite_X1_1.set_input(spaccumulator1_3.out_crd_inner())
            fiberwrite_X0_2.set_input(spaccumulator1_3.out_crd_outer())
            if tiled_done and not tile_signalled:
                mem_model_b.check_if_done(tiled_done)
                mem_model_c.check_if_done(tiled_done)
                tile_signalled = True
            else:
                mem_model_b.check_if_done(False)
                mem_model_c.check_if_done(False)
        fiberlookup_Bi00.update()
        fiberlookup_Bk00.update()
        repsiggen_i00.update()
        repeat_Ci00.update()
        fiberlookup_Ck00.update()
        fiberlookup_Cj00.update()
        repsiggen_j00.update()
        repeat_Bj00.update()
        glb_model_b.update(time_cnt)
        glb_model_c.update(time_cnt)
        if flag_glb:
            fiberlookup_Bi0.update()
            fiberlookup_Bk0.update()
            repsiggen_i0.update()
            repeat_Ci0.update()
            fiberlookup_Ck0.update()
            fiberlookup_Cj0.update()
            repsiggen_j0.update()
            repeat_Bj0.update()
        mem_model_b.update(time_cnt)
        mem_model_c.update(time_cnt)

        tiled_done = False
        if flag:
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

            tiled_done = fiberwrite_X0_2.out_done() and fiberwrite_X1_1.out_done() and fiberwrite_Xvals_0.out_done()
            if not tiled_done:
                tile_signalled = False
            if tiled_skip:
                tiled_done = True

            if tiled_done and check_flag:
                array.append(time_cnt - initialize_cntr)
                array2.append(str(B_i00) + "," + str(B_k00) + "," + str(C_j00) + ":" + "," +
                              str(B_i0) + "," + str(B_k0) + "," + str(C_j0))
                check_flag = False
                if not tiled_skip:
                    fiberwrite_X0_2.autosize()
                    fiberwrite_X1_1.autosize()
                    fiberwrite_Xvals_0.autosize()

                    out_crds = [fiberwrite_X0_2.get_arr(), fiberwrite_X1_1.get_arr()]
                    out_segs = [fiberwrite_X0_2.get_seg_arr(), fiberwrite_X1_1.get_seg_arr()]
                    out_vals = fiberwrite_Xvals_0.get_arr()
                if debug_sim:
                    pass
                if check_gold:
                    assert B_k00 == C_k00 and B_k0 == C_k0
                    if not tiled_skip:
                        check_gold_matmul_tiled([B_i00, B_k00, B_i0, B_k0],
                                                [C_k00, C_j00, C_k0, C_j0], None,
                                                debug_sim, out_crds=out_crds, out_segs=out_segs,
                                                out_val=out_vals, out_format="ss01")
                if report_stats and not tiled_skip:
                    stats_dict["mul_6_ops"] += mul_6.return_statistics()["cycle_operation"]
                    stats_dict["spacc1_3_rmw_ops"].append(spaccumulator1_3.return_statistics()["rmw_ops"])
                else:
                    stats_dict["mul_6_ops"] += 0
                    stats_dict["spacc1_3_rmw_ops"].append(0)
            if debug_sim and glb_model_b.out_done() == "D":
                print(mem_model_c.token(), " ", mem_model_b.token())
                print("GLB reader done ", glb_model_x.out_done(), " ", mem_model_x.out_done())
                print(glb_model_c.token(), " ", glb_model_b.token())
            if debug_sim and mem_model_c.token() == "D" and mem_model_b.token() == "D":
                print("Mem reader done ", glb_model_x.out_done(), " ", mem_model_x.out_done())
                print(glb_model_c.token() == "D" and glb_model_b.token() == "D")
            if tiled_done:
                if mem_model_c.out_done():
                    if glb_model_c.out_done():
                        if mem_model_b.out_done():
                            if glb_model_b.out_done():
                                done = True
        if not tiled_skip:
            time_cnt += 1
        tiled_skip = False
    print(array)
    print("compute ", sum(array))
    print(array2)
    print("TOTAL_TIME = ", time_cnt)
    if report_stats:
        print("\t Mul ops:", stats_dict["mul_6_ops"])
        print("\t Acc ops:", sum(stats_dict["spacc1_3_rmw_ops"]))
