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
from sam.sim.src.tiling.generate_tile_crd import *
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
def test_matmul_ikj_tiled_sparse(samBench, ssname, check_gold, debug_sim, report_stats,
                                 skip_empty, yaml_name, nbuffer, fill=0):
    stats_dict = {"mul_6_ops": 0, "spacc1_3_rmw_ops": [], "out_arr_size": 0, "repsiggen_i_17_total_rep": 0,
                  "repsiggen_j_10_total_rep": 0, "repsiggen_i_17_max_rep": 0, "repsiggen_j_10_max_rep": 0,
                  "fiberlookup_Bi_19_done": 0, "fiberlookup_Bk_14_done": 0, "fiberlookup_Bi_19_ttl": 0,
                  "fiberlookup_Bk_14_ttl": 0, "repsiggen_i_17_done": 0,
                  "repeat_Ci_16_done": 0, "repeat_Bj_9_done": 0, "fiberlookup_Ck_15_done": 0,
                  "fiberlookup_Ck_15_ttl": 0, "intersectk_13_done": 0, "crdhold_5_done": 0,
                  "fiberlookup_Cj_12_done": 0, "fiberlookup_Cj_12_ttl": 0,
                  "arrayvals_C_8_done": 0, "crdhold_4_done": 0, "repsiggen_j_10_done": 0,
                  "arrayvals_B_7_done": 0, "mul_6_done": 0, "spaccumulator1_3_done": 0, "spaccumulator1_3_inner_done": 0,
                  "spaccumulator1_3_outer_done": 0, "spaccumulator1_3_vals_done": 0}
    # Helper Datastructures
    # Sizes for the full matrix (to get software loop sizes)
    with open(os.path.join(sam_home, "tiles/matmul_ikj/tensor_sizes"), "rb") as ff:
        sizes_dict_level_full = pickle.load(ff)
    # Make sure the tiles were correctly done
    assert sizes_dict_level_full["B"][0] == sizes_dict_level_full["C"][1]
    # GLB tile's sizes in a dict
    with open(os.path.join(sam_home, "tiles/matmul_ikj/hw_level_0_sizes"), "rb") as ff:
        sizes_dict_level0 = pickle.load(ff)
    # Mem tile's sizes in a dict
    with open(os.path.join(sam_home, "tiles/matmul_ikj/hw_level_1_sizes"), "rb") as ff:
        sizes_dict_level1 = pickle.load(ff)
    # Tile sizes at GLB and memory level
    with open(os.path.join(sam_home, "sam/sim/src/tiling/" + yaml_name), "r") as stream:
        loop_config = yaml.safe_load(stream)
    # Loading the same thing again (redundant)
    with open(os.path.join(sam_home, "./sam/sim/src/tiling/" + yaml_name), "r") as stream:
        memory_config = yaml.safe_load(stream)

    struct = {"i00": 1 + int(sizes_dict_level_full["B"][0]) // (loop_config["Glb_tile_size"] * loop_config["Mem_tile_size"]),
              "k00": 1 + int(sizes_dict_level_full["B"][1]) // (loop_config["Glb_tile_size"] * loop_config["Mem_tile_size"]),
              "j00": 1 + int(sizes_dict_level_full["C"][1]) // (loop_config["Glb_tile_size"] * loop_config["Mem_tile_size"]),
              "i0": int(loop_config["Glb_tile_size"]), "k0": int(loop_config["Glb_tile_size"]),
              "j0": int(loop_config["Glb_tile_size"])}

    cnt_i = 0
    # Get an array that allows us to get the cordinates from the reference values
    # Called ref_glb_convertor to be able to use the sizes_dict_level0
    # Get seg and crd arrays of mem tiles and glb tiles in glb_arr
    ref_glb_convertor, glb_arr = generate_tile_crd_glb_matmul(struct, sizes_dict_level0)
    ref_to_crd_convertor = dict()

    # Intitialize software loops
    fiberlookup_Bi00 = CompressedCrdRdScan(crd_arr=glb_arr["B_crd0"], seg_arr=glb_arr["B_seg0"], debug=debug_sim)
    fiberlookup_Bk00 = CompressedCrdRdScan(crd_arr=glb_arr["B_crd1"], seg_arr=glb_arr["B_seg1"], debug=debug_sim)
    repsiggen_i00 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Ci00 = Repeat(debug=debug_sim, statistics=report_stats)
    fiberlookup_Ck00 = CompressedCrdRdScan(crd_arr=glb_arr["C_crd0"], seg_arr=glb_arr["C_seg0"], debug=debug_sim)
    intersect_00 = Intersect2(debug=debug_sim)
    fiberlookup_Cj00 = CompressedCrdRdScan(crd_arr=glb_arr["C_crd1"], seg_arr=glb_arr["C_seg1"], debug=debug_sim)
    repsiggen_j00 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Bj00 = Repeat(debug=debug_sim, statistics=report_stats)

    # Reference for software loops
    in_ref_B00 = [0, 'D']
    in_ref_C00 = [0, 'D']
    in_ref_B0 = []
    in_ref_C0 = []
    done = False
    time_cnt = 0
    time_cnt_sum = 0
    first_d = False
    # Intitialize memory blocks
    glb_model_b = memory_block(name="GLB_B", skip_blocks=skip_empty, nbuffer=nbuffer,
                               element_size=memory_config["Bytes_per_element"], size=memory_config["Glb_memory"],
                               bandwidth=memory_config["Glb_tile_bandwidth"], latency=memory_config["Global_Glb_latency"],
                               debug=debug_sim, pipeline_en=True, statistics=report_stats)
    glb_model_c = memory_block(name="GLB_C", skip_blocks=skip_empty, nbuffer=nbuffer,
                               element_size=memory_config["Bytes_per_element"], size=memory_config["Glb_memory"],
                               bandwidth=memory_config["Glb_tile_bandwidth"], latency=memory_config["Global_Glb_latency"],
                               debug=debug_sim, pipeline_en=True, statistics=report_stats)
    mem_model_b = memory_block(name="B", skip_blocks=skip_empty, nbuffer=nbuffer,
                               element_size=memory_config["Bytes_per_element"],
                               size=memory_config["Mem_memory"],
                               bandwidth=memory_config["Mem_tile_bandwidth"] / memory_config["Mem_tiles"],
                               latency=memory_config["Glb_Mem_latency"], debug=debug_sim, statistics=report_stats)
    mem_model_c = memory_block(name="C", skip_blocks=skip_empty, nbuffer=nbuffer,
                               element_size=memory_config["Bytes_per_element"],
                               size=memory_config["Mem_memory"],
                               bandwidth=memory_config["Mem_tile_bandwidth"] / memory_config["Mem_tiles"],
                               latency=memory_config["Glb_Mem_latency"], debug=debug_sim, statistics=report_stats)
    # Output memory blocks Not used with sparse tiling figure out later
    mem_model_x = output_memory_block(name="X", element_size=memory_config["Bytes_per_element"],
                                      level="mem2glb", bandwidth=memory_config["Mem_tile_bandwidth"],
                                      latency=memory_config["Glb_Mem_latency"], debug=debug_sim,
                                      loop_order=[struct["i0"], struct["k0"], struct["j0"]])
    glb_model_x = output_memory_block(name="X", element_size=memory_config["Bytes_per_element"],
                                      level="glb2global", bandwidth=memory_config["Glb_tile_bandwidth"],
                                      latency=memory_config["Glb_Mem_latency"], debug=debug_sim,
                                      loop_order=[struct["i00"], struct["k00"], struct["j00"], struct["i0"], struct["k0"],
                                      struct["j0"]])
    # seperate flags used to make sure python doesnt cause initiazation errors if a block isnt initialized yet
    flag_glb = False
    flag = False
    # Flags for tile output
    tiled_done = False
    tile_signalled = False
    # So gold checking and atatistics are collected once
    check_flag = True
    # Old skipping tiled redundant
    tiled_skip = False
    if report_stats or True:
        array = []
        array2 = []
        array3 = []
        array4 = []
        glb_b_fifo = []
        glb_c_fifo = []
        b_vals_size = 0
        c_vals_size = 0
        tile_id_b = []
        tile_id_c = []

    while not done and time_cnt < TIMEOUT:
        if debug_sim:
            print(time_cnt)
        # Software loop update every cycle
        if len(in_ref_B00) > 0:
            fiberlookup_Bi00.set_in_ref(in_ref_B00.pop(0))
        fiberlookup_Bk00.set_in_ref(fiberlookup_Bi00.out_ref())
        repsiggen_i00.set_istream(fiberlookup_Bi00.out_crd())
        if len(in_ref_C00) > 0:
            repeat_Ci00.set_in_ref(in_ref_C00.pop(0))
        repeat_Ci00.set_in_repsig(repsiggen_i00.out_repsig())
        fiberlookup_Ck00.set_in_ref(repeat_Ci00.out_ref())
        intersect_00.set_in1(fiberlookup_Ck00.out_ref(), fiberlookup_Ck00.out_crd())
        intersect_00.set_in2(fiberlookup_Bk00.out_ref(), fiberlookup_Bk00.out_crd())
        fiberlookup_Cj00.set_in_ref(intersect_00.out_ref1())
        repsiggen_j00.set_istream(fiberlookup_Cj00.out_crd())
        repeat_Bj00.set_in_ref(intersect_00.out_ref2())
        repeat_Bj00.set_in_repsig(repsiggen_j00.out_repsig())

        if glb_model_b.valid_tile() and glb_model_c.valid_tile():
            # Valid tile is in glb
            # get keys
            keys = [glb_model_b.token(), glb_model_c.token()]
            # Initialize memory array
            # Get an array that allows us to get the cordinates from the reference values
            # Called ref_to_crd_convertor to be able to use the sizes_dict_level0
            # Get seg and crd arrays of mem tiles in mem_arr
            ref_to_crd_convertor, mem_arr = generate_tile_crd_mem_matmul(struct, sizes_dict_level1, keys,
                                                                         ref_glb_convertor, ref_to_crd_convertor)
            # New GLB tile: Reintialize controllers at the glb level
            flag_glb = True
            in_ref_B0 = [0, 'D']
            in_ref_C0 = [0, 'D']
            fiberlookup_Bi0 = CompressedCrdRdScan(crd_arr=mem_arr["B_crd0"], seg_arr=mem_arr["B_seg0"], debug=debug_sim)
            fiberlookup_Bk0 = CompressedCrdRdScan(crd_arr=mem_arr["B_crd1"], seg_arr=mem_arr["B_seg1"], debug=debug_sim)
            repsiggen_i0 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
            repeat_Ci0 = Repeat(debug=debug_sim, statistics=report_stats)
            fiberlookup_Ck0 = CompressedCrdRdScan(crd_arr=mem_arr["C_crd0"], seg_arr=mem_arr["C_seg0"], debug=debug_sim)
            intersect_0 = Intersect2(debug=debug_sim)
            fiberlookup_Cj0 = CompressedCrdRdScan(crd_arr=mem_arr["C_crd1"], seg_arr=mem_arr["C_seg1"], debug=debug_sim)
            repsiggen_j0 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
            repeat_Bj0 = Repeat(debug=debug_sim, statistics=report_stats)
            # print("INITIALIZE Mem loop at ", time_cnt)
            glb_model_b.valid_tile_recieved()
            glb_model_c.valid_tile_recieved()

        if isinstance(repeat_Bj00.out_ref(), int):
            # Get size of tile from a the datastructure
            # Get the coordinates for the tile
            B_ = ref_glb_convertor["B"][repeat_Bj00.out_ref()].split("_")
            glb_model_b.add_tile(repeat_Bj00.out_ref(), sizes_dict_level0["B"][(int(B_[0]), int(B_[1]))])
        else:
            # Add Done token with size 8
            glb_model_b.add_tile(repeat_Bj00.out_ref(), 8)
        if isinstance(fiberlookup_Cj00.out_ref(), int):
            # Add an actual tile for the datastructure
            # Get crds from the reference
            C_ = ref_glb_convertor["C"][fiberlookup_Cj00.out_ref()].split("_")
            glb_model_c.add_tile(fiberlookup_Cj00.out_ref(), sizes_dict_level0["C"][(int(C_[0]), int(C_[1]))])
        else:
            # Add Done tile
            glb_model_c.add_tile(fiberlookup_Cj00.out_ref(), 8)
        # Evict tile and move ahead
        glb_model_b.check_if_done(mem_model_b.out_done_in())
        glb_model_c.check_if_done(mem_model_c.out_done_in())
        # GLB controllers run
        if flag_glb:
            if len(in_ref_B0) > 0:
                fiberlookup_Bi0.set_in_ref(in_ref_B0.pop(0))
            fiberlookup_Bk0.set_in_ref(fiberlookup_Bi0.out_ref())
            repsiggen_i0.set_istream(fiberlookup_Bi0.out_crd())
            if len(in_ref_C0) > 0:
                repeat_Ci0.set_in_ref(in_ref_C0.pop(0))
            repeat_Ci0.set_in_repsig(repsiggen_i0.out_repsig())
            fiberlookup_Ck0.set_in_ref(repeat_Ci0.out_ref())
            intersect_0.set_in1(fiberlookup_Ck0.out_ref(), fiberlookup_Ck0.out_crd())
            intersect_0.set_in2(fiberlookup_Bk0.out_ref(), fiberlookup_Bk0.out_crd())
            fiberlookup_Cj0.set_in_ref(intersect_0.out_ref1())
            repsiggen_j0.set_istream(fiberlookup_Cj0.out_crd())
            repeat_Bj0.set_in_ref(intersect_0.out_ref2())
            repeat_Bj0.set_in_repsig(repsiggen_j0.out_repsig())

            if isinstance(glb_model_b.token(), int):
                B_ct = ref_glb_convertor["B"][glb_model_b.token()].split("_")
                B_k00__ = int(B_ct[1])
                B_i00__ = int(B_ct[0])
                if isinstance(repeat_Bj0.out_ref(), int):
                    # Send the next channel the reference for the mem tile being loaded + current glb tile so can
                    # use it to correctly address into the datastructure
                    # Get crds for mem tile value using the datastructure
                    # Each memtile block output has both the glb tile address + mem tile address backed in to use the
                    # helper datastructures
                    mem_b_t = ref_to_crd_convertor["B_" + str(glb_model_b.token())][repeat_Bj0.out_ref()].split("_")
                    mem_model_b.add_tile(repeat_Bj0.out_ref(),
                                         sizes_dict_level1["B"][(B_i00__, B_k00__, int(mem_b_t[0]), int(mem_b_t[1]))],
                                         glb_model_b.token())
                else:
                    mem_model_b.add_tile(repeat_Bj0.out_ref(), 8, glb_model_b.token())

            if isinstance(glb_model_c.token(), int):
                C_ct = ref_glb_convertor["C"][glb_model_c.token()].split("_")
                C_k00__ = int(C_ct[0])
                C_j00__ = int(C_ct[1])
                if isinstance(fiberlookup_Cj0.out_ref(), int):
                    # Send the next channel the reference for the mem tile being loaded + current glb tile so can
                    # use it to correctly address into the datastructure
                    # Get crds for mem tile value using the datastructure
                    mem_c_t = ref_to_crd_convertor["C_" + str(glb_model_c.token())][fiberlookup_Cj0.out_ref()].split("_")
                    mem_model_c.add_tile(fiberlookup_Cj0.out_ref(),
                                         sizes_dict_level1["C"][(C_k00__, C_j00__, int(mem_c_t[0]), int(mem_c_t[1]))],
                                         glb_model_c.token())
                else:
                    mem_model_c.add_tile(fiberlookup_Cj0.out_ref(), 8, glb_model_c.token())

        # Mem tile ready for working compute
        if mem_model_b.valid_tile() and mem_model_c.valid_tile():
            # mem tile kicks in
            flag = True
            in_ref_B = [0, 'D']
            in_ref_C = [0, 'D']
            if nbuffer:
                # Get the GLB reference from the memtile by divuiding it by 1000000 (hard coded for now)
                # Later replace this with a crd hold block
                # Use the reference to get the crds for the glb level
                # (since the size datastructure uses coordinates not refrences
                B_glb = ref_glb_convertor["B"][mem_model_b.token() // 1000000].split("_")
                B_k00 = int(B_glb[1])
                B_i00 = int(B_glb[0])
                # Get the mem tile refrence by modulo 1000000
                # Use the reference to get the crds for the mem level
                # (since the size datastructure uses coordinates not refrences)
                B_mem = ref_to_crd_convertor["B_" +
                                             str(mem_model_b.token() // 1000000)][mem_model_b.token() % 1000000].split("_")
                B_k0 = int(B_mem[1])
                B_i0 = int(B_mem[0])
            else:
                B_glb = ref_glb_convertor["B"][glb_model_b.token()].split("_")
                B_k00 = int(B_glb[1])
                B_i00 = int(B_glb[0])
                B_mem = ref_to_crd_convertor["B_" +
                                             str(mem_model_b.token() // 1000000)][mem_model_b.token() % 1000000].split("_")
                B_k0 = int(B_mem[1])
                B_i0 = int(B_mem[0])

            if nbuffer:
                # Get the GLB reference from the memtile by divuiding it by 1000000 (hard coded for now)
                # can use a crd hold block
                # Use the reference to get the crds for the glb level
                # (since the size datastructure uses coordinates not refrences
                C_glb = ref_glb_convertor["C"][mem_model_c.token() // 1000000].split("_")
                C_j00 = int(C_glb[1])
                C_k00 = int(C_glb[0])
                # Get the mem tile refrence by modulo 1000000
                # Use the reference to get the crds for the mem level
                # (since the size datastructure uses coordinates not refrences
                C_mem = ref_to_crd_convertor["C_" +
                                             str(mem_model_c.token() // 1000000)][mem_model_c.token() % 1000000].split("_")
                C_j0 = int(C_mem[1])
                C_k0 = int(C_mem[0])
            else:
                C_glb = ref_glb_convertor["C"][glb_model_c.token()].split("_")
                C_j00 = int(C_glb[1])
                C_k00 = int(C_glb[0])
                C_mem = ref_to_crd_convertor["C_" +
                                             str(mem_model_c.token() // 1000000)][mem_model_c.token() % 1000000].split("_")
                C_j0 = int(C_mem[1])
                C_k0 = int(C_mem[0])

            if B_k0 != C_k0:
                print("B ", B_i00, B_k00, B_i0, B_k0)
                print("C ", C_k00, C_j00, C_k0, C_j0)
                assert False
            # ABove allows us to get coordinates, allows reading in correct tiles since the files are stored with respect to
            # their coordinates not references
            # Load files for current tile
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
                # if sparse tiling case shouldnt happen
                assert False
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
                # if sparse tiling cant happen
                assert False

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
            # Shape was incorrect, so get it from loop config value to get it
            B_shape = [loop_config["Mem_tile_size"], loop_config["Mem_tile_size"]]
            C_shape = [loop_config["Mem_tile_size"], loop_config["Mem_tile_size"]]
            # tile id as a string for bookeeping
            b_tile_id = str(B_i00) + "_" + str(B_k00) + "_" + str(B_i0) + "_" + str(B_k0)
            c_tile_id = str(C_k00) + "_" + str(C_j00) + "_" + str(C_k0) + "_" + str(C_j0)
            # Add sizes with statistics
            if report_stats and b_tile_id not in tile_id_b:
                b_vals_size += len(B_vals)
                tile_id_b.append(b_tile_id)
            if report_stats and c_tile_id not in tile_id_c:
                c_vals_size += len(C_vals)
                tile_id_c.append(c_tile_id)

            fiberlookup_Bi_19 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats)
            fiberlookup_Bk_14 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats)
            repsiggen_i_17 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
            repeat_Ci_16 = Repeat(debug=debug_sim, statistics=report_stats)
            fiberlookup_Ck_15 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim, statistics=report_stats)
            intersectk_13 = Intersect2(debug=debug_sim, statistics=report_stats)
            crdhold_5 = CrdHold(debug=debug_sim, statistics=report_stats)
            fiberlookup_Cj_12 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim, statistics=report_stats)
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
            fiberwrite_X0_2 = CompressWrScan(seg_size=2, size=B_shape[0], fill=fill, debug=debug_sim, statistics=report_stats)
            # print("INITIALIZE compute loop at ", time_cnt)
            initialize_cntr = time_cnt
            mem_model_b.valid_tile_recieved()
            mem_model_c.valid_tile_recieved()
            check_flag = True
            # Chekc if current tile has already been computed on
            if str(B_i00) + "," + str(B_k00) + "," + str(C_j00) + ":" + \
                    "," + str(B_i0) + "," + str(B_k0) + "," + str(C_j0) in array2:
                print(str(B_i00) + "," + str(B_k00) + "," + str(C_j00) +
                      ":" + "," + str(B_i0) + "," + str(B_k0) + "," + str(C_j0))
                print(ref_to_crd_convertor["B_" + str(mem_model_b.token() // 1000000)])
                assert False

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
            # If tile computed on move ahead
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
        intersect_00.update()
        fiberlookup_Cj00.update()
        repsiggen_j00.update()
        repeat_Bj00.update()
        glb_model_b.update(time_cnt)
        glb_model_c.update(time_cnt)
        # print("000000000000000000000000000000000000000000000000000000000000000")
        if flag_glb:
            fiberlookup_Bi0.update()
            fiberlookup_Bk0.update()
            repsiggen_i0.update()
            repeat_Ci0.update()
            fiberlookup_Ck0.update()
            intersect_0.update()
            fiberlookup_Cj0.update()
            repsiggen_j0.update()
            repeat_Bj0.update()
        # print("1111111111111111111111111111111111111111111111111111111111111111")
        # print("1111111111111111111111111111111111111111111111111111111111111111")
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
            # Not required used to skip tiles
            if not tiled_done:
                tile_signalled = False
            if tiled_skip:
                tiled_done = True
            if tiled_done and check_flag:
                # Add tile time for tile computation
                first_d = True
                if report_stats:
                    array.append(time_cnt - initialize_cntr)
                # Check if current tile was repeated
                if str(B_i00) + "," + str(B_k00) + "," + str(C_j00) + \
                        ":" + "," + str(B_i0) + "," + str(B_k0) + "," + str(C_j0) in array2:
                    print(str(B_i00) + "," + str(B_k00) + "," + str(C_j00) +
                          ":" + "," + str(B_i0) + "," + str(B_k0) + "," + str(C_j0))
                    print(array2)
                    print(ref_to_crd_convertor["B_" + str(mem_model_b.token() // 1000000)])
                    assert False

                check_flag = False
                if not tiled_skip:
                    fiberwrite_X0_2.autosize()
                    fiberwrite_X1_1.autosize()
                    fiberwrite_Xvals_0.autosize()
                    out_crds = [fiberwrite_X0_2.get_arr(), fiberwrite_X1_1.get_arr()]
                    out_segs = [fiberwrite_X0_2.get_seg_arr(), fiberwrite_X1_1.get_seg_arr()]
                    out_vals = fiberwrite_Xvals_0.get_arr()
                    array4.append(len(out_vals))
                if debug_sim:
                    pass
                if check_gold:
                    assert B_k00 == C_k00 and B_k0 == C_k0
                    if C_j0 == 0 and C_k0 == 0:
                        print("Checking gold... ", B_i00, B_k00, B_i0, B_k0, C_k00, C_j00, C_k0, C_j0, " ", tiled_skip)
                        # check_gold_matmul_tiled([B_i00, B_k00, B_i0, B_k0], [C_k00, C_j00, C_k0, C_j0],
                        #                         None, debug_sim, out_crds=out_crds, out_segs=out_segs,
                        #                         out_val=out_vals, out_format="ss01")
                if report_stats and not tiled_skip:
                    stats_dict["mul_6_ops"] += mul_6.return_statistics()["cycle_operation"]
                    stats_dict["spacc1_3_rmw_ops"].append(spaccumulator1_3.return_statistics()["rmw_ops"])
                    stats_dict["out_arr_size"] += fiberwrite_Xvals_0.return_statistics()["size"]
                else:
                    stats_dict["mul_6_ops"] += 0
                    stats_dict["spacc1_3_rmw_ops"].append(0)

                if report_stats:
                    stats_dict["repsiggen_i_17_total_rep"] += repsiggen_i_17.cycles_curr_total
                    stats_dict["repsiggen_j_10_total_rep"] += repsiggen_j_10.cycles_curr_total
                    stats_dict["repsiggen_i_17_max_rep"] += repsiggen_i_17.cycles_curr_max
                    stats_dict["repsiggen_j_10_max_rep"] += repsiggen_j_10.cycles_curr_max
                    stats_dict["fiberlookup_Bi_19_done"] += fiberlookup_Bi_19.return_statistics_base()["done_cycles"] -\
                        fiberlookup_Bi_19.return_statistics_base()["start_cycle"]
                    stats_dict["fiberlookup_Bk_14_done"] += fiberlookup_Bk_14.return_statistics_base()["done_cycles"] -\
                        fiberlookup_Bk_14.return_statistics_base()["start_cycle"]
                    stats_dict["repsiggen_i_17_done"] += repsiggen_i_17.return_statistics_base()["done_cycles"]
                    stats_dict["repeat_Ci_16_done"] += repeat_Ci_16.return_statistics_base()["done_cycles"]
                    stats_dict["fiberlookup_Ck_15_done"] += fiberlookup_Ck_15.return_statistics_base()["done_cycles"] -\
                        fiberlookup_Ck_15.return_statistics_base()["start_cycle"]
                    stats_dict["intersectk_13_done"] += intersectk_13.return_statistics_base()["done_cycles"]
                    stats_dict["crdhold_5_done"] += crdhold_5.return_statistics_base()["done_cycles"]
                    stats_dict["fiberlookup_Cj_12_done"] += fiberlookup_Cj_12.return_statistics_base()["done_cycles"] -\
                        fiberlookup_Cj_12.return_statistics_base()["start_cycle"]
                    stats_dict["fiberlookup_Bi_19_ttl"] += fiberlookup_Bi_19.return_statistics_base()["done_cycles"]
                    stats_dict["fiberlookup_Bk_14_ttl"] += fiberlookup_Bk_14.return_statistics_base()["done_cycles"]
                    stats_dict["fiberlookup_Ck_15_ttl"] += fiberlookup_Ck_15.return_statistics_base()["done_cycles"]
                    stats_dict["fiberlookup_Cj_12_ttl"] += fiberlookup_Cj_12.return_statistics_base()["done_cycles"]
                    stats_dict["arrayvals_C_8_done"] += arrayvals_C_8.return_statistics_base()["done_cycles"]
                    stats_dict["crdhold_4_done"] += crdhold_4.return_statistics_base()["done_cycles"]
                    stats_dict["repsiggen_j_10_done"] += repsiggen_j_10.return_statistics_base()["done_cycles"]
                    stats_dict["repeat_Bj_9_done"] += repeat_Bj_9.return_statistics_base()["done_cycles"]
                    stats_dict["arrayvals_B_7_done"] += arrayvals_B_7.return_statistics_base()["done_cycles"]
                    stats_dict["mul_6_done"] += mul_6.return_statistics_base()["done_cycles"]
                    stats_dict["spaccumulator1_3_done"] += spaccumulator1_3.return_statistics_base()["done_cycles"]
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
            if first_d and not fiberwrite_X0_2.return_block_start() and not fiberwrite_X1_1.return_block_start() \
                    and not fiberwrite_Xvals_0.return_block_start() and not arrayvals_C_8.return_block_start() \
                    and not arrayvals_B_7.return_block_start():
                time_cnt_sum += 1

        if not tiled_skip:
            time_cnt += 1
        tiled_skip = False

    print("-")
    if report_stats:
        print(b_vals_size, c_vals_size)
        print(len(array), len(array2), len(array3))
        print("outputs:= ", sum(array3))
        print("outputs compressed: ", sum(array4))
        print("compute ", sum(array))
    print("tiles: ", len(sizes_dict_level1["B"].keys()))
    print("TOTAL_TIME = ", time_cnt, time_cnt_sum - 2 * len(array) + 2)
    if report_stats:
        print("\t Mul ops:", stats_dict["mul_6_ops"])
        print("\t Acc ops:", sum(stats_dict["spacc1_3_rmw_ops"]))
        print("\t Out size:", stats_dict["out_arr_size"])
        print("\t Stats:", glb_model_b.print_stats())
        print("\t Stats:", glb_model_c.print_stats())
        print("\t Stats:", mem_model_b.print_stats())
        print("\t Stats:", mem_model_c.print_stats())
        for a in stats_dict.keys():
            if "spacc1_3" not in a or "spacc1_3_rmw_ops" not in a:
                print(a)
                if isinstance(stats_dict[a], float) or isinstance(stats_dict[a], int):
                    print(stats_dict[a] / max(len(array), len(array2)))
