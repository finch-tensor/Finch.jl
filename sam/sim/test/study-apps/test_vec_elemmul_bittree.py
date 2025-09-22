import pytest
import random
import os
import time

from sam.sim.src.rd_scanner import BVRdScan, CompressedCrdRdScan
from sam.sim.src.bitvector import BV, BVDrop
from sam.sim.src.wr_scanner import ValsWrScan, CompressWrScan
from sam.sim.src.joiner import IntersectBV2
from sam.sim.src.compute import Multiply2
from sam.sim.src.array import Array
from sam.sim.src.split import Split
from sam.sim.src.base import remove_emptystr

from sam.sim.test.test import TIMEOUT, check_arr, bv, inner_bv, read_inputs

cwd = os.getcwd()
synthetic_dir = os.getenv('SYNTHETIC_PATH', default=os.path.join(cwd, 'synthetic'))


# NOTE: This is the full vector elementwise multiplication as a bitvector
# @pytest.mark.vec
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.synth
@pytest.mark.parametrize("run_length", [1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 200, 300, 400])
@pytest.mark.parametrize("vectype", ["random", "runs", "blocks"])
@pytest.mark.parametrize("sparsity", [0.2, 0.6, 0.8, 0.9, 0.95, 0.975, 0.9875, 0.99375])
@pytest.mark.parametrize("sf", [16, 32, 64, 256, 512])
def test_vec_elemmul_bv_split(samBench, run_length, vectype, sparsity, backpressure, depth, sf, debug_sim, size=2000, fill=0):
    inner_fiber_cnt = int(size / sf) + 1

    if vectype == "random":
        b_dirname = os.path.join(synthetic_dir, vectype, "compressed", "B_" + vectype + "_sp_" + str(sparsity))
    elif vectype == "runs":
        # b_dirname = os.path.join(synthetic_dir, vectype, "compressed", "runs_0_100_200")
        b_dirname = os.path.join(synthetic_dir, vectype, "compressed", f"runs_rl_{run_length}_nnz_400")
    elif vectype == "blocks":
        b_dirname = os.path.join(synthetic_dir, vectype, "compressed", f"B_blocks_400_{run_length}")

    b0_seg_filename = os.path.join(b_dirname, "tensor_B_mode_0_seg")
    b_seg0 = read_inputs(b0_seg_filename)
    b0_crd_filename = os.path.join(b_dirname, "tensor_B_mode_0_crd")
    b_crd0 = read_inputs(b0_crd_filename)
    b_vals_filename = os.path.join(b_dirname, "tensor_B_mode_vals")
    b_vals = read_inputs(b_vals_filename, float)

    if vectype == "random":
        c_dirname = os.path.join(synthetic_dir, vectype, "compressed", "C_" + vectype + "_sp_" + str(sparsity))
    elif vectype == "runs":
        # c_dirname = os.path.join(synthetic_dir, vectype, "compressed", "runs_0_100_200")
        c_dirname = os.path.join(synthetic_dir, vectype, "compressed", f"runs_rl_{run_length}_nnz_400")
    elif vectype == "blocks":
        c_dirname = os.path.join(synthetic_dir, vectype, "compressed", f"C_blocks_400_{run_length}")
        # c_dirname = os.path.join(synthetic_dir, vectype, "compressed", "C_blocks_20_20")

    c0_seg_filename = os.path.join(c_dirname, "tensor_C_mode_0_seg")
    c_seg0 = read_inputs(c0_seg_filename)
    c0_crd_filename = os.path.join(c_dirname, "tensor_C_mode_0_crd")
    c_crd0 = read_inputs(c0_crd_filename)

    c_vals_filename = os.path.join(c_dirname, "tensor_C_mode_vals")
    c_vals = read_inputs(c_vals_filename, float)

    if debug_sim:
        print("Compressed VECTOR 1:\n", b_seg0, "\n", b_crd0, "\n", b_vals)
        print("Compressed VECTOR 2:\n", c_seg0, "\n", c_crd0, "\n", c_vals)

    gold_bv1_1 = [bv([int(elem / sf) for elem in b_crd0])]
    gold_bv1_0 = inner_bv(b_crd0, size, sf)
    gold_bv1_0 += (inner_fiber_cnt - len(gold_bv1_0)) * [0]

    gold_bv2_1 = [bv([int(elem / sf) for elem in c_crd0])]
    gold_bv2_0 = inner_bv(c_crd0, size, sf)
    gold_bv2_0 += (inner_fiber_cnt - len(gold_bv2_0)) * [0]

    gold_crd = sorted(set(b_crd0) & set(c_crd0))
    gold_seg = [0, len(gold_crd)]
    gold_vals = []

    gold_bv1 = []
    gold_bv0 = []
    if gold_crd:
        gold_vals = [b_vals[b_crd0.index(i)] * c_vals[c_crd0.index(i)] for i in gold_crd]
        gold_bv1 = [bv([int(elem / sf) for elem in gold_crd])]
        gold_bv0 = inner_bv(gold_crd, size, sf)

    if debug_sim:
        print("Compressed RESULT  :\n", gold_seg, "\n", gold_crd, "\n", gold_vals)
        print("BV arr1 0", gold_bv1_0)
        print("BV arr1 1", gold_bv1_1)
        print("BV arr2 0", gold_bv2_0)
        print("BV arr2 1", gold_bv2_1)

    crdscan1 = CompressedCrdRdScan(seg_arr=b_seg0, crd_arr=b_crd0, debug=debug_sim,
                                   back_en=backpressure, depth=int(depth))
    crdscan2 = CompressedCrdRdScan(seg_arr=c_seg0, crd_arr=c_crd0, debug=debug_sim,
                                   back_en=backpressure, depth=int(depth))
    split1 = Split(split_factor=sf, orig_crd=False, debug=debug_sim,
                   back_en=backpressure, depth=int(depth))
    split2 = Split(split_factor=sf, orig_crd=False, debug=debug_sim,
                   back_en=backpressure, depth=int(depth))

    bv1_0 = BV(debug=debug_sim, back_en=backpressure, depth=int(depth))
    bv1_1 = BV(debug=debug_sim, back_en=backpressure, depth=int(depth))
    bv2_0 = BV(debug=debug_sim, back_en=backpressure, depth=int(depth))
    bv2_1 = BV(debug=debug_sim, back_en=backpressure, depth=int(depth))

    wrscan1_0 = ValsWrScan(size=inner_fiber_cnt, fill=fill, back_en=backpressure, depth=int(depth))
    wrscan1_1 = ValsWrScan(size=1, fill=fill, back_en=backpressure, depth=int(depth))
    wrscan2_0 = ValsWrScan(size=inner_fiber_cnt, fill=fill, back_en=backpressure, depth=int(depth))
    wrscan2_1 = ValsWrScan(size=1, fill=fill, back_en=backpressure, depth=int(depth))

    in_ref1 = [0, 'D']
    in_ref2 = [0, 'D']
    done = False
    time1 = 0
    out_split1_0 = []
    out_split1_1 = []
    out_split2_0 = []
    out_split2_1 = []
    while not done and time1 < TIMEOUT:
        if len(in_ref1) > 0:
            crdscan1.set_in_ref(in_ref1.pop(0), "")
        if len(in_ref2) > 0:
            crdscan2.set_in_ref(in_ref2.pop(0), "")
        split1.set_in_crd(crdscan1.out_crd(), crdscan1)
        split2.set_in_crd(crdscan2.out_crd(), crdscan2)
        out_split1_0.append(split1.out_inner_crd())
        out_split1_1.append(split1.out_outer_crd())
        out_split2_0.append(split2.out_inner_crd())
        out_split2_1.append(split2.out_outer_crd())

        bv1_0.set_in_crd(split1.out_inner_crd(), split1)
        bv1_1.set_in_crd(split1.out_outer_crd(), split1)
        bv2_0.set_in_crd(split2.out_inner_crd(), split2)
        bv2_1.set_in_crd(split2.out_outer_crd(), split2)
        wrscan1_0.set_input(bv1_0.out_bv_int(), bv1_0)
        wrscan1_1.set_input(bv1_1.out_bv_int(), bv1_1)
        wrscan2_0.set_input(bv2_0.out_bv_int(), bv2_0)
        wrscan2_1.set_input(bv2_1.out_bv_int(), bv2_1)

        crdscan1.update()
        crdscan2.update()
        split1.update()
        split2.update()
        bv1_0.update()
        bv1_1.update()
        bv2_0.update()
        bv2_1.update()
        wrscan1_0.update()
        wrscan1_1.update()
        wrscan2_0.update()
        wrscan2_1.update()

        # print("\nTimestep", time1, "\t Done -- \n",
        #       "\nRdScan1:", crdscan1.out_done(), "\tRdScan2:", crdscan2.out_done(),
        #       "\nSplit1:", split1.out_done(), "\tSplit2:", split2.out_done(),
        #       "\nBV:", bv1_0.out_done(), bv1_1.out_done(), bv2_0.out_done(), bv2_1.out_done(),
        #       "\nWrScan:", wrscan1_0.out_done(), wrscan1_1.out_done(), wrscan2_0.out_done(), wrscan2_1.out_done()
        #       )
        done = wrscan2_0.out_done() and wrscan2_1.out_done() and wrscan1_1.out_done() and wrscan1_0.out_done()
        time1 += 1

    if debug_sim:
        print(remove_emptystr(out_split1_0))
        print(remove_emptystr(out_split1_1))
        print(remove_emptystr(out_split2_0))
        print(remove_emptystr(out_split2_1))

        print([bin(i) for i in wrscan1_0.get_arr()])
        print([bin(i) for i in wrscan1_1.get_arr()])
        print([bin(i) for i in wrscan2_0.get_arr()])
        print([bin(i) for i in wrscan2_1.get_arr()])

    check_arr(wrscan1_0, gold_bv1_0)
    check_arr(wrscan1_1, gold_bv1_1)
    check_arr(wrscan2_0, gold_bv2_0)
    check_arr(wrscan2_1, gold_bv2_1)

    bvscan1_0 = BVRdScan(bv_arr=wrscan1_0.get_arr(), debug=debug_sim)
    bvscan1_1 = BVRdScan(bv_arr=wrscan1_1.get_arr(), debug=debug_sim)
    bvscan2_0 = BVRdScan(bv_arr=wrscan2_0.get_arr(), debug=debug_sim)
    bvscan2_1 = BVRdScan(bv_arr=wrscan2_1.get_arr(), debug=debug_sim)

    inter0 = IntersectBV2(debug=debug_sim)
    inter1 = IntersectBV2(debug=debug_sim)
    val1 = Array(init_arr=b_vals, debug=debug_sim)
    val2 = Array(init_arr=c_vals, debug=debug_sim)
    mul = Multiply2(debug=debug_sim)
    bvdrop = BVDrop(debug=debug_sim)
    oval_wrscan = ValsWrScan(size=size, fill=fill)
    wrscan0 = ValsWrScan(size=size, fill=fill)
    wrscan1 = ValsWrScan(size=1, fill=fill)

    in_ref1 = [0, 'D']
    in_ref2 = [0, 'D']
    done = False
    time2 = 0
    while not done and time2 < TIMEOUT:
        if len(in_ref1) > 0:
            bvscan1_1.set_in_ref(in_ref1.pop(0), "")
        if len(in_ref2) > 0:
            bvscan2_1.set_in_ref(in_ref2.pop(0), "")

        inter1.set_in1(bvscan1_1.out_ref(), bvscan1_1.out_bv(), bvscan1_1)
        inter1.set_in2(bvscan2_1.out_ref(), bvscan2_1.out_bv(), bvscan2_1)

        bvscan1_0.set_in_ref(inter1.out_ref1(), inter1)

        bvscan2_0.set_in_ref(inter1.out_ref2(), inter1)

        inter0.set_in1(bvscan1_0.out_ref(), bvscan1_0.out_bv(), bvscan1_0)
        inter0.set_in2(bvscan2_0.out_ref(), bvscan2_0.out_bv(), bvscan2_0)

        val1.set_load(inter0.out_ref1(), inter0)
        val2.set_load(inter0.out_ref2(), inter0)
        mul.set_in1(val1.out_load(), val1)
        mul.set_in2(val2.out_load(), val2)

        oval_wrscan.set_input(mul.out_val(), mul)

        bvdrop.set_inner_bv(inter0.out_bv(), inter0)
        bvdrop.set_outer_bv(inter1.out_bv(), inter1)
        wrscan0.set_input(bvdrop.out_bv_inner(), bvdrop)
        wrscan1.set_input(bvdrop.out_bv_outer(), bvdrop)

        bvscan1_1.update()
        bvscan2_1.update()
        inter1.update()
        bvscan1_0.update()
        bvscan2_0.update()
        inter0.update()
        val1.update()
        val2.update()
        mul.update()
        oval_wrscan.update()
        bvdrop.update()
        wrscan0.update()
        wrscan1.update()

        print("Timestep", time2, "\t Done --",
              "\nRdScan1:", bvscan1_0.out_done(), bvscan2_0.out_done(), bvscan1_1.out_done(), bvscan2_1.out_done(),
              "\nInter:", inter0.out_done(), inter1.out_done(),
              "\nArr:", val1.out_done(), val2.out_done(),
              "\tMul:", mul.out_done(),
              "\nOutVal:", oval_wrscan.out_done(),
              "\tOutBV1:", wrscan1.out_done(), "\tOutBV0:", wrscan0.out_done()
              )
        done = wrscan0.out_done() and wrscan1.out_done() and oval_wrscan.out_done()
        time2 += 1

    if debug_sim:
        print("TOTAL TIME:", time1 + time2)
        print(oval_wrscan.get_arr())
        print(wrscan0.get_arr())
        print(gold_bv0)
        print(wrscan1.get_arr())
        print(gold_bv1)

    check_arr(oval_wrscan, gold_vals)
    if gold_crd:
        check_arr(wrscan0, gold_bv0)
        check_arr(wrscan1, gold_bv1)

    def bench():
        time.sleep(0.0001)

    extra_info = dict()
    extra_info["cycles_reformat"] = time1
    extra_info["cycles"] = time2
    extra_info["vectype"] = vectype
    extra_info["sparsity"] = sparsity
    extra_info["run_length"] = run_length
    extra_info["block_size"] = run_length
    extra_info["format"] = "bittree"
    extra_info["split_factor"] = sf
    extra_info["test_name"] = "test_vec_elemmul_bv_split"

    samBench(bench, extra_info)
