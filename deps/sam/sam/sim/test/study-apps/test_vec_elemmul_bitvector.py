import math
import pytest
import random
import os
import time

from sam.sim.src.rd_scanner import BVRdScan, CompressedCrdRdScan
from sam.sim.src.bitvector import BV, ChunkBV, BVDrop
from sam.sim.src.wr_scanner import ValsWrScan, CompressWrScan
from sam.sim.src.joiner import IntersectBV2
from sam.sim.src.compute import Multiply2
from sam.sim.src.array import Array
from sam.sim.src.split import Split
from sam.sim.src.base import remove_emptystr

from sam.sim.test.test import TIMEOUT, check_arr, get_bv, read_inputs

cwd = os.getcwd()
synthetic_dir = os.getenv('SYNTHETIC_PATH', default=os.path.join(cwd, 'synthetic'))


# NOTE: This is the full vector elementwise multiplication as a bitvector
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.synth
@pytest.mark.parametrize("run_length", [1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 200, 300, 400])
@pytest.mark.parametrize("vectype", ["random", "runs", "blocks"])
@pytest.mark.parametrize("sparsity", [0.2, 0.6, 0.8, 0.9, 0.95, 0.975, 0.9875, 0.99375])
# @pytest.mark.parametrize("nnz", [1, 10, 100, 500])
@pytest.mark.parametrize("sf", [16, 32, 64, 256, 512])
def test_vec_elemmul_bv(samBench, run_length, vecname, vectype, sparsity, sf, debug_sim,
                        backpressure, depth, max_val=999, size=2000, fill=0):
    inner_fiber_cnt = int(size / sf) + 1

    if vectype == "random":
        b_dirname = os.path.join(synthetic_dir, vectype, "compressed", "B_" + vectype + "_sp_" + str(sparsity))
    elif vectype == "runs":
        # b_dirname = os.path.join(synthetic_dir, vectype, "compressed", "runs_0_100_200")
        b_dirname = os.path.join(synthetic_dir, vectype, "compressed", f"runs_rl_{run_length}_nnz_400")
    elif vectype == "blocks":
        # b_dirname = os.path.join(synthetic_dir, vectype, "compressed", "B_blocks_20_20")
        b_dirname = os.path.join(synthetic_dir, vectype, "compressed", f"B_blocks_400_{run_length}")

    b0_seg_filename = os.path.join(b_dirname, "tensor_B_mode_0_seg")
    seg_arr1 = read_inputs(b0_seg_filename)
    b0_crd_filename = os.path.join(b_dirname, "tensor_B_mode_0_crd")
    crd_arr1 = read_inputs(b0_crd_filename)
    b_vals_filename = os.path.join(b_dirname, "tensor_B_mode_vals")
    vals_arr1 = read_inputs(b_vals_filename, float)

    if vectype == "random":
        c_dirname = os.path.join(synthetic_dir, vectype, "compressed", "C_" + vectype + "_sp_" + str(sparsity))
    elif vectype == "runs":
        # c_dirname = os.path.join(synthetic_dir, vectype, "compressed", "runs_0_100_200")
        c_dirname = os.path.join(synthetic_dir, vectype, "compressed", f"runs_rl_{run_length}_nnz_400")
    elif vectype == "blocks":
        c_dirname = os.path.join(synthetic_dir, vectype, "compressed", f"C_blocks_400_{run_length}")
        # c_dirname = os.path.join(synthetic_dir, vectype, "compressed", "C_blocks_20_20")

    c0_seg_filename = os.path.join(c_dirname, "tensor_C_mode_0_seg")
    seg_arr2 = read_inputs(c0_seg_filename)
    c0_crd_filename = os.path.join(c_dirname, "tensor_C_mode_0_crd")
    crd_arr2 = read_inputs(c0_crd_filename)
    c_vals_filename = os.path.join(c_dirname, "tensor_C_mode_vals")
    vals_arr2 = read_inputs(c_vals_filename, float)

    # crd_arr1 = [random.randint(0, max_val) for _ in range(nnz)]
    # crd_arr1 = sorted(set(crd_arr1))
    # seg_arr1 = [0, len(crd_arr1)]
    # vals_arr1 = [random.randint(0, max_val) for _ in range(len(crd_arr1))]

    # crd_arr2 = [random.randint(0, max_val) for _ in range(nnz)]
    # crd_arr2 = sorted(set(crd_arr2))
    # seg_arr2 = [0, len(crd_arr2)]
    # vals_arr2 = [random.randint(0, max_val) for _ in range(len(crd_arr2))]

    if debug_sim:
        print("Compressed VECTOR 1:\n", seg_arr1, "\n", crd_arr1, "\n", vals_arr1)
        print("Compressed VECTOR 2:\n", seg_arr2, "\n", crd_arr2, "\n", vals_arr2)

    gold_crd = sorted(set(crd_arr1) & set(crd_arr2))
    gold_seg = [0, len(gold_crd)]
    gold_vals = []

    gold_bv = []
    if gold_crd:
        gold_vals = [vals_arr1[crd_arr1.index(i)] * vals_arr2[crd_arr2.index(i)] for i in gold_crd]
        full_bv = get_bv(gold_crd + ['S0', 'D'])
        for elem in full_bv:
            if isinstance(elem, str) and elem[0] == '0':
                elem = int(elem, 2)
                for i in range(0, math.ceil(size / sf)):
                    gold_bv.append(bin((elem >> (sf * i)) & ((1 << sf) - 1)))
            else:
                gold_bv.append(elem)
        # gold_bv = [int(elem,2) for elem in gold_bv if elem[0] == '0' and int(elem, 2) > 0]
        gold_bv = [int(elem, 2) for elem in gold_bv[:-2]]

    if debug_sim:
        print("Compressed RESULT  :\n", gold_seg, "\n", gold_crd, "\n", gold_vals)
        print(full_bv)
        print(gold_bv)

    crdscan1 = CompressedCrdRdScan(seg_arr=seg_arr1, crd_arr=crd_arr1, debug=debug_sim, back_en=backpressure,
                                   depth=int(depth))
    crdscan2 = CompressedCrdRdScan(seg_arr=seg_arr2, crd_arr=crd_arr2, debug=debug_sim, back_en=backpressure,
                                   depth=int(depth))

    bv1 = BV(debug=debug_sim, back_en=backpressure, depth=int(depth))
    bv2 = BV(debug=debug_sim, back_en=backpressure, depth=int(depth))

    bvchunk1 = ChunkBV(width=sf, size=size, debug=debug_sim, back_en=backpressure, depth=int(depth))
    bvchunk2 = ChunkBV(width=sf, size=size, debug=debug_sim, back_en=backpressure, depth=int(depth))

    inter = IntersectBV2(emit_zeros=True, debug=debug_sim, back_en=backpressure, depth=int(depth))
    val1 = Array(init_arr=vals_arr1, debug=debug_sim, back_en=backpressure, depth=int(depth))
    val2 = Array(init_arr=vals_arr2, debug=debug_sim, back_en=backpressure, depth=int(depth))

    mul = Multiply2(debug=debug_sim, back_en=backpressure, depth=int(depth))

    oval_wrscan = ValsWrScan(size=size, fill=fill, back_en=backpressure, depth=int(depth))
    wrscan0 = ValsWrScan(size=size, fill=fill, back_en=backpressure, depth=int(depth))

    in_ref1 = [0, 'D']
    in_ref2 = [0, 'D']
    done = False
    time1 = 0

    temp1 = []
    temp2 = []
    temp3 = []
    temp4 = []
    temp5 = []
    while not done and time1 < TIMEOUT:
        if len(in_ref1) > 0:
            crdscan1.set_in_ref(in_ref1.pop(0), "")
        if len(in_ref2) > 0:
            crdscan2.set_in_ref(in_ref2.pop(0), "")
        bv1.set_in_crd(crdscan1.out_crd(), crdscan1)
        bv2.set_in_crd(crdscan2.out_crd(), crdscan2)
        bvchunk1.set_in_bv(bv1.out_bv_int(), bv1)
        bvchunk2.set_in_bv(bv2.out_bv_int(), bv2)

        temp1.append(bvchunk1.out_bv())
        temp2.append(bvchunk1.out_ref())
        temp3.append(bvchunk2.out_bv())
        temp4.append(bvchunk2.out_ref())

        inter.set_in1(bvchunk1.out_ref(), bvchunk1.out_bv_int(), bvchunk1)
        inter.set_in2(bvchunk2.out_ref(), bvchunk2.out_bv_int(), bvchunk2)

        temp5.append(inter.out_bv())

        val1.set_load(inter.out_ref1(), inter)
        val2.set_load(inter.out_ref2(), inter)
        mul.set_in1(val1.out_load(), val1)
        mul.set_in2(val2.out_load(), val2)

        oval_wrscan.set_input(mul.out_val(), mul)

        wrscan0.set_input(inter.out_bv(), inter)

        crdscan1.update()
        crdscan2.update()
        bv1.update()
        bv2.update()
        bvchunk1.update()
        bvchunk2.update()
        inter.update()
        val1.update()
        val2.update()
        mul.update()
        oval_wrscan.update()
        wrscan0.update()

        print("Timestep", time1, "\t Done --",
              "\nRdScan1:", crdscan1.out_done(), crdscan2.out_done(), bv1.out_done(), bv2.out_done(),
              "\nBVChunk:", bvchunk2.out_done(), bvchunk1.out_done(),
              "\nInter:", inter.out_done(),
              "\nArr:", val1.out_done(), val2.out_done(),
              "\tMul:", mul.out_done(),
              "\nOutVal:", oval_wrscan.out_done(), "\tOutBV0:", wrscan0.out_done()
              )
        done = wrscan0.out_done() and oval_wrscan.out_done()
        time1 += 1

    oval_wrscan.autosize()
    wrscan0.autosize()

    if debug_sim:
        print(remove_emptystr(temp1))
        print(remove_emptystr(temp2))

        print(remove_emptystr(temp3))
        print(remove_emptystr(temp4))
        print(remove_emptystr(temp5))
        print(oval_wrscan.get_arr())
        print(gold_vals)
        print(wrscan0.get_arr())
        print(gold_bv)

    check_arr(oval_wrscan, gold_vals)
    if gold_crd:
        check_arr(wrscan0, gold_bv)

    def bench():
        time.sleep(0.0001)

    extra_info = dict()
    extra_info["cycles_reformat"] = ""
    extra_info["cycles"] = time1
    extra_info["vectype"] = vectype
    extra_info["sparsity"] = sparsity
    extra_info["run_length"] = run_length
    extra_info["block_size"] = run_length
    extra_info["format"] = "bitvector"
    extra_info["split_factor"] = sf
    extra_info["test_name"] = "test_vec_elemmul_bv"

    samBench(bench, extra_info)
