import pytest
import random
import os
import time

from sam.sim.src.rd_scanner import CompressedCrdRdScan
from sam.sim.src.wr_scanner import ValsWrScan, CompressWrScan
from sam.sim.src.joiner import Intersect2
from sam.sim.src.crd_manager import CrdDrop
from sam.sim.src.compute import Multiply2
from sam.sim.src.array import Array
from sam.sim.src.split import Split
from sam.sim.src.base import remove_emptystr

from sam.sim.test.test import TIMEOUT, check_arr, check_seg_arr, remove_zeros, read_inputs

cwd = os.getcwd()
synthetic_dir = os.getenv('SYNTHETIC_PATH', default=os.path.join(cwd, 'synthetic'))


def inner_crd(ll, size, sf):
    result = []
    for i in range(int(size / sf) + 2):
        temp = [elem % sf for elem in ll if max((i - 1) * sf, 0) <= elem < i * sf]
        if temp:
            result += temp
    return result


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
def test_vec_elemmul_split(samBench, run_length, vectype, sparsity, vecname, sf, debug_sim, backpressure, depth,
                           max_val=999, size=2000, fill=0):
    inner_fiber_cnt = int(size / sf) + 1

    if vectype == "random":
        b_dirname = os.path.join(synthetic_dir, vectype, "compressed", "B_" + vectype + "_sp_" + str(sparsity))
    elif vectype == "runs":
        # b_dirname = os.path.join(synthetic_dir, vectype, "compressed", "runs_0_100_200")
        b_dirname = os.path.join(synthetic_dir, vectype, "compressed", f"runs_rl_{run_length}_nnz_400")
    elif vectype == "blocks":
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

    gold_crd1_1 = sorted(set([int(elem / sf) for elem in crd_arr1]))
    gold_crd1_0 = inner_crd(crd_arr1, size, sf)

    gold_crd2_1 = sorted(set([int(elem / sf) for elem in crd_arr2]))
    gold_crd2_0 = inner_crd(crd_arr2, size, sf)

    gold_crd = sorted(set(crd_arr1) & set(crd_arr2))
    gold_seg = [0, len(gold_crd)]
    gold_vals = []

    gold_crd1 = []
    gold_crd0 = []

    if gold_crd:
        gold_vals = [vals_arr1[crd_arr1.index(i)] * vals_arr2[crd_arr2.index(i)] for i in gold_crd]
        gold_crd1 = sorted(set([int(elem / sf) for elem in gold_crd]))
        gold_crd0 = inner_crd(gold_crd, size, sf)

    if debug_sim:
        print("Compressed RESULT  :\n", gold_seg, "\n", gold_crd, "\n", gold_vals)
        print("Crd arr1 0", gold_crd1_0)
        print("Crd arr1 1", gold_crd1_1)
        print("Crd arr2 0", gold_crd2_0)
        print("Crd arr2 1", gold_crd2_1)

    crdscan1 = CompressedCrdRdScan(seg_arr=seg_arr1, crd_arr=crd_arr1, debug=debug_sim, back_en=backpressure, depth=int(depth))
    crdscan2 = CompressedCrdRdScan(seg_arr=seg_arr2, crd_arr=crd_arr2, debug=debug_sim, back_en=backpressure, depth=int(depth))
    split1 = Split(split_factor=sf, orig_crd=False, debug=debug_sim, back_en=backpressure, depth=int(depth))
    split2 = Split(split_factor=sf, orig_crd=False, debug=debug_sim, back_en=backpressure, depth=int(depth))

    wrscan1_0 = CompressWrScan(seg_size=inner_fiber_cnt + 1, size=size, fill=fill, debug=debug_sim,
                               back_en=backpressure, depth=int(depth))
    wrscan1_1 = CompressWrScan(seg_size=2, size=inner_fiber_cnt, fill=fill, debug=debug_sim,
                               back_en=backpressure, depth=int(depth))
    wrscan2_0 = CompressWrScan(seg_size=inner_fiber_cnt + 1, size=size, fill=fill, debug=debug_sim,
                               back_en=backpressure, depth=int(depth))
    wrscan2_1 = CompressWrScan(seg_size=2, size=inner_fiber_cnt, fill=fill, debug=debug_sim,
                               back_en=backpressure, depth=int(depth))
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
        if debug_sim:
            print(remove_emptystr(out_split1_0))
            print(remove_emptystr(out_split1_1))
            print(remove_emptystr(out_split2_0))
            print(remove_emptystr(out_split2_1))

        wrscan1_0.set_input(split1.out_inner_crd(), split1)
        wrscan1_1.set_input(split1.out_outer_crd(), split1)
        wrscan2_0.set_input(split2.out_inner_crd(), split2)
        wrscan2_1.set_input(split2.out_outer_crd(), split2)

        crdscan1.update()
        crdscan2.update()
        split1.update()
        split2.update()
        wrscan1_0.update()
        wrscan1_1.update()
        wrscan2_0.update()
        wrscan2_1.update()

        print("Timestep", time1, "\t Done -- \n",
              "\nRdScan1:", crdscan1.out_done(), "\tRdScan2:", crdscan2.out_done(),
              "\nSplit1:", split1.out_done(), "\tSplit2:", split2.out_done(),
              "\nWrScan:", wrscan1_0.out_done(), wrscan1_1.out_done(), wrscan2_0.out_done(), wrscan2_1.out_done()
              )
        done = wrscan2_0.out_done() and wrscan2_1.out_done() and wrscan1_1.out_done() and wrscan1_0.out_done()
        time1 += 1

    wrscan1_0.autosize()
    wrscan1_1.autosize()
    wrscan2_0.autosize()
    wrscan2_1.autosize()

    if debug_sim:
        print(wrscan1_0.get_arr())
        print(gold_crd1_0)
        print(wrscan1_1.get_arr())
        print(gold_crd1_1)
        print(wrscan2_0.get_arr())
        print(gold_crd2_0)
        print(wrscan2_1.get_arr())
        print(gold_crd2_1)

    check_arr(wrscan1_0, gold_crd1_0)
    check_arr(wrscan1_1, gold_crd1_1)
    check_arr(wrscan2_0, gold_crd2_0)
    check_arr(wrscan2_1, gold_crd2_1)

    print("Datastructure Write Done")

    rdscan1_0 = CompressedCrdRdScan(seg_arr=wrscan1_0.get_seg_arr(), crd_arr=wrscan1_0.get_arr(), skip=False,
                                    debug=debug_sim)
    rdscan1_1 = CompressedCrdRdScan(seg_arr=wrscan1_1.get_seg_arr(), crd_arr=wrscan1_1.get_arr(), skip=False,
                                    debug=debug_sim)
    rdscan2_0 = CompressedCrdRdScan(seg_arr=wrscan2_0.get_seg_arr(), crd_arr=wrscan2_0.get_arr(), skip=False,
                                    debug=debug_sim)
    rdscan2_1 = CompressedCrdRdScan(seg_arr=wrscan2_1.get_seg_arr(), crd_arr=wrscan2_1.get_arr(), skip=False,
                                    debug=debug_sim)

    inter0 = Intersect2(debug=debug_sim)
    inter1 = Intersect2(debug=debug_sim)
    val1 = Array(init_arr=vals_arr1, debug=debug_sim)
    val2 = Array(init_arr=vals_arr2, debug=debug_sim)
    mul = Multiply2(debug=debug_sim)
    crddrop = CrdDrop(debug=debug_sim)
    oval_wrscan = ValsWrScan(size=size, fill=fill)
    wrscan0 = CompressWrScan(seg_size=inner_fiber_cnt + 1, size=size, fill=fill)
    wrscan1 = CompressWrScan(seg_size=2, size=inner_fiber_cnt, fill=fill)

    in_ref1 = [0, 'D']
    in_ref2 = [0, 'D']
    done = False
    time2 = 0
    while not done and time2 < TIMEOUT:
        if len(in_ref1) > 0:
            rdscan1_1.set_in_ref(in_ref1.pop(0))
        if len(in_ref2) > 0:
            rdscan2_1.set_in_ref(in_ref2.pop(0))

        inter1.set_in1(rdscan1_1.out_ref(), rdscan1_1.out_crd())
        inter1.set_in2(rdscan2_1.out_ref(), rdscan2_1.out_crd())

        rdscan1_0.set_in_ref(inter1.out_ref1())

        rdscan2_0.set_in_ref(inter1.out_ref2())

        inter0.set_in1(rdscan1_0.out_ref(), rdscan1_0.out_crd())
        inter0.set_in2(rdscan2_0.out_ref(), rdscan2_0.out_crd())

        val1.set_load(inter0.out_ref1())
        val2.set_load(inter0.out_ref2())
        mul.set_in1(val1.out_load())
        mul.set_in2(val2.out_load())

        oval_wrscan.set_input(mul.out_val())

        crddrop.set_inner_crd(inter0.out_crd())
        crddrop.set_outer_crd(inter1.out_crd())

        wrscan0.set_input(crddrop.out_crd_inner())
        wrscan1.set_input(crddrop.out_crd_outer())

        rdscan1_1.update()
        rdscan2_1.update()
        inter1.update()
        rdscan1_0.update()
        rdscan2_0.update()
        inter0.update()
        val1.update()
        val2.update()
        mul.update()
        oval_wrscan.update()
        crddrop.update()
        wrscan0.update()
        wrscan1.update()
        print("Timestep", time2, "\t Done --",
              "\nRdScan1:", rdscan1_0.out_done(), rdscan2_0.out_done(), rdscan1_1.out_done(), rdscan2_1.out_done(),
              "\nInter:", inter0.out_done(), inter1.out_done(),
              "\nArr:", val1.out_done(), val2.out_done(),
              "\tMul:", mul.out_done(),
              "\nOutVal:", oval_wrscan.out_done(),
              "\nCrdDrop:", crddrop.out_done(),
              "\nOutWrScan1:", wrscan1.out_done(), "\tOutWrScan0:", wrscan0.out_done()
              )
        done = wrscan0.out_done() and wrscan1.out_done() and oval_wrscan.out_done()
        time2 += 1

    wrscan0.autosize()
    wrscan1.autosize()
    oval_wrscan.autosize()

    if debug_sim:
        print("TOTAL TIME:", time1 + time2)
        print(oval_wrscan.get_arr())
        print(wrscan0.get_arr())
        print(gold_crd0)
        print(wrscan1.get_arr())
        print(gold_crd1)

    check_arr(oval_wrscan, gold_vals)
    if gold_crd:
        check_arr(wrscan0, gold_crd0)
        check_arr(wrscan1, gold_crd1)

    def bench():
        time.sleep(0.0001)

    extra_info = dict()
    extra_info["cycles_reformat"] = time1
    extra_info["cycles"] = time2
    extra_info["vectype"] = vectype
    extra_info["sparsity"] = sparsity
    extra_info["run_length"] = run_length
    extra_info["block_size"] = run_length
    extra_info["format"] = "compressed"
    extra_info["split_factor"] = sf
    extra_info["test_name"] = "test_vec_elemmul_split"

    samBench(bench, extra_info)
