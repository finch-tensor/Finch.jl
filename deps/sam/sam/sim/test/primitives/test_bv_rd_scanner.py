import copy
import pytest

from sam.sim.src.rd_scanner import BVRdScan
from sam.sim.test.test import TIMEOUT

# TODO: figure out if bv_arr should be one long bit or a list of bits
arr_dict1 = {"bv": [0b1011], "dim": 4, "in_ref": [0, 'D'],
             "out_bv": [0b1011, 'S0', 'D'], "out_ref": [0, 'S0', 'D']}

arr_dict2 = {"bv": [0b1011, 0b0101, 0b1000], "dim": 4, "in_ref": [1, 2, 'S0', 'D'],
             "out_bv": [0b0101, 'S0', 0b1000, 'S1', 'D'],
             "out_ref": [3, 'S0', 5, 'S1', 'D']}

arr_dict3 = {"bv": [0b1011, 0b0110], "dim": 4, "in_ref": [0, 1, 'S0', 'D'],
             "out_bv": [0b1011, 'S0', 0b0110, 'S1', 'D'],
             "out_ref": [0, 'S0', 3, 'S1', 'D']}


@pytest.mark.parametrize("arrs", [arr_dict1, arr_dict2, arr_dict3])
def test_rd_scan_bv_direct_nd(arrs, debug_sim):
    bv_arr = arrs["bv"]
    dim = arrs["dim"]

    gold_bv = arrs["out_bv"]
    gold_ref = arrs["out_ref"]
    assert (len(gold_bv) == len(gold_ref))

    bvscan = BVRdScan(bv_arr=bv_arr, dim=dim)

    in_ref = copy.deepcopy(arrs["in_ref"])
    done = False
    time = 0
    out_bv = []
    out_ref = []
    while not done and time < TIMEOUT:
        if len(in_ref) > 0:
            bvscan.set_in_ref(in_ref.pop(0))

        bvscan.update()

        out_bv.append(bvscan.out_bv())
        out_ref.append(bvscan.out_ref())

        print("Timestep", time, "\t Crd:", bvscan.out_bv(), "\t Ref:", bvscan.out_ref())

        done = bvscan.done
        time += 1

    if debug_sim:
        print("Gold BV:", gold_bv)
        print("Out BV:", out_bv)
        print("Gold Ref:", gold_ref)
        print("Out Ref:", out_ref)

    assert (out_bv == gold_bv)
    assert (out_ref == gold_ref)
