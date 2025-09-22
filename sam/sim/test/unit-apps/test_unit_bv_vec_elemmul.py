import pytest
import random

from sam.sim.src.rd_scanner import BVRdScan, CompressedCrdRdScan
from sam.sim.src.bitvector import BV, BVDrop
from sam.sim.src.wr_scanner import ValsWrScan, CompressWrScan
from sam.sim.src.joiner import IntersectBV2
from sam.sim.src.compute import Multiply2
from sam.sim.src.array import Array
from sam.sim.src.split import Split
from sam.sim.src.base import remove_emptystr

from sam.sim.test.test import TIMEOUT, check_arr, check_seg_arr, remove_zeros


def bv(ll):
    result = 0
    for elem in ll:
        result |= 1 << elem
    return result


def inner_bv(ll, size, sf):
    result = []
    for i in range(int(size / sf) + 2):
        temp = bv([elem % sf for elem in ll if max((i - 1) * sf, 0) <= elem < i * sf])
        if temp:
            result.append(temp)
    return result


@pytest.mark.parametrize("nnz", [1, 10, 100, 500, 1000])
def test_vec_bv_split(nnz, debug_sim, backpressure, depth, max_val=999, size=1000, fill=0):
    sf = 32

    crd_arr1 = [random.randint(0, max_val) for _ in range(nnz)]
    crd_arr1 = sorted(set(crd_arr1))
    seg_arr1 = [0, len(crd_arr1)]
    vals_arr1 = [random.randint(0, max_val) for _ in range(len(crd_arr1))]

    crd_arr2 = [random.randint(0, max_val) for _ in range(nnz)]
    crd_arr2 = sorted(set(crd_arr2))
    seg_arr2 = [0, len(crd_arr2)]
    vals_arr2 = [random.randint(0, max_val) for _ in range(len(crd_arr2))]

    if debug_sim:
        print("Compressed VECTOR 1:\n", seg_arr1, "\n", crd_arr1, "\n", vals_arr1)
        print("Compressed VECTOR 2:\n", seg_arr2, "\n", crd_arr2, "\n", vals_arr2)

    gold_bv1_1 = [bv([int(elem / sf) for elem in crd_arr1])]
    gold_bv1_0 = inner_bv(crd_arr1, size, sf)
    gold_bv1_0 += (32 - len(gold_bv1_0)) * [0]

    gold_bv2_1 = [bv([int(elem / sf) for elem in crd_arr2])]
    gold_bv2_0 = inner_bv(crd_arr2, size, sf)
    gold_bv2_0 += (32 - len(gold_bv2_0)) * [0]

    crdscan1 = CompressedCrdRdScan(seg_arr=seg_arr1, crd_arr=crd_arr1, debug=debug_sim,
                                   back_en=backpressure, depth=int(depth))
    crdscan2 = CompressedCrdRdScan(seg_arr=seg_arr2, crd_arr=crd_arr2, debug=debug_sim,
                                   back_en=backpressure, depth=int(depth))
    split1 = Split(split_factor=sf, orig_crd=False, debug=debug_sim, back_en=backpressure, depth=int(depth))
    split2 = Split(split_factor=sf, orig_crd=False, debug=debug_sim, back_en=backpressure, depth=int(depth))

    bv1_0 = BV(debug=debug_sim, back_en=backpressure, depth=int(depth))
    bv1_1 = BV(debug=debug_sim, back_en=backpressure, depth=int(depth))
    bv2_0 = BV(debug=debug_sim, back_en=backpressure, depth=int(depth))
    bv2_1 = BV(debug=debug_sim, back_en=backpressure, depth=int(depth))

    wrscan1_0 = ValsWrScan(size=int(size / sf) + 1, fill=fill, back_en=backpressure, depth=int(depth))
    wrscan1_1 = ValsWrScan(size=1, fill=fill, back_en=backpressure, depth=int(depth))
    wrscan2_0 = ValsWrScan(size=int(size / sf) + 1, fill=fill, back_en=backpressure, depth=int(depth))
    wrscan2_1 = ValsWrScan(size=1, fill=fill, back_en=backpressure, depth=int(depth))

    in_ref1 = [0, 'D']
    in_ref2 = [0, 'D']
    done = False
    time = 0
    out_split1_0 = []
    out_split1_1 = []
    out_split2_0 = []
    out_split2_1 = []
    while not done and time < TIMEOUT:
        if len(in_ref1) > 0:
            crdscan1.set_in_ref(in_ref1.pop(0), "")
        if len(in_ref2) > 0:
            crdscan2.set_in_ref(in_ref2.pop(0), "")

        split1.set_in_crd(crdscan1.out_crd(), crdscan1)

        split2.set_in_crd(crdscan2.out_crd(), crdscan2)

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

        out_split1_0.append(split1.out_inner_crd())
        out_split1_1.append(split1.out_outer_crd())
        out_split2_0.append(split2.out_inner_crd())
        out_split2_1.append(split2.out_outer_crd())

        print("Timestep", time, "\t Done -- \n",
              "\nRdScan1:", crdscan1.out_done(), "\tRdScan2:", crdscan2.out_done(),
              "\nSplit1:", split1.out_done(), "\tSplit2:", split2.out_done(),
              "\nBV:", bv1_0.out_done(), bv1_1.out_done(), bv2_0.out_done(), bv2_1.out_done(),
              "\nWrScan:", wrscan1_0.out_done(), wrscan1_1.out_done(), wrscan2_0.out_done(), wrscan2_1.out_done()
              )

        done = wrscan2_0.out_done() and wrscan2_1.out_done() and wrscan1_1.out_done() and wrscan1_0.out_done()
        time += 1

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


# TODO: BV already set vecmul ONLY and then combined
@pytest.mark.parametrize("nnz", [1, 10, 100, 500, 1000])
def test_mat_elemmul_bvonly(nnz, debug_sim, backpressure, depth, max_val=1000, size=1001, fill=0):
    assert (size > max_val)
    sf = 32

    crd_arr1 = [random.randint(0, max_val) for _ in range(nnz)]
    crd_arr1 = sorted(set(crd_arr1))
    seg_arr1 = [0, len(crd_arr1)]
    vals_arr1 = [random.randint(0, max_val) for _ in range(len(crd_arr1))]

    crd_arr2 = [random.randint(0, max_val) for _ in range(nnz)]
    crd_arr2 = sorted(set(crd_arr2))
    seg_arr2 = [0, len(crd_arr2)]
    vals_arr2 = [random.randint(0, max_val) for _ in range(len(crd_arr2))]

    if debug_sim:
        print("Compressed VECTOR 1:\n", seg_arr1, "\n", crd_arr1, "\n", vals_arr1)
        print("Compressed VECTOR 2:\n", seg_arr2, "\n", crd_arr2, "\n", vals_arr2)

    gold_bv1_1 = [bv([int(elem / sf) for elem in crd_arr1])]
    gold_bv1_0 = inner_bv(crd_arr1, size, sf)
    gold_bv1_0 += (32 - len(gold_bv1_0)) * [0]

    gold_bv2_1 = [bv([int(elem / sf) for elem in crd_arr2])]
    gold_bv2_0 = inner_bv(crd_arr2, size, sf)
    gold_bv2_0 += (32 - len(gold_bv2_0)) * [0]

    gold_crd = sorted(set(crd_arr1) & set(crd_arr2))
    gold_seg = [0, len(gold_crd)]
    gold_vals = []

    gold_bv1 = []
    gold_bv0 = []
    if gold_crd:
        gold_vals = [vals_arr1[crd_arr1.index(i)] * vals_arr2[crd_arr2.index(i)] for i in gold_crd]
        gold_bv1 = [bv([int(elem / sf) for elem in gold_crd])]
        gold_bv0 = inner_bv(gold_crd, size, sf)

    if debug_sim:
        print("Compressed RESULT  :\n", gold_seg, "\n", gold_crd, "\n", gold_vals)
        print("BV arr1 0", gold_bv1_0)
        print("BV arr1 1", gold_bv1_1)
        print("BV arr2 0", gold_bv2_0)
        print("BV arr2 1", gold_bv2_1)

    bvscan1_0 = BVRdScan(bv_arr=gold_bv1_0, debug=debug_sim, back_en=backpressure, depth=int(depth))
    bvscan1_1 = BVRdScan(bv_arr=gold_bv1_1, debug=debug_sim, back_en=backpressure, depth=int(depth))
    bvscan2_0 = BVRdScan(bv_arr=gold_bv2_0, debug=debug_sim, back_en=backpressure, depth=int(depth))
    bvscan2_1 = BVRdScan(bv_arr=gold_bv2_1, debug=debug_sim, back_en=backpressure, depth=int(depth))

    inter0 = IntersectBV2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    inter1 = IntersectBV2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    val1 = Array(init_arr=vals_arr1, debug=debug_sim, back_en=backpressure, depth=int(depth))
    val2 = Array(init_arr=vals_arr2, debug=debug_sim, back_en=backpressure, depth=int(depth))
    mul = Multiply2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    bvdrop = BVDrop(debug=debug_sim, back_en=backpressure, depth=int(depth))
    oval_wrscan = ValsWrScan(size=size, fill=fill, back_en=backpressure, depth=int(depth))
    wrscan0 = ValsWrScan(size=size, fill=fill, back_en=backpressure, depth=int(depth))
    wrscan1 = ValsWrScan(size=1, fill=fill, back_en=backpressure, depth=int(depth))

    temp1 = []
    temp2 = []
    temp3 = []
    temp4 = []

    in_ref1 = [0, 'D']
    in_ref2 = [0, 'D']
    done = False
    time = 0
    while not done and time < TIMEOUT:
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

        temp3.append(inter0.out_bv())
        temp4.append(inter1.out_bv())
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

        temp1.append(inter0.out_ref1())
        print(remove_emptystr(temp1))
        temp2.append(inter0.out_ref2())
        print(remove_emptystr(temp2))

        print("Timestep", time, "\t Done --",
              "\nRdScan1:", bvscan1_0.out_done(), bvscan2_0.out_done(), bvscan1_1.out_done(), bvscan2_1.out_done(),
              "\nInter:", inter0.out_done(), inter1.out_done(),
              "\nArr:", val1.out_done(), val2.out_done(),
              "\tMul:", mul.out_done(),
              "\nOutVal:", oval_wrscan.out_done(),
              "\tOutBV1:", wrscan1.out_done(), "\tOutBV0:", wrscan0.out_done()
              )
        done = wrscan0.out_done() and wrscan1.out_done() and oval_wrscan.out_done()
        time += 1

    if debug_sim:
        print(oval_wrscan.get_arr())
        print(temp3)
        print(temp4)
        print(wrscan0.get_arr())
        print(gold_bv0)
        print(wrscan1.get_arr())
        print(gold_bv1)

    check_arr(oval_wrscan, gold_vals)
    if gold_crd:
        check_arr(wrscan0, gold_bv0)
        check_arr(wrscan1, gold_bv1)


# NOTE: This is the full vector elementwise multiplication as a bitvector
@pytest.mark.parametrize("sf", [16, 32, 64, 128])
@pytest.mark.parametrize("nnz", [1, 10, 100, 500, 1000])
def test_vec_elemmul_bv_split(nnz, sf, debug_sim, backpressure, depth, max_val=999, size=1000, fill=0):
    inner_fiber_cnt = int(size / sf) + 1

    crd_arr1 = [random.randint(0, max_val) for _ in range(nnz)]
    crd_arr1 = sorted(set(crd_arr1))
    seg_arr1 = [0, len(crd_arr1)]
    vals_arr1 = [random.randint(0, max_val) for _ in range(len(crd_arr1))]

    crd_arr2 = [random.randint(0, max_val) for _ in range(nnz)]
    crd_arr2 = sorted(set(crd_arr2))
    seg_arr2 = [0, len(crd_arr2)]
    vals_arr2 = [random.randint(0, max_val) for _ in range(len(crd_arr2))]

    if debug_sim:
        print("Compressed VECTOR 1:\n", seg_arr1, "\n", crd_arr1, "\n", vals_arr1)
        print("Compressed VECTOR 2:\n", seg_arr2, "\n", crd_arr2, "\n", vals_arr2)

    gold_bv1_1 = [bv([int(elem / sf) for elem in crd_arr1])]
    gold_bv1_0 = inner_bv(crd_arr1, size, sf)
    gold_bv1_0 += (inner_fiber_cnt - len(gold_bv1_0)) * [0]

    gold_bv2_1 = [bv([int(elem / sf) for elem in crd_arr2])]
    gold_bv2_0 = inner_bv(crd_arr2, size, sf)
    gold_bv2_0 += (inner_fiber_cnt - len(gold_bv2_0)) * [0]

    gold_crd = sorted(set(crd_arr1) & set(crd_arr2))
    gold_seg = [0, len(gold_crd)]
    gold_vals = []

    gold_bv1 = []
    gold_bv0 = []
    if gold_crd:
        gold_vals = [vals_arr1[crd_arr1.index(i)] * vals_arr2[crd_arr2.index(i)] for i in gold_crd]
        gold_bv1 = [bv([int(elem / sf) for elem in gold_crd])]
        gold_bv0 = inner_bv(gold_crd, size, sf)

    if debug_sim:
        print("Compressed RESULT  :\n", gold_seg, "\n", gold_crd, "\n", gold_vals)
        print("BV arr1 0", gold_bv1_0)
        print("BV arr1 1", gold_bv1_1)
        print("BV arr2 0", gold_bv2_0)
        print("BV arr2 1", gold_bv2_1)

    crdscan1 = CompressedCrdRdScan(seg_arr=seg_arr1, crd_arr=crd_arr1, debug=debug_sim, back_en=backpressure, depth=int(depth))
    crdscan2 = CompressedCrdRdScan(seg_arr=seg_arr2, crd_arr=crd_arr2, debug=debug_sim, back_en=backpressure, depth=int(depth))
    split1 = Split(split_factor=sf, orig_crd=False, debug=debug_sim, back_en=backpressure, depth=int(depth))
    split2 = Split(split_factor=sf, orig_crd=False, debug=debug_sim, back_en=backpressure, depth=int(depth))

    bv1_0 = BV(debug=debug_sim, back_en=backpressure, depth=int(depth))
    bv1_1 = BV(debug=debug_sim, back_en=backpressure, depth=int(depth))
    bv2_0 = BV(debug=debug_sim, back_en=backpressure, depth=int(depth))
    bv2_1 = BV(debug=debug_sim, back_en=backpressure, depth=int(depth))

    wrscan1_0 = ValsWrScan(size=inner_fiber_cnt, fill=fill, back_en=backpressure, depth=int(depth))
    wrscan1_1 = ValsWrScan(size=1, fill=fill, back_en=backpressure, depth=int(depth))
    wrscan2_0 = ValsWrScan(size=inner_fiber_cnt, fill=fill, back_en=backpressure, depth=int(depth))
    wrscan2_1 = ValsWrScan(size=1, fill=fill, back_en=backpressure, depth=int(depth))

    # MAKE BIT-TREE
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

        out_split1_0.append(split1.out_inner_crd())
        out_split1_1.append(split1.out_outer_crd())
        out_split2_0.append(split2.out_inner_crd())
        out_split2_1.append(split2.out_outer_crd())

        print("Timestep", time1, "\t Done -- \n",
              "\nRdScan1:", crdscan1.out_done(), "\tRdScan2:", crdscan2.out_done(),
              "\nSplit1:", split1.out_done(), "\tSplit2:", split2.out_done(),
              "\nBV:", bv1_0.out_done(), bv1_1.out_done(), bv2_0.out_done(), bv2_1.out_done(),
              "\nWrScan:", wrscan1_0.out_done(), wrscan1_1.out_done(), wrscan2_0.out_done(), wrscan2_1.out_done()
              )
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

    # COMPUTE ON BIT-TREE
    bvscan1_0 = BVRdScan(bv_arr=wrscan1_0.get_arr(), debug=debug_sim, back_en=backpressure, depth=int(depth))
    bvscan1_1 = BVRdScan(bv_arr=wrscan1_1.get_arr(), debug=debug_sim, back_en=backpressure, depth=int(depth))
    bvscan2_0 = BVRdScan(bv_arr=wrscan2_0.get_arr(), debug=debug_sim, back_en=backpressure, depth=int(depth))
    bvscan2_1 = BVRdScan(bv_arr=wrscan2_1.get_arr(), debug=debug_sim, back_en=backpressure, depth=int(depth))

    inter0 = IntersectBV2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    inter1 = IntersectBV2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    val1 = Array(init_arr=vals_arr1, debug=debug_sim, back_en=backpressure, depth=int(depth))
    val2 = Array(init_arr=vals_arr2, debug=debug_sim, back_en=backpressure, depth=int(depth))
    mul = Multiply2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    bvdrop = BVDrop(debug=debug_sim, back_en=backpressure, depth=int(depth))
    oval_wrscan = ValsWrScan(size=size, fill=fill, back_en=backpressure, depth=int(depth))
    wrscan0 = ValsWrScan(size=size, fill=fill, back_en=backpressure, depth=int(depth))
    wrscan1 = ValsWrScan(size=1, fill=fill, back_en=backpressure, depth=int(depth))

    in_ref1 = [0, 'D']
    in_ref2 = [0, 'D']
    done = False
    time2 = 0
    while not done and time1 < TIMEOUT:
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
