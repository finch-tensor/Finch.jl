import pytest
import random

from sam.sim.src.rd_scanner import CompressedCrdRdScan
from sam.sim.src.wr_scanner import ValsWrScan, CompressWrScan
from sam.sim.src.joiner import Intersect2
from sam.sim.src.compute import Multiply2
from sam.sim.src.array import Array
from sam.sim.src.base import remove_emptystr

from sam.sim.test.test import TIMEOUT, check_arr, check_seg_arr

# arrs_dict1 = {
#     "seg1": [0, 10],
#     "crd1": [54, 71, 323, 537, 549, 558, 683, 787, 860, 929],
#     "seg2": [0, 10],
#     "crd2": [227, 280, 389, 486, 530, 606, 738, 744, 877, 996]
# }
#
#
# @pytest.mark.parametrize("arrs", [arrs_dict1])
# def test_vec_elemmul_direct_skip_c_c_c(arrs, debug_sim, max_val=1000, size=1001, fill=0):
#     assert(size > max_val)
#
#     crd_arr1 = [random.randint(0, max_val) for _ in range(nnz)]
#     crd_arr1 = sorted(set(crd_arr1))
#     seg_arr1 = [0, len(crd_arr1)]
#     vals_arr1 = [random.randint(0, max_val) for _ in range(len(crd_arr1))]
#
#     crd_arr2 = [random.randint(0, max_val) for _ in range(nnz)]
#     crd_arr2 = sorted(set(crd_arr2))
#     seg_arr2 = [0, len(crd_arr2)]
#     vals_arr2 = [random.randint(0, max_val) for _ in range(len(crd_arr2))]
#
#     if debug_sim:
#         print("Compressed VECTOR 1:\n", seg_arr1, "\n", crd_arr1, "\n", vals_arr1)
#         print("Compressed VECTOR 2:\n", seg_arr2, "\n", crd_arr2, "\n", vals_arr2)
#
#     gold_crd = sorted(set(crd_arr1) & set(crd_arr2))
#     gold_seg = [0, len(gold_crd)]
#     gold_vals = []
#     if gold_crd:
#         gold_vals = [vals_arr1[crd_arr1.index(i)] * vals_arr2[crd_arr2.index(i)] for i in gold_crd]
#
#     if debug_sim:
#         print("Compressed RESULT  :\n", gold_seg, "\n", gold_crd, "\n", gold_vals)
#
#     crdscan1 = CompressedCrdRdScan(seg_arr=seg_arr1, crd_arr=crd_arr1, debug=debug_sim, skip=True)
#     crdscan2 = CompressedCrdRdScan(seg_arr=seg_arr2, crd_arr=crd_arr2, debug=debug_sim, skip=True)
#     inter = Intersect2(debug=debug_sim, skip=True)
#
#     in_ref1 = [0, 'D']
#     in_ref2 = [0, 'D']
#     out_rdscan1 = []
#     out_rdscan2 = []
#     out_skip1 = []
#     out_skip2 = []
#     done = False
#     time = 0
#     while not done and time < TIMEOUT:
#         if len(in_ref1) > 0:
#             crdscan1.set_in_ref(in_ref1.pop(0))
#         crdscan1.update()
#         if len(in_ref2) > 0:
#             crdscan2.set_in_ref(in_ref2.pop(0))
#         crdscan2.update()
#
#         inter.set_in1(crdscan1.out_ref(), crdscan1.out_crd())
#         inter.set_in2(crdscan2.out_ref(), crdscan2.out_crd())
#         out_rdscan1.append(crdscan1.out_crd())
#         out_rdscan2.append(crdscan2.out_crd())
#         inter.update()
#
#         crdscan2.set_crd_skip(inter.out_crd_skip2())
#         crdscan1.set_crd_skip(inter.out_crd_skip1())
#
#         out_skip1.append(inter.out_crd_skip1())
#         out_skip2.append(inter.out_crd_skip2())
#
#         val1.set_load(inter.out_ref1())
#         val2.set_load(inter.out_ref2())
#         val1.update()
#         val2.update()
#         mul.set_in1(val1.out_load())
#         mul.set_in2(val2.out_load())
#         mul.update()
#         oval_wrscan.set_input(mul.out_val())
#         oval_wrscan.update()
#         ocrd_wrscan.set_input(inter.out_crd())
#         ocrd_wrscan.update()
#
#         print("Timestep", time, "\t Done --",
#               "\tRdScan1:", crdscan1.out_done(), "\tRdScan2:", crdscan2.out_done(),
#               "\tInter:", inter.out_done(),
#               "\tArr:", val1.out_done(), val2.out_done(),
#               "\tMul:", mul.out_done(),
#               "\tOutVal:", oval_wrscan.out_done(),
#               "\tOutCrd:", ocrd_wrscan.out_done())
#         done = ocrd_wrscan.out_done()
#         time += 1
#
#     if debug_sim:
#         print("Out Crd1:", out_rdscan1)
#         print("Out Crd2:", out_rdscan2)
#         print("Skip1:", out_skip1)
#         print("Skip2:", out_skip2)
#         print("Out Crd1:", remove_emptystr(out_rdscan1))
#         print("Out Crd2:", remove_emptystr(out_rdscan2))
#         print("Skip1:", remove_emptystr(out_skip1))
#         print("Skip2:", remove_emptystr(out_skip2))
#     check_arr(oval_wrscan, gold_vals)
#     check_arr(ocrd_wrscan, gold_crd)
#     check_seg_arr(ocrd_wrscan, gold_seg)


@pytest.mark.parametrize("nnz", [1, 10, 100, 500, 1000])
def test_vec_elemmul_skip_c_c_c(nnz, debug_sim, backpressure, depth, max_val=1000, size=1001, fill=0):
    assert (size > max_val)

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

    gold_crd = sorted(set(crd_arr1) & set(crd_arr2))
    gold_seg = [0, len(gold_crd)]
    gold_vals = []
    if gold_crd:
        gold_vals = [vals_arr1[crd_arr1.index(i)] * vals_arr2[crd_arr2.index(i)] for i in gold_crd]

    if debug_sim:
        print("Compressed RESULT  :\n", gold_seg, "\n", gold_crd, "\n", gold_vals)

    crdscan1 = CompressedCrdRdScan(seg_arr=seg_arr1, crd_arr=crd_arr1, debug=debug_sim, skip=True,
                                   back_en=backpressure, depth=int(depth))
    crdscan2 = CompressedCrdRdScan(seg_arr=seg_arr2, crd_arr=crd_arr2, debug=debug_sim, skip=True,
                                   back_en=backpressure, depth=int(depth))
    inter = Intersect2(debug=debug_sim, skip=True, back_en=backpressure, depth=int(depth))
    val1 = Array(init_arr=vals_arr1, debug=debug_sim, back_en=backpressure, depth=int(depth))
    val2 = Array(init_arr=vals_arr2, debug=debug_sim, back_en=backpressure, depth=int(depth))
    mul = Multiply2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    oval_wrscan = ValsWrScan(size=size, fill=fill, back_en=backpressure, depth=int(depth))
    ocrd_wrscan = CompressWrScan(size=size, seg_size=size, fill=fill, back_en=backpressure, depth=int(depth))

    in_ref1 = [0, 'D']
    in_ref2 = [0, 'D']
    out_rdscan1 = []
    out_rdscan2 = []
    out_skip1 = []
    out_skip2 = []
    done = False
    time = 0
    while not done and time < TIMEOUT:
        if len(in_ref1) > 0:
            crdscan1.set_in_ref(in_ref1.pop(0), "")
        if len(in_ref2) > 0:
            crdscan2.set_in_ref(in_ref2.pop(0), "")

        crdscan2.set_crd_skip(inter.out_crd_skip2(), inter)
        crdscan1.set_crd_skip(inter.out_crd_skip1(), inter)

        inter.set_in1(crdscan1.out_ref(), crdscan1.out_crd(), crdscan1)
        inter.set_in2(crdscan2.out_ref(), crdscan2.out_crd(), crdscan2)
        out_rdscan1.append(crdscan1.out_crd())
        out_rdscan2.append(crdscan2.out_crd())

        out_skip1.append(inter.out_crd_skip1())
        out_skip2.append(inter.out_crd_skip2())

        val1.set_load(inter.out_ref1(), inter)
        val2.set_load(inter.out_ref2(), inter)
        mul.set_in1(val1.out_load(), val1)
        mul.set_in2(val2.out_load(), val2)
        oval_wrscan.set_input(mul.out_val(), mul)
        ocrd_wrscan.set_input(inter.out_crd(), inter)

        crdscan1.update()
        crdscan2.update()
        inter.update()
        val1.update()
        val2.update()
        mul.update()
        oval_wrscan.update()
        ocrd_wrscan.update()

        print("Timestep", time, "\t Done --",
              "\tRdScan1:", crdscan1.out_done(), "\tRdScan2:", crdscan2.out_done(),
              "\tInter:", inter.out_done(),
              "\tArr:", val1.out_done(), val2.out_done(),
              "\tMul:", mul.out_done(),
              "\tOutVal:", oval_wrscan.out_done(),
              "\tOutCrd:", ocrd_wrscan.out_done())
        done = ocrd_wrscan.out_done()
        time += 1

    if debug_sim:
        print("Out Crd1:", out_rdscan1)
        print("Out Crd2:", out_rdscan2)
        print("Skip1:", out_skip1)
        print("Skip2:", out_skip2)
        print("Out Crd1:", remove_emptystr(out_rdscan1))
        print("Out Crd2:", remove_emptystr(out_rdscan2))
        print("Skip1:", remove_emptystr(out_skip1))
        print("Skip2:", remove_emptystr(out_skip2))
    check_arr(oval_wrscan, gold_vals)
    check_arr(ocrd_wrscan, gold_crd)
    check_seg_arr(ocrd_wrscan, gold_seg)
