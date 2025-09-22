import pytest
import random

from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.wr_scanner import ValsWrScan, CompressWrScan
from sam.sim.src.joiner import Intersect2
from sam.sim.src.compute import Multiply2
from sam.sim.src.array import Array

from sam.sim.src.base import remove_emptystr
from sam.sim.test.test import TIMEOUT, check_arr, check_seg_arr


@pytest.mark.parametrize("dim1", [4, 16, 32, 64])
def test_unit_vec_elemmul_u_u_u(dim1, debug_sim, backpressure, depth, max_val=1000, size=100, fill=0):
    in_vec1 = [random.randint(0, max_val) for _ in range(dim1)]
    in_vec2 = [random.randint(0, max_val) for _ in range(dim1)]

    if debug_sim:
        print("VECTOR 1:", in_vec1)
        print("VECTOR 2:", in_vec2)

    assert (len(in_vec1) == len(in_vec2))

    gold_vec = [in_vec1[i] * in_vec2[i] for i in range(len(in_vec1))]

    rdscan = UncompressCrdRdScan(dim=dim1, debug=debug_sim, back_en=backpressure, depth=int(depth))
    val1 = Array(init_arr=in_vec1, debug=debug_sim, back_en=backpressure, depth=int(depth))
    val2 = Array(init_arr=in_vec2, debug=debug_sim, back_en=backpressure, depth=int(depth))
    mul = Multiply2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    wrscan = ValsWrScan(size=size, fill=fill, debug=debug_sim, back_en=backpressure, depth=int(depth))

    in_ref = [0, 'D']
    done = False
    time = 0
    while not done and time < TIMEOUT:
        if len(in_ref) > 0:
            rdscan.set_in_ref(in_ref.pop(0), "")
        val1.set_load(rdscan.out_ref(), rdscan)
        val2.set_load(rdscan.out_ref(), rdscan)
        mul.set_in1(val1.out_load(), val1)
        mul.set_in2(val2.out_load(), val2)
        wrscan.set_input(mul.out_val(), mul)

        rdscan.update()
        val1.update()
        val2.update()
        mul.update()
        wrscan.update()

        print("Timestep", time, "\t Done --", "\tRdScan:", rdscan.out_done(),
              "\tArr:", val1.out_done(), val2.out_done(),
              "\tMul:", mul.out_done(), "\tWrScan:", wrscan.out_done())

        done = wrscan.out_done()
        time += 1

    check_arr(wrscan, gold_vec)


@pytest.mark.parametrize("nnz", [1, 10, 100, 500, 1000])
def test_unit_vec_elemmul_u_c_c(nnz, debug_sim, backpressure, depth, max_val=1000, size=1001, fill=0):
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

    out_crd = sorted(set(crd_arr1) & set(crd_arr2))
    if out_crd:
        out_val = [vals_arr1[crd_arr1.index(i)] * vals_arr2[crd_arr2.index(i)] for i in out_crd]
        gold_vec = [out_val[out_crd.index(i)] if i in out_crd else 0 for i in range(max(out_crd) + 1)]
    else:
        out_val = ''
        gold_vec = [fill] * size

    if debug_sim:
        print("Compressed RESULT  :\n", out_crd, "\n", out_val, "\n", gold_vec)

    crdscan1 = CompressedCrdRdScan(seg_arr=seg_arr1, crd_arr=crd_arr1, debug=debug_sim, back_en=backpressure, depth=int(depth))
    crdscan2 = CompressedCrdRdScan(seg_arr=seg_arr2, crd_arr=crd_arr2, debug=debug_sim, back_en=backpressure, depth=int(depth))
    inter = Intersect2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    val1 = Array(init_arr=vals_arr1, debug=debug_sim, back_en=backpressure, depth=int(depth))
    val2 = Array(init_arr=vals_arr2, debug=debug_sim, back_en=backpressure, depth=int(depth))
    mul = Multiply2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    oval = Array(size=size, fill=fill, debug=debug_sim, back_en=backpressure, depth=int(depth))
    wrscan = ValsWrScan(size=size, fill=fill, debug=debug_sim, back_en=backpressure, depth=int(depth))

    in_ref1 = [0, 'D']
    in_ref2 = [0, 'D']
    done = False
    time = 0
    while not done and time < TIMEOUT:
        if len(in_ref1) > 0:
            crdscan1.set_in_ref(in_ref1.pop(0), "")
        if len(in_ref2) > 0:
            crdscan2.set_in_ref(in_ref2.pop(0), "")
        inter.set_in1(crdscan1.out_ref(), crdscan1.out_crd(), crdscan1)
        inter.set_in2(crdscan2.out_ref(), crdscan2.out_crd(), crdscan2)
        val1.set_load(inter.out_ref1(), inter)
        val2.set_load(inter.out_ref2(), inter)
        mul.set_in1(val1.out_load(), val1)
        mul.set_in2(val2.out_load(), val2)
        wrscan.set_input(mul.out_val(), mul)
        # oval.set_store(inter.out_crd(), mul.out_val())

        crdscan1.update()
        crdscan2.update()
        inter.update()
        val1.update()
        val2.update()
        mul.update()
        wrscan.update()

        # oval.update()

        print("Timestep", time, "\t Done --",
              "\tRdScan1:", crdscan1.out_done(), "\tRdScan2:", crdscan2.out_done(),
              "\tInter:", inter.out_done(),
              "\tArr:", val1.out_done(), val2.out_done(),
              "\tMul:", mul.out_done(), "\tOutVal:", oval.out_done())

        done = wrscan.out_done()
        time += 1

    # check_arr(oval, gold_vec)

    if out_val:
        check_arr(wrscan, out_val)
    else:
        check_arr(wrscan, gold_vec)


@pytest.mark.parametrize("nnz", [1, 10, 100, 500, 1000])
def test_unit_vec_elemmul_c_c_c(nnz, debug_sim, backpressure, depth, max_val=1000, size=1001, fill=0):
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

    crdscan1 = CompressedCrdRdScan(seg_arr=seg_arr1, crd_arr=crd_arr1, debug=debug_sim, back_en=backpressure, depth=int(depth))
    crdscan2 = CompressedCrdRdScan(seg_arr=seg_arr2, crd_arr=crd_arr2, debug=debug_sim, back_en=backpressure, depth=int(depth))
    inter = Intersect2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    val1 = Array(init_arr=vals_arr1, debug=debug_sim, back_en=backpressure, depth=int(depth))
    val2 = Array(init_arr=vals_arr2, debug=debug_sim, back_en=backpressure, depth=int(depth))
    mul = Multiply2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    oval_wrscan = ValsWrScan(size=size, fill=fill, back_en=backpressure, depth=int(depth))
    ocrd_wrscan = CompressWrScan(size=size, seg_size=size, fill=fill, back_en=backpressure, depth=int(depth))

    in_ref1 = [0, 'D']
    in_ref2 = [0, 'D']
    done = False
    time = 0
    while not done and time < TIMEOUT:
        if len(in_ref1) > 0:
            crdscan1.set_in_ref(in_ref1.pop(0), "")
        if len(in_ref2) > 0:
            crdscan2.set_in_ref(in_ref2.pop(0), "")
        inter.set_in1(crdscan1.out_ref(), crdscan1.out_crd(), crdscan1)
        inter.set_in2(crdscan2.out_ref(), crdscan2.out_crd(), crdscan2)
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

    check_arr(oval_wrscan, gold_vals)
    check_arr(ocrd_wrscan, gold_crd)
    check_seg_arr(ocrd_wrscan, gold_seg)


@pytest.mark.parametrize("nnz", [1, 10, 100, 500, 1000])
def test_unit_vec_elemmul_c_c_u(nnz, debug_sim, backpressure, depth, dim=1000, size=1000, fill=0):
    assert (size >= dim)
    crd_arr1 = [random.randint(0, dim - 1) for _ in range(nnz)]
    crd_arr1 = sorted(set(crd_arr1))
    seg_arr1 = [0, len(crd_arr1)]
    vals_arr1 = [random.randint(0, dim - 1) for _ in range(len(crd_arr1))]

    # arr2 val == crd
    vals_arr2 = [i for i in range(dim)]

    if debug_sim:
        print("Compressed VECTOR 1:\n", seg_arr1, "\n", crd_arr1, "\n", vals_arr1)

    gold_crd = sorted(set(crd_arr1))
    gold_seg = [0, len(gold_crd)]
    gold_vals = []
    if gold_crd:
        gold_vals = [vals_arr1[crd_arr1.index(i)] * i for i in gold_crd]

    if debug_sim:
        print("Compressed RESULT  :\n", gold_seg, "\n", gold_crd, "\n", gold_vals)

    crdscan1 = CompressedCrdRdScan(seg_arr=seg_arr1, crd_arr=crd_arr1, debug=debug_sim, back_en=backpressure, depth=int(depth))
    crdscan2 = UncompressCrdRdScan(dim=dim, debug=debug_sim, back_en=backpressure, depth=int(depth))
    inter = Intersect2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    val1 = Array(init_arr=vals_arr1, debug=debug_sim, back_en=backpressure, depth=int(depth))
    val2 = Array(init_arr=vals_arr2, debug=debug_sim, back_en=backpressure, depth=int(depth))
    mul = Multiply2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    oval_wrscan = ValsWrScan(size=size, fill=fill, back_en=backpressure, depth=int(depth))
    ocrd_wrscan = CompressWrScan(size=size, seg_size=size, fill=fill, back_en=backpressure, depth=int(depth))

    in_ref1 = [0, 'D']
    in_ref2 = [0, 'D']
    done = False
    time = 0
    while not done and time < TIMEOUT:
        if len(in_ref1) > 0:
            crdscan1.set_in_ref(in_ref1.pop(0), "")
        if len(in_ref2) > 0:
            crdscan2.set_in_ref(in_ref2.pop(0), "")
        inter.set_in1(crdscan1.out_ref(), crdscan1.out_crd(), crdscan1)
        inter.set_in2(crdscan2.out_ref(), crdscan2.out_crd(), crdscan2)
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

    check_arr(oval_wrscan, gold_vals)
    check_arr(ocrd_wrscan, gold_crd)
    check_seg_arr(ocrd_wrscan, gold_seg)
