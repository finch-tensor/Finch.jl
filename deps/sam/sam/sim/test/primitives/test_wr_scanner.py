import pytest

from sam.sim.src.wr_scanner import ValsWrScan

from sam.sim.test.test import *


@pytest.mark.parametrize("dim1", [4, 16, 32, 64])
def test_val_wr_scan_1d(dim1, debug_sim, max_val=1000, size=100, fill=0):
    in_val = [random.randint(0, max_val) for _ in range(dim1)] + ['S0', 'D']

    gold_val = in_val[:-2]

    wrscan = ValsWrScan(size=size, fill=fill, debug=debug_sim)

    done = False
    time = 0
    while not done and time < TIMEOUT:
        if len(in_val) > 0:
            wrscan.set_input(in_val.pop(0))

        wrscan.update()

        print("Timestep", time)

        done = wrscan.out_done()
        time += 1

    check_arr(wrscan, gold_val)


@pytest.mark.parametrize("nnz", [1, 10, 100, 500, 1000])
def test_comp_wr_scan_1d(nnz, debug_sim, max_val=1000, size=1001, fill=0):
    in_val = [random.randint(0, max_val) for _ in range(nnz)]
    in_val = sorted(set(in_val)) + ['S0', 'D']

    if debug_sim:
        print("Crd Stream:\n", in_val)

    gold_crd = in_val[:-2]
    gold_seg = [0, len(gold_crd)]

    if debug_sim:
        print("Gold Crd:\n", gold_crd)
        print("Gold Seg:\n", gold_seg)

    wrscan = CompressWrScan(size=size, seg_size=size, fill=fill, debug=debug_sim)

    done = False
    time = 0
    while not done and time < TIMEOUT:
        if len(in_val) > 0:
            wrscan.set_input(in_val.pop(0))

        wrscan.update()

        print("Timestep", time, "\t WrScan:", wrscan.out_done())

        done = wrscan.out_done()
        time += 1

    check_arr(wrscan, gold_crd)
    check_seg_arr(wrscan, gold_seg)


arrs_dict1 = {"in_crd": [0, 2, 3, 'S0', 0, 2, 3, 'S1', 'D'], "gold_crd": [0, 2, 3, 0, 2, 3],
              "gold_seg": [0, 3, 6]}


@pytest.mark.parametrize("arrs", [arrs_dict1])
def test_comp_wr_scan_direct(arrs, debug_sim, size=1001, fill=0):
    in_val = copy.deepcopy(arrs["in_crd"])

    if debug_sim:
        print("Crd Stream:\n", in_val)

    gold_crd = arrs["gold_crd"]
    gold_seg = arrs["gold_seg"]

    if debug_sim:
        print("Gold Crd:\n", gold_crd)
        print("Gold Seg:\n", gold_seg)

    wrscan = CompressWrScan(size=size, seg_size=size, fill=fill, debug=debug_sim)

    done = False
    time = 0
    while not done and time < TIMEOUT:
        if len(in_val) > 0:
            wrscan.set_input(in_val.pop(0))

        wrscan.update()

        print("Timestep", time, "\t WrScan:", wrscan.out_done())

        done = wrscan.out_done()
        time += 1

    check_arr(wrscan, gold_crd)
    check_seg_arr(wrscan, gold_seg)
