import copy
import pytest

from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.test.test import TIMEOUT
from sam.sim.src.base import remove_emptystr


########################################
# Uncompressed Read Scanner Unit Tests #
########################################

@pytest.mark.parametrize("dim", [1, 2, 4, 16, 32, 100])
def test_rd_scan_u_direct_1d(dim, debug_sim):
    gold_crd = [x for x in range(dim)]
    gold_crd.append('S0')
    gold_crd.append('D')
    gold_ref = gold_crd
    assert (len(gold_crd) == len(gold_ref))

    urs = UncompressCrdRdScan(dim=dim, debug=debug_sim)

    in_ref = [0, 'D']
    done = False
    time = 0
    out_crd = []
    out_ref = []
    while not done and time < TIMEOUT:
        if len(in_ref) > 0:
            urs.set_in_ref(in_ref.pop(0))

        urs.update()

        out_crd.append(urs.out_crd())
        out_ref.append(urs.out_ref())

        print("Timestep", time, "\t Crd:", urs.out_crd(), "\t Ref:", urs.out_ref())

        done = urs.done
        time += 1

    assert (out_crd == gold_crd)
    assert (out_ref == gold_ref)


@pytest.mark.parametrize("dim", [1, 2, 4, 16, 32, 100])
def test_rd_scan_u_direct_2d(dim, debug_sim):
    cnt = [x for x in range(dim)]
    gold_crd = (cnt + ['S0']) * 3 + cnt + ['S1', 'D']
    gold_ref = cnt + ['S0'] + [dim + x for x in cnt] + ['S0'] + [2 * dim + x for x in cnt] + ['S0'] + \
        [3 * dim + x for x in cnt] + ['S1', 'D']
    assert (len(gold_crd) == len(gold_ref))

    urs = UncompressCrdRdScan(dim=dim, debug=debug_sim)

    in_ref = [0, 1, 2, 3, 'S0', 'D']
    done = False
    time = 0
    out_crd = []
    out_ref = []
    while not done and time < TIMEOUT:
        if len(in_ref) > 0:
            urs.set_in_ref(in_ref.pop(0))

        urs.update()

        out_crd.append(urs.out_crd())
        out_ref.append(urs.out_ref())

        print("Timestep", time, "\t Crd:", urs.out_crd(), "\t Ref:", urs.out_ref())

        done = urs.done
        time += 1

    assert (out_crd == gold_crd)
    assert (out_ref == gold_ref)


@pytest.mark.parametrize("dim", [1, 2, 4, 16, 32, 100])
def test_rd_scan_u_direct_3d(dim, debug_sim):
    cnt = [x for x in range(dim)]
    gold_crd = (cnt + ['S0']) * 2 + cnt + ['S1'] + (cnt + ['S0']) + cnt + ['S1'] + (cnt + ['S0']) + cnt + ['S2', 'D']
    gold_ref = cnt + ['S0'] + [dim + x for x in cnt] + ['S0'] + [2 * dim + x for x in cnt] + ['S1'] + \
        [3 * dim + x for x in cnt] + ['S0'] + [4 * dim + x for x in cnt] + ['S1'] + \
        [5 * dim + x for x in cnt] + ['S0'] + \
        [6 * dim + x for x in cnt] + ['S2', 'D']
    assert (len(gold_crd) == len(gold_ref))

    urs = UncompressCrdRdScan(dim=dim, debug=debug_sim)

    in_ref = [0, 1, 2, 'S0', 3, 4, 'S0', 5, 6, 'S1', 'D']
    done = False
    time = 0
    out_crd = []
    out_ref = []
    while not done and time < TIMEOUT:
        if len(in_ref) > 0:
            urs.set_in_ref(in_ref.pop(0))

        urs.update()

        out_crd.append(urs.out_crd())
        out_ref.append(urs.out_ref())

        print("Timestep", time, "\t Crd:", urs.out_crd(), "\t Ref:", urs.out_ref())

        done = urs.done
        time += 1

    assert (out_crd == gold_crd)
    assert (out_ref == gold_ref)


######################################
# Compressed Read Scanner Unit Tests #
######################################

def test_rd_scan_c_direct_1d(debug_sim):
    seg_arr = [0, 3]
    crd_arr = [0, 1, 3]

    gold_crd = crd_arr + ['S0', 'D']
    gold_ref = [x for x in range(len(crd_arr))] + ['S0', 'D']

    assert (len(gold_crd) == len(gold_ref))

    crdscan = CompressedCrdRdScan(seg_arr=seg_arr, crd_arr=crd_arr, debug=debug_sim)

    in_ref = [0, 'D']
    done = False
    time = 0
    out_crd = []
    out_ref = []
    while not done and time < TIMEOUT:
        if len(in_ref) > 0:
            crdscan.set_in_ref(in_ref.pop(0))

        crdscan.update()

        out_crd.append(crdscan.out_crd())
        out_ref.append(crdscan.out_ref())

        print("Timestep", time, "\t Crd:", crdscan.out_crd(), "\t Ref:", crdscan.out_ref())

        done = crdscan.done
        time += 1

    assert (out_crd == gold_crd)
    assert (out_ref == gold_ref)


arr_dict1 = {"seg": [0, 2, 3, 4], "crd": [0, 2, 2, 2], "in_ref": [0, 1, 2, 'S0', 'D'],
             "out_crd": [0, 2, 'S0', 2, 'S0', 2, 'S1', 'D'], "out_ref": [0, 1, 'S0', 2, 'S0', 3, 'S1', 'D']}
arr_dict2 = {"seg": [0, 3, 4, 6], "crd": [0, 2, 3, 0, 2, 3], "in_ref": [0, 0, 'S0', 1, 'S0', 2, 'S1', 'D'],
             "out_crd": [0, 2, 3, 'S0', 0, 2, 3, 'S1', 0, 'S1', 2, 3, 'S2', 'D'],
             "out_ref": [0, 1, 2, 'S0', 0, 1, 2, 'S1', 3, 'S1', 4, 5, 'S2', 'D']}
arr_dict3 = {"seg": [0, 4], "crd": [0, 1, 2, 3], "in_ref": [0, 'D'],
             "out_crd": [0, 1, 2, 3, 'S0', 'D'], "out_ref": [0, 1, 2, 3, 'S0', 'D']}
arr_dict4 = {"seg": [0, 1], "crd": [28], "in_ref": [0, 'S0', 'S0', 0, 'S0', 'D'],
             "out_crd": [28, 'S1', 'S1', 28, 'S1', 'D'], "out_ref": [0, 'S1', 'S1', 0, 'S1', 'D']}
arr_dict5 = {"seg": [0, 1], "crd": [28], "in_ref": [0, 'S0', '', '', 'S0', '',
                                                    '', 0, '', '', 'S0', '', '', 'D'],
             "out_crd": [28, 'S1', 'S1', 28, 'S1', 'D'], "out_ref": [0, 'S1', 'S1', 0, 'S1', 'D']}


@pytest.mark.parametrize("arrs", [arr_dict1, arr_dict2, arr_dict3, arr_dict4, arr_dict5])
def test_rd_scan_c_direct_nd(arrs, debug_sim):
    seg_arr = arrs["seg"]
    crd_arr = arrs["crd"]

    gold_crd = arrs["out_crd"]
    gold_ref = arrs["out_ref"]
    assert (len(gold_crd) == len(gold_ref))

    crdscan = CompressedCrdRdScan(seg_arr=seg_arr, crd_arr=crd_arr, debug=debug_sim)

    in_ref = copy.deepcopy(arrs["in_ref"])
    done = False
    time = 0
    out_crd = []
    out_ref = []
    while not done and time < TIMEOUT:
        if len(in_ref) > 0:
            crdscan.set_in_ref(in_ref.pop(0))

        crdscan.update()

        out_crd.append(crdscan.out_crd())
        out_ref.append(crdscan.out_ref())

        print("Timestep", time, "\t Crd:", crdscan.out_crd(), "\t Ref:", crdscan.out_ref())

        done = crdscan.done
        time += 1

    assert (remove_emptystr(out_crd) == gold_crd)
    assert (remove_emptystr(out_ref) == gold_ref)


arr_dict1 = {"seg": [0, 2, 3, 4], "crd": [0, 1, 2, 3], "in_ref": [0, 1, 'N', 2, 'N', 'S0', 'D'],
             "out_crd": [0, 1, 'S0', 2, 'S0', 'N', 'S0', 3, 'S0', 'N', 'S1', 'D'],
             "out_ref": [0, 1, 'S0', 2, 'S0', 'N', 'S0', 3, 'S0', 'N', 'S1', 'D']}
arr_dict2 = {"seg": [0, 3, 4, 6], "crd": [0, 2, 3, 0, 2, 3], "in_ref": [0, 'N', 1, 'N', 2, 'S0', 'D'],
             "out_crd": [0, 2, 3, 'S0', 'N', 'S0', 0, 'S0', 'N', 'S0', 2, 3, 'S1', 'D'],
             "out_ref": [0, 1, 2, 'S0', 'N', 'S0', 3, 'S0', 'N', 'S0', 4, 5, 'S1', 'D']}
arr_dict3 = {"seg": [0, 4, 5, 5, 7, 10, 11], "crd": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             "in_ref": [0, 1, 'N', 'N', 'S0', 2, 3, 'S0', 'N', 'N', 'S0', 4, 5, 'S0', 'N', 'N', 'S1', 'D'],
             "out_crd": [0, 1, 2, 3, 'S0', 4, 'S0', 'N', 'S0', 'N', 'S1', 'S0', 5, 6, 'S1', 'N', 'S0', 'N', 'S1',
                         7, 8, 9, 'S0', 10, 'S1', 'N', 'S0', 'N', 'S2', 'D'],
             "out_ref": [0, 1, 2, 3, 'S0', 4, 'S0', 'N', 'S0', 'N', 'S1', 'S0', 5, 6, 'S1', 'N', 'S0', 'N', 'S1',
                         7, 8, 9, 'S0', 10, 'S1', 'N', 'S0', 'N', 'S2', 'D']}


@pytest.mark.parametrize("arrs", [arr_dict1, arr_dict2, arr_dict3])
def test_rd_scan_direct_c_0tkn_nd(arrs, debug_sim):
    seg_arr = arrs["seg"]
    crd_arr = arrs["crd"]

    gold_crd = arrs["out_crd"]
    gold_ref = arrs["out_ref"]
    assert (len(gold_crd) == len(gold_ref))

    crdscan = CompressedCrdRdScan(seg_arr=seg_arr, crd_arr=crd_arr, debug=debug_sim)

    in_ref = copy.deepcopy(arrs["in_ref"])
    done = False
    time = 0
    out_crd = []
    out_ref = []
    while not done and time < TIMEOUT:
        if len(in_ref) > 0:
            crdscan.set_in_ref(in_ref.pop(0))

        crdscan.update()

        out_crd.append(crdscan.out_crd())
        out_ref.append(crdscan.out_ref())

        print("Timestep", time, "\t Crd:", crdscan.out_crd(), "\t Ref:", crdscan.out_ref())

        done = crdscan.done
        time += 1

    assert (out_crd == gold_crd)
    assert (out_ref == gold_ref)


arr_dict1 = {"seg": [0, 10], "crd": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "in_ref": [0, 'D'],
             "in_skip": [2, 7, 'S0'],
             "out_crd": [2, 7, 'S0', 'D'],
             "out_ref": [2, 7, 'S0', 'D']}
arr_dict2 = {"seg": [0, 10], "crd": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "in_ref": [0, 'D'],
             "in_skip": ['', '', '', 0, 7, 'S0'],
             "out_crd": [0, 1, 2, 3, 7, 'S0', 'D'],
             "out_ref": [0, 1, 2, 3, 7, 'S0', 'D']}
arr_dict3 = {"seg": [0, 5, 10], "crd": [0, 1, 2, 3, 4, 0, 1, 2, 3, 4], "in_ref": [0, 1, 'S0', 'D'],
             "in_skip": [2, 'S0', 3, 'S1'],
             "out_crd": [2, 'S0', 3, 'S1', 'D'],
             "out_ref": [2, 'S0', 8, 'S1', 'D']}
arr_dict4 = {"seg": [0, 5, 10], "crd": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "in_ref": [0, 1, 'S0', 'D'],
             "in_skip": ['', '', '', '', '', 2, 'S0', 9, 'S1'],
             "out_crd": [0, 1, 2, 3, 4, 'S0', 5, 9, 'S1', 'D'],
             "out_ref": [0, 1, 2, 3, 4, 'S0', 5, 9, 'S1', 'D']}

arr_dict5 = {"seg": [0, 10], "crd": [54, 71, 323, 537, 549, 558, 683, 787, 860, 929], "in_ref": [0, 'D'],
             "in_skip": [227, 389, 606, 738, 877, 996, 'S0'],
             "out_crd": [323, 537, 683, 787, 929, 'S0', 'D'],
             "out_ref": [2, 3, 6, 7, 9, 'S0', 'D']}
arr_dict6 = {"seg": [0, 10], "crd": [227, 280, 389, 486, 530, 606, 738, 744, 877, 996], "in_ref": [0, 'D'],
             "in_skip": [323, 537, 683, 787, 929, 'S0'],
             "out_crd": [389, 606, 738, 877, 996, 'S0', 'D'],
             "out_ref": [2, 5, 6, 8, 9, 'S0', 'D']}
arr_dict7 = {"seg": [0, 10], "crd": [29, 40, 178, 273, 637, 721, 744, 763, 855, 975], "in_ref": [0, 'D'],
             "in_skip": ['', 112, '', '', 238, '', 338, '', '', '', '', '', '', 852, '', '', '', '', 996, '', '', 'S0'],
             "out_crd": [29, 178, 273, 637, 721, 744, 763, 855, 975, 'S0', 'D'],
             "out_ref": [0, 2, 3, 4, 5, 6, 7, 8, 9, 'S0', 'D']}


@pytest.mark.parametrize("arrs", [arr_dict1, arr_dict2, arr_dict3, arr_dict4, arr_dict5, arr_dict6, arr_dict7])
def test_rd_scan_direct_c_skip_nd(arrs, debug_sim):
    seg_arr = arrs["seg"]
    crd_arr = arrs["crd"]

    gold_crd = arrs["out_crd"]
    gold_ref = arrs["out_ref"]
    assert (len(gold_crd) == len(gold_ref))

    crdscan = CompressedCrdRdScan(seg_arr=seg_arr, crd_arr=crd_arr, debug=debug_sim)

    in_ref = copy.deepcopy(arrs["in_ref"])
    in_skip = copy.deepcopy(arrs["in_skip"])
    done = False
    time = 0
    out_crd = []
    out_ref = []
    while not done and time < TIMEOUT:
        if len(in_ref) > 0:
            crdscan.set_in_ref(in_ref.pop(0))
        if len(in_skip) > 0:
            crdscan.set_crd_skip(in_skip.pop(0))

        crdscan.update()

        out_crd.append(crdscan.out_crd())
        out_ref.append(crdscan.out_ref())

        print("Timestep", time, "\t Crd:", crdscan.out_crd(), "\t Ref:", crdscan.out_ref())

        done = crdscan.done
        time += 1

    print(out_crd)
    print(out_ref)
    out_crd = remove_emptystr(out_crd)
    out_ref = remove_emptystr(out_ref)
    print(out_crd)
    print(out_ref)
    assert (out_crd == gold_crd)
    assert (out_ref == gold_ref)
