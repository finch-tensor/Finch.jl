import copy
import pytest

from sam.sim.src.accumulator import SparseAccumulator1, SparseAccumulator2
from sam.sim.src.base import remove_emptystr
from sam.sim.test.test import TIMEOUT

arrs_dict1 = {'ocrd_in': [0, 0, 0, 2, 2, 2, 2, 2, 2, 'D'],
              'icrd_in': [0, 2, 3, 0, 2, 3, 0, 2, 3, 'D'],
              'val_in': [50, 5, 10, 40, 4, 8, -40, 33, 36, 'D'],
              'ocrd_gold': [0, 2, 'S0', 'D'],
              'icrd_gold': [0, 2, 3, 'S0', 0, 2, 3, 'S1', 'D'],
              'val_gold': [50, 5, 10, 'S0', 0, 37, 44, 'S1', 'D']}


@pytest.mark.parametrize("arrs", [arrs_dict1])
def test_spacc1_direct(arrs, debug_sim):
    icrd = copy.deepcopy(arrs['icrd_in'])
    ocrd = copy.deepcopy(arrs['ocrd_in'])
    val = copy.deepcopy(arrs['val_in'])

    gold_ocrd = copy.deepcopy(arrs['ocrd_gold'])
    gold_icrd = copy.deepcopy(arrs['icrd_gold'])
    gold_val = copy.deepcopy(arrs['val_gold'])

    sa = SparseAccumulator1(val_stkn=True, debug=debug_sim)

    done = False
    time = 0
    out_ocrd = []
    out_icrd = []
    out_val = []
    while not done and time < TIMEOUT:
        if len(icrd) > 0:
            sa.set_inner_crdpt(icrd.pop(0))
        if len(ocrd) > 0:
            sa.set_outer_crdpt(ocrd.pop(0))
        if len(val) > 0:
            sa.set_val(val.pop(0))

        sa.update()

        out_ocrd.append(sa.out_outer_crd())
        out_icrd.append(sa.out_inner_crd())
        out_val.append(sa.out_val())

        print("Timestep", time, "\t Done:", sa.out_done())

        done = sa.out_done()
        time += 1

    out_ocrd = remove_emptystr(out_ocrd)
    out_icrd = remove_emptystr(out_icrd)
    out_val = remove_emptystr(out_val)

    if debug_sim:
        print("Outer Crd: ", out_ocrd)
        print("Inner Crd: ", out_icrd)
        print("Vals: ", out_val)

    assert (out_ocrd == gold_ocrd)
    assert (out_icrd == gold_icrd)
    assert (out_val == gold_val)


arrs_dict1 = {'crd1_in': [0, 2, 3, 0, 0, 2, 2, 2, 3, 3, 3, 3, 3, 0, 'D'],
              'crd0_in': [0, 2, 3, 0, 2, 0, 0, 3, 0, 2, 3, 0, 2, 0, 'D'],
              'val_in': [50, 5, 10, 40, 4, 8, -40, 33, 36, 1, 2, 3, 4, 5, 'D'],
              'crd1_gold': [0, 2, 3, 'S0', 'D'],
              'crd0_gold': [0, 2, 'S0', 0, 2, 3, 'S0', 0, 2, 3, 'S1', 'D'],
              'val_gold': [95.0, 4.0, 'S0', -32.0, 5.0, 33.0, 'S0', 39.0, 5.0, 12.0, 'S1', 'D']}

arrs_dict2 = {'crd1_in': [0, 0, 0, 1, 1, 1, 1, 0, 0, 'D'],
              'crd0_in': [0, 2, 3, 0, 2, 3, 4, 2, 3, 'D'],
              'val_in': [50, 5, 10, 40, 4, 8, -40, 33, 36, 'D'],
              'crd1_gold': [0, 1, 'S0', 'D'],
              'crd0_gold': [0, 2, 3, 'S0', 0, 2, 3, 4, 'S1', 'D'],
              'val_gold': [50, 38, 46, 'S0', 40, 4, 8, -40, 'S1', 'D']}


@pytest.mark.parametrize("arrs", [arrs_dict1, arrs_dict2])
def test_spacc2_direct(arrs, debug_sim):
    crd1 = copy.deepcopy(arrs['crd1_in'])
    crd0 = copy.deepcopy(arrs['crd0_in'])
    val = copy.deepcopy(arrs['val_in'])

    gold_crd1 = copy.deepcopy(arrs['crd1_gold'])
    gold_crd0 = copy.deepcopy(arrs['crd0_gold'])
    gold_val = copy.deepcopy(arrs['val_gold'])

    sa = SparseAccumulator2(val_stkn=True, debug=debug_sim)

    done = False
    time = 0
    out_crd1 = []
    out_crd0 = []
    out_val = []
    while not done and time < TIMEOUT:
        if len(crd0) > 0:
            sa.set_crd_inner(crd0.pop(0))
        if len(crd1) > 0:
            sa.set_crd_outer(crd1.pop(0))
        if len(val) > 0:
            sa.set_val(val.pop(0))

        sa.update()

        out_crd1.append(sa.out_crd_outer())
        out_crd0.append(sa.out_crd_inner())
        out_val.append(sa.out_val())

        print("Timestep", time, "\t Done:", sa.out_done())

        done = sa.out_done()
        time += 1

    out_crd1 = remove_emptystr(out_crd1)
    out_crd0 = remove_emptystr(out_crd0)
    out_val = remove_emptystr(out_val)

    if debug_sim:
        print("Crd1: ", out_crd1)
        print("Crd0: ", out_crd0)
        print("Vals: ", out_val)

    assert (out_crd1 == gold_crd1)
    assert (out_crd0 == gold_crd0)
    assert (out_val == gold_val)
