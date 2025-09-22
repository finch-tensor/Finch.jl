import pytest
import copy

from sam.sim.src.base import remove_emptystr
from sam.sim.src.joiner import Union2
from sam.sim.src.rd_scanner import CompressedCrdRdScan
from sam.sim.test.test import TIMEOUT


arrs_dict1 = {'crd1_in': [0, 1, 3, 5, 'S0', 'D'],
              'crd2_in': [0, 2, 3, 4, 'S0', 'D'],
              'ref1_in': [0, 1, 2, 3, 'S0', 'D'],
              'ref2_in': [0, 1, 2, 3, 'S0', 'D'],
              'crd_gold': [0, 1, 2, 3, 4, 5, 'S0', 'D'],
              'ref1_gold': [0, 1, 'N', 2, 'N', 3, 'S0', 'D'],
              'ref2_gold': [0, 'N', 1, 2, 3, 'N', 'S0', 'D']}

arrs_dict2 = {'crd1_in': [0, 1, 'S0', 2, 3, 'S0', 'S0', 4, 5, 'S1', 'D'],
              'crd2_in': [1, 2, 3, 'S0', 'S0', 0, 1, 2, 'S0', 'S1', 'D'],
              'ref1_in': [0, 1, 'S0', 2, 3, 'S0', 'S0', 4, 5, 'S1', 'D'],
              'ref2_in': [0, 1, 2, 'S0', 'S0', 2, 3, 4, 'S0', 'S1', 'D'],
              'crd_gold': [0, 1, 2, 3, 'S0', 2, 3, 'S0', 0, 1, 2, 'S0', 4, 5, 'S1', 'D'],
              'ref1_gold': [0, 1, 'N', 'N', 'S0', 2, 3, 'S0', 'N', 'N', 'N', 'S0', 4, 5, 'S1', 'D'],
              'ref2_gold': ['N', 0, 1, 2, 'S0', 'N', 'N', 'S0', 2, 3, 4, 'S0', 'N', 'N', 'S1', 'D']}


@pytest.mark.parametrize("arrs", [arrs_dict1, arrs_dict2])
def test_union_direct_nd(arrs, debug_sim):
    crd1 = copy.deepcopy(arrs['crd1_in'])
    ref1 = copy.deepcopy(arrs['ref1_in'])
    crd2 = copy.deepcopy(arrs['crd2_in'])
    ref2 = copy.deepcopy(arrs['ref2_in'])

    crd_gold = copy.deepcopy(arrs['crd_gold'])
    ref1_gold = copy.deepcopy(arrs['ref1_gold'])
    ref2_gold = copy.deepcopy(arrs['ref2_gold'])

    if debug_sim:
        print("Gold Crd:", crd_gold)
        print('Gold Ref1:', ref1_gold)
        print('Gold Ref2:', ref2_gold)

    union = Union2(debug=debug_sim)

    done = False
    time = 0
    out_crd = []
    out_ref1 = []
    out_ref2 = []
    while not done and time < TIMEOUT:
        if len(crd1) > 0:
            union.set_in1(ref1.pop(0), crd1.pop(0))
        if len(crd2) > 0:
            union.set_in2(ref2.pop(0), crd2.pop(0))

        union.update()

        out_crd.append(union.out_crd())
        out_ref1.append(union.out_ref1())
        out_ref2.append(union.out_ref2())

        print("Timestep", time, "\t Crd:", union.out_crd(), "\t Ref1:", union.out_ref1(), "\t Ref2:", union.out_ref2())

        done = union.done
        time += 1

    out_crd = remove_emptystr(out_crd)
    out_ref1 = remove_emptystr(out_ref1)
    out_ref2 = remove_emptystr(out_ref2)

    assert (out_crd == crd_gold)
    assert (out_ref1 == ref1_gold)
    assert (out_ref2 == ref2_gold)
