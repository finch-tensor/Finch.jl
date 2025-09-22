import copy
import pytest

from functools import reduce

# from sam.sim.src.bitvector import BV
from sam.sim.src.joiner import IntersectBV2
from sam.sim.src.base import remove_emptystr
from sam.sim.test.test import TIMEOUT


def get_bv(crd):
    gold_bv = []
    temp = []
    for x in crd:
        if isinstance(x, int):
            temp.append(x)
        else:
            if temp:
                gold_bv.append(bin(reduce(lambda a, b: a | b, [0b1 << i for i in temp])))
                temp = []
            gold_bv.append(x)
    return gold_bv


arrs_dict1 = {'bv1_in': [0b1101, 'S0', 'D'],
              'bv2_in': [0b1100, 'S0', 'D'],
              'ref1_in': [1, 'S0', 'D'],
              'ref2_in': [3, 'S0', 'D'],
              'bv_gold': [0b1100, 'S0', 'D'],
              'ref1_gold': [2, 3, 'S0', 'D'],
              'ref2_gold': [3, 4, 'S0', 'D']
              }
arrs_dict2 = {'bv1_in': [0b1010, 'S0', 0b0001, 'S1', 'D'],
              'bv2_in': [0b1101, 'S0', 0b0110, 'S1', 'D'],
              'ref1_in': [3, 'S0', 5, 'S1', 'D'],
              'ref2_in': [0, 'S0', 3, 'S1', 'D'],
              'bv_gold': [0b1000, 'S0', 'S1', 'D'],  # TODO: see if the stop token needs to be removed
              'ref1_gold': [4, 'S0', 'S1', 'D'],
              'ref2_gold': [2, 'S0', 'S1', 'D']
              }


# , arrs_dict3, arrs_dict4, arrs_dict5, arrs_dict6,
#                                   arrs_dict7, arrs_dict8, arrs_dict9, arrs_dict10, arrs_dict11, arrs_dict12,
#                                   arrs_dict13, arrs_dict14, arrs_dict15, arrs_dict16, arrs_dict17, arrs_dict18
@pytest.mark.parametrize("arrs", [arrs_dict1, arrs_dict2])
def test_intersectbv_direct(arrs, debug_sim):
    bv1 = copy.deepcopy(arrs['bv1_in'])
    bv2 = copy.deepcopy(arrs['bv2_in'])
    ref1 = copy.deepcopy(arrs['ref1_in'])
    ref2 = copy.deepcopy(arrs['ref2_in'])

    gold_bv = copy.deepcopy(arrs['bv_gold'])
    gold_ref1 = copy.deepcopy(arrs['ref1_gold'])
    gold_ref2 = copy.deepcopy(arrs['ref2_gold'])

    intersect_bv = IntersectBV2(debug=debug_sim)

    done = False
    time = 0
    out_bv = []
    out_ref1 = []
    out_ref2 = []
    while not done and time < TIMEOUT:
        if len(bv1) > 0 and len(ref1) > 0:
            intersect_bv.set_in1(ref1.pop(0), bv1.pop(0))

        if len(bv2) > 0 and len(ref2) > 0:
            intersect_bv.set_in2(ref2.pop(0), bv2.pop(0))

        intersect_bv.update()

        out_bv.append(intersect_bv.out_bv())
        out_ref1.append(intersect_bv.out_ref1())
        out_ref2.append(intersect_bv.out_ref2())

        print("Timestep", time, "\t Done:", intersect_bv.out_done())

        done = intersect_bv.out_done()
        time += 1

    out_bv = remove_emptystr(out_bv)
    out_ref1 = remove_emptystr(out_ref1)
    out_ref2 = remove_emptystr(out_ref2)

    if debug_sim:
        print("BV: ", out_bv)
        print("Ref1:", out_ref1)
        print("Ref2:", out_ref2)

    assert (out_bv == gold_bv)
    assert (out_ref1 == gold_ref1)
    assert (out_ref2 == gold_ref2)
