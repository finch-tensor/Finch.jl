import copy
import pytest

from sam.sim.src.split import Split
from sam.sim.src.bitvector import BV
from sam.sim.src.base import remove_emptystr
from sam.sim.test.test import TIMEOUT, get_bv


arrs_dict1 = {'crd_in': [0, 2, 3, 9, 11, 12, 'S0', 'D']}
arrs_dict2 = {'crd_in': [0, 2, 3, 'S0', 'D']}
arrs_dict3 = {'crd_in': [0, 2, 3, 'S0', 9, 11, 'S0', 12, 'S1', 'D']}

arrs_dict4 = {'crd_in': [0, 2, 3, 'S0', 'D']}
arrs_dict5 = {'crd_in': [0, 'S0', 'D']}
arrs_dict6 = {'crd_in': [0, 2, 3, 'S1', 'D']}

arrs_dict7 = {'crd_in': [0, 2, 3, 'S0', 1, 3, 'S0', 0, 'S1', 'D']}
arrs_dict8 = {'crd_in': [0, 1, 'S0', 0, 1, 'S0', 0, 'S1', 'D']}
arrs_dict9 = {'crd_in': [0, 'S0', 2, 3, 'S1', 1, 'S0', 3, 'S1', 0, 'S2', 'D']}

arrs_dict10 = {'crd_in': [0, 1, 2, 3, 'S0', 0, 1, 2, 3, 'S0', 0, 1, 2, 3, 'S1', 'D']}
arrs_dict11 = {'crd_in': [0, 1, 'S0', 0, 1, 'S0', 0, 1, 'S1', 'D']}
arrs_dict12 = {'crd_in': [0, 1, 'S0', 2, 3, 'S1', 0, 1, 'S0', 2, 3, 'S1', 0, 1, 'S0', 2, 3, 'S2', 'D']}

arrs_dict13 = {'crd_in': [0, 4, 8, 12, 16, 'S0', 'D']}
arrs_dict14 = {'crd_in': [0, 1, 2, 3, 4, 'S0', 'D']}
arrs_dict15 = {'crd_in': [0, 'S0', 4, 'S0', 8, 'S0', 12, 'S0', 16, 'S1', 'D']}

arrs_dict16 = {'crd_in': [0, 2, 3, 9, 11, 12, 'S0', 'D']}
arrs_dict17 = {'crd_in': [0, 1, 3, 4, 'S0', 'D']}
arrs_dict18 = {'crd_in': [0, 2, 'S0', 3, 'S0', 9, 11, 'S0', 12, 'S1', 'D']}

arrs_dict19 = {'crd_in': ['S0', 'D']}


@pytest.mark.parametrize("arrs", [arrs_dict1, arrs_dict2, arrs_dict3, arrs_dict4, arrs_dict5, arrs_dict6,
                                  arrs_dict7, arrs_dict8, arrs_dict9, arrs_dict10, arrs_dict11, arrs_dict12,
                                  arrs_dict13, arrs_dict14, arrs_dict15, arrs_dict16, arrs_dict17, arrs_dict18,
                                  arrs_dict19])
def test_bv_direct(arrs, debug_sim):
    crd = copy.deepcopy(arrs['crd_in'])
    gold_bv = get_bv(crd)
    if debug_sim:
        print("Gold:", gold_bv)

    bv = BV(debug=debug_sim)

    done = False
    time = 0
    out_bv = []
    while not done and time < TIMEOUT:
        if len(crd) > 0:
            bv.set_in_crd(crd.pop(0))

        bv.update()

        out_bv.append(bv.out_bv())

        print("Timestep", time, "\t Done:", bv.out_done())

        done = bv.out_done()
        time += 1

    out_bv = remove_emptystr(out_bv)

    if debug_sim:
        print("BV: ", out_bv)

    assert (out_bv == gold_bv)


arrs_dict1 = {'crd_in': [0, 2, 3, 9, 11, 12, 'S0', 'D'],
              'ocrd_gold': [0, 2, 3, 'S0', 'D'],
              'icrd_gold': [0, 2, 3, 'S0', 9, 11, 'S0', 12, 'S1', 'D']
              }

arrs_dict2 = {'crd_in': [0, 2, 3, 'S0', 'D'],
              'ocrd_gold': [0, 'S0', 'D'],
              'icrd_gold': [0, 2, 3, 'S1', 'D']
              }

arrs_dict3 = {'crd_in': [0, 2, 3, 'S0', 1, 3, 'S0', 0, 'S1', 'D'],
              'ocrd_gold': [0, 1, 'S0', 0, 1, 'S0', 0, 'S1', 'D'],
              'icrd_gold': [0, 'S0', 2, 3, 'S1', 1, 'S0', 3, 'S1', 0, 'S2', 'D']
              }

arrs_dict4 = {'crd_in': [0, 1, 2, 3, 'S0', 0, 1, 2, 3, 'S0', 0, 1, 2, 3, 'S1', 'D'],
              'ocrd_gold': [0, 1, 'S0', 0, 1, 'S0', 0, 1, 'S1', 'D'],
              'icrd_gold': [0, 1, 'S0', 2, 3, 'S1', 0, 1, 'S0', 2, 3, 'S1', 0, 1, 'S0', 2, 3, 'S2', 'D']
              }

arrs_dict5 = {'crd_in': [0, 4, 8, 12, 16, 'S0', 'D'],
              'ocrd_gold': [0, 1, 2, 3, 4, 'S0', 'D'],
              'icrd_gold': [0, 'S0', 4, 'S0', 8, 'S0', 12, 'S0', 16, 'S1', 'D']
              }

arrs_dict7 = {'crd_in': [0, 2, 3, 9, 11, 12, 'S0', 'D'],
              'ocrd_gold': [0, 1, 3, 4, 'S0', 'D'],
              'icrd_gold': [0, 2, 'S0', 3, 'S0', 9, 11, 'S0', 12, 'S1', 'D']
              }


@pytest.mark.parametrize("arrs", [(arrs_dict1, 4), (arrs_dict2, 4), (arrs_dict3, 2), (arrs_dict4, 2), (arrs_dict5, 4),
                                  (arrs_dict7, 3)])
def test_split_bv_intersect_direct(arrs, debug_sim):
    split_factor = arrs[1]
    arrs = arrs[0]

    crd = copy.deepcopy(arrs['crd_in'])

    gold_ocrd = copy.deepcopy(arrs['ocrd_gold'])
    gold_icrd = copy.deepcopy(arrs['icrd_gold'])
    gold_icrd = [x % split_factor if isinstance(x, int) else x for x in gold_icrd]

    gold_ibv = get_bv(gold_icrd)
    gold_obv = get_bv(gold_ocrd)

    if debug_sim:
        print("Gold Outer BV:", gold_ibv)
        print("Gold Inner BV:", gold_obv)

    split = Split(split_factor=split_factor, orig_crd=False, debug=debug_sim)

    obv = BV(debug=debug_sim)
    ibv = BV(debug=debug_sim)

    done = False
    time = 0
    out_ocrd = []
    out_icrd = []
    out_obv = []
    out_ibv = []
    while not done and time < TIMEOUT:
        if len(crd) > 0:
            split.set_in_crd(crd.pop(0))

        obv.set_in_crd(split.out_outer_crd())

        ibv.set_in_crd(split.out_inner_crd())

        split.update()
        obv.update()
        ibv.update()

        out_ocrd.append(split.out_outer_crd())
        out_icrd.append(split.out_inner_crd())

        out_obv.append(obv.out_bv())
        out_ibv.append(ibv.out_bv())

        print("Timestep", time, "\t Done:", split.out_done())

        done = split.out_done() and obv.out_done() and ibv.out_done()
        time += 1

    out_ocrd = remove_emptystr(out_ocrd)
    out_icrd = remove_emptystr(out_icrd)
    out_ibv = remove_emptystr(out_ibv)
    out_obv = remove_emptystr(out_obv)

    if debug_sim:
        print("Outer BV: ", out_obv)
        print("Inner BV: ", out_ibv)

    assert out_ocrd == gold_ocrd, "Outer crds after split do not match gold"
    assert out_icrd == gold_icrd, "Inner crds after split do not match gold"
    assert out_obv == gold_obv, "Outer crds after bv do not match gold"
    assert out_ibv == gold_ibv, "Inner crds after bv do not match gold"
