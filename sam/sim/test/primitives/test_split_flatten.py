import copy
import pytest

from sam.sim.src.split import Split
from sam.sim.src.flatten import Flatten
from sam.sim.src.base import remove_emptystr
from sam.sim.test.test import TIMEOUT

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
arrs_dict8 = {'crd_in': [0, 1, 2, 3, 4, 5, '', 6, 7, 'S0', 'D'],
              'ocrd_gold': [0, 1, 'S0', 'D'],
              'icrd_gold': [0, 1, 2, 3, 'S0', 4, 5, 6, 7, 'S1', 'D']
              }


@pytest.mark.parametrize("orig", [True, False])
@pytest.mark.parametrize("arrs", [(arrs_dict1, 4), (arrs_dict2, 4), (arrs_dict3, 2), (arrs_dict4, 2), (arrs_dict5, 4),
                                  (arrs_dict7, 3), (arrs_dict8, 4)])
def test_split_direct(arrs, orig, debug_sim):
    split_factor = arrs[1]
    arrs = arrs[0]

    crd = copy.deepcopy(arrs['crd_in'])

    gold_ocrd = copy.deepcopy(arrs['ocrd_gold'])
    gold_icrd = copy.deepcopy(arrs['icrd_gold'])

    if not orig:
        gold_icrd = [x % split_factor if isinstance(x, int) else x for x in gold_icrd]

    split = Split(split_factor=split_factor, orig_crd=orig, debug=debug_sim)

    done = False
    time = 0
    out_ocrd = []
    out_icrd = []
    while not done and time < TIMEOUT:
        if len(crd) > 0:
            split.set_in_crd(crd.pop(0))

        split.update()

        out_ocrd.append(split.out_outer_crd())
        out_icrd.append(split.out_inner_crd())

        print("Timestep", time, "\t Done:", split.out_done())

        done = split.out_done()
        time += 1

    out_ocrd = remove_emptystr(out_ocrd)
    out_icrd = remove_emptystr(out_icrd)

    if debug_sim:
        print("Outer Crd: ", out_ocrd)
        print("Inner Crd: ", out_icrd)

    assert (out_ocrd == gold_ocrd)
    assert (out_icrd == gold_icrd)


@pytest.mark.parametrize("arrs", [(arrs_dict1, 4), (arrs_dict2, 4), (arrs_dict3, 2), (arrs_dict4, 2), (arrs_dict5, 4),
                                  (arrs_dict7, 3)])
def test_flat_direct(arrs, debug_sim):
    split_factor = arrs[1]
    arrs = arrs[0]

    gold_crd = copy.deepcopy(arrs['crd_in'])

    ocrd = copy.deepcopy(arrs['ocrd_gold'])
    icrd = [x % split_factor if isinstance(x, int) else x for x in copy.deepcopy(arrs['icrd_gold'])]

    flat = Flatten(split_factor=split_factor, debug=debug_sim)

    done = False
    time = 0
    out_crd = []
    while not done and time < TIMEOUT:
        if len(ocrd) > 0:
            flat.set_in_outer_crd(ocrd.pop(0))

        if len(icrd) > 0:
            flat.set_in_inner_crd(icrd.pop(0))

        flat.update()

        out_crd.append(flat.out_crd())

        print("Timestep", time, "\t Done:", flat.out_done())

        done = flat.out_done()
        time += 1

    out_crd = remove_emptystr(out_crd)

    if debug_sim:
        print("Crd: ", out_crd)

    assert (out_crd == gold_crd)


@pytest.mark.parametrize("arrs", [(arrs_dict1, 4), (arrs_dict2, 4), (arrs_dict3, 2), (arrs_dict4, 2), (arrs_dict5, 4),
                                  (arrs_dict7, 3)])
def test_split_flat_direct(arrs, debug_sim):
    orig = False
    split_factor = arrs[1]
    arrs = arrs[0]

    crd = copy.deepcopy(arrs['crd_in'])

    gold_ocrd = copy.deepcopy(arrs['ocrd_gold'])
    gold_icrd = copy.deepcopy(arrs['icrd_gold'])

    if not orig:
        gold_icrd = [x % split_factor if isinstance(x, int) else x for x in gold_icrd]

    split = Split(split_factor=split_factor, orig_crd=orig)

    flat = Flatten(split_factor=split_factor, debug=debug_sim)

    done = False
    time = 0
    out_ocrd = []
    out_icrd = []
    out_crd = []
    while not done and time < TIMEOUT:
        if len(crd) > 0:
            split.set_in_crd(crd.pop(0))

        flat.set_in_outer_crd(split.out_outer_crd())
        flat.set_in_inner_crd(split.out_inner_crd())

        split.update()
        flat.update()

        out_ocrd.append(split.out_outer_crd())
        out_icrd.append(split.out_inner_crd())
        out_crd.append(flat.out_crd())

        print("Timestep", time, "\t Split Done:", split.out_done(), "\t Flat Done:", flat.out_done())

        done = split.out_done() and flat.out_done()
        time += 1

    if debug_sim:
        print(out_ocrd)
        print(out_icrd)

    out_ocrd = remove_emptystr(out_ocrd)
    out_icrd = remove_emptystr(out_icrd)
    out_crd = remove_emptystr(out_crd)

    if debug_sim:
        print("Outer Crd:", out_ocrd)
        print("Inner Crd:", out_icrd)
        print("Crd:", out_crd)

    assert (out_ocrd == gold_ocrd)
    assert (out_icrd == gold_icrd)
    assert (out_crd == arrs['crd_in'])
