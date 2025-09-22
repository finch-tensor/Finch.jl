import copy
import pytest

from sam.sim.src.crd_manager import CrdHold
from sam.sim.src.base import remove_emptystr
from sam.sim.test.test import TIMEOUT

arrs_dict1 = {'ocrd_in': [0, 1, 2, 'S0', 'D'],
              'icrd_in': [0, 2, 'S0', 2, 'S0', 2, 'S1', 'D'],
              'gold': [0, 0, 'S0', 1, 'S0', 2, 'S1', 'D']}

arrs_dict2 = {'ocrd_in': [0, 1, 2, 5, 'S0', 'D'],
              'icrd_in': [1, 2, 5, 'S0', 2, 'S0', 2, 'S0', 2, 3, 4, 5, 'S1', 'D'],
              'gold': [0, 0, 0, 'S0', 1, 'S0', 2, 'S0', 5, 5, 5, 5, 'S1', 'D']}

arrs_dict3 = {'ocrd_in': [0, 2, 'S0', 3, 'S0', 4, 'S1', 'D'],
              'icrd_in': [0, 2, 3, 'S0', 0, 2, 3, 'S1', 0, 'S1', 2, 3, 'S2', 'D'],
              'gold': [0, 0, 0, 'S0', 2, 2, 2, 'S1', 3, 'S1', 4, 4, 'S2', 'D']}


@pytest.mark.parametrize("arrs", [arrs_dict1, arrs_dict2, arrs_dict3])
def test_crd_hold_nd(arrs, debug_sim, max_val=1000):
    icrd = copy.deepcopy(arrs['icrd_in'])
    ocrd = copy.deepcopy(arrs['ocrd_in'])

    gold = copy.deepcopy(arrs['gold'])

    ch = CrdHold(debug=debug_sim)

    done = False
    time = 0
    out = []
    while not done and time < TIMEOUT:
        if len(icrd) > 0:
            ch.set_inner_crd(icrd.pop(0))
        if len(ocrd) > 0:
            ch.set_outer_crd(ocrd.pop(0))

        ch.update()

        out.append(ch.out_crd_outer())

        print("Timestep", time, "\t Done:", ch.out_done(), "\t Out:", ch.out_crd_outer())

        done = ch.out_done()
        time += 1

    out = remove_emptystr(out)
    assert (out == gold)
