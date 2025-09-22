import copy
import pytest

from sam.sim.src.crd_manager import CrdPtConverter
from sam.sim.src.base import remove_emptystr, increment_stkn
from sam.sim.test.test import TIMEOUT

arrs_dict1 = {'ocrd_in': [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 'D'],
              'icrd_in': [0, 0, 1, 0, 1, 2, 3, 1, 1, 1, 'D'],
              'ocrd_gold': [0, 1, 2, 'S0', 'D'],
              'icrd_gold': [0, 0, 1, 'S0', 0, 1, 2, 3, 'S0', 1, 1, 1, 'S0', 'D']}

arrs_dict2 = {'ocrd_in': [0, 0, 0, 1, 1, 1, 1, 2, 'D'],
              'icrd_in': [0, 0, 1, 0, 1, 2, 3, 1, 'D'],
              'ocrd_gold': [0, 1, 2, 'S0', 'D'],
              'icrd_gold': [0, 0, 1, 'S0', 0, 1, 2, 3, 'S0', 1, 'S0', 'D']}

arrs_dict3 = {'ocrd_in': [0, 0, 1, 'S0', 0, 1, 2, 3, 'S0', 1, 1, 1, 'S0', 'D'],
              'icrd_in': [0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 'D'],
              'ocrd_gold': [0, 1, 'S0', 0, 1, 2, 3, 'S0', 1, 'S1', 'D'],
              'icrd_gold': [0, 1, 'S0', 0, 'S1', 0, 'S0', 1, 'S0', 0, 'S0', 1, 'S1', 0, 1, 2, 'S1', 'D']}


@pytest.mark.parametrize("arrs", [arrs_dict1, arrs_dict2, arrs_dict3])
@pytest.mark.parametrize("last_level", [True, False])
def test_crdpt_converter_direct(arrs, last_level, debug_sim):
    icrd = copy.deepcopy(arrs['icrd_in'])
    ocrd = copy.deepcopy(arrs['ocrd_in'])

    gold_ocrd = copy.deepcopy(arrs['ocrd_gold'])
    gold_icrd = copy.deepcopy(arrs['icrd_gold'])
    if last_level:
        gold_icrd[-2] = increment_stkn(gold_icrd[-2])

    cpc = CrdPtConverter(last_level=last_level, debug=debug_sim)

    done = False
    time = 0
    out_ocrd = []
    out_icrd = []
    while not done and time < TIMEOUT:
        if len(icrd) > 0:
            cpc.set_inner_crdpt(icrd.pop(0))
        if len(ocrd) > 0:
            cpc.set_outer_crdpt(ocrd.pop(0))

        cpc.update()

        out_ocrd.append(cpc.out_crd_outer())
        out_icrd.append(cpc.out_crd_inner())

        print("Timestep", time, "\t Done:", cpc.out_done(),
              "\n Curr in ocrd:", cpc.outer_crdpt, "\t Curr in icrd:", cpc.inner_crdpt,
              "\n Curr out ocrd:", cpc.curr_ocrd, "\t Curr out icrd:", cpc.curr_icrd,
              "\t Prev ocrd:", cpc.prev_ocrdpt, "\t Prev ocrd:", cpc.prev_ocrd,
              "\n Emit Tkn:", cpc.emit_stkn, "\t Emit Done:", cpc.emit_done, "\t Prev Stkn:", cpc.prev_stkn)

        done = cpc.out_done()
        time += 1

    out_ocrd = remove_emptystr(out_ocrd)
    out_icrd = remove_emptystr(out_icrd)

    if debug_sim:
        print("Outer Crd: ", out_ocrd)
        print("Inner Crd: ", out_icrd)

    assert (out_ocrd == gold_ocrd)
    assert (out_icrd == gold_icrd)
