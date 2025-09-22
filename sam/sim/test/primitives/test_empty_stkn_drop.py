import copy
import pytest

from sam.sim.src.base import is_stkn, larger_stkn
from sam.sim.src.token import EmptyFiberStknDrop
from sam.sim.src.base import remove_emptystr
from sam.sim.test.test import TIMEOUT


arrs_dict1 = {'in': [0, 1, 'S0', 'D']}
arrs_dict2 = {'in': [1, 'S0', 'S1', 'D']}
arrs_dict3 = {'in': [0, 1, 'S0', 'D']}
arrs_dict4 = {'in': [1, 'S0', 1, 'S1', 'D']}
arrs_dict5 = {'in': [0, 1, 2, 3, 'S0', 'D']}
arrs_dict6 = {'in': [1, 'S0', 1, 'S0', 'S0', 1, 'S1', 'D']}
arrs_dict7 = {'in': [0, 1, 3, 'S0', 'D']}
arrs_dict8 = {'in': [8, 'S0', 'S1', 'D']}
arrs_dict9 = {'in': [8, 9, 'S0', 10, 'S0', 'S0', 'S1', 11, 'S1', 'S0', 'S2', 'D']}
arrs_dict10 = {'in': ['S0', 'S1', 8, 9, 'S0', 10, 'S0', 'S0', 'S1', 11, 'S1', 'S0', 'S2', 'D']}


@pytest.mark.parametrize("arrs", [arrs_dict1, arrs_dict2, arrs_dict3, arrs_dict4, arrs_dict5, arrs_dict6, arrs_dict7,
                                  arrs_dict8, arrs_dict9, arrs_dict10])
def test_empty_stkn_drop(arrs, debug_sim):
    ival = copy.deepcopy(arrs['in'])

    gold = []
    prev_stkn = False
    stkn = None
    leading_stkn = True
    for item in arrs['in']:
        if not leading_stkn:
            if is_stkn(item):
                stkn = item if stkn is None else larger_stkn(stkn, item)
                prev_stkn = True
            elif prev_stkn:
                gold.append(stkn)
                stkn = None
                prev_stkn = False

        if not is_stkn(item):
            leading_stkn = False
            gold.append(item)

    td = EmptyFiberStknDrop(debug=debug_sim)

    done = False
    time = 0
    out = []
    while not done and time < TIMEOUT:
        if len(ival) > 0:
            td.set_in_stream(ival.pop(0))

        td.update()

        out.append(td.out_val())

        print("Timestep", time, "\t Done:", td.out_done(), "\t Out:", td.out_val())

        done = td.out_done()
        time += 1

    out = remove_emptystr(out)

    if debug_sim:
        print("Out:", out)
        print("Gold:", gold)

    assert (out == gold)
