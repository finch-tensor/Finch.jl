import copy
import pytest

from sam.sim.src.base import is_stkn
from sam.sim.src.token import StknDrop
from sam.sim.src.base import remove_emptystr
from sam.sim.test.test import TIMEOUT

arrs_dict1 = {'in': [0, 1, 'S0', 'D']}
arrs_dict2 = {'in': [1, 'S0', 'S1', 'D']}
arrs_dict3 = {'in': [0, 1, 'S0', 'D']}
arrs_dict4 = {'in': [1, 'S0', 1, 'S1', 'D']}
arrs_dict5 = {'in': [0, 1, 2, 3, 'S0', 'D']}
arrs_dict6 = {'in': [1, 'S0', 1, 'S0', 'S0', 1, 'S1', 'D']}
arrs_dict7 = {'in': [0, 1, 3, 'S0', 'D']}


@pytest.mark.parametrize("arrs", [arrs_dict1, arrs_dict2, arrs_dict3, arrs_dict4, arrs_dict5, arrs_dict6, arrs_dict7])
def test_tkn_drop(arrs, debug_sim):
    ival = copy.deepcopy(arrs['in'])

    gold = [item for item in arrs['in'] if not is_stkn(item)]

    td = StknDrop(debug=debug_sim)

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
        print(out)
        print(gold)

    assert (out == gold)
