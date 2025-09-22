import pytest
import copy
from sam.sim.src.repeater import RepeatSigGen, Repeat
from sam.sim.src.base import remove_emptystr

from sam.sim.test.test import TIMEOUT, gen_stream, dedup_adj

####################
# Test Repeat Signal Generator
##################
arr_dict1 = {'istream': [0, 1, 'S0', 2, 'S0', 3, 'S1', 'D'],
             'gold': ['R', 'R', 'S', 'R', 'S', 'R', 'S', 'D']}
arr_dict2 = {'istream': [0, 1, 2, 'S0', 0, 1, 2, 'S1', 3, 'S1', 4, 5, 'S2', 'D'],
             'gold': ['R', 'R', 'R', 'S', 'R', 'R', 'R', 'S', 'R', 'S', 'R', 'R', 'S', 'D']}
arr_dict3 = {'istream': [0, 1, 2, 'S0', 0, 1, 2, 'S1', 'S1', 4, 5, 'S2', 'D'],
             'gold': ['R', 'R', 'R', 'S', 'R', 'R', 'R', 'S', 'S', 'R', 'R', 'S', 'D']}


@pytest.mark.parametrize("arrs", [arr_dict1, arr_dict2, arr_dict3])
def test_repeat_gen_direct(arrs, debug_sim):
    in_stream = copy.deepcopy(arrs['istream'])
    gold = copy.deepcopy(arrs['gold'])

    repsig = RepeatSigGen(debug=debug_sim)

    done = False
    time = 0
    out = []
    while not done and time < TIMEOUT:
        if len(in_stream) > 0:
            repsig.set_istream(in_stream.pop(0))

        repsig.update()

        out.append(repsig.out_repeat())

        print("Timestep", time, "\t Done:", repsig.out_done(), "\t Repeat Sig:", repsig.out_repeat())

        done = repsig.out_done()
        time += 1

    out = remove_emptystr(out)
    assert (out == gold)


@pytest.mark.parametrize("max_val", [4, 16, 32, 64])
@pytest.mark.parametrize("nd", [1, 2, 3, 4, 5])
def test_repeat_gen_random_nd(max_val, nd, debug_sim):
    in_stream = gen_stream(n=nd, max_val=max_val, max_nnz=max_val)

    if debug_sim:
        print("Input Stream:", in_stream)

    gold = dedup_adj(in_stream)
    gold = ['R' if isinstance(x, int) else x if x == 'D' else 'S' for x in gold]

    repsig = RepeatSigGen(debug=debug_sim)

    done = False
    time = 0
    out = []
    while not done and time < TIMEOUT:
        if len(in_stream) > 0:
            repsig.set_istream(in_stream.pop(0))

        repsig.update()

        out.append(repsig.out_repeat())

        print("Timestep", time, "\t Done:", repsig.out_done(), "\t Repeat Sig:", repsig.out_repeat())

        done = repsig.out_done()
        time += 1

    out = remove_emptystr(out)
    assert (out == gold)


####################
# Test Repeater
##################
arr_dict1 = {'in_ref': [0, 1, 2, 'S0', 'D'],
             'repeat': ['R', 'R', 'S', 'R', 'S', 'R', 'S', 'D'],
             'gold': [0, 0, 'S0', 1, 'S0', 2, 'S1', 'D']}
arr_dict2 = {'in_ref': [0, 1, 'S0', 2, 'S0', 3, 'S1', 'D'],
             'repeat': ['R', 'R', 'R', 'S', 'R', 'R', 'R', 'S', 'R', 'S', 'R', 'R', 'S', 'D'],
             'gold': [0, 0, 0, 'S0', 1, 1, 1, 'S1', 2, 'S1', 3, 3, 'S2', 'D']}
arr_dict3 = {'in_ref': [0, 'S0', 1, 2, 3, 'S1', 'D'],
             'repeat': ['R', 'R', 'R', 'S', 'R', 'R', 'R', 'S', 'R', 'S', 'R', 'R', 'S', 'D'],
             'gold': [0, 0, 0, 'S1', 1, 1, 1, 'S0', 2, 'S0', 3, 3, 'S2', 'D']}
arr_dict4 = {'in_ref': [0, 'D'],
             'repeat': ['R', 'R', 'S', 'D'],
             'gold': [0, 0, 'S0', 'D']}
arr_dict5 = {'in_ref': [0, 1, 'S0', 'D'],
             'repeat': ['R', 'R', 'R', 'S'] * 2 + ['D'],
             'gold': [0, 0, 0, 'S0', 1, 1, 1, 'S1', 'D']}
arr_dict6 = {'in_ref': [0, 'S0', 'S0', 'S0', 'S0', 'S0', 'S0', 'S0', 'S1', 'D'],
             'repeat': ['R', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'D'],
             'gold': [0, 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S1', 'S2', 'D']}
arr_dict7 = {'in_ref': [0, 'S0', 1, 'S1', 'D'],
             'repeat': ['S', 'R', 'S', 'D'],
             'gold': ['S1', 1, 'S2', 'D']}
arr_dict8 = {'in_ref': ['N', 0, 1, 'S0', 'D'],
             'repeat': ['R', 'S', 'R', 'S', 'R', 'S', 'D'],
             'gold': ['N', 'S0', 0, 'S0', 1, 'S1', 'D']}


@pytest.mark.parametrize("arrs", [arr_dict1, arr_dict2, arr_dict3, arr_dict4, arr_dict5, arr_dict6, arr_dict7,
                                  arr_dict8])
def test_repeat_direct(arrs, debug_sim):
    in_ref = copy.deepcopy(arrs['in_ref'])
    in_repeat = copy.deepcopy(arrs['repeat'])
    gold = copy.deepcopy(arrs['gold'])

    rep = Repeat(debug=debug_sim)

    done = False
    time = 0
    out = []
    while not done and time < TIMEOUT:
        if len(in_ref) > 0:
            rep.set_in_ref(in_ref.pop(0))
        if len(in_repeat) > 0:
            rep.set_in_repeat(in_repeat.pop(0))

        rep.update()

        out.append(rep.out_ref())

        print("Timestep", time, "\t Done:", rep.out_done(), "\t Repeat Sig:", rep.out_ref())

        done = rep.out_done()
        time += 1

    out = remove_emptystr(out)
    assert (out == gold)
