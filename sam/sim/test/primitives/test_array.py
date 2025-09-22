import pytest
import random
import copy
from sam.sim.src.array import Array
from sam.sim.test.test import TIMEOUT


@pytest.mark.parametrize("dim1", [4, 16, 32, 64])
def test_arr_load_1d(dim1, debug_sim, max_val=1000):
    in_val = [x for x in range(dim1)] + ['S0', 'D']

    gold_val = [random.randint(0, max_val) for _ in range(dim1)]

    arr = Array(init_arr=gold_val, debug=debug_sim)

    done = False
    time = 0
    out_val = []
    while not done and time < TIMEOUT:
        if len(in_val) > 0:
            arr.set_load(in_val.pop(0))

        arr.update()

        out_val.append(arr.out_load())

        print("Timestep", time, "\t Out:", arr.out_load())

        done = arr.out_done()
        time += 1

    # Assert the array stores only the values
    assert (arr.get_arr() == gold_val)

    # Assert that the loaded streams also pass through tokens
    assert (out_val == gold_val + ['S0', 'D'])


@pytest.mark.parametrize("dim1", [4, 16, 32, 64])
def test_arr_store_1d(dim1, debug_sim, max_val=1000):
    in_addr = [x for x in range(dim1)] + ['S0', 'D']
    in_val = [random.randint(0, max_val) for _ in range(dim1)] + ['S0', 'D']
    assert (len(in_val) == len(in_addr))

    gold_val = in_val[:-2]

    arr = Array(debug=debug_sim)
    arr.resize(len(gold_val))

    done = False
    time = 0
    out_val = []
    count = 0
    while not done and time < TIMEOUT:
        if len(in_val) > 0:
            arr.set_store(in_addr.pop(0), in_val.pop(0))

        arr.update()

        out_val.append(arr.out_load())

        print("Timestep", time, "\t Out:", arr.out_load())

        done = arr.out_done()
        time += 1

    # Assert the array stores only the values
    assert (arr.get_arr() == gold_val)


arrs_dict1 = {'ref': [0, 1, 'N', 2, 'N', 3, 'S0', 'D'],
              'arr': [1, 2, 3, 4],
              'gold': [1, 2, 0, 3, 0, 4, 'S0', 'D']}
arrs_dict2 = {'ref': [0, 'N', 1, 2, 3, 'N', 'S0', 'D'],
              'arr': [1, 2, 3, 4],
              'gold': [1, 0, 2, 3, 4, 0, 'S0', 'D']}
arrs_dict3 = {'ref': [0, 1, 'S0', 2, 3, 'S0', 'N', 'S0', 4, 5, 'S1', 'D'],
              'arr': [1, 2, 3, 4, 5, 6],
              'gold': [1, 2, 'S0', 3, 4, 'S0', 0, 'S0', 5, 6, 'S1', 'D']}
arrs_dict4 = {'ref': [0, 1, 2, 'S0', 'N', 'S0', 2, 3, 'N', 4, 'N', 'S0', 'N', 'S1', 'D'],
              'arr': [1, 2, 3, 4, 5],
              'gold': [1, 2, 3, 'S0', 0, 'S0', 3, 4, 0, 5, 0, 'S0', 0, 'S1', 'D']}
arrs_dict5 = {'ref': [0, 1, 'N', 'N', 'S0', 2, 3, 'S0', 'N', 'N', 'N', 'S0', 4, 5, 'S1', 'D'],
              'arr': [1, 2, 3, 4, 5, 6],
              'gold': [1, 2, 0, 0, 'S0', 3, 4, 'S0', 0, 0, 0, 'S0', 5, 6, 'S1', 'D']}
arrs_dict6 = {'ref': ['N', 0, 1, 2, 'S0', 'N', 'N', 'S0', 2, 3, 4, 'S0', 'N', 'N', 'S1', 'D'],
              'arr': [1, 2, 3, 4, 5],
              'gold': [0, 1, 2, 3, 'S0', 0, 0, 'S0', 3, 4, 5, 'S0', 0, 0, 'S1', 'D']}


@pytest.mark.parametrize("arrs", [arrs_dict1, arrs_dict2, arrs_dict3, arrs_dict4, arrs_dict5, arrs_dict6])
def test_arr_load_direct_0tkn(arrs, debug_sim):
    ref = copy.deepcopy(arrs['ref'])
    arr = copy.deepcopy(arrs['arr'])
    gold_val = copy.deepcopy(arrs['gold'])

    arr = Array(init_arr=arr, debug=debug_sim)

    done = False
    time = 0
    out_val = []
    while not done and time < TIMEOUT:
        if len(ref) > 0:
            arr.set_load(ref.pop(0))

        arr.update()

        out_val.append(arr.out_load())

        print("Timestep", time, "\t Out:", arr.out_load())

        done = arr.out_done()
        time += 1

    # Assert the array stores only the values
    assert (out_val == gold_val)
