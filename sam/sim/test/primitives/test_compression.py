import pytest
from sam.sim.src.compression import ValDropper
from sam.sim.src.base import remove_emptystr
from sam.sim.test.test import TIMEOUT
import numpy as np


@pytest.mark.parametrize("dim1", [8])
def test_compress_1d(dim1, debug_sim):
    nums = np.random.choice([0, 1], size=dim1, p=[.4, .6])
    in1 = nums.tolist() + ['S0', 'D']
    crd_nums = np.arange(dim1)
    crd = crd_nums.tolist() + ['S0', 'D']
    # assert (len(in1) == len(in1))

    gold_val = nums[nums != 0].tolist() + ['S0', 'D']
    gold_crd = np.delete(crd_nums, np.where(nums == 0)).tolist() + ['S0', 'D']

    comp = ValDropper(debug=debug_sim)

    done = False
    time = 0

    out_val = []
    out_crd = []

    while not done and time < TIMEOUT:

        if len(in1) > 0:
            comp.set_val(in1.pop(0))
            comp.set_crd(crd.pop(0))

        comp.update()
        out_val.append(comp.out_val())
        out_crd.append(comp.out_crd())

        if debug_sim:
            print("Timestep", time, "\t Out:", comp.out_val())

        done = comp.out_done()
        time += 1

    out_val = remove_emptystr(out_val)
    out_crd = remove_emptystr(out_crd)
    print("Ref val:", gold_val)
    print("Out val:", out_val)

    print("Ref crd:", gold_crd)
    print("Out crd:", out_crd)

    assert (out_val == gold_val)
    assert (out_crd == gold_crd)

# @pytest.mark.parametrize("dim1", [4, 16, 32, 64])
# def test_exp_1d(dim1, debug_sim):
#     in1 = [x for x in range(dim1)] + ['S0', 'D']
#     in2 = None
#     # assert (len(in1) == len(in1))

#     gold_val = np.exp(np.arange(dim1)).tolist() + ['S0', 'D']

#     exp1 = Exp(debug=debug_sim)

#     done = False
#     time = 0
#     out_val = []
#     exp1.set_in2(in2)
#     while not done and time < TIMEOUT:
#         if len(in1) > 0:
#             exp1.set_in1(in1.pop(0))

#         exp1.update()

#         out_val.append(exp1.out_val())

#         print("Timestep", time, "\t Out:", exp1.out_val())

#         done = exp1.out_done()
#         time += 1

#     out_val = remove_emptystr(out_val)
#     print("Ref:", gold_val)
#     print("Out:", out_val)

#     assert (out_val == gold_val)
