import pytest
from sam.sim.src.base import remove_emptystr
from sam.sim.src.compute import Add2, Multiply2
from sam.sim.test.primitives.test_intersect import TIMEOUT


@pytest.mark.parametrize("dim1", [4, 16, 32, 64])
@pytest.mark.parametrize("neg1", [True, False])
@pytest.mark.parametrize("neg2", [True, False])
def test_add_1d(dim1, neg1, neg2, debug_sim):
    in1 = [x for x in range(dim1)] + ['S0', 'D']
    in2 = [2 * x for x in range(dim1)] + ['S0', 'D']
    assert (len(in1) == len(in1))

    gold_val = [3 * x for x in range(dim1)] + ['S0', 'D']
    if neg1 and neg2:
        gold_val = [-3 * x for x in range(dim1)] + ['S0', 'D']
    elif neg1:
        gold_val = [x for x in range(dim1)] + ['S0', 'D']
    elif neg2:
        gold_val = [-x for x in range(dim1)] + ['S0', 'D']

    add = Add2(debug=debug_sim, neg1=neg1, neg2=neg2)

    done = False
    time = 0
    out_val = []
    while not done and time < TIMEOUT:
        if len(in1) > 0:
            add.set_in1(in1.pop(0))
        if len(in2) > 0:
            add.set_in2(in2.pop(0))

        add.update()

        out_val.append(add.out_val())

        print("Timestep", time, "\t Out:", add.out_val())

        done = add.out_done()
        time += 1

    out_val = remove_emptystr(out_val)

    assert (out_val == gold_val)


@pytest.mark.parametrize("dim1", [4, 16, 32, 64])
def test_mul_1d(dim1, debug_sim):
    in1 = [x for x in range(dim1)] + ['S0', 'D']
    in2 = [2 * x for x in range(dim1)] + ['S0', 'D']
    assert (len(in1) == len(in1))

    gold_val = [2 * x ** 2 for x in range(dim1)] + ['S0', 'D']

    mul = Multiply2(debug=debug_sim)

    done = False
    time = 0
    out_val = []
    while not done and time < TIMEOUT:
        if len(in1) > 0:
            mul.set_in1(in1.pop(0))
        if len(in2) > 0:
            mul.set_in2(in2.pop(0))

        mul.update()

        out_val.append(mul.out_val())

        print("Timestep", time, "\t Out:", mul.out_val())

        done = mul.out_done()
        time += 1

    out_val = remove_emptystr(out_val)

    assert (out_val == gold_val)
