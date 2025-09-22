import copy
import math

import pytest

from functools import reduce

from sam.sim.src.split import Split
from sam.sim.src.bitvector import BV, ChunkBV
from sam.sim.src.base import remove_emptystr
from sam.sim.test.test import TIMEOUT, get_bv


arrs_dict1 = {'crd_in': [0, 2, 3, 9, 11, 12, 'S0', 'D']}
arrs_dict2 = {'crd_in': [0, 2, 3, 9, 11, 'S0', 'D']}
arrs_dict3 = {'crd_in': ['S0', 'D']}


@pytest.mark.parametrize("arrs", [arrs_dict1, arrs_dict2, arrs_dict3])
def test_bv_chunk_direct(arrs, debug_sim):
    sf = 4
    size = 16
    crd = copy.deepcopy(arrs['crd_in'])
    gold_bv = get_bv(crd)

    gold_bv_chunk = []
    for elem in gold_bv:
        if isinstance(elem, str) and elem[0] == '0':
            elem = int(elem, 2)
            for i in range(0, math.ceil(size / sf)):
                gold_bv_chunk.append(bin((elem >> (sf * i)) & ((1 << sf) - 1)))
        else:
            gold_bv_chunk.append(elem)

    if debug_sim:
        print("Gold:", gold_bv)

    bv = BV(debug=debug_sim)
    bvchunk = ChunkBV(width=sf, size=size, debug=debug_sim)
    done = False
    time = 0
    out_bv = []
    out_bv_chunk = []
    while not done and time < TIMEOUT:
        if len(crd) > 0:
            bv.set_in_crd(crd.pop(0))

        bvchunk.set_in_bv(bv.out_bv_int())

        bv.update()
        bvchunk.update()

        out_bv.append(bv.out_bv())
        out_bv_chunk.append(bvchunk.out_bv())

        print("Timestep", time, "\t Done:", bv.out_done(), bvchunk.out_done())

        done = bvchunk.out_done()
        time += 1

    out_bv = remove_emptystr(out_bv)

    if debug_sim:
        print("BV: ", out_bv)
        print("BV Chunked:", out_bv_chunk)

    assert (out_bv == gold_bv)
    assert remove_emptystr(out_bv_chunk) == gold_bv_chunk
