import pytest
import copy
import random
from sam.sim.src.accumulator import Reduce
from sam.sim.src.rd_scanner import CompressedCrdRdScan
from sam.sim.src.wr_scanner import CompressWrScan, ValsWrScan
from sam.sim.src.base import remove_emptystr
from sam.sim.test.test import *

arrs_dict0 = {'in_val': [5, 5, 'S0', 5, 'S0', 4, 8, 'S0', 4, 3, 'S0', 4, 3, 'S1', 'D'],
              'gold_val': [10, 5, 12, 7, 7, 'S0', 'D']}
arrs_dict1 = {'in_val': [1, 2, 3, 4, 5, 6, 7, 8, 9, 'S2', 'D'],
              'gold_val': [45, 'S1', 'D']}
arrs_dict2 = {'in_val': [5, 5, 'S0', 'S0', 4, 8, 'S0', 4, 3, 'S0', 4, 3, 'S1', 'D'],
              'gold_val': [10, 0, 12, 7, 7, 'S0', 'D']}
arrs_dict2 = {'in_val': [5, 5, 'S0', 'S0', 4, 8, 'S0', 4, 3, 'S0', 'S1', 'D'],
              'gold_val': [10, 0, 12, 7, 0, 'S0', 'D']}


@pytest.mark.parametrize("arrs", [arrs_dict0, arrs_dict1])
def test_reduce_direct_nd(arrs, debug_sim):
    in_val = copy.deepcopy(arrs['in_val'])
    gold_val = copy.deepcopy(arrs['gold_val'])

    red = Reduce(debug=debug_sim)

    done = False
    time = 0
    out_val = []
    while not done and time < TIMEOUT:
        if len(in_val) > 0:
            red.set_in_val(in_val.pop(0))

        red.update()

        out_val.append(red.out_val())

        print("Timestep", time, "\t Red:", red.out_val(), "\t Ref1:", )

        done = red.done
        time += 1

    out_val = remove_emptystr(out_val)

    assert (out_val == gold_val)


# FIXME: Need to get this test working with reduce
@pytest.mark.skip
@pytest.mark.parametrize("dim", [4, 16, 32, 64])
def test_reduce_random_2d(dim, debug_sim, max_val=1000, fill=0):
    in_mat_crds1, in_mat_segs1 = gen_n_comp_arrs(2, dim)
    in_mat_vals1 = gen_val_arr(len(in_mat_crds1[-1]), max_val, -max_val)

    in1_tup = convert_point_tuple(get_point_list(in_mat_crds1, in_mat_segs1, in_mat_vals1))

    nd1 = convert_point_tuple_ndarr(in1_tup, dim)
    gold_nd = np.sum(nd1, 1)
    gold_tup = convert_ndarr_point_tuple(gold_nd)

    if debug_sim:
        print("Pts Mat1:", in1_tup)
        print("Dense Mat1:", nd1)
        print("Dense Gold:", gold_nd)
        print("Gold:", gold_tup)

    rdscan_B1 = CompressedCrdRdScan(crd_arr=in_mat_crds1[0], seg_arr=in_mat_segs1[0], debug=debug_sim)
    rdscan_B2 = CompressedCrdRdScan(crd_arr=in_mat_crds1[1], seg_arr=in_mat_segs1[1], debug=debug_sim)

    val_B = Array(init_arr=in_mat_vals1, debug=debug_sim)
    red = Reduce(debug=debug_sim)

    vals_X = ValsWrScan(size=dim * dim, fill=fill, debug=debug_sim)
    wrscan_X1 = CompressWrScan(seg_size=2, size=dim, fill=fill, debug=debug_sim)

    in_ref_B = [0, 'D']
    done = False
    time = 0
    while not done and time < TIMEOUT:
        if len(in_ref_B) > 0:
            rdscan_B1.set_in_ref(in_ref_B.pop(0))

        rdscan_B2.set_in_ref(rdscan_B1.out_ref())

        val_B.set_load(rdscan_B2.out_ref())

        red.set_in_val(val_B.out_load())

        vals_X.set_input(red.out_val())

        wrscan_X1.set_input(rdscan_B1.out_crd())

        rdscan_B1.update()
        rdscan_B2.update()
        val_B.update()
        red.update()
        vals_X.update()
        wrscan_X1.update()

        print("Timestep", time, "\t Done --",
              "\tRdScan B1:", rdscan_B1.out_done(),
              "\tReduce:", red.out_done(),
              "\tArr:", val_B.out_done(),
              "\tWrScan:", vals_X.out_done(),
              "\tWrScan X1:", wrscan_X1.out_done(),
              )

        done = wrscan_X1.out_done()
        time += 1

    wrscan_X1.autosize()
    vals_X.autosize()

    out_crds = [wrscan_X1.get_arr()]
    out_segs = [wrscan_X1.get_seg_arr()]
    out_val = vals_X.get_arr()

    if debug_sim:
        print(out_segs)
        print(out_crds)
        print(out_val)

    if out_val == []:
        assert out_val == gold_tup
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        assert (check_point_tuple(out_tup, gold_tup))
