import pytest

from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.wr_scanner import ValsWrScan
from sam.sim.src.joiner import Intersect2
from sam.sim.src.compute import Multiply2
from sam.sim.src.crd_manager import CrdDrop
from sam.sim.src.base import remove_emptystr

from sam.sim.test.test import *


@pytest.mark.parametrize("dim1", [4, 16, 32, 64])
@pytest.mark.parametrize("dim2", [4, 16, 32, 64])
def test_unit_mat_elemmul_uu_uu_uu(dim1, dim2, debug_sim, backpressure, depth, max_val=1000, fill=0):
    size = dim1 * dim2 + 1
    in_mat1 = [random.randint(0, max_val) for _ in range(dim1 * dim2)]
    in_mat2 = [random.randint(0, max_val) for _ in range(dim1 * dim2)]

    if debug_sim:
        print("Mat 1:", in_mat1)
        print("Mat 2:", in_mat2)

    assert (len(in_mat1) == len(in_mat2))
    print(len(in_mat1))

    gold_vec = [in_mat1[i] * in_mat2[i] for i in range(len(in_mat1))]

    rdscan_d1 = UncompressCrdRdScan(dim=dim1, debug=debug_sim, back_en=backpressure, depth=int(depth))
    rdscan_d2 = UncompressCrdRdScan(dim=dim2, debug=debug_sim, back_en=backpressure, depth=int(depth))

    val1 = Array(init_arr=in_mat1, debug=debug_sim, back_en=backpressure, depth=int(depth))
    val2 = Array(init_arr=in_mat2, debug=debug_sim, back_en=backpressure, depth=int(depth))
    mul = Multiply2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    wrscan = ValsWrScan(size=size, fill=fill, debug=debug_sim, back_en=backpressure, depth=int(depth))

    in_ref = [0, 'D']
    done = False
    time = 0
    while not done and time < TIMEOUT:
        if len(in_ref) > 0:
            rdscan_d1.set_in_ref(in_ref.pop(0), "")

        rdscan_d2.set_in_ref(rdscan_d1.out_ref(), rdscan_d1)

        val1.set_load(rdscan_d2.out_ref(), rdscan_d2)
        val2.set_load(rdscan_d2.out_ref(), rdscan_d2)

        mul.set_in1(val1.out_load(), val1)
        mul.set_in2(val2.out_load(), val2)

        wrscan.set_input(mul.out_val(), mul)

        rdscan_d1.update()
        rdscan_d2.update()
        val1.update()
        val2.update()
        mul.update()
        wrscan.update()

        print("Timestep", time, "\t Done --",
              "\tRdScan D1:", rdscan_d1.out_done(),
              "\tRdScan D2:", rdscan_d2.out_done(),
              "\tArr:", val1.out_done(), val2.out_done(),
              "\tMul:", mul.out_done(), "\tWrScan:", wrscan.out_done())

        done = wrscan.out_done()
        time += 1

    check_arr(wrscan, gold_vec)


arrs_dict0 = {'pts1': [[1, 2, -778]],
              'seg_arrs1': [[0, 1], [0, 1]],
              'crd_arrs1': [[1], [2]],
              'val_arr1': [-778],
              'pts2': [[0, 0, -252], [0, 2, -816], [1, 1, 95], [1, 2, 939], [1, 3, -627], [2, 0, 422], [2, 1, 470],
                       [3, 0, -948]],
              'seg_arrs2': [[0, 4], [0, 2, 5, 7, 8]],
              'crd_arrs2': [[0, 1, 2, 3], [0, 2, 1, 2, 3, 0, 1, 0]],
              'val_arr2': [-252, -816, 95, 939, -627, 422, 470, -948]
              }
arrs_dict1 = {
    'pts1': [(0, 0, -586), (0, 2, 22), (1, 0, 716), (2, 0, 715), (2, 2, -566), (3, 0, 427), (3, 1, 114), (3, 2, -893),
             (3, 3, -146)],
    'seg_arrs1': [[0, 4], [0, 2, 3, 5, 9]],
    'crd_arrs1': [[0, 1, 2, 3], [0, 2, 0, 0, 2, 0, 1, 2, 3]],
    'val_arr1': [-586, 22, 716, 715, -566, 427, 114, -893, -146],
    'pts2': [(0, 0, 985), (3, 0, -131), (3, 1, 718)],
    'seg_arrs2': [[0, 2], [0, 1, 3]],
    'crd_arrs2': [[0, 3], [0, 0, 1]],
    'val_arr2': [985, -131, 718]
}
arrs_dict2 = {
    'seg_arrs1': [[0, 2], [0, 3, 4]],
    'crd_arrs1': [[0, 1], [0, 1, 3, 0]],
    'val_arr1': [559, 728, -79, 95],
    'seg_arrs2': [[0, 4], [0, 2, 3, 4, 7]],
    'crd_arrs2': [[0, 1, 2, 3], [1, 2, 3, 0, 0, 2, 3]],
    'val_arr2': [704, -631, -377, -780, 337, 338, -83],
    'pts1': [(0, 0, 559), (0, 1, 728), (0, 3, -79), (1, 0, 95)],
    'pts2': [(0, 1, 704), (0, 2, -631), (1, 3, -377), (2, 0, -780), (3, 0, 337), (3, 2, 338), (3, 3, -83)]
}
arrs_dict3 = {
    'seg_arrs1': [[0, 3], [0, 1, 3, 4]],
    'crd_arrs1': [[0, 1, 2], [0, 2, 3, 2]],
    'val_arr1': [-158, -102, -674, -107],
    'seg_arrs2': [[0, 3], [0, 1, 3, 4]],
    'crd_arrs2': [[0, 1, 3], [1, 2, 3, 1]],
    'val_arr2': [399, 848, -325, -677],
    'pts1': [(0, 0, -158), (1, 2, -102), (1, 3, -674), (2, 2, -107)],
    'pts2': [(0, 1, 399), (1, 2, 848), (1, 3, -325), (3, 1, -677)],
}


@pytest.mark.parametrize("arrs", [arrs_dict0, arrs_dict1, arrs_dict2, arrs_dict3])
def test_unit_mat_elemmul_direct_cc_cc_cc(arrs, debug_sim, backpressure, depth, dim=4, fill=0):
    in_mat_crds1 = copy.deepcopy(arrs['crd_arrs1'])
    in_mat_segs1 = copy.deepcopy(arrs['seg_arrs1'])
    in_mat_vals1 = copy.deepcopy(arrs['val_arr1'])
    in_mat_crds2 = copy.deepcopy(arrs['crd_arrs2'])
    in_mat_segs2 = copy.deepcopy(arrs['seg_arrs2'])
    in_mat_vals2 = copy.deepcopy(arrs['val_arr2'])

    in1_tup = copy.deepcopy(arrs['pts1'])
    in2_tup = copy.deepcopy(arrs['pts2'])

    nd1 = convert_point_tuple_ndarr(in1_tup, dim).astype(int)
    nd2 = convert_point_tuple_ndarr(in2_tup, dim).astype(int)
    gold_nd = np.multiply(nd1, nd2)
    gold_tup = convert_ndarr_point_tuple(gold_nd)

    if debug_sim:
        print("Pts Mat1:", in1_tup)
        print("Pts Mat2:", in2_tup)
        print("Dense Mat1:", nd1)
        print("Dense Mat2:", nd2)
        print("Dense Gold:", gold_nd)
        print("Gold:", gold_tup)

    rdscan_B1 = CompressedCrdRdScan(crd_arr=in_mat_crds1[0], seg_arr=in_mat_segs1[0], debug=debug_sim,
                                    back_en=backpressure, depth=int(depth))
    rdscan_B2 = CompressedCrdRdScan(crd_arr=in_mat_crds1[1], seg_arr=in_mat_segs1[1], debug=debug_sim,
                                    back_en=backpressure, depth=int(depth))

    rdscan_C1 = CompressedCrdRdScan(crd_arr=in_mat_crds2[0], seg_arr=in_mat_segs2[0], debug=debug_sim,
                                    back_en=backpressure, depth=int(depth))
    rdscan_C2 = CompressedCrdRdScan(crd_arr=in_mat_crds2[1], seg_arr=in_mat_segs2[1], debug=debug_sim,
                                    back_en=backpressure, depth=int(depth))

    val_B = Array(init_arr=in_mat_vals1, debug=debug_sim, back_en=backpressure, depth=int(depth))
    val_C = Array(init_arr=in_mat_vals2, debug=debug_sim, back_en=backpressure, depth=int(depth))
    inter1 = Intersect2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    inter2 = Intersect2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    mul = Multiply2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    drop = CrdDrop(debug=debug_sim, back_en=backpressure, depth=int(depth))
    vals_X = ValsWrScan(size=dim * dim, fill=fill, debug=debug_sim, back_en=backpressure, depth=int(depth))
    wrscan_X1 = CompressWrScan(seg_size=2, size=dim, fill=fill, back_en=backpressure, depth=int(depth))
    wrscan_X2 = CompressWrScan(seg_size=dim + 1, size=dim * dim, fill=fill, back_en=backpressure, depth=int(depth))

    in_ref_B = [0, 'D']
    in_ref_C = [0, 'D']
    out_inter1 = []
    out_drop = []
    in_drop = []
    done = False
    time = 0
    while not done and time < TIMEOUT:
        if len(in_ref_B) > 0:
            rdscan_B1.set_in_ref(in_ref_B.pop(0), "")

        if len(in_ref_C) > 0:
            rdscan_C1.set_in_ref(in_ref_C.pop(0), "")

        inter1.set_in1(rdscan_B1.out_ref(), rdscan_B1.out_crd(), rdscan_B1)
        inter1.set_in2(rdscan_C1.out_ref(), rdscan_C1.out_crd(), rdscan_C1)

        rdscan_B2.set_in_ref(inter1.out_ref1(), inter1)

        rdscan_C2.set_in_ref(inter1.out_ref2(), inter1)

        inter2.set_in1(rdscan_B2.out_ref(), rdscan_B2.out_crd(), rdscan_B2)
        inter2.set_in2(rdscan_C2.out_ref(), rdscan_C2.out_crd(), rdscan_C2)

        val_B.set_load(inter2.out_ref1(), inter2)
        val_C.set_load(inter2.out_ref2(), inter2)

        mul.set_in1(val_B.out_load(), val_B)
        mul.set_in2(val_C.out_load(), val_C)

        drop.set_outer_crd(inter1.out_crd(), inter1)
        drop.set_inner_crd(inter2.out_crd(), inter2)

        vals_X.set_input(mul.out_val(), mul)

        # wrscan_X1.set_input(inter1.out_crd())
        wrscan_X1.set_input(drop.out_crd_outer(), drop)

        wrscan_X2.set_input(inter2.out_crd(), inter2)

        rdscan_B1.update()
        rdscan_C1.update()
        inter1.update()
        rdscan_B2.update()
        rdscan_C2.update()
        inter2.update()
        val_B.update()
        val_C.update()
        mul.update()
        drop.update()
        vals_X.update()
        wrscan_X1.update()
        wrscan_X2.update()

        out_drop.append(drop.out_crd_outer())
        out_inter1.append(inter1.out_crd())
        in_drop.append(inter2.out_crd())

        print("Timestep", time, "\t Done --",
              "\tRdScan B1:", rdscan_B1.out_done(), "\tRdScan B2:", rdscan_B2.out_done(),
              "\tInter1:", inter1.out_done(),
              "\tRdScan C1:", rdscan_C1.out_done(), "\tRdScan C2:", rdscan_C2.out_done(),
              "\tInter2:", inter2.out_done(),
              "\tArr:", val_B.out_done(), val_C.out_done(),
              "\tMul:", mul.out_done(),
              "\tDrop:", drop.out_done(),
              "\tWrScan:", vals_X.out_done(),
              "\tWrScan X1:", wrscan_X1.out_done(), "\tWrScan X2:", wrscan_X2.out_done(),
              )

        done = wrscan_X2.out_done()
        time += 1

    wrscan_X2.autosize()
    wrscan_X1.autosize()
    vals_X.autosize()

    out_crds = [wrscan_X1.get_arr(), wrscan_X2.get_arr()]
    out_segs = [wrscan_X1.get_seg_arr(), wrscan_X2.get_seg_arr()]
    out_val = vals_X.get_arr()

    if debug_sim:
        print(out_segs)
        print(out_crds)
        print(out_val)

        print("OUT INTER1:", remove_emptystr(out_inter1))
        print("OUT INTER1:", out_inter1)
        print("IN DROP:", remove_emptystr(in_drop))
        print("IN DROP:", in_drop)
        print("OUT DROP:", remove_emptystr(out_drop))
        print("OUT DROP:", out_drop)

    if not out_val:
        assert out_val == gold_tup
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        assert (check_point_tuple(out_tup, gold_tup))


@pytest.mark.parametrize("dim", [4, 16, 32, 64])
def test_unit_mat_elemmul_cc_cc_cc(dim, debug_sim, backpressure, depth, max_val=1000, fill=0):
    in_mat_crds1, in_mat_segs1 = gen_n_comp_arrs(2, dim)
    in_mat_vals1 = gen_val_arr(len(in_mat_crds1[-1]), max_val, -max_val)
    in_mat_crds2, in_mat_segs2 = gen_n_comp_arrs(2, dim)
    in_mat_vals2 = gen_val_arr(len(in_mat_crds2[-1]), max_val, -max_val)

    if debug_sim:
        print("Mat 1:", in_mat_segs1, in_mat_crds1, in_mat_vals1)
        print("Mat 2:", in_mat_segs2, in_mat_crds2, in_mat_vals2)

    in1_tup = convert_point_tuple(get_point_list(in_mat_crds1, in_mat_segs1, in_mat_vals1))
    in2_tup = convert_point_tuple(get_point_list(in_mat_crds2, in_mat_segs2, in_mat_vals2))

    nd1 = convert_point_tuple_ndarr(in1_tup, dim)
    nd2 = convert_point_tuple_ndarr(in2_tup, dim)
    gold_nd = np.multiply(nd1, nd2)
    gold_tup = convert_ndarr_point_tuple(gold_nd)

    if debug_sim:
        print("Pts Mat1:", in1_tup)
        print("Pts Mat2:", in2_tup)
        print("Dense Mat1:", nd1)
        print("Dense Mat2:", nd2)
        print("Dense Gold:", gold_nd)
        print("Gold:", gold_tup)

    rdscan_B1 = CompressedCrdRdScan(crd_arr=in_mat_crds1[0], seg_arr=in_mat_segs1[0], debug=debug_sim,
                                    back_en=backpressure, depth=int(depth))
    rdscan_B2 = CompressedCrdRdScan(crd_arr=in_mat_crds1[1], seg_arr=in_mat_segs1[1], debug=debug_sim,
                                    back_en=backpressure, depth=int(depth))

    rdscan_C1 = CompressedCrdRdScan(crd_arr=in_mat_crds2[0], seg_arr=in_mat_segs2[0], debug=debug_sim,
                                    back_en=backpressure, depth=int(depth))
    rdscan_C2 = CompressedCrdRdScan(crd_arr=in_mat_crds2[1], seg_arr=in_mat_segs2[1], debug=debug_sim,
                                    back_en=backpressure, depth=int(depth))

    val_B = Array(init_arr=in_mat_vals1, debug=debug_sim, back_en=backpressure, depth=int(depth))
    val_C = Array(init_arr=in_mat_vals2, debug=debug_sim, back_en=backpressure, depth=int(depth))
    inter1 = Intersect2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    inter2 = Intersect2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    mul = Multiply2(debug=debug_sim, back_en=backpressure, depth=int(depth))
    drop = CrdDrop(debug=debug_sim, back_en=backpressure, depth=int(depth))
    vals_X = ValsWrScan(size=dim * dim, fill=fill, debug=debug_sim, back_en=backpressure, depth=int(depth))
    wrscan_X1 = CompressWrScan(seg_size=2, size=dim, fill=fill, back_en=backpressure, depth=int(depth))
    wrscan_X2 = CompressWrScan(seg_size=dim + 1, size=dim * dim, fill=fill, back_en=backpressure, depth=int(depth))

    in_ref_B = [0, 'D']
    in_ref_C = [0, 'D']
    done = False
    time = 0
    while not done and time < TIMEOUT:
        if len(in_ref_B) > 0:
            rdscan_B1.set_in_ref(in_ref_B.pop(0), "")

        if len(in_ref_C) > 0:
            rdscan_C1.set_in_ref(in_ref_C.pop(0), "")

        inter1.set_in1(rdscan_B1.out_ref(), rdscan_B1.out_crd(), rdscan_B1)
        inter1.set_in2(rdscan_C1.out_ref(), rdscan_C1.out_crd(), rdscan_C1)

        rdscan_B2.set_in_ref(inter1.out_ref1(), inter1)

        rdscan_C2.set_in_ref(inter1.out_ref2(), inter1)

        inter2.set_in1(rdscan_B2.out_ref(), rdscan_B2.out_crd(), rdscan_B2)
        inter2.set_in2(rdscan_C2.out_ref(), rdscan_C2.out_crd(), rdscan_C2)

        val_B.set_load(inter2.out_ref1(), inter2)
        val_C.set_load(inter2.out_ref2(), inter2)

        mul.set_in1(val_B.out_load(), val_B)
        mul.set_in2(val_C.out_load(), val_C)

        drop.set_outer_crd(inter1.out_crd(), inter1)
        drop.set_inner_crd(inter2.out_crd(), inter2)

        vals_X.set_input(mul.out_val(), mul)

        wrscan_X1.set_input(drop.out_crd_outer(), drop)

        wrscan_X2.set_input(inter2.out_crd(), inter2)

        rdscan_B1.update()
        rdscan_C1.update()
        inter1.update()
        rdscan_B2.update()
        rdscan_C2.update()
        inter2.update()
        val_B.update()
        val_C.update()
        mul.update()
        drop.update()
        vals_X.update()
        wrscan_X1.update()
        wrscan_X2.update()

        print("Timestep", time, "\t Done --",
              "\tRdScan B1:", rdscan_B1.out_done(), "\tRdScan B2:", rdscan_B2.out_done(),
              "\tInter1:", inter1.out_done(),
              "\tRdScan C1:", rdscan_C1.out_done(), "\tRdScan C2:", rdscan_C2.out_done(),
              "\tInter2:", inter2.out_done(),
              "\tArr:", val_B.out_done(), val_C.out_done(),
              "\tMul:", mul.out_done(),
              "\tDrop:", drop.out_done(),
              "\tWrScan:", vals_X.out_done(),
              "\tWrScan X1:", wrscan_X1.out_done(), "\tWrScan X2:", wrscan_X2.out_done(),
              )
        done = wrscan_X2.out_done()
        time += 1

    wrscan_X2.autosize()
    wrscan_X1.autosize()
    vals_X.autosize()

    out_crds = [wrscan_X1.get_arr(), wrscan_X2.get_arr()]
    out_segs = [wrscan_X1.get_seg_arr(), wrscan_X2.get_seg_arr()]
    out_val = vals_X.get_arr()

    if debug_sim:
        print(out_segs)
        print(out_crds)
        print(out_val)

    if not out_val:
        assert out_val == gold_tup
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        assert (check_point_tuple(out_tup, gold_tup))
