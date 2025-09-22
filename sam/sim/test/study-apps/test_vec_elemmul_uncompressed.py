import pytest
import random
import time
import os

from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.wr_scanner import ValsWrScan, CompressWrScan
from sam.sim.src.joiner import Intersect2
from sam.sim.src.compute import Multiply2
from sam.sim.src.array import Array

from sam.sim.test.test import TIMEOUT, check_arr, check_seg_arr, read_inputs

cwd = os.getcwd()
synthetic_dir = os.getenv('SYNTHETIC_PATH', default=os.path.join(cwd, 'synthetic'))


@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.synth
@pytest.mark.parametrize("run_length", [1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 200, 300, 400])
@pytest.mark.parametrize("vectype", ["random", "runs", "blocks"])
@pytest.mark.parametrize("sparsity", [0.2, 0.6, 0.8, 0.9, 0.95, 0.975, 0.9875, 0.99375])
def test_unit_vec_elemmul_u_u_u(samBench, run_length, vectype, sparsity, debug_sim, backpressure, depth,
                                dim1=2000, max_val=1000, fill=0):

    if vectype == "random":
        b_dirname = os.path.join(synthetic_dir, vectype, "uncompressed", "B_" + vectype + "_sp_" + str(sparsity))
    elif vectype == "runs":
        # b_dirname = os.path.join(synthetic_dir, vectype, "uncompressed", "runs_0_100_200")
        b_dirname = os.path.join(synthetic_dir, vectype, "uncompressed", f"runs_rl_{run_length}_nnz_400")
    elif vectype == "blocks":
        # b_dirname = os.path.join(synthetic_dir, vectype, "uncompressed", "B_blocks_20_20")
        b_dirname = os.path.join(synthetic_dir, vectype, "uncompressed", f"B_blocks_400_{run_length}")

    b_vals_filename = os.path.join(b_dirname, "tensor_B_mode_vals")
    in_vec1 = read_inputs(b_vals_filename, float)

    if vectype == "random":
        c_dirname = os.path.join(synthetic_dir, vectype, "uncompressed", "C_" + vectype + "_sp_" + str(sparsity))
    elif vectype == "runs":
        # c_dirname = os.path.join(synthetic_dir, vectype, "uncompressed", "runs_0_100_200")
        c_dirname = os.path.join(synthetic_dir, vectype, "uncompressed", f"runs_rl_{run_length}_nnz_400")
    elif vectype == "blocks":
        # c_dirname = os.path.join(synthetic_dir, vectype, "uncompressed", "C_blocks_20_20")
        c_dirname = os.path.join(synthetic_dir, vectype, "uncompressed", f"C_blocks_400_{run_length}")

    c_vals_filename = os.path.join(c_dirname, "tensor_C_mode_vals")
    in_vec2 = read_inputs(c_vals_filename, float)

    # in_vec1 = [random.randint(0, max_val) for _ in range(dim1)]
    # in_vec2 = [random.randint(0, max_val) for _ in range(dim1)]

    if debug_sim:
        print("VECTOR 1:", in_vec1)
        print("VECTOR 2:", in_vec2)

    assert (len(in_vec1) == len(in_vec2))

    gold_vec = [in_vec1[i] * in_vec2[i] for i in range(len(in_vec1))]

    rdscan = UncompressCrdRdScan(dim=dim1, debug=debug_sim, back_en=backpressure, depth=int(depth))
    val1 = Array(init_arr=in_vec1, debug=debug_sim, back_en=backpressure, depth=int(depth))
    val2 = Array(init_arr=in_vec2, debug=debug_sim, back_en=backpressure, depth=int(depth))
    mul = Multiply2(debug=debug_sim, back_en=backpressure, dpeth=int(depth))
    wrscan = ValsWrScan(size=dim1, fill=fill, debug=debug_sim, back_en=backpressure, depth=int(depth))

    in_ref = [0, 'D']
    done = False
    time_cnt = 0
    while not done and time_cnt < TIMEOUT:
        if len(in_ref) > 0:
            rdscan.set_in_ref(in_ref.pop(0), "")
        val1.set_load(rdscan.out_ref(), rdscan)
        val2.set_load(rdscan.out_ref(), rdscan)
        mul.set_in1(val1.out_load(), val1)
        mul.set_in2(val2.out_load(), val2)
        wrscan.set_input(mul.out_val(), mul)

        rdscan.update()
        val1.update()
        val2.update()
        mul.update()
        wrscan.update()

        print("Timestep", time_cnt, "\t Done --", "\tRdScan:", rdscan.out_done(),
              "\tArr:", val1.out_done(), val2.out_done(),
              "\tMul:", mul.out_done(), "\tWrScan:", wrscan.out_done())
        done = wrscan.out_done()
        time_cnt += 1

    check_arr(wrscan, gold_vec)

    def bench():
        time.sleep(0.0001)

    extra_info = dict()
    extra_info["cycles_reformat"] = ""
    extra_info["cycles"] = time_cnt
    extra_info["vectype"] = vectype
    extra_info["sparsity"] = sparsity
    extra_info["run_length"] = run_length
    extra_info["block_size"] = run_length
    extra_info["format"] = "dense"
    extra_info["split_factor"] = 1
    extra_info["test_name"] = "test_unit_vec_elemmul_u_u_u"

    samBench(bench, extra_info)
