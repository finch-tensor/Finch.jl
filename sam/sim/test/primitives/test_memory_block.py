import copy
import pytest
import os
from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.channel import memory_block
from sam.sim.test.test import TIMEOUT
from sam.sim.src.base import remove_emptystr
import os
import csv
import pickle
import yaml
cwd = os.getcwd()
formatted_dir = os.getenv('TILED_SUITESPARSE_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))
sam_home = os.getenv('SAM_HOME')

################################
# Unit test for memory channel #
################################
arr_dict1 = {"seg": [0, 2, 3, 4], "crd": [0, 2, 2, 2], "in_ref": [0, 1, 2, 'S0', 'D'],
             "out_crd": [0, 2, 'S0', 2, 'S0', 2, 'S1', 'D'], "out_ref": [0, 1, 'S0', 2, 'S0', 3, 'S1', 'D']}
arr_dict2 = {"seg": [0, 3, 4, 6], "crd": [0, 2, 3, 0, 2, 3], "in_ref": [0, 0, 'S0', 1, 'S0', 2, 'S1', 'D'],
             "out_crd": [0, 2, 3, 'S0', 0, 2, 3, 'S1', 0, 'S1', 2, 3, 'S2', 'D'],
             "out_ref": [0, 1, 2, 'S0', 0, 1, 2, 'S1', 3, 'S1', 4, 5, 'S2', 'D']}
arr_dict3 = {"seg": [0, 4], "crd": [0, 1, 2, 3], "in_ref": [0, 'D'],
             "out_crd": [0, 1, 2, 3, 'S0', 'D'], "out_ref": [0, 1, 2, 3, 'S0', 'D']}
arr_dict4 = {"seg": [0, 1], "crd": [28], "in_ref": [0, 'S0', 'S0', 0, 'S0', 'D'],
             "out_crd": [28, 'S1', 'S1', 28, 'S1', 'D'], "out_ref": [0, 'S1', 'S1', 0, 'S1', 'D']}
arr_dict5 = {"seg": [0, 1], "crd": [28], "in_ref": [0, 'S0', '', '', 'S0', '',
                                                    '', 0, '', '', 'S0', '', '', 'D'],
             "out_crd": [28, 'S1', 'S1', 28, 'S1', 'D'], "out_ref": [0, 'S1', 'S1', 0, 'S1', 'D']}


@pytest.mark.parametrize("arrs", [arr_dict1, arr_dict2, arr_dict3, arr_dict4, arr_dict5])
def test_memory_block_nbuffer(arrs, debug_sim, skip_empty, yaml_name, report_stats, nbuffer):
    nbuffer = True
    seg_arr = arrs["seg"]
    crd_arr = arrs["crd"]

    gold_crd = arrs["out_crd"]
    gold_ref = arrs["out_ref"]
    assert (len(gold_crd) == len(gold_ref))
    with open(os.path.join(sam_home + "/sam/sim/src/tiling/", yaml_name), "r") as stream:
        memory_config = yaml.safe_load(stream)
    crdscan = CompressedCrdRdScan(seg_arr=seg_arr, crd_arr=crd_arr, debug=debug_sim)
    mem_blk = memory_block(name="GLB_B", skip_blocks=skip_empty, nbuffer=nbuffer,
                           element_size=memory_config["Bytes_per_element"], size=memory_config["Glb_memory"],
                           bandwidth=memory_config["Glb_tile_bandwidth"] // memory_config["Glb_tiles"],
                           latency=memory_config["Global_Glb_latency"],
                           debug=debug_sim, pipeline_en=True, statistics=report_stats)
    in_ref = copy.deepcopy(arrs["in_ref"])
    done = False
    time = 0
    out_crd = []
    out_ref = []
    out_gold = []
    while not done and time < TIMEOUT:
        if len(in_ref) > 0:
            crdscan.set_in_ref(in_ref.pop(0))
        crdscan.update()
        out_crd.append(crdscan.out_crd())
        out_ref.append(crdscan.out_ref())
        if isinstance(crdscan.out_ref(), int):
            out_gold.append(crdscan.out_ref())
        done = crdscan.done
        time += 1
    assert (remove_emptystr(out_crd) == gold_crd)
    assert (remove_emptystr(out_ref) == gold_ref)
    time = 0
    inp_arr = out_ref.copy()
    i = -1
    out = []
    done = False
    while not done and time < TIMEOUT:
        while len(inp_arr) > 0:
            if isinstance(inp_arr[0], int):
                mem_blk.add_tile(inp_arr.pop(0), 100)
            else:
                mem_blk.add_tile(inp_arr.pop(0), 8)
        if mem_blk.valid_tile():
            out.append(mem_blk.token())
            i = 0
            mem_blk.valid_tile_received()
        if i > -1:
            i += 1
        if i == 1:
            i = -1
            mem_blk.check_if_done(True)
        else:
            mem_blk.check_if_done(False)
        mem_blk.update(time)
        done = mem_blk.out_done()
        # mem_blk.print_debug()
        # print(i)
        time += 1
    # print(out_ref)
    # print(out)
    # print(out_gold)
    assert out_gold == out


@pytest.mark.parametrize("arrs", [arr_dict1, arr_dict2, arr_dict3, arr_dict4, arr_dict5])
def test_memory_block(arrs, debug_sim, skip_empty, yaml_name, report_stats, nbuffer):
    nbuffer = False
    seg_arr = arrs["seg"]
    crd_arr = arrs["crd"]
    gold_crd = arrs["out_crd"]
    gold_ref = arrs["out_ref"]
    assert (len(gold_crd) == len(gold_ref))
    with open(os.path.join(sam_home + "/sam/sim/src/tiling/", yaml_name), "r") as stream:
        memory_config = yaml.safe_load(stream)
    crdscan = CompressedCrdRdScan(seg_arr=seg_arr, crd_arr=crd_arr, debug=debug_sim)
    mem_blk = memory_block(name="GLB_B", skip_blocks=skip_empty, nbuffer=nbuffer,
                           element_size=memory_config["Bytes_per_element"], size=memory_config["Glb_memory"],
                           bandwidth=memory_config["Glb_tile_bandwidth"] // memory_config["Glb_tiles"],
                           latency=memory_config["Global_Glb_latency"],
                           debug=debug_sim, pipeline_en=True, statistics=report_stats)
    in_ref = copy.deepcopy(arrs["in_ref"])
    done = False
    time = 0
    out_crd = []
    out_ref = []
    out_gold = []
    while not done and time < TIMEOUT:
        if len(in_ref) > 0:
            crdscan.set_in_ref(in_ref.pop(0))

        crdscan.update()

        out_crd.append(crdscan.out_crd())
        out_ref.append(crdscan.out_ref())
        if isinstance(crdscan.out_ref(), int):
            out_gold.append(crdscan.out_ref())
        done = crdscan.done
        time += 1
    assert (remove_emptystr(out_crd) == gold_crd)
    assert (remove_emptystr(out_ref) == gold_ref)
    time = 0
    inp_arr = out_ref.copy()
    i = -1
    out = []
    done = False
    while not done and time < TIMEOUT:
        while len(inp_arr) > 0:
            if isinstance(inp_arr[0], int):
                mem_blk.add_tile(inp_arr.pop(0), 100)
            else:
                mem_blk.add_tile(inp_arr.pop(0), 8)
        if mem_blk.valid_tile():
            out.append(mem_blk.token())
            i = 0
            mem_blk.valid_tile_received()
        if i > -1:
            i += 1
        if i == 1:
            i = -1
            mem_blk.check_if_done(True)
        else:
            mem_blk.check_if_done(False)
        mem_blk.update(time)
        done = mem_blk.out_done()
        # mem_blk.print_debug()
        # print(i)
        time += 1

    # print(out_ref)
    # print(out)
    # print(out_gold)
    # print("_______")
    assert out_gold == out
