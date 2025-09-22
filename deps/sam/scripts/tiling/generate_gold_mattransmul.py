import scipy
import scipy.sparse
import os
import scipy.io
import numpy as np
import yaml
import math
import pickle
import argparse

from pathlib import Path
from scripts.util.util import round_sparse


def generate_gold_mattransmul_tiled(tile_crd_b, tile_crd_c, tile_crd_d, dirname, out_format="ss01"):
    # CSR
    formatted_dir = f"./tiles/mat_mattransmul/mtx"

    B_dir = "tensor_B_tile_"
    for a in tile_crd_b:
        B_dir += str(a) + "_"
    C_dir = "tensor_c_tile_"
    for a in tile_crd_c:
        C_dir += str(a) + "_"
    d_dir = "tensor_d_tile_"
    for a in tile_crd_d:
        d_dir += str(a) + "_"

    B_dir = B_dir[0:-1] + ".mtx"
    C_dir = C_dir[0:-1] + ".mtx"
    d_dir = d_dir[0:-1] + ".mtx"
    # print(B_dir, " ", C_dir)
    B_filename = os.path.join(formatted_dir, B_dir)
    C_filename = os.path.join(formatted_dir, C_dir)
    d_filename = os.path.join(formatted_dir, d_dir)
    # print()
    # print(B_filename)
    # print(C_filename)
    # print(d_filename)
    # print()
    if os.path.exists(B_filename) and os.path.exists(C_filename) and os.path.exists(d_filename):
        B_scipy = scipy.io.mmread(B_filename)
        itr = 0
        # print("\nB_scipy: ", B_scipy)
        for i, j, v in zip(B_scipy.row, B_scipy.col, B_scipy.data):
            # print(B_scipy.data)
            # print(i, " ", j, " ", v)
            B_scipy.data[itr] = round_sparse(B_scipy.data[itr])
            # if B_scipy.data[itr] < 1 and B_scipy.data[itr] > 0:
            #    B_scipy.data[itr] = 1
            # elif B_scipy.data[itr] < 0 and B_scipy.data[itr] > -1:
            #    B_scipy.data[itr] = -1
            # else:
            #    B_scipy.data[itr] = int(B_scipy.data[itr])
            itr += 1
        B_scipy = B_scipy.tocsr()

        C_scipy = scipy.io.mmread(C_filename)
        # print(C_filename)
        # print("\nC_scipy: ", C_scipy)
        # print("___________________")
        # print(B_scipy)
        itr = 0
        for i, j, v in zip(C_scipy.row, C_scipy.col, C_scipy.data):
            C_scipy.data[itr] = round_sparse(C_scipy.data[itr])
            itr += 1
        C_scipy = C_scipy.tocsr()
        C_scipy = np.transpose(C_scipy)

        d_scipy = scipy.io.mmread(d_filename)
        # print("\nd_scipy: ", d_scipy)

        itr = 0
        for i, j, v in zip(d_scipy.row, d_scipy.col, d_scipy.data):
            d_scipy.data[itr] = round_sparse(d_scipy.data[itr])

            itr += 1
        d_scipy = d_scipy.tocsr()
        d_scipy = np.transpose(d_scipy)

        # gold_nd = (B_scipy @ C_scipy)
        # gold_nd = B_scipy.dot(C_scipy)

        # constants
        alpha = 2
        beta = 2

        print("B_scipy.shape: ", B_scipy.shape)
        print("C_scipy.shape: ", C_scipy.shape)
        print("d_scipy.shape: ", d_scipy.shape)

        gold_nd = alpha * (B_scipy @ C_scipy) + beta * d_scipy
        # print(gold_nd)

        gold_out = gold_nd.tocoo()
        assert (tile_crd_b[1] == tile_crd_c[0] and tile_crd_b[3] == tile_crd_c[1] and
                tile_crd_b[0] == tile_crd_d[0] and tile_crd_b[2] == tile_crd_d[1])
        # assert tile_crd_b[1] == tile_crd_c[0] and tile_crd_b[3] == tile_crd_c[2]
        scipy.io.mmwrite(
            dirname + "out_" + str(tile_crd_b[0]) + "_" + str(tile_crd_b[1]) + "_" + str(tile_crd_b[3]) + "_" +
            str(tile_crd_b[2]) + "_" + str(tile_crd_c[0]) + "_" + str(tile_crd_c[1]) +
            "_" + str(tile_crd_d[0]) + "_" + str(tile_crd_d[1]) + ".mtx", gold_out)
    elif os.path.exists(d_filename):
        d_scipy = scipy.io.mmread(d_filename)
        # print("\nd_scipy: ", d_scipy)

        itr = 0
        for i, j, v in zip(d_scipy.row, d_scipy.col, d_scipy.data):
            d_scipy.data[itr] = d_scipy.data[itr]

            itr += 1
        d_scipy = d_scipy.tocsr()
        # d_scipy = np.transpose(d_scipy)

        # gold_nd = (B_scipy @ C_scipy)
        # gold_nd = B_scipy.dot(C_scipy)

        # constants
        alpha = 2
        beta = 2

        # print(d_scipy.todense())
        gold_nd = beta * d_scipy
        # print(gold_nd)
        if (np.count_nonzero(gold_nd.todense()) == 0):
            print("output is all zero")
            return

        gold_out = gold_nd.tocoo()
        # assert tile_crd_b[1] == tile_crd_c[0] and tile_crd_b[3] == tile_crd_c[1]
        # and tile_crd_b[0] == tile_crd_d[0] and tile_crd_b[2] == tile_crd_d[1]
        # assert tile_crd_b[1] == tile_crd_c[0] and tile_crd_b[3] == tile_crd_c[2]
        scipy.io.mmwrite(
            dirname + "out_" + str(tile_crd_b[0]) + "_" + str(tile_crd_b[1]) +
            "_" + str(tile_crd_b[3]) + "_" + str(tile_crd_b[2]) + "_" +
            str(tile_crd_c[0]) + "_" + str(tile_crd_c[1]) + "_" +
            str(tile_crd_d[0]) + "_" + str(tile_crd_d[1]) + ".mtx", gold_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate tiled output gold")
    parser.add_argument("--yaml_name", type=str, default="memory_config_real.yaml")
    args = parser.parse_args()
    outdir = f"./tiles/mat_mattransmul/output/"
    outpath = Path(outdir)
    outpath.mkdir(parents=True)

    # generate_gold_matmul_tiled([0, 1, 2, 9], [1, 0, 9, 0], outdir)

    # generate_gold_matmul_tiled([0, 1, 0, 7], [1, 0, 7, 0], outdir)
    # quit()    with open("/nobackup/rsharma3/Sparsity/simulator/old_sam/sam/tiles/matmul_ikj/tensor_sizes", "rb") as ff:

    with open(f"./tiles/mat_mattransmul/tensor_sizes", "rb") as ff:
        sizes_dict_level_full = pickle.load(ff)

    with open("./sam/sim/src/tiling/" + args.yaml_name, "r") as stream:
        loop_config = yaml.safe_load(stream)

    print()
    print("sizes_dict_level_full", sizes_dict_level_full)
    print()
    print("loop_config", loop_config)

    struct = {
        "j00": 1 + int(sizes_dict_level_full["B"][0]) // (loop_config["Glb_tile_size"] * loop_config["Mem_tile_size"]),
        "i00": 1 + int(sizes_dict_level_full["c"][0]) // (loop_config["Glb_tile_size"] * loop_config["Mem_tile_size"]),
        "i0": loop_config["Glb_tile_size"], "j0": loop_config["Glb_tile_size"]}

    print()
    print(struct)

    # print(struct)
    # # quit()
    for i00 in range(struct["i00"]):
        for j00 in range(struct["j00"]):
            for i0 in range(struct["i0"]):
                for j0 in range(struct["j0"]):
                    generate_gold_mattransmul_tiled([j00, i00, j0, i0], [i00, i0], [j00, j0], outdir)
