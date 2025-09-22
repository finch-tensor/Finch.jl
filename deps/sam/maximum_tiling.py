import glob
import sys
import numpy as np
import scipy
import os
import re


class EarlyReturn():
    pass


def get_files_from_dir(path, operands):
    operand_files = {}
    for operand in operands:
        operand_files[operand] = glob.glob(os.path.join(path, f"*{operand}*.mtx"))

    return operand_files


def get_tile_id(string):
    indices = [m.start() for m in re.finditer("tile", string)]
    if len(indices) >= 2:
        substring = string[indices[1] + len("tile") + 1:]
        substring = substring.rstrip(".mtx")
        numbers = substring.split("_")
    return numbers


def pair_tiles(app_name):
    path = f"tiles/{app_name}/mtx"
    tile_pairing = {}

    operands = []
    if "matmul" in app_name:
        operands = ["B", "C"]
        operand_files = get_files_from_dir(path, operands)
        b_tensors = operand_files["B"]
        c_tensors = operand_files["C"]

        tile = 0
        for b in b_tensors:
            for c in c_tensors:
                b_loc = get_tile_id(b)
                c_loc = get_tile_id(c)
                if (b_loc[1] == c_loc[0] and b_loc[3] == c_loc[2]):
                    tile_pairing[tile] = [b, c]
                    tile += 1
    elif "elemmul" in app_name:
        operands = ["B", "C"]
        operand_files = get_files_from_dir(path, operands)
        b_tensors = operand_files["B"]
        c_tensors = operand_files["C"]

        tile = 0
        for b in b_tensors:
            for c in c_tensors:
                b_loc = get_tile_id(b)
                c_loc = get_tile_id(c)
                if (b_loc == c_loc):
                    tile_pairing[tile] = [b, c]
                    tile += 1
    elif "elemadd3" in app_name:
        operands = ["B", "C", "D"]
        operand_files = get_files_from_dir(path, operands)
        b_tensors = operand_files["B"]
        c_tensors = operand_files["C"]
        d_tensors = operand_files["D"]

        tile = 0
        for b in b_tensors:
            for c in c_tensors:
                b_loc = get_tile_id(b)
                c_loc = get_tile_id(c)
                if (b_loc != c_loc):
                    continue

                for d in d_tensors:
                    d_loc = get_tile_id(d)
                    if (b_loc == c_loc and c_loc == d_loc):
                        tile_pairing[tile] = [b, c, d]
                        tile += 1

    elif "mat_mask_tri" in app_name:
        operands = ["B", "C", "D"]
        operand_files = get_files_from_dir(path, operands)
        b_tensors = operand_files["B"]
        c_tensors = operand_files["C"]
        d_tensors = operand_files["D"]

        tile = 0
        for b in b_tensors:
            for c in c_tensors:
                b_loc = get_tile_id(b)
                c_loc = get_tile_id(c)
                if not (b_loc[0] == c_loc[0] and b_loc[2] == c_loc[2]):
                    continue

                for d in d_tensors:
                    d_loc = get_tile_id(d)
                    if (c_loc[1] == d_loc[0] and c_loc[3] == d_loc[2] and b_loc[1] == d_loc[1] and
                            b_loc[3] == d_loc[3] and b_loc[0] == c_loc[0] and b_loc[2] == c_loc[2]):
                        tile_pairing[tile] = [b, c, d]
                        tile += 1
    elif "mat_vecmul_iter" in app_name:
        operands = ["B", "C", "D", "E", "f"]
        operand_files = get_files_from_dir(path, operands)
        b_tensors = operand_files["B"]
        c_tensors = operand_files["C"]
        d_tensors = operand_files["D"]
        e_tensors = operand_files["E"]
        f_tensors = operand_files["f"]

        tile = 0

        for b in b_tensors:
            for c in c_tensors:
                b_loc = get_tile_id(b)
                c_loc = get_tile_id(c)
                if not (b_loc[1] == c_loc[0] and b_loc[3] == c_loc[2]):
                    continue
                for d in d_tensors:
                    d_loc = get_tile_id(d)
                    # check k coord
                    if not (c_loc[1] == d_loc[0] and c_loc[3] == d_loc[2]):
                        continue
                    for e in e_tensors:
                        e_loc = get_tile_id(e)
                        # check l coord
                        if not (d_loc[1] == e_loc[0] and d_loc[3] == e_loc[2]):
                            continue
                        for f in f_tensors:
                            f_loc = get_tile_id(f)
                            if (d_loc[1] == e_loc[0] and d_loc[3] == e_loc[2] and
                                c_loc[1] == d_loc[0] and c_loc[3] == d_loc[2] and
                                b_loc[1] == c_loc[0] and b_loc[3] == c_loc[2] and
                                    e_loc[1] == f_loc[0] and e_loc[3] == f_loc[1]):
                                tile_pairing[tile] = [b, c, d, e, f]
                                tile += 1

    return tile_pairing


def read_mtx(mtx_path):
    matrix = scipy.io.mmread(mtx_path)
    arr = np.array(matrix.todense())
    return arr


def compute_outputs(tile_pairing, app_name, limit=900):
    for key, value in tile_pairing.items():
        if "matmul" in app_name:
            B_mat = read_mtx(value[0])
            C_mat = read_mtx(value[1])
            C_mat = np.transpose(C_mat)
            out = np.matmul(B_mat, C_mat)
            if np.count_nonzero(out) > limit or np.count_nonzero(B_mat) > limit or np.count_nonzero(C_mat) > limit:
                # if np.any(out):
                print("tile = ", key)
                print("B_tile_ID = ", value[0])
                print("C_tile_ID = ", value[1])
                print("out = ", out)
                print("count = ", np.count_nonzero(out))
                return EarlyReturn()
        elif "elemmul" in app_name:
            B_mat = read_mtx(value[0])
            C_mat = read_mtx(value[1])
            out = np.multiply(B_mat, C_mat)
            # if np.any(out):
            if np.count_nonzero(out) > limit or np.count_nonzero(B_mat) > limit or np.count_nonzero(C_mat) > limit:
                # if np.count_nonzero(out) > limit or (np.count_nonzero(B_mat) + np.count_nonzero(C_mat)) > limit:
                print("tile = ", key)
                print("B_tile_ID = ", value[0])
                print("C_tile_ID = ", value[1])
                print("out = ", out)
                print("count = ", np.count_nonzero(out))
                return EarlyReturn()
        elif "elemadd3" in app_name:
            B_mat = read_mtx(value[0])
            C_mat = read_mtx(value[1])
            D_mat = read_mtx(value[2])

            out = np.add(np.add(B_mat, C_mat), D_mat)
            # if np.any(out):
            if np.count_nonzero(out) > limit or np.count_nonzero(B_mat) > limit or np.count_nonzero(
                    C_mat) > limit or np.count_nonzero(D_mat) > limit:
                # if np.count_nonzero(out) > limit or (np.count_nonzero(B_mat) + np.count_nonzero(C_mat)) > limit:
                print("tile = ", key)
                print("B_tile_ID = ", value[0])
                print("C_tile_ID = ", value[1])
                print("D_tile_ID = ", value[2])
                print("out = ", out)
                print("count = ", np.count_nonzero(out))
                return EarlyReturn()
        elif "mat_mask_tri" in app_name:
            B_mat = read_mtx(value[0])
            C_mat = read_mtx(value[1])
            D_mat = read_mtx(value[2])
            D_mat = np.transpose(D_mat)
            out = np.sum(np.multiply(np.matmul(C_mat, D_mat), B_mat))
            if np.count_nonzero(out) > limit or np.count_nonzero(B_mat) > limit or np.count_nonzero(
                    C_mat) > limit or np.count_nonzero(D_mat) > limit:
                print("tile = ", key)
                print("B_tile_ID = ", value[0])
                print("C_tile_ID = ", value[1])
                print("D_tile_ID = ", value[2])
                print("out = ", out)
                print("count = ", np.count_nonzero(out))
                return EarlyReturn()
        elif "mat_vecmul_iter" in app_name:
            B_mat = read_mtx(value[0])
            C_mat = read_mtx(value[1])
            D_mat = read_mtx(value[2])
            E_mat = read_mtx(value[3])
            f_mat = read_mtx(value[4])
            # we transpose bc we swap in copy formatted
            f_mat = np.transpose(f_mat)
            out = np.matmul(np.matmul(np.matmul(np.matmul(B_mat, C_mat), D_mat), E_mat), f_mat)
            if np.any(out):
                # if np.count_nonzero(out) > limit or np.count_nonzero(B_mat) > limit or
                # np.count_nonzero(C_mat) > limit or np.count_nonzero(D_mat) > limit or
                # np.count_nonzero(E_mat) > limit or np.count_nonzero(f_mat) > limit:
                print("tile = ", key)
                print("B_tile_ID = ", value[0])
                print("C_tile_ID = ", value[1])
                print("D_tile_ID = ", value[2])
                print("E_tile_ID = ", value[3])
                print("f_tile_ID = ", value[4])
                print("out = ", out)
                print("count = ", np.count_nonzero(out))
                breakpoint()
                return EarlyReturn()
    return None


def find_optimal_tilesize(app_name, datum, initial=30, step_size=10):
    tile_size = initial
    max_tile_size = initial
    prev_tile_pairing = None

    # while True:
    for _ in range(50):
        call_tiling = f"python3 setup_tiling_mat.py {app_name} {datum} {tile_size} > temp.txt"
        os.system(call_tiling)
        print(call_tiling)

        tile_pairing = pair_tiles(app_name)
        exit_status = compute_outputs(tile_pairing, app_name)
        if isinstance(exit_status, EarlyReturn):
            max_tile_size = tile_size - step_size
            return max_tile_size, prev_tile_pairing

        tile_size += step_size
        print("***********************")
        print("tile size = ", tile_size)
        print("***********************")
        prev_tile_pairing = tile_pairing

    return tile_size, prev_tile_pairing


if __name__ == "__main__":
    max_list = {}
    # for i in range(1, 11):
    app_name = "matmul_ijk"
    datum = "N_biocarta"

    # tile_pairing = pair_tiles(app_name)
    # compute_outputs(tile_pairing, app_name)

    max_tile_size, tile_pairing = find_optimal_tilesize(app_name, datum, initial=40, step_size=10)
    print("-" * 20)
    print(f"MAX TILESIZE for {app_name}, {datum}: {max_tile_size}")
    print(f"NUMBER OF TILES: {len(tile_pairing.keys())}")
    print("-" * 20)

    max_list[datum] = [max_tile_size, len(tile_pairing.keys())]

    call_tiling = f"python3 setup_tiling_mat.py {app_name} {datum} {max_tile_size} > temp.txt"
    os.system(call_tiling)
    print(call_tiling)

    # print(max_list)
