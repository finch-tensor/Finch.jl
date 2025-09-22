import shutil
import glob
import subprocess
import os
import json

# test = "bcsstm26"
# test = "rel5"
test = "qiulp"
# test = "adder_dcop_30"
# test = "n4c6-b1"
# app_name = "mat_residual"
# app_name = "matmul_ijk"
# app_name = "matmul_ijk"
# app_name = "mat_mattrpython3ansmul"
app_name = "mat_elemmul"
const_val = 2  # only for mat_mattransmul


tiles_accumulation = {}

b_tensors = glob.glob(f"/home/avb03/sam/tiles/{app_name}/formatted/tensor_B*")
c_tensors = glob.glob(f"/home/avb03/sam/tiles/{app_name}/formatted/tensor_C*")
d_tensors = glob.glob(f"/home/avb03/sam/tiles/{app_name}/formatted/tensor_D*")

print("b_tensors: ", b_tensors)
print("c_tensors: ", c_tensors)
print("d_tensors: ", d_tensors)

# b_tensors = glob.glob(f"/aha/garnet/tiles_{app_name}_{test}/formatted/tensor_B*")
# c_tensors = glob.glob(f"/aha/garnet/tiles_{app_name}_{test}/formatted/tensor_C*")

b_vec_tensors = glob.glob(f"/aha/garnet/tiles_{app_name}_{test}/{app_name}/formatted/tensor_b*")
c_vec_tensors = glob.glob(f"/home/avb03/sam/tiles/{app_name}/formatted/tensor_c*")
print("c_vec_tensors: ", c_vec_tensors)
d_vec_tensors = glob.glob(f"/aha/garnet/tiles_{app_name}_{test}/{app_name}/formatted/tensor_d*")

d_loc_paired = []
b_loc_paired = []

if not os.path.exists("SPARSE_TESTS/MAT_TMP_DIR"):
    os.makedirs("SPARSE_TESTS/MAT_TMP_DIR")

os.system(f"rm -rf SPARSE_TESTS/{app_name}*")
os.system(f"rm -rf SPARSE_TESTS/MAT_TMP_DIR/tile*")

tile = 0

os.chdir("SPARSE_TESTS")

if app_name == "matmul_ijk":
    for b in b_tensors:
        for c in c_tensors:
            tile_str = "tile" + str(tile)
            b_loc = b[-7:]
            c_loc = c[-7:]
            b_loc = b_loc.split("_")
            c_loc = c_loc.split("_")
            if (b_loc[1] == c_loc[0] and b_loc[3] == c_loc[2]):
                print(b, c)

                if b_loc[2] not in tiles_accumulation:
                    tiles_accumulation[b_loc[2]] = []

                tiles_accumulation[b_loc[2]].append(tile_str)

                if not os.path.exists(f"./MAT_TMP_DIR/{tile_str}"):
                    os.mkdir(f"./MAT_TMP_DIR/{tile_str}")

                shutil.copy(f"{b}/B0_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_0_crd")
                shutil.copy(f"{b}/B0_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_0_seg")

                shutil.copy(f"{b}/B1_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_1_crd")
                shutil.copy(f"{b}/B1_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_1_seg")

                shutil.copy(f"{b}/B_vals.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_vals")

                shutil.copy(f"{b}/B_shape.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_shape")

                shutil.copy(f"{c}/C0_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_1_crd")
                shutil.copy(f"{c}/C0_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_1_seg")

                shutil.copy(f"{c}/C1_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_0_crd")
                shutil.copy(f"{c}/C1_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_0_seg")

                shutil.copy(f"{c}/C_vals.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_vals")

                shutil.copy(f"{c}/C_shape.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_shape")

                tile = tile + 1
elif app_name == "mat_elemadd" or app_name == "mat_elemmul":
    for b in b_tensors:
        for c in c_tensors:
            tile_str = "tile" + str(tile)
            b_loc = b[-7:]
            c_loc = c[-7:]
            b_loc = b_loc.split("_")
            c_loc = c_loc.split("_")
            if (b_loc == c_loc):
                print(b, c)
                if not os.path.exists(f"./MAT_TMP_DIR/{tile_str}"):
                    os.mkdir(f"./MAT_TMP_DIR/{tile_str}")
                shutil.copy(f"{b}/B0_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_0_crd")
                shutil.copy(f"{b}/B0_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_0_seg")

                shutil.copy(f"{b}/B1_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_1_crd")
                shutil.copy(f"{b}/B1_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_1_seg")

                shutil.copy(f"{b}/B_vals.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_vals")

                shutil.copy(f"{b}/B_shape.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_shape")

                shutil.copy(f"{c}/C0_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_0_crd")
                shutil.copy(f"{c}/C0_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_0_seg")

                shutil.copy(f"{c}/C1_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_1_crd")
                shutil.copy(f"{c}/C1_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_1_seg")

                shutil.copy(f"{c}/C_vals.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_vals")

                shutil.copy(f"{c}/C_shape.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_shape")

                # subprocess.call(["aha",
                #     "regress",
                #     "fast"],
                #     text=True)

                tile = tile + 1
                # print("we are on tile ", tile)
elif app_name == "mat_mattransmul":
    for b in b_tensors:
        for c in c_vec_tensors:
            for d in d_vec_tensors:
                tile_str = "tile" + str(tile)
                b_loc = b[-7:]
                c_loc = c[-3:]
                d_loc = d[-3:]

                b_loc = b_loc.split("_")
                c_loc = c_loc.split("_")
                d_loc = d_loc.split("_")

                if (b_loc[1] == c_loc[0] and b_loc[3] == c_loc[1] and b_loc[0] == d_loc[0] and b_loc[2] == d_loc[1]):
                    # if(b_loc[1] == d_loc[0] and b_loc[3] == d_loc[1] and b_loc[0] == c_loc[0] and b_loc[2] == c_loc[1]):
                    d_loc_paired.append(d_loc)

                    print(f"\n ----- TILE {tile} ----- \n")
                    print("B is: ", b)  # in build_tb, B == C, c == d, d == f. (#FIXME: change build_tb)
                    print("C is: ", c)
                    print("d is: ", d)
                    print(f"\n ----- TILE {tile} ----- \n")
                    if not os.path.exists(f"./MAT_TMP_DIR/{tile_str}"):
                        os.mkdir(f"./MAT_TMP_DIR/{tile_str}")

                    shutil.copy(f"{b}/B0_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_1_crd")
                    shutil.copy(f"{b}/B0_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_1_seg")

                    shutil.copy(f"{b}/B1_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_0_crd")
                    shutil.copy(f"{b}/B1_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_0_seg")

                    shutil.copy(f"{b}/B_vals.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_vals")

                    shutil.copy(f"{b}/B_shape.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_shape")

                    shutil.copy(f"{c}/c1_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_d_mode_1_crd")
                    shutil.copy(f"{c}/c1_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_d_mode_1_seg")

                    shutil.copy(f"{c}/c0_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_d_mode_0_crd")
                    shutil.copy(f"{c}/c0_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_d_mode_0_seg")

                    shutil.copy(f"{d}/d1_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_f_mode_1_crd")
                    shutil.copy(f"{d}/d1_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_f_mode_1_seg")

                    shutil.copy(f"{d}/d0_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_f_mode_0_crd")
                    shutil.copy(f"{d}/d0_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_f_mode_0_seg")

                    shutil.copy(f"{c}/c_vals.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_d_mode_vals")
                    shutil.copy(f"{c}/c_shape.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_d_mode_shape")

                    shutil.copy(f"{d}/d_vals.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_f_mode_vals")
                    shutil.copy(f"{d}/d_shape.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_f_mode_shape")

                    with open(f"./MAT_TMP_DIR/{tile_str}/tensor_b_mode_vals", 'w') as file:
                        file.write(str(const_val))

                    with open(f"./MAT_TMP_DIR/{tile_str}/tensor_e_mode_vals", 'w') as file:
                        file.write(str(const_val))

                    tile = tile + 1
                elif d_loc not in d_loc_paired:
                    # case: B and c tiles are zero but d is nonzero. We have all d tiles. Just
                    # take a B and c tile, copy it and make it zero.'
                    d_loc_paired.append(d_loc)
                    print(f"\n ----- TILE D-unpaired {tile} ----- \n")
                    print("B (zero tile) is: ", b)  # in build_tb, B == C, c == d, d == f. (#FIXME: change build_tb)
                    print("C (zero tile) is: ", c)
                    print("d is: ", d)
                    print(f"\n ----- TILE D-unpaired {tile} ----- \n")

                    if not os.path.exists(f"./MAT_TMP_DIR/{tile_str}"):
                        os.mkdir(f"./MAT_TMP_DIR/{tile_str}")

                    shutil.copy(f"{b}/B0_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_1_crd")
                    shutil.copy(f"{b}/B0_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_1_seg")

                    shutil.copy(f"{b}/B1_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_0_crd")
                    shutil.copy(f"{b}/B1_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_0_seg")

                    shutil.copy(f"{b}/B_vals.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_vals")

                    # clear out C vals
                    with open(f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_vals", 'r+') as file:
                        contents = file.read()
                        contents = contents.replace(contents, str(0))
                        file.seek(0)
                        file.write(contents)

                    shutil.copy(f"{b}/B_shape.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_shape")

                    shutil.copy(f"{c}/c1_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_d_mode_1_crd")
                    shutil.copy(f"{c}/c1_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_d_mode_1_seg")

                    shutil.copy(f"{c}/c0_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_d_mode_0_crd")
                    shutil.copy(f"{c}/c0_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_d_mode_0_seg")

                    shutil.copy(f"{d}/d1_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_f_mode_1_crd")
                    shutil.copy(f"{d}/d1_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_f_mode_1_seg")

                    shutil.copy(f"{d}/d0_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_f_mode_0_crd")
                    shutil.copy(f"{d}/d0_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_f_mode_0_seg")

                    shutil.copy(f"{c}/c_vals.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_d_mode_vals")

                    # clear out d vals
                    with open(f"./MAT_TMP_DIR/{tile_str}/tensor_d_mode_vals", 'r+') as file:
                        contents = file.read()
                        contents = contents.replace(contents, str(0))
                        file.seek(0)
                        file.write(contents)

                    shutil.copy(f"{c}/c_shape.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_d_mode_shape")

                    shutil.copy(f"{d}/d_vals.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_f_mode_vals")
                    shutil.copy(f"{d}/d_shape.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_f_mode_shape")

                    with open(f"./MAT_TMP_DIR/{tile_str}/tensor_b_mode_vals", 'w') as file:
                        file.write(str(const_val))

                    with open(f"./MAT_TMP_DIR/{tile_str}/tensor_e_mode_vals", 'w') as file:
                        file.write(str(const_val))

                    tile = tile + 1
    print("d_loc_paired: ", d_loc_paired)
elif app_name == "mat_vecmul_ij":
    for b in b_tensors:
        for c in c_vec_tensors:
            tile_str = "tile" + str(tile)
            b_loc = b[-7:]
            c_loc = c[-3:]

            b_loc = b_loc.split("_")
            c_loc = c_loc.split("_")

            # if(b_loc[1] == c_loc[0] and b_loc[3] == c_loc[1] and b_loc[0] == d_loc[0] and b_loc[2] == d_loc[1]):
            if (b_loc[1] == c_loc[0] and b_loc[3] == c_loc[1]):
                print(b, c)
                if not os.path.exists(f"./MAT_TMP_DIR/{tile_str}"):
                    os.mkdir(f"./MAT_TMP_DIR/{tile_str}")

                shutil.copy(f"{b}/B0_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_0_crd")
                shutil.copy(f"{b}/B0_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_0_seg")

                shutil.copy(f"{b}/B1_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_1_crd")
                shutil.copy(f"{b}/B1_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_1_seg")

                shutil.copy(f"{b}/B_vals.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_vals")

                shutil.copy(f"{b}/B_shape.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_shape")

                # shutil.copy(f"{c}/c1_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_c_mode_1_crd")
                # shutil.copy(f"{c}/c1_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_c_mode_1_seg")

                shutil.copy(f"{c}/c0_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_c_mode_0_crd")
                shutil.copy(f"{c}/c0_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_c_mode_0_seg")

                shutil.copy(f"{c}/c_vals.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_c_mode_vals")
                shutil.copy(f"{c}/c_shape.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_c_mode_shape")

                tile = tile + 1
elif app_name == "mat_residual":
    for b in b_vec_tensors:
        for c in c_tensors:
            for d in d_vec_tensors:
                tile_str = "tile" + str(tile)
                b_loc = b[-3:]
                c_loc = c[-7:]
                d_loc = d[-3:]

                b_loc = b_loc.split("_")
                c_loc = c_loc.split("_")
                d_loc = d_loc.split("_")

                # if(b_loc[1] == c_loc[0] and b_loc[3] == c_loc[1] and b_loc[0] == d_loc[0] and b_loc[2] == d_loc[1]):
                if (c_loc[0] == b_loc[0] and c_loc[2] == b_loc[1] and c_loc[1] == d_loc[0] and c_loc[3] == d_loc[1]):
                    print(b, c, d)
                    b_loc_paired.append(b_loc)

                    if not os.path.exists(f"./MAT_TMP_DIR/{tile_str}"):
                        os.mkdir(f"./MAT_TMP_DIR/{tile_str}")

                    shutil.copy(f"{c}/C0_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_0_crd")
                    shutil.copy(f"{c}/C0_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_0_seg")

                    shutil.copy(f"{c}/C1_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_1_crd")
                    shutil.copy(f"{c}/C1_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_1_seg")

                    shutil.copy(f"{c}/C_vals.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_vals")

                    shutil.copy(f"{c}/C_shape.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_shape")

                    shutil.copy(f"{b}/b1_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_b_mode_1_crd")
                    shutil.copy(f"{b}/b1_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_b_mode_1_seg")

                    shutil.copy(f"{b}/b0_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_b_mode_0_crd")
                    shutil.copy(f"{b}/b0_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_b_mode_0_seg")

                    shutil.copy(f"{d}/d1_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_d_mode_1_crd")
                    shutil.copy(f"{d}/d1_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_d_mode_1_seg")

                    shutil.copy(f"{d}/d0_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_d_mode_0_crd")
                    shutil.copy(f"{d}/d0_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_d_mode_0_seg")

                    shutil.copy(f"{b}/b_vals.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_b_mode_vals")
                    shutil.copy(f"{b}/b_shape.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_b_mode_shape")

                    shutil.copy(f"{d}/d_vals.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_d_mode_vals")
                    shutil.copy(f"{d}/d_shape.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_d_mode_shape")

                    tile = tile + 1
                elif b_loc not in b_loc_paired:
                    b_loc_paired.append(b_loc)

                    if not os.path.exists(f"./MAT_TMP_DIR/{tile_str}"):
                        os.mkdir(f"./MAT_TMP_DIR/{tile_str}")

                    shutil.copy(f"{c}/C0_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_0_crd")
                    shutil.copy(f"{c}/C0_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_0_seg")

                    shutil.copy(f"{c}/C1_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_1_crd")
                    shutil.copy(f"{c}/C1_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_1_seg")

                    shutil.copy(f"{c}/C_vals.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_vals")

                    # clear out C vals
                    with open(f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_vals", 'r+') as file:
                        contents = file.read()
                        contents = contents.replace(contents, str(0))
                        file.seek(0)
                        file.write(contents)

                    shutil.copy(f"{c}/C_shape.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_shape")

                    shutil.copy(f"{b}/b1_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_b_mode_1_crd")
                    shutil.copy(f"{b}/b1_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_b_mode_1_seg")

                    shutil.copy(f"{b}/b0_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_b_mode_0_crd")
                    shutil.copy(f"{b}/b0_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_b_mode_0_seg")

                    shutil.copy(f"{d}/d1_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_d_mode_1_crd")
                    shutil.copy(f"{d}/d1_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_d_mode_1_seg")

                    shutil.copy(f"{d}/d0_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_d_mode_0_crd")
                    shutil.copy(f"{d}/d0_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_d_mode_0_seg")

                    shutil.copy(f"{b}/b_vals.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_b_mode_vals")
                    shutil.copy(f"{b}/b_shape.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_b_mode_shape")

                    shutil.copy(f"{d}/d_vals.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_d_mode_vals")
                    shutil.copy(f"{d}/d_shape.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_d_mode_shape")

                    # clear out d vals
                    with open(f"./MAT_TMP_DIR/{tile_str}/tensor_d_mode_vals", 'r+') as file:
                        contents = file.read()
                        contents = contents.replace(contents, str(0))
                        file.seek(0)
                        file.write(contents)

                    tile = tile + 1

elif app_name == "mat_sddmm":
    for b in b_tensors:
        for c in c_tensors:
            for d in d_tensors:
                tile_str = "tile" + str(tile)

                b_loc = b[-7:]
                c_loc = c[-7:]
                d_loc = d[-7:]

                b_loc = b_loc.split("_")
                c_loc = c_loc.split("_")
                d_loc = d_loc.split("_")

                # first j, then i (k is a free coordinate)
                if (b_loc[0] == d_loc[1] and b_loc[2] == d_loc[3] and b_loc[1] == c_loc[0] and b_loc[3] == c_loc[2]):
                    print(b, c, d)
                    if not os.path.exists(f"./MAT_TMP_DIR/{tile_str}"):
                        os.mkdir(f"./MAT_TMP_DIR/{tile_str}")

                    shutil.copy(f"{b}/B0_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_0_crd")
                    shutil.copy(f"{b}/B0_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_0_seg")

                    shutil.copy(f"{b}/B1_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_1_crd")
                    shutil.copy(f"{b}/B1_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_1_seg")

                    shutil.copy(f"{b}/B_vals.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_vals")

                    shutil.copy(f"{b}/B_shape.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_shape")

                    shutil.copy(f"{c}/C0_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_1_crd")
                    shutil.copy(f"{c}/C0_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_1_seg")

                    shutil.copy(f"{c}/C1_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_0_crd")
                    shutil.copy(f"{c}/C1_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_0_seg")

                    shutil.copy(f"{c}/C_vals.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_vals")

                    shutil.copy(f"{c}/C_shape.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_shape")

                    tile = tile + 1

elif app_name == "mat_elemadd3":
    for b in b_tensors:
        for c in c_tensors:
            for d in d_tensors:
                tile_str = "tile" + str(tile)
                b_loc = b[-7:]
                c_loc = c[-7:]
                b_loc = b_loc.split("_")
                c_loc = c_loc.split("_")
                d_loc = d[-7:]
                d_loc = d_loc.split("_")

                # if(b_loc == c_loc and b_loc != d_loc):
                #     b_equal_c_no_d += 1
                # if(c_loc == d_loc and b_loc != c_loc):
                #     c_equal_d_no_b += 1
                # if(b_loc == d_loc and b_loc != c_loc):
                #     b_equal_d_no_c += 1

                if (b_loc == c_loc and b_loc == d_loc):
                    print(b, c, d)
                    if not os.path.exists(f"./MAT_TMP_DIR/{tile_str}"):
                        os.mkdir(f"./MAT_TMP_DIR/{tile_str}")
                    shutil.copy(f"{b}/B0_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_0_crd")
                    shutil.copy(f"{b}/B0_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_0_seg")

                    shutil.copy(f"{b}/B1_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_1_crd")
                    shutil.copy(f"{b}/B1_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_1_seg")

                    shutil.copy(f"{b}/B_vals.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_vals")

                    shutil.copy(f"{b}/B_shape.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_B_mode_shape")

                    shutil.copy(f"{c}/C0_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_0_crd")
                    shutil.copy(f"{c}/C0_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_0_seg")

                    shutil.copy(f"{c}/C1_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_1_crd")
                    shutil.copy(f"{c}/C1_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_1_seg")

                    shutil.copy(f"{c}/C_vals.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_vals")

                    shutil.copy(f"{c}/C_shape.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_C_mode_shape")

                    shutil.copy(f"{d}/D0_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_D_mode_0_crd")
                    shutil.copy(f"{d}/D0_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_D_mode_0_seg")

                    shutil.copy(f"{d}/D1_crd.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_D_mode_1_crd")
                    shutil.copy(f"{d}/D1_seg.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_D_mode_1_seg")

                    shutil.copy(f"{d}/D_vals.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_D_mode_vals")

                    shutil.copy(f"{d}/D_shape.txt", f"./MAT_TMP_DIR/{tile_str}/tensor_D_mode_shape")

                    # subprocess.call(["aha",
                    #     "regress",
                    #     "fast"],
                    #     text=True)

                    tile = tile + 1
                    # print("we are on tile ", tile)

print("tiles_accumulation: ", tiles_accumulation)

with open("../tiles_accumulation.json", "w") as file:
    json.dump(tiles_accumulation, file)

print("there are ", tile, " tiles")
