import subprocess
import glob
import shutil
import os
import re
import sys

from sam.util import SUITESPARSE_PATH

# Usage: python3 setup_tiling_mat.py <app_name> <data_file> <tile_size> <docker_path>


# PARAMS
data = [sys.argv[2]]
tilesizes = [int(sys.argv[3])]
app_name = sys.argv[1]
docker_path = sys.argv[4]

print("TILESIZES: ", tilesizes)
print("DATA: ", data)


def write_to_line(file_path, line_number, new_content):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    if line_number > len(lines) or line_number < 1:
        # Line number is out of range
        return

    lines[line_number - 1] = new_content + '\n'

    with open(file_path, 'w') as file:
        file.writelines(lines)


def replace_ones_with_zeros(mtx_file):
    with open(mtx_file, 'r') as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        values = line.split()
        if len(values) >= 3:
            values[2] = '0'
        new_lines.append(' '.join(values))

    with open(mtx_file, 'w') as file:
        file.writelines(new_lines)


i = 0
for datum in data:
    tilesize = tilesizes[i]

    yaml_file = "sam/sim/src/tiling/memory_config_onyx.yaml"
    mem_tile_line = f"Mem_tile_size: {tilesize}"
    print(mem_tile_line)
    write_to_line(yaml_file, 19, mem_tile_line)

    rmdir = f"rm -rf tiles/{app_name}"
    os.system(rmdir)

    print(f"{SUITESPARSE_PATH}/{datum}.mtx")
    mtx_file = glob.glob(f"{SUITESPARSE_PATH}/{datum}.mtx")[0]
    os.makedirs("extensor_mtx", exist_ok=True)
    shutil.copyfile(mtx_file, f"extensor_mtx/{datum}.mtx")

    command = f"./scripts/suitesparse_memory_model_runner.sh {datum} {app_name}"
    os.system(command)

    docker_clean = f"docker exec {docker_path} rm -r /aha/garnet/tiles_{app_name}_{datum}"
    print(docker_clean)
    os.system(docker_clean)

    docker_copy_command = f"docker cp tiles {docker_path}:/aha/garnet/tiles_{app_name}_{datum}"
    print(docker_copy_command)
    os.system(docker_copy_command)

    i = i + 1
