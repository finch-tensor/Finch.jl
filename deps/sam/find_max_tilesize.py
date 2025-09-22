import os


def write_to_line(file_path, line_number, new_content):
    with open(file_path, "r") as file:
        lines = file.readlines()

    if line_number > len(lines) or line_number < 1:
        # Line number is out of range
        return

    lines[line_number - 1] = new_content + "\n"

    with open(file_path, "w") as file:
        file.writelines(lines)


def check_keyword_in_output(command, keyword):
    # Run the command and redirect the output to a file
    os.system(f"{command} > output.txt")

    # Read the contents of the file
    with open("output.txt", "r") as file:
        output = file.read()

    # Check if the keyword is present in the output
    if keyword in output:
        # Optionally, you can delete the output file
        os.remove("output.txt")
        return True
    else:
        # Optionally, you can delete the output file
        os.remove("output.txt")
        return False


tile_size = 300
step = 10

for _ in range(20):
    print("********************")
    print("tile size: ", tile_size)
    print("step: ", step)

    yaml_file = "sam/sim/src/tiling/memory_config_onyx.yaml"
    mem_tile_line = f"Mem_tile_size: {tile_size}"
    print(mem_tile_line)
    write_to_line(yaml_file, 19, mem_tile_line)

    run_setup_script = "python3 setup_tiling_mat.py > temp.txt"
    os.system(run_setup_script)
    print(run_setup_script)

    run_tile_pairing = "python3 tile_pairing.py > temp.txt"
    os.system(run_tile_pairing)
    print(run_tile_pairing)

    run_count = "python3 count_nnz_tiling.py"
    print(run_count)

    if (check_keyword_in_output(run_count, "error")) is False:
        tile_size += step
        step *= 2
    else:
        print("****************Tile broken!")
        tile_size -= step
        step //= 2

    if tile_size == 450:
        break

    if step == 0:
        if _ >= 15:
            step = 10
        else:
            break

print("max tile size: ", tile_size)
