
if __name__ == "__main__":

    with open('./suitesparse_valid.txt', 'r') as total_file:
        all_runs = total_file.readlines()

    total_num = len(all_runs)

    num_per = total_num // 8
    for i in range(8):
        with open(f"./temp_{i}.txt", "w+") as new_tmp:
            if i == 7:
                new_tmp.writelines(all_runs[num_per * i:])
            else:
                new_tmp.writelines(all_runs[num_per * i: num_per * (i + 1)])
