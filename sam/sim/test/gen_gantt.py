import matplotlib.pyplot as plt
import csv


def gen_gantt(extra_info, testname):
    block_list = []
    start_list = []
    finish_list = []
    duration_list = []

    start_c = ''
    finish_c = ''
    sam_name = ''

    for k in extra_info.keys():
        if "done_cycles" in k:
            sam_name = k.split('/')[0]
            finish_c = extra_info[k]
            if not isinstance(finish_c, int):
                finish_list.insert(0, int(finish_c))
            else:
                finish_list.insert(0, finish_c)
        elif ("start_cycle" in k) and (sam_name in k.split('/')[0]):
            '''
            We assume that the info to extra_info is added
            in the same order each block is updated.
            If we assume the opposite order, use append function
            instea of insert.
            (e.g.) block_list.insert(0, sam_name) -> block_list.append(sam_name)
            '''
            block_list.insert(0, sam_name)
            start_c = extra_info[k]
            if not isinstance(start_c, int):
                start_list.insert(0, int(start_c))
                duration_list.insert(0, finish_c - int(start_c))
            else:
                start_list.insert(0, start_c)
                duration_list.insert(0, finish_c - start_c)

    back_depth = 'N'  # assume there is no back pressure for default
    if "backpressure" in extra_info.keys() and extra_info["backpressure"]:
        back_depth = extra_info["depth"]

    # Writing cycle info to csv file
    with open(testname + '_' + extra_info["dataset"] + '_back_' + back_depth + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["block", "start", "finish", "duration", "valid_ops"])
        for idx, block in reversed(list(enumerate(block_list))):
            writer.writerow([block, start_list[idx], finish_list[idx], duration_list[idx], '-'])

    # Print all the statistics to a text file
    text_file = open(testname + '_' + extra_info["dataset"] + '_back_' + back_depth + ".txt", "w")
    for k in extra_info.keys():
        if "/" in k:
            text_file.write(k + ": " + str(extra_info[k]) + "\n")
    text_file.close()

    # Creating gantt chart
    plt.barh(y=block_list, width=duration_list, left=start_list)
    file_name = testname + '_' + extra_info["dataset"] + "_back_" + back_depth + ".png"
    plt.savefig(file_name, bbox_inches="tight")
