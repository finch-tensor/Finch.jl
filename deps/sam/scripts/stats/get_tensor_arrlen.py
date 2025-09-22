# python scripts/stats/get_tensor_arrlen.py <formatted_dir> <output.csv>

import argparse
import os
import csv


# This is using the old CSF file types
def write_csv(path, outpath):
    with open(outpath, 'w+', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerow(["name", "ss01-len", "ss10-len"])
        for dir in os.listdir(path):
            tensor_dirname = os.path.join(path, dir)
            if os.path.isdir(tensor_dirname):
                tensorname = dir
                filename = os.path.join(tensor_dirname, "orig", "ss01", "B0_seg.txt")
                ss01_len = ''
                if os.path.isfile(filename, 'r'):
                    with open(filename) as f:
                        last_line = f.readlines()[-1]
                        ss01_len = last_line

                ss10_len = ''
                filename = os.path.join(tensor_dirname, "orig", "ss10", "B1_seg.txt")
                if os.path.isfile(filename, 'r'):
                    with open(filename) as f:
                        last_line = f.readlines()[-1]
                        ss10_len = last_line

                writer.writerow([tensorname, ss01_len, ss10_len])


parser = argparse.ArgumentParser()
parser.add_argument('target_directory', type=str, help="Directory containing suitesparse formated directories")
parser.add_argument('output_csv_name', type=str, help="Name of the CSV to generate")
args = parser.parse_args()
write_csv(args.target_directory, args.output_csv_name)
