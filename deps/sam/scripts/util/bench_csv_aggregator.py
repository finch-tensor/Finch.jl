import csv
import os
import glob
import sys
import tqdm
import argparse

csv.field_size_limit(sys.maxsize)


def aggregateTacoBenches(folder, outfile, taco=False, labelSet=None):
    first = True
    outputFile = open(outfile, 'w+')
    writer = csv.writer(outputFile, delimiter=',')
    # If labelSet is not None, we're going to collect all of the
    # valid labels we've seen, and output them to a file later.
    validLabels = set()
    # Loop through all files with a csv extension.
    with open(outfile, 'w+') as outputFile:
        writer = csv.writer(outputFile, delimiter=',')
        for fname in tqdm.tqdm(glob.glob(os.path.join(folder, "*.csv"))):
            # Open up the file.
            with open(fname, 'r') as f:
                # Discard the first 9 lines. This corresponds to the
                # google-benchmark generated header.
                if taco:
                    for i in range(0, 9):
                        f.readline()
                # Open the rest of the file as a CSV.
                reader = csv.reader(f)
                # Attempt to read the header from CSV. If this fails,
                # the benchmark might have failed in the middle. So,
                # just continue on to the next file.
                try:
                    header = next(reader)
                except Exception as e:
                    continue
                # Find the column that contains label. We're going to skip
                # entries that have a skip marker in the label.
                # labelIdx = header.index("label", 0)
                if first:
                    header.append("original_filename")
                    writer.writerow(header)
                    first = False

                # Find the column that contains label. We're going to skip
                # entries that have a skip marker in the label.
                if taco:
                    labelIdx = header.index("label", 0)
                for row in reader:
                    # if taco and "SKIPPED" not in row[labelIdx]:
                    #   validLabels.add(row[labelIdx])
                    row.append(fname)
                    writer.writerow(row)
    # Write out the set of valid labels.
    if labelSet is not None:
        with open(labelSet, 'w+') as validSet:
            for ll in validLabels:
                validSet.write(ll)
                validSet.write("\n")


parser = argparse.ArgumentParser()
parser.add_argument('target_directory', type=str, help="Directory containing CSV's to aggregate")
parser.add_argument('output_csv_name', type=str, help="Name of the CSV to generate")
parser.add_argument('--label_set_file', type=str, default=None, help='Set to output all valid labels seen to a file')
parser.add_argument('--taco', action='store_true', help='Flag to aggregate TACO csvs, default is numpy')
args = parser.parse_args()
aggregateTacoBenches(args.target_directory, args.output_csv_name, taco=args.taco, labelSet=args.label_set_file)
