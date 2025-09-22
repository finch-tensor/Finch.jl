import json
import csv
import argparse
import os

UNWANTED_KEYS = ['group', 'params', 'options']
FLATTEN_KEYS = ['extra_info', 'stats']
RESULTS_DIR = 'results/'


def convert(json_names, csv_names):
    for json_name, csv_name in zip(json_names, csv_names):
        print("Converting", json_name)

        if os.stat(json_name).st_size == 0:
            print(json_name + " was empty...continuing")
            continue

        with open(json_name) as json_file:
            data = json.load(json_file)
        benchmark_data = data['benchmarks']

        filtered_benchmark_data = [{k: v for k, v in d.items() if k not in UNWANTED_KEYS} for d in benchmark_data]

        flattened_benchmark_data = []
        for d in filtered_benchmark_data:
            flattened_data = dict()
            for k, v in d.items():
                if k in FLATTEN_KEYS:
                    for vk, vv in v.items():
                        flattened_data[vk] = vv
                else:
                    flattened_data[k] = v
            flattened_benchmark_data.append(flattened_data)

        csv_file = open(csv_name, 'w')
        csv_writer = csv.writer(csv_file)

        count = 0
        for benchmark in flattened_benchmark_data:
            if count == 0:
                header = benchmark.keys()
                csv_writer.writerow(header)
                count += 1
            csv_writer.writerow(benchmark.values())

        csv_file.close()


parser = argparse.ArgumentParser()
parser.add_argument('--json_name', type=str, default=None, help="Input JSON file name. --all overrides this flag")
parser.add_argument('--csv_name', type=str, default=None, help="Output CSV file name")
parser.add_argument('--all', action='store_true', default=False,
                    help='Convert all files in results/numpy from json to csv')

args = parser.parse_args()

json_name = args.json_name
if args.all:
    json_files = [RESULTS_DIR + pos_json for pos_json in os.listdir(RESULTS_DIR) if pos_json.endswith('.json')]
    print("JSON Files being converted")
    print(json_files)
    csv_files = [os.path.splitext(json_file)[0] + '.csv' for json_file in json_files]

else:
    if args.json_name is None:
        raise ValueError('Set --json_name or pass --all.')
    else:
        json_files = [args.json_name]
        if args.csv_name is None:
            csv_name = os.path.splitext(json_name)[0] + '.csv'
        else:
            csv_name = args.csv_name
        csv_files = [csv_name]

convert(json_files, csv_files)
