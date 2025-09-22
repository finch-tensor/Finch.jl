import csv
import argparse
import os
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from ast import literal_eval

LEGEND = {5000: ['-^', 'y'], 10000: ['-D', 'r'], 25000: ['-s', 'b'], 50000: ['-o', 'g']}


def plot_memory_model(input_csv, outfile, default_outfile="./fig15.pdf"):
    dim_sizes = dict()
    cycles = dict()

    with open(input_csv) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Get the nnz of this matrix
            nnz = int(row['nnz'])
            if nnz not in cycles:
                cycles[nnz] = []
            cycles[nnz].append(int(row['cycles']))

            if nnz not in dim_sizes:
                dim_sizes[nnz] = []
            dim_sizes[nnz].append(int(row['dim_size']))

#        dims, mtx_names, outer_stream_done, outer_stream_idle, outer_stream_nonctrl, \
#            outer_stream_stop, inner_stream_done, inner_stream_idle, inner_stream_nonctrl, inner_stream_stop =\
#            zip(*sorted(zip(dims, mtx_names, outer_stream_done, outer_stream_idle, outer_stream_nonctrl,
#                            outer_stream_stop, inner_stream_done, inner_stream_idle, inner_stream_nonctrl,
#                            inner_stream_stop)))
#

        plt.figure(figsize=(7, 4), dpi=80)

        for nnz, dim_size_list in dim_sizes.items():
            # Sort the dim_size_list and permute cycles to match using the Schwartzian Transform
            dim_size_sorted, cycles_sorted = zip(*sorted(zip(dim_size_list, cycles[nnz])))
            print(dim_size_sorted)
            print(cycles_sorted)
            marker, color = LEGEND[nnz]
            plt.plot(dim_size_sorted, cycles_sorted, marker, color=color, label=str(nnz) + " NNZ", markeredgecolor='black')
        plt.title("Recreation of Figure 19 from the ExTensor Paper,\n SAM Paper Figure 15")
        plt.xticks(fontsize=14)
        plt.xlabel("Matrix Dimension Size", fontsize=16)

        plt.gca().yaxis.get_offset_text().set_fontsize(14)
        plt.yticks(fontsize=14)
        plt.ylabel("Runtime (Cycles)", fontsize=16)

        plt.tight_layout()
        plt.legend(fontsize=12, loc='upper right')

        # Need default outfile location for copying script out of docker
        plt.savefig(default_outfile)
        plt.savefig(outfile)
#        plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('input_csv_name', type=str, help="Name of the CSV to plot")
parser.add_argument('output_plot_name', type=str, help="Name of the plot to generate")
args = parser.parse_args()

plot_memory_model(args.input_csv_name, args.output_plot_name)
