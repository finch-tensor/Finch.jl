import csv
import argparse
import os
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from ast import literal_eval


def plot_stream_overhead(input_csv, outfile, default_outfile="./fig14.pdf"):
    mtx_names = []
    cycles = []
    # Inner stream counts
    inner_stream_idle = []
    inner_stream_done = []
    inner_stream_nonctrl = []
    inner_stream_stop = []

    # Outer stream counts
    outer_stream_idle = []
    outer_stream_done = []
    outer_stream_nonctrl = []
    outer_stream_stop = []
    with open(input_csv) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Get matrix name from full file path
            mtx_name = row['dataset']
            mtx_names.append(mtx_name)

            cycles.append(int(row['cycles']))

            # Only ever 1 done token at the end of a stream
            outer_total = 1 + int(row['stream_Bi_empty']) + int(row['stream_Bi_noncontrol']) + \
                int(row['stream_Bi_stop'])
            outer_stream_done.append(1 / outer_total)
            outer_stream_idle.append(int(row['stream_Bi_empty']) / outer_total)
            outer_stream_nonctrl.append(int(row['stream_Bi_noncontrol']) / outer_total)
            outer_stream_stop.append(int(row['stream_Bi_stop']) / outer_total)

            # Only ever 1 done token at the end of a stream
            inner_total = 1 + int(row['stream_Bj_empty']) + int(row['stream_Bj_noncontrol']) + \
                int(row['stream_Bj_stop'])
            inner_stream_done.append(1 / inner_total)
            inner_stream_idle.append(int(row['stream_Bj_empty']) / inner_total)
            inner_stream_nonctrl.append(int(row['stream_Bj_noncontrol']) / inner_total)
            inner_stream_stop.append(int(row['stream_Bj_stop']) / inner_total)

        cycles, mtx_names, outer_stream_done, outer_stream_idle, outer_stream_nonctrl, \
            outer_stream_stop, inner_stream_done, inner_stream_idle, inner_stream_nonctrl, inner_stream_stop =\
            zip(*sorted(zip(cycles, mtx_names, outer_stream_done, outer_stream_idle, outer_stream_nonctrl,
                            outer_stream_stop, inner_stream_done, inner_stream_idle, inner_stream_nonctrl,
                            inner_stream_stop)))

        nrows = len(mtx_names)

        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle('Stream Overhead Plots, Figure 14')

        red_patch = mpatches.Patch(color='red', label='stop')
        blue_patch = mpatches.Patch(color='blue', label='noncontrol')
        yellow_patch = mpatches.Patch(color='yellow', label='done')
        green_patch = mpatches.Patch(color='green', label='idle')
        plt.figlegend(handles=[yellow_patch, green_patch, red_patch, blue_patch])

        ax1.bar(mtx_names, outer_stream_nonctrl, color='b')
        ax1.bar(mtx_names, outer_stream_stop, bottom=outer_stream_nonctrl, color='r')
        ax1.bar(mtx_names, outer_stream_done, bottom=[outer_stream_nonctrl[i] +
                outer_stream_stop[i] for i in range(nrows)], color='y')
        ax1.bar(mtx_names, outer_stream_idle, bottom=[outer_stream_nonctrl[i] +
                outer_stream_stop[i] + outer_stream_done[i] for i in range(nrows)],
                color='g')
        ax1.set_xticklabels([])
        ax1.title.set_text("Outer-level Bi Stream Breakdown")

        ax2.bar(mtx_names, inner_stream_nonctrl, color='b')
        ax2.bar(mtx_names, inner_stream_stop, bottom=inner_stream_nonctrl, color='r')
        ax2.bar(mtx_names, inner_stream_done, bottom=[inner_stream_nonctrl[i] +
                inner_stream_stop[i] for i in range(nrows)], color='y')
        ax2.bar(mtx_names, inner_stream_idle, bottom=[inner_stream_nonctrl[i] +
                inner_stream_stop[i] + inner_stream_done[i] for i in range(nrows)],
                color='g')
        ax2.set_xticklabels(mtx_names, fontsize=10, rotation=45)
        ax2.title.set_text("Inner-level Bj Stream Breakdown")

        fig.subplots_adjust(bottom=0.25)
        plt.xlabel("Suitesparse Matrix Name")

        # Need a default outfile location for copying script out of the docker
        plt.savefig(default_outfile)
        plt.savefig(outfile)
        # plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('input_csv_name', type=str, help="Name of the CSV to plot")
parser.add_argument('output_plot_name', type=str, help="Name of the plot to generate")
args = parser.parse_args()

plot_stream_overhead(args.input_csv_name, args.output_plot_name)
