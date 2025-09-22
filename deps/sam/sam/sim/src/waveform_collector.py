import csv
from .channel import *
from collections import defaultdict
import waterfall_chart
import matplotlib.pyplot as plt
import pandas as pd


class waveform_collector():
    def __init__(self, memory_blocks):
        self.memory_blocks = memory_blocks
        self.stats_dict = defaultdict()
        self.waveform = defaultdict()

        for a in self.memory_blocks:
            stats_dict = a.stats_base()
            for key in stats_dict.keys():
                self.stats_dict[key] = stats_dict[key]
            stats_dict = a.stats_cycle2()
            for key in stats_dict.keys():
                self.waveform[key] = []

    def add_memory_block(mem_block):
        if isinstance(mem_block, memory_block):
            self.memory_blocks.append(mem_block)
            stats_dict = mem_block.stats_base()
            for key in stats_dict.keys():
                self.stats_dict[key] = stats_dict[key]
                self.waveform[key] = []
        else:
            assert False

    def update(self):
        for mem_block in self.memory_blocks:
            stats_dict = mem_block.stats_cycle()
            for key in stats_dict.keys():
                if stats_dict[key]:
                    self.stats_dict[key] += 1

            stats_dict = mem_block.stats_cycle2()
            for key in stats_dict.keys():
                if stats_dict[key]:
                    self.waveform[key].append(1)
                else:
                    self.waveform[key].append(0)

    def print_waveform(self):
        for a in self.waveform.keys():
            print(a, len(self.waveform[a]))
            print(self.waveform[a][:min(100, len(self.waveform))])

    def save_as_dict(self, path):
        df = pd.DataFrame.from_dict(self.waveform, orient='index').transpose()
        df.to_csv(path)
        # with open(path, "w") as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames=self.waveform.keys())
        #     writer.writeheader()
        #     # for data in self.waveform:
        #     writer.writerow(self.waveform)

    def waveform_plot(self):
        for a in self.waveform.keys():
            pass
            # waterfall_chart.plot(range(len(self.waveform[a])), self.waveform[a])
            # plt.plot(self.waveform[a][:100])
            # waterfall_chart.plot(range(100), self.waveform[a][:min(100, 1000)])
