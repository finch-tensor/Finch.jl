from abc import ABC

from .base import *
from sam.sim.src.array import Array


class WrScan(Primitive, ABC):
    def __init__(self, size=1024, fill=0, backpressure_en=False, depth=1, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.fill = fill

        self.size_init = size
        self.fill_init = fill

        self.input = []
        self.arr = Array(size=size, fill=self.fill, debug=self.debug)
        self.blk_start_ = False
        self.backpressure_en = False
        if self.backpressure_en:
            self.ready_backpressure = True
            self.depth = depth
            self.fifo_avail = True
            self.data_valid = True

    def fifo_available(self):
        return self.fifo_avail

    def set_backpressure(self, backpressure):
        if not backpressure:
            self.ready_backpressure = False

    def check_backpressure(self):
        if self.backpressure_en:
            copy_backpressure = self.ready_backpressure
            self.ready_backpressure = True
            return copy_backpressure
        return True

    def update_ready(self):
        if self.backpressure_en and len(self.input) > self.depth:
            self.fifo_avail = False
        else:
            self.fifo_avail = True

    def set_input(self, val, parent=None):
        # Make sure streams have correct token type
        assert (isinstance(val, int) or isinstance(val, float) or val in valid_tkns or val is None)

        if val != '' and val is not None:
            # print("Add input:", self.name, val)
            self.blk_start_ = True
            self.input.append(val)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail)

    def return_block_start(self):
        return self.blk_start_

    def clear_arr(self):
        self.arr.clear(self.fill)

    def resize_arr(self, size):
        self.size = size
        self.arr.resize(self.size)

    def get_arr(self):
        return self.arr.get_arr()

    def return_fifo(self):
        return self.input

    def set_fifo(self, fifo):
        self.input = fifo

    @abstractmethod
    def reset(self):
        pass


class ValsWrScan(WrScan):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.curr_addr = 0

    def update(self):
        self.update_done()
        self.update_ready()
        if self.done:
            return
            if self.debug:
                print("RESET FOR VALS", self.input)
            if self.debug:
                print("post reset: ", self.arr.out_done())

        if (len(self.input) > 0):
            self.block_start = False

        if self.backpressure_en:
            self.data_valid = False
        if (self.backpressure_en and self.check_backpressure()) or not self.backpressure_en:
            if self.backpressure_en:
                self.data_valid = True

            if (len(self.input) > 0):
                self.block_start = False

            if len(self.input) > 0:
                val = self.input.pop(0)

                if not is_stkn(val) and val != 'D':
                    self.arr.set_store(self.curr_addr, val)
                    self.curr_addr += 1
                else:
                    self.arr.set_store(val, val)

                self.arr.update()
                self.done = self.arr.out_done()
            if self.debug:
                print("Vals Wr scanner print ", self.done, self.curr_addr)

    def reset(self):
        # print("reset vals")
        # arr_fifo = self.return_fifo()
        self.done = False
        self.curr_addr = 0
        # print("RESET VALS to ", self.size_init, self.fill)
        # self.clear_arr()
        self.arr = Array(size=self.size_init, fill=self.fill_init, debug=self.debug)
        # self.set_fifo(arr_fifo)

    def autosize(self):
        self.resize_arr(self.curr_addr)

    def return_statistics(self):
        if self.get_stats:
            stats_dict = {"size": self.curr_addr}
            stats_dict.update(super().return_statistics())
        else:
            stats_dict = {}
        return stats_dict


# Unique compressed (not from points)
class CompressWrScan(WrScan):
    def __init__(self, seg_size=0, level=0, **kwargs):
        super().__init__(**kwargs)
        # FIXME: Either use this later or remove
        self.level = level

        self.curr_addr = 0
        self.curr_seg_addr = 1
        self.curr_crd_cnt = 0

        self.end_fiber = True

        self.seg_size = seg_size
        self.seg_size_init = seg_size
        self.seg_arr = Array(size=self.seg_size, fill=0, debug=self.debug)

    def update(self):
        self.update_done()
        self.update_ready()
        if self.done:
            return
            if self.debug:
                print("RESET WR SCAN ", self.input)
            if self.debug:
                print("post reset: ", self.arr.out_done())

        if len(self.input) > 0:
            self.block_start = False

        if self.backpressure_en:
            self.data_valid = False
        if (self.backpressure_en and self.check_backpressure()) or not self.backpressure_en:
            if self.backpressure_en:
                self.data_valid = True
        if len(self.input) > 0:
            in_crd = self.input.pop(0)
            if not is_stkn(in_crd) and in_crd != 'D':
                self.arr.set_store(self.curr_addr, in_crd)
                self.curr_addr += 1
                self.curr_crd_cnt += 1
                self.end_fiber = False
            elif is_stkn(in_crd) and not self.end_fiber:
                self.seg_arr.set_store(self.curr_seg_addr, self.curr_crd_cnt)
                self.curr_seg_addr += 1
                self.end_fiber = True
                self.arr.set_store(in_crd, in_crd)
            else:
                self.arr.set_store(in_crd, in_crd)

            self.arr.update()
            self.seg_arr.update()
            self.done = self.arr.out_done()

        if self.debug:
            print("DEBUG: WR SCAN: \t "
                  "name: ", self.name, self.done,
                  "\t Curr crd addr:", self.curr_addr, "\t curr crd cnt:", self.curr_crd_cnt, "\t curr seg addr:",
                  self.curr_seg_addr,
                  "\t end fiber:", self.end_fiber, "\t", self.input)

    def reset(self):
        # print("reset crd arr")
        self.done = False
        self.curr_addr = 0
        self.curr_seg_addr = 1
        self.curr_crd_cnt = 0
        self.end_fiber = True
        # arr_fifo = self.return_fifo()
        # self.clear_seg_arr()
        # self.clear_arr()
        self.seg_arr = Array(size=self.seg_size_init, fill=0, debug=self.debug)
        self.arr = Array(size=self.size_init, fill=self.fill_init, debug=self.debug)
        # self.set_fifo(arr_fifo)

    def clear_seg_arr(self):
        self.seg_arr.clear(self.fill)

    def resize_seg_arr(self, size):
        self.seg_size = size
        self.seg_arr.resize(self.seg_size)

    def get_seg_arr(self):
        return self.seg_arr.get_arr()

    def autosize(self):
        self.resize_seg_arr(self.curr_seg_addr)
        self.resize_arr(self.curr_crd_cnt)

    def return_statistics(self):
        if self.get_stats:
            stats_dict = {}
            stats_dict["seg_size"] = self.seg_size
            stats_dict["arr_size"] = self.size
            stats_dict.update(super().return_statistics())
        else:
            stats_dict = {}
        return stats_dict


# Unique compressed (not from points)
class UncompressWrScan(WrScan, ABC):
    def __init__(self, seg_size=0, level=0, **kwargs):
        super().__init__(**kwargs)

        self.input = None
        self.dim = ''

    def update(self):
        self.update_done()
        if len(self.input) > 0:
            self.block_start = False

        if isinstance(self.input, int):
            self.dim = self.input
        elif self.input == 'D':
            self.done = True

    def set_in_dim(self, dim):
        if dim != '' and not is_stkn(dim):
            self.input = dim

    def out_dim(self):
        return self.dim
