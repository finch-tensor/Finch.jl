import numpy as np


def hash_tile(tile_id):
    hash_val = 0
    num = len(tile_id) // 2
    cnt = 0
    encoder = 1
    encoder_diff = 10000
    for a in reversed(tile_id):
        hash_val += encoder * a
        encoder *= encoder_diff
        cnt += 1
    return hash_val, hash_val // (encoder_diff ** num)


def get_glb_tile_id(tile_id, num=3):
    if isinstance(tile_id, int):
        encoder_diff = 10000
        return tile_id // (encoder_diff ** num)
    return "D"


def get_mem_tile_id(tile_id, num=3):
    if isinstance(tile_id, int):
        encoder_diff = 10000
        return tile_id % (encoder_diff ** num)
    return "D"


class output_memory_block():
    def __init__(self, name="B", element_size=2, level=None, indexes=2,
                 size=1000 * 2, debug=False, latency=2, bandwidth=2, length=1,
                 mode="all_unpacked", loop_order=[2, 2, 2]):
        self.name = name
        self.level = level
        self.size = size // 2
        self.bandwidth = bandwidth
        self.latency = latency
        self.element_size = element_size
        self.length = length
        self.tile_ptrs_glb = []
        self.tile_ptrs_fifo = []  # virtual
        self.tile_ptrs_size = []
        self.mode = mode
        self.timestamp = None
        self.ready = True
        self.loading = False
        self.valid = False
        self.curr_tile = None
        self.curr_size = 0
        self.old_tile = None
        self.debug = debug
        self.outputed = False
        self.done = False
        self.done_received = False
        self.valid_processed = False
        self.child_ready = False
        self.indexes = indexes
        loop_order = [a - 1 for a in loop_order]
        self.loop_order, _ = hash_tile(loop_order)
        if self.level == "glb2global":
            self.mem_loop_limit = get_mem_tile_id(self.loop_order)

    def out_done(self):
        return self.done

    def final_done(self):
        if self.debug:
            print(self.level, " done check-- ", self.curr_tile, " ", self.loop_order)
        if self.level == "mem2glb" and self.done:
            if get_mem_tile_id(self.curr_tile) == self.loop_order:
                return True
        if self.level == "glb2global" and self.curr_tile == self.loop_order and self.done:
            return True
        return False

    def out_ready(self):
        return self.ready

    def return_if_loading(self):
        return self.loading

    def check_if_done(self, arr):
        if isinstance(arr, bool):
            self.done_received = arr
            return
        token = True
        if len(arr) > 0:
            for i in arr:
                if not i:
                    token = False
                    self.done_received = False
                    return
        self.done_received = True
        return

    def add_upstream(self, tilecoord, data, valid):
        if not valid:
            self.valid_processed = False
        if valid and not self.valid_processed and self.level == "mem2glb":
            self.valid_processed = True
            limits = len(data) - 1
            if tilecoord == "D":
                self.tile_ptrs_fifo.append("D")
                self.tile_ptrs_size.append(0)
                return
            size_levels = [0] * (limits // 2)
            size_vals = len(data[len(data) - 1])
            for lengths in range(len(size_levels)):
                size_levels[lengths] = len(data[lengths]) + len(data[lengths + limits // 2])
            if self.mode == "not_consolidated":
                max_levels_size = max(max(size_levels), size_vals)
            elif self.mode == "all_unpacked":
                max_levels_size = np.sum(np.asarray(size_levels)) + size_vals
            else:
                print(self.mode + " not found")
                assert False
            self.tile_ptrs_size.append(max_levels_size)
            tile_hash, tile_glb = hash_tile(tilecoord)
            if self.debug:
                print("Tile pointer vales = ", tile_hash, " ", tile_glb)
            # tilecoord
            self.tile_ptrs_fifo.append(tile_hash)

        if valid and not self.valid_processed and self.level == "glb2global":
            self.valid_processed = True
            self.tile_ptrs_fifo.append(tilecoord)
            # size = data
            self.tile_ptrs_size.append(data)

    def update(self, cyclenum):
        if self.level == "mem2glb":
            if self.ready and len(self.tile_ptrs_fifo) > 0:
                if self.curr_tile is not None:
                    self.old_tile = self.curr_tile
                self.curr_tile = self.tile_ptrs_fifo.pop(0)
                self.curr_size = self.tile_ptrs_size.pop(0)
                self.loading = True
                self.ready = False
                self.done = False
                self.timestamp = cyclenum
            elif self.loading and cyclenum > self.timestamp + self.compute_latency(self.curr_tile):
                self.loading = False
                self.done = True
                # self.curr_tile = "D"
            elif self.done and self.child_ready:
                self.ready = True

        if self.level == "glb2global":
            self.old_tile = None
            if self.ready and (not self.done) and \
                    get_mem_tile_id(self.curr_tile) == self.mem_loop_limit:
                self.loading = True
                self.ready = False
                self.done = False
                self.timestamp = cyclenum
            elif self.ready and len(self.tile_ptrs_fifo) > 0 and (self.curr_tile is None or self.done):
                self.curr_tile = self.tile_ptrs_fifo.pop(0)
                self.done = False
                self.ready = True
            elif self.ready and len(self.tile_ptrs_fifo) > 0 and \
                    (get_glb_tile_id(self.curr_tile) == get_glb_tile_id(self.tile_ptrs_fifo[0])):
                self.curr_tile = self.tile_ptrs_fifo.pop(0)
                self.curr_size += self.tile_ptrs_size.pop(0)
                self.done = False
                self.ready = True
            elif self.ready and len(self.tile_ptrs_fifo) > 0 and self.curr_tile is not None \
                    and get_glb_tile_id(self.curr_tile) != get_glb_tile_id(self.tile_ptrs_fifo[0]):
                self.loading = True
                self.ready = False
                self.done = False
                self.timestamp = cyclenum
            elif self.loading and cyclenum > self.timestamp + self.compute_latency(self.curr_tile):
                self.ready = True
                self.curr_size = 0
                self.done = True
                self.loading = False
                if len(self.tile_ptrs_fifo) > 0:
                    self.curr_tile = self.tile_ptrs_fifo[0]
        if self.debug:
            if self.loading:
                print(self.level, " loads stuff : ", self.curr_tile, " ", get_glb_tile_id(self.curr_tile),
                      self.curr_tile // (10000 ** 3), get_mem_tile_id(self.curr_tile))
            print("Output ", self.level, " valid: ", self.valid, " ready: ", self.ready, " loading: ",
                  self.loading, " done: ", self.done, " curr tile: ", self.curr_tile, " Done ",
                  self.done, " valid processed: ", self.valid_processed, " Child ready ",
                  self.child_ready, " , ", self.curr_size,
                  "       ", self.tile_ptrs_fifo)

    def set_child_ready(self, token):
        self.child_ready = token

    def token(self):
        return self.curr_tile

    def get_size(self):
        return self.curr_size

    def check_tile(self, tile_id):
        if tile_id in self.tile_ptrs_ids:
            return True
        return False

    def compute_latency(self, tile):
        if self.mode == "not_consolidated":
            return self.latency + self.curr_size // (self.bandwidth)
        if self.mode == "all_unpacked":
            return self.latency + (self.curr_size * self.element_size) // (self.bandwidth)
        else:
            print(self.mode, " not found")
            assert False

    def input_token(self):
        if self.downstream_token == "D":
            return True

    def input_token_(self, token):
        self.downstream_token = token


# FIXME: Follow code style and fix class naming convention and make sure it's base is primitive...
class memory_block():
    def __init__(self, name="B", skip_blocks=False, element_size=2, level=None, indexes=2,
                 size=1000 * 2, nbuffer=False, latency=10, debug=False, bandwidth=2,
                 length=1, mode="all_unpacked", pipeline_en=False, statistics=False):
        self.name = name
        self.skip_blocks = skip_blocks
        self.level = level
        self.size = size // element_size
        self.latency = latency
        self.element_size = element_size
        self.bandwidth = bandwidth
        self.length = length
        self.tile_ptrs = []
        self.nbuffer = nbuffer
        self.tile_ptrs_fifo = []  # virtual
        self.tile_ptrs_size = []
        self.mode = mode
        self.timestamp = None
        self.ready = True
        self.loading = False
        self.valid = False
        self.curr_tile = None
        self.curr_size = None
        self.old_tile = None
        self.debug = debug
        self.outputed = False
        self.done = False
        self.signalled = False
        self.downstream_token = None
        self.done_received = False
        self.valid_processed = False
        self.done_processed = False
        self.indexes = indexes
        self.full_buff = False  # True
        if self.nbuffer or self.full_buff:
            self.ready = False
            self.loading_tile = None
            self.load_size = 0
            self.curr_tile = None
            self.next_tile = None
            self.curr_size = 0
            self.tile_ptrs = []
            self.repeat_pattern = []
            self.tile_sizes = []
            self.timestamp = None
            self.loading = False
            self.done_in = False
            self.remove_size = 0
            self.if_latency_ = []
            self.if_latency = True
            self.pipeline_en = pipeline_en
        self.get_stats = statistics
        if self.get_stats:
            self.load_cycles = 0
            self.valid_tile_cycles = 0
            self.not_ready_cycles = 0
            self.num_tiles = 0
            self.if_repeat = 0
            self.repeat_dist = 0
            self.rep_true = False
            self.load_not_valid = 0
            self.not_load_valid = 0

    def stats_base(self):
        temp_dict = {self.name + "_load_cycle": 0, self.name + "_valid_cycle": 0, self.name + "_ready_cycle": 0,
                     self.name + "_ready_not_valid_not_load_cycle": 0, self.name + "_not_ready_valid_not_load_cycle": 0,
                     self.name + "_not_ready_not_valid_load": 0, self.name + "_ready_valid_not_load": 0,
                     self.name + "_ready_not_valid_load": 0, self.name + "_not_ready_valid_load": 0,
                     self.name + "_ready_valid_load": 0}
        return temp_dict

    def stats_cycle(self):
        temp_dict = {self.name + "_load_cycle": self.loading, self.name + "_valid_cycle": self.valid,
                     self.name + "_ready_cycle": self.ready,
                     self.name + "_ready_not_valid_not_load_cycle": self.ready and not self.valid and not self.loading,
                     self.name + "_not_ready_valid_not_load_cycle": not self.ready and self.valid and not self.loading,
                     self.name + "_not_ready_not_valid_load": not self.ready and not self.valid and self.loading,
                     self.name + "_ready_valid_not_load": self.ready and self.valid and not self.loading,
                     self.name + "_ready_not_valid_load": self.ready and not self.valid and self.loading,
                     self.name + "_not_ready_valid_load": not self.ready and self.valid and self.loading,
                     self.name + "_ready_valid_load": self.ready and self.valid and self.loading}
        # temp_dict = {self.name + "_load_cycle": self.loading, self.name + "_valid_cycle": self.valid,
        #              self.name + "_valid_not_load_cycle": self.valid and not self.loading,
        #              self.name + "_not_valid_load": not self.valid and self.loading}
        return temp_dict

    def stats_cycle2(self):
        temp_dict = {self.name + "_load_cycle": self.loading,
                     self.name + "_valid_cycle": self.valid,
                     self.name + "_ready_cycle": self.ready}
        # temp_dict = {self.name + "_load_cycle": self.loading, self.name + "_valid_cycle": self.valid,
        #              self.name + "_valid_not_load_cycle": self.valid and not self.loading,
        #              self.name + "_not_valid_load": not self.valid and self.loading}
        return temp_dict

    def update_stats(self):
        if self.get_stats:
            if self.loading:
                self.load_cycles += 1
            if self.valid:
                self.valid_tile_cycles += 1
            if not self.ready:
                self.not_ready_cycles += 1
            if self.nbuffer:
                self.num_tiles = max(self.num_tiles, len(self.tile_ptrs))
                if self.rep_true:
                    self.if_repeat += 1
                else:
                    self.if_repeat = 0
                self.repeat_dist = max(self.repeat_dist, self.if_repeat)
            if self.loading and not self.valid:
                self.load_not_valid += 1
            if not self.loading and self.valid:
                self.not_load_valid += 1

    def return_stats(self):
        if self.get_stats:
            stats_dict = {self.name + "_load_cycles": self.load_cycles,
                          self.name + "_valid_cycles": self.valid_tile_cycles,
                          self.name + "_ready_cycles": self.not_ready_cycles,
                          self.name + "_max_tile_nums": self.num_tiles, self.name + "_repeat_dist": self.repeat_dist,
                          self.name + "_load_not_valid": self.load_not_valid,
                          self.name + "_not_load_valid": self.not_load_valid}
        else:
            stats_dict = {}
        return stats_dict

    def print_stats(self):
        stats_dict = self.return_stats()
        print(stats_dict)

    def out_done(self):
        if self.nbuffer or self.full_buff:
            return (len(self.tile_ptrs_fifo) == 0) and (len(self.tile_ptrs) < 2) and self.done
        return self.done

    def out_done_in(self):
        if self.nbuffer or self.full_buff:
            return self.done_in
        return self.out_done()

    def add_tile(self, tile_ptr, size, upperlvl=0):
        if tile_ptr == "D":
            self.tile_ptrs_fifo.append("D")
            self.tile_ptrs_size.append(0)
            if self.nbuffer:
                self.if_latency_.append(True)
            return
        if self.nbuffer or self.full_buff:
            if tile_ptr != "" and isinstance(tile_ptr, int):
                self.done = False
                if isinstance(upperlvl, int):
                    # upperlvl = 0
                    self.tile_ptrs_fifo.append(tile_ptr + upperlvl * 1000000)
                    self.if_latency_.append(True)
                else:
                    self.tile_ptrs_fifo.append(tile_ptr)
                    self.if_latency_.append(True)
                self.tile_ptrs_size.append(size)
                # print("Tile ptrs fifo: ", self.name, self.tile_ptrs_fifo)
        else:
            if tile_ptr != "" and isinstance(tile_ptr, int):
                self.done = False
                self.tile_ptrs_fifo.append(tile_ptr)
                self.tile_ptrs_size.append(size)

    def set_downstream_token(self, token):
        self.downstream_token = token

    def return_token(self):
        if self.done:
            return "D"
        if self.loading:
            return "L"

    def return_if_loading(self):
        return self.loading

    def check_if_done(self, arr):
        if isinstance(arr, bool):
            self.done_received = arr
            return
        token = True
        if len(arr) > 0:
            for i in arr:
                if not i:
                    token = False
                    self.done_received = False
                    return
        self.done_received = True
        return

    def print_debug(self):
        print("evit case", self.done_in, self.curr_size, self.load_size, self.remove_size, ":",
              self.size, self.name, " valid: ", self.valid, " ready: ", self.ready, " loading: ",
              self.loading, " done: ", self.done, " downstream token: ", self.downstream_token,
              " Done received and processed ", self.done_received, " ", self.done_processed, " : current tile: ",
              self.curr_tile, " full tles ", self.tile_ptrs, self.loading_tile, self.tile_ptrs_fifo, " ", self.tile_sizes)

    def return_next(self, ref_to_crd_map=None):
        if self.nbuffer:
            if self.next_tile is None:
                return None
            else:
                return self.next_tile
        return None

    def update(self, cyclenum):
        self.update_stats()
        if self.nbuffer:
            self.done_in = False
            if len(self.if_latency_) != len(self.tile_ptrs_fifo):
                assert False
            assert len(self.tile_ptrs) == len(self.tile_sizes)
            if len(self.tile_sizes) > 0:
                if self.loading_tile is None:
                    pass
                    # assert sum(self.tile_sizes) == self.curr_size
                else:
                    pass
                    # assert sum(self.tile_sizes) + self.load_size == self.curr_size

            assert len(self.tile_ptrs_fifo) == len(self.tile_ptrs_size)
            if self.done_received and len(self.tile_ptrs) > 0:  # and not self.done_processed:
                assert self.curr_size == sum(self.tile_sizes) or \
                    self.curr_size == (sum(self.tile_sizes) + self.load_size) or \
                    self.curr_size == (sum(self.tile_sizes) + self.load_size + self.remove_size)
                self.outputed = False
                self.done_processed = True
                self.valid = False
                tile = self.tile_ptrs.pop(0)
                if len(self.tile_ptrs) == 0:
                    self.old_tile = None  # tile
                else:
                    self.old_tile = None

                if self.tile_sizes[0] > 0:
                    self.remove_size = self.tile_sizes.pop(0)
                else:
                    self.tile_sizes.pop(0)

                if len(self.tile_ptrs) > 0:
                    if tile != self.tile_ptrs[0]:
                        self.curr_size -= self.remove_size
                        if self.curr_size != self.load_size + sum(self.tile_sizes) + 0:
                            print(self.curr_size + self.remove_size, self.load_size, self.tile_sizes,
                                  self.remove_size, "::", self.tile_ptrs, self.curr_tile, self.tile_ptrs_fifo)
                            assert False
                        self.remove_size = 0
                else:
                    self.curr_size -= self.remove_size
                    if self.curr_size != self.load_size + sum(self.tile_sizes) + 0:
                        print(self.curr_size + self.remove_size, self.load_size, self.tile_sizes,
                              self.remove_size, "::", self.tile_ptrs, self.curr_tile, self.tile_ptrs_fifo)
                        assert False
                    self.remove_size = 0

                if len(self.tile_ptrs) > 0:
                    self.curr_tile = self.tile_ptrs[0]
                if len(self.tile_ptrs) > 1:
                    self.next_tile = self.tile_ptrs[1]

            if self.done_processed and not self.done_received:
                self.done_processed = False

            if len(self.tile_ptrs) > 0:
                self.curr_tile = self.tile_ptrs[0]
                self.valid = True
                if self.curr_tile == "D" and len(self.tile_ptrs) < 2 and len(self.tile_ptrs_fifo) == 0:
                    # self.timestamp = None
                    self.ready = True
                    # self.loading = False
                    self.valid = False
                    self.outputed = False
                    self.done = True
                    self.done_received = False
                    self.done_processed = False
                    if self.debug:
                        print("Done case:: ", self.name, " valid: ", self.valid, " ready: ", self.ready,
                              " loading: ", self.loading, " done: ", self.done, " downstream token: ", self.downstream_token,
                              " Done received and processed ", self.done_received, " ", self.done_processed,
                              " : current tile: ", self.curr_tile, " full tles ", self.tile_ptrs)
                elif self.curr_tile == "D":
                    while len(self.tile_ptrs) > 0 and self.tile_ptrs[0] == "D":
                        self.tile_ptrs.pop(0)
                        # print("REMOVE DONE ", self.name, self.tile_ptrs_fifo, " ", self.ready)
                        assert self.tile_sizes.pop(0) == 0
                        if len(self.tile_ptrs) > 0:
                            self.curr_tile = self.tile_ptrs[0]
                    if len(self.tile_ptrs) == 0:
                        self.curr_tile = "D"
                        self.valid = False
                    if len(self.tile_ptrs) > 0:
                        self.curr_tile = self.tile_ptrs[0]
                        self.valid = True

            if len(self.tile_ptrs) > 0:
                self.curr_tile = self.tile_ptrs[0]
            if len(self.tile_ptrs) > 1:
                self.next_tile = self.tile_ptrs[1]
            else:
                self.next_tile = None

            if self.curr_tile == "D" and len(self.tile_ptrs) < 2 and len(self.tile_ptrs_fifo) == 0:
                # self.timestamp = None
                self.ready = True
                # self.loading = False
                self.valid = False
                self.outputed = False
                self.done = True
                self.done_received = False
                self.done_processed = False
                if self.debug:
                    print("Done case:: ", self.name, " valid: ", self.valid, " ready: ", self.ready,
                          " loading: ", self.loading, " done: ", self.done, " downstream token: ",
                          self.downstream_token, " Done received and processed ", self.done_received,
                          " ", self.done_processed, " : current tile: ", self.curr_tile, " full tles ", self.tile_ptrs)
            # Determines Ready
            if self.curr_size != self.load_size + sum(self.tile_sizes) + self.remove_size:
                print(self.curr_size, self.load_size, self.tile_sizes, self.remove_size,
                      "::", self.tile_ptrs, self.curr_tile, self.tile_ptrs_fifo)
                assert False
            if len(self.tile_ptrs_fifo) > 0:
                if len(self.tile_ptrs) > 0 and self.tile_ptrs_fifo[0] != self.tile_ptrs[-1]:
                    assert self.tile_ptrs_size[0] < self.size
                    if self.tile_ptrs_size[0] + self.curr_size < self.size:
                        self.ready = True
                    else:
                        self.ready = False
                if len(self.tile_ptrs) > 0 and self.tile_ptrs_fifo[0] == self.tile_ptrs[-1]:
                    self.ready = True
                if len(self.tile_ptrs) == 0 and self.curr_size < self.size:
                    if self.curr_size != self.load_size + self.remove_size:
                        print(self.curr_size, self.load_size, self.tile_sizes, self.remove_size,
                              "::", self.tile_ptrs, self.curr_tile, self.tile_ptrs_fifo)
                    assert self.curr_size == self.load_size + self.remove_size
                    self.ready = True
            else:
                self.ready = True
            # Dram pipelining
            if self.pipeline_en and self.ready and self.loading:
                temp_size = 0
                for i in range(len(self.if_latency_)):
                    temp_size += self.tile_ptrs_size[i]
                    if self.curr_size + temp_size < self.size:
                        self.if_latency_[i] = False
                    else:
                        break

            # Actual transfer of data
            if self.ready and len(self.tile_ptrs_fifo) > 0 and \
                    len(self.tile_ptrs) > 0 and self.tile_ptrs[-1] == self.tile_ptrs_fifo[0] and not self.loading:
                if self.tile_ptrs_fifo[0] == "D":
                    self.done_in = True
                if self.get_stats:
                    self.rep_true = True
                tile = self.tile_ptrs_fifo.pop(0)
                # Remove the tile's latency record
                self.if_latency_.pop(0)
                self.tile_ptrs_size.pop(0)
                self.tile_ptrs.append(tile)
                self.tile_sizes.append(0)
            elif self.ready and len(self.tile_ptrs_fifo) > 0 and \
                    (self.tile_ptrs_fifo[0] == "D" or len(self.tile_ptrs) == 0 or
                        self.tile_ptrs[-1] != self.tile_ptrs_fifo[0]) and not self.loading:
                # print(self.loading)
                if self.curr_size != self.load_size + sum(self.tile_sizes) + self.remove_size:
                    print(self.name, " ", self.curr_size, self.load_size, self.tile_sizes,
                          self.remove_size, "::", self.tile_ptrs, self.curr_tile,
                          self.loading_tile, self.tile_ptrs_fifo)
                    assert False
                if self.get_stats:
                    self.rep_true = False
                self.loading = True
                tile = self.tile_ptrs_fifo.pop(0)
                self.load_size = self.tile_ptrs_size[0]
                self.curr_size += self.tile_ptrs_size.pop(0)
                self.loading_tile = tile
                self.timestamp = cyclenum
                self.if_latency = self.if_latency_.pop(0)
                if self.curr_size != self.load_size + sum(self.tile_sizes) + self.remove_size:
                    print(self.name, " ", self.curr_size, self.load_size, self.tile_sizes,
                          self.remove_size, "::", self.tile_ptrs, self.curr_tile,
                          self.loading_tile, self.tile_ptrs_fifo)
                    assert False

            elif self.loading and cyclenum > self.compute_latency(self.load_size, self.if_latency) + self.timestamp:
                self.loading = False
                if self.get_stats:
                    self.rep_true = False
                if self.loading_tile == "D" and (len(self.tile_ptrs) > 1 or len(self.tile_ptrs_fifo) > 1):
                    self.done_in = True
                    # self.load_size = 0
                elif self.loading_tile == "D":
                    self.done = True
                    self.done_in = True
                self.tile_ptrs.append(self.loading_tile)
                self.tile_sizes.append(self.load_size)
                # self.repeat_pattern.append("S")
                self.load_size = 0
                self.loading_tile = None

        else:
            self.signalled = False
            if self.ready and len(self.tile_ptrs_fifo) > 0:
                if self.curr_tile != "D":
                    self.old_tile = self.curr_tile
                self.curr_tile = self.tile_ptrs_fifo.pop(0)
                self.curr_size = self.tile_ptrs_size.pop(0)
                if self.mode == "all_unpacked":
                    assert self.curr_size < (self.size)
                elif self.mode == "not_consolidated":
                    assert self.curr_size < self.size
                else:
                    print(self.mode + " not found")
                if self.curr_tile == "D":
                    self.timestamp = None
                    self.ready = True
                    self.loading = False
                    self.valid = False
                    # self.curr_tile = None
                    self.outputed = False
                    self.done = True
                    self.signalled = True
                    self.done_received = False
                    self.done_processed = False
                    if self.debug:
                        print(self.name, " valid: ", self.valid, " ready: ", self.ready,
                              " loading: ", self.loading, " done: ", self.done, " downstream token: ",
                              self.downstream_token, " Done received and processed ", self.done_received,
                              " ", self.done_processed, " : current tile: ", self.curr_tile)
                    return
                self.timestamp = cyclenum
                self.ready = False
                self.loading = True
                self.valid = False
                # self.done_processed = False
            elif self.loading and cyclenum < self.timestamp + self.compute_latency(self.curr_tile):
                pass
            elif self.loading and cyclenum > self.timestamp + self.compute_latency(self.curr_tile):
                self.loading = False
                self.valid = True
                self.outputed = False
            elif self.done_received and not self.done_processed:
                self.ready = True
                self.done_processed = True
                self.valid = False
            elif not self.ready and not self.loading and self.input_token():
                self.ready = True
                self.valid = False
            if self.done_processed and not self.done_received:
                self.done_processed = False
        if self.debug:
            if self.nbuffer:
                print(self.name, self.curr_size, self.size, self.old_tile, " done in ", self.done_in,
                      " valid: ", self.valid, " ready: ", self.ready, " loading: ", self.loading,
                      " done: ", self.done, " downstream token: ", self.downstream_token,
                      " Done received and processed ", self.done_received, " ", self.done_processed,
                      " : current tile: ", self.curr_tile, " ", self.tile_ptrs, " ", self.loading_tile, " ",
                      self.tile_ptrs_fifo, "----------")
            else:
                print(self.name, self.old_tile, " valid: ", self.valid, " ready: ", self.ready, " loading: ",
                      self.loading, " done: ", self.done, " downstream token: ", self.downstream_token,
                      " Done received and processed ", self.done_received, " ", self.done_processed,
                      " : current tile: ", self.curr_tile, " ", self.tile_ptrs[0:10], " ",
                      self.tile_ptrs_fifo[0:10], "----------")

    def remove_tile(self, tile_ptr=None, tile_id=-1):
        if tile_ptr is not None:
            index = self.tile_ptrs_ids.index(tile_ptr.getid())
            self.tile_ptrs_ids.pop(index)
            self.tile_ptrs_tile.pop(index)

    def pop_tile(self, lower_level_done):
        if lower_level_done:
            self.ready = True

    def pop_tile_after(self, cycle_num, time):
        if isinstance(self.curr_tile, int) and cycle_num > self.timestamp + self.compute_latency(self.curr_tile) + time:
            self.ready = True

    def token(self):
        return self.curr_tile

    def get_size(self):
        return self.curr_size

    def check_tile(self, tile_id):
        if tile_id in self.tile_ptrs_ids:
            return True
        return False

    def compute_latency(self, tile, if_latency=True):
        if self.nbuffer or self.full_buff:
            # if self.loading_tile == "D":
            #     return 1
            if self.skip_blocks and self.curr_size == 0 and self.loading_tile is not None:
                if self.pipeline_en and not if_latency:
                    return 1
                else:
                    return self.latency
            if self.loading_tile is not None and self.old_tile is not None and self.loading_tile == self.old_tile:
                return 1
            if self.pipeline_en and not if_latency:
                return 1 + (tile * self.element_size) // (self.bandwidth)
            return self.latency + (tile * self.element_size) // (self.bandwidth)
        else:
            if self.skip_blocks and self.curr_size == 0 and self.curr_tile is not None:
                return self.latency
            if self.curr_tile == self.old_tile:
                return 1
            if self.mode == "all_unpacked":
                return self.latency + (self.curr_size * self.element_size) // (self.bandwidth)
            elif self.mode == "not_consolidated":
                return self.latency + self.curr_size // (self.bandwidth)
            else:
                print(self.mode + " not found")
                assert False

    def valid_tile(self):
        if self.valid and not self.outputed:
            return True

    def if_valid(self):
        return self.valid

    def valid_tile_recieved(self):
        # if self.debug:
        # print(self.name, " returns ", self.curr_tile)
        self.outputed = True

    def valid_tile_received(self):
        # if self.debug:
        # print(self.name, " returns ", self.curr_tile)
        self.outputed = True

    def if_stop(self):
        if self.done:
            return True

    def input_token(self):
        if self.downstream_token == "D":
            return True

    def input_token_(self, token):
        self.downstream_token = token
