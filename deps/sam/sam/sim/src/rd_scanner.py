from .base import *


#################
# Read Scanners
#################


class CrdRdScan(Primitive, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.curr_ref = ''
        self.curr_crd = ''

        self.in_ref = []
        if self.backpressure_en:
            self.fifo_avail = True
            self.data_valid = True
            self.ready_backpressure = True

    def set_in_ref(self, in_ref, parent=None):
        if in_ref != '' and in_ref is not None:
            self.in_ref.append(in_ref)
        if self.backpressure_en and parent != "":
            parent.set_backpressure(self.fifo_avail)

    def set_backpressure(self, backpressure):
        if not backpressure:
            self.ready_backpressure = False

    def check_backpressure(self):
        if self.backpressure_en:
            copy_backpressure = self.ready_backpressure
            self.ready_backpressure = True
            return copy_backpressure
        return True

    def out_ref(self, child=None):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.curr_ref

    def out_crd(self, child=None):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.curr_crd

    def update_ready(self):
        if self.backpressure_en:
            if len(self.in_ref) > self.depth:
                self.fifo_avail = False
            else:
                self.fifo_avail = True

    def fifo_debug(self):
        print("Crd rd: ", self.in_ref)


# TODO: figure out how uncompressed read scans work with 'N' tokens
class UncompressCrdRdScan(CrdRdScan):
    def __init__(self, dim=0, depth=4, **kwargs):
        super().__init__(**kwargs)

        self.start_addr = 0
        self.stop_addr = dim

        self.curr_in_ref = 0

        self.meta_dim = dim

        self.end_fiber = False
        self.emit_tkn = False

        self.begin = True
        if self.backpressure_en:
            self.depth = depth
            self.data_valid = True
            self.fifo_avail = True
            self.ready_backpressure = True

    def update(self):
        self.update_done()
        self.update_ready()
        if self.backpressure_en:
            self.data_valid = False
        if (self.backpressure_en and self.check_backpressure()) or not self.backpressure_en:
            if len(self.in_ref) > 0:
                self.block_start = False
            if self.backpressure_en:
                self.data_valid = True
            if self.emit_tkn and len(self.in_ref) > 0:
                next_in = self.in_ref[0]
                if is_stkn(next_in):
                    self.in_ref.pop(0)
                    stkn = increment_stkn(next_in)
                else:
                    stkn = 'S0'
                self.curr_crd = stkn
                self.curr_ref = stkn
                self.emit_tkn = False
                return
            elif self.end_fiber and len(self.in_ref) > 0:
                self.curr_in_ref = self.in_ref.pop(0)
                if self.curr_in_ref == 'D':
                    self.curr_crd = 'D'
                    self.curr_ref = 'D'
                    self.done = True
                    self.end_fiber = False
                    return
                elif is_stkn(self.curr_in_ref):
                    if len(self.in_ref) > 0:
                        next_in = self.in_ref[0]
                        if is_stkn(next_in):
                            self.in_ref.pop(0)
                            stkn = increment_stkn(next_in)
                        else:
                            stkn = 'S0'
                        self.curr_crd = stkn
                        self.curr_ref = stkn
                    else:
                        self.curr_crd = ''
                        self.curr_ref = ''
                        self.emit_tkn = True
                else:
                    self.curr_crd = 0
                    self.curr_ref = self.curr_crd + (self.curr_in_ref * self.meta_dim)
                    self.end_fiber = False
                    return
            elif self.end_fiber or self.emit_tkn:
                self.curr_crd = ''
                self.curr_ref = ''
                return

            if is_stkn(self.curr_crd) or self.begin:
                self.begin = False
                if len(self.in_ref) > 0:
                    self.curr_in_ref = self.in_ref.pop(0)
                    if self.curr_in_ref == 'D':
                        self.curr_crd = 'D'
                        self.curr_ref = 'D'
                        self.done = True
                        return
                    elif is_stkn(self.curr_in_ref):
                        stkn = increment_stkn(self.curr_in_ref)
                        self.curr_crd = stkn
                        self.curr_ref = stkn
                        return
                    elif is_0tkn(self.curr_in_ref):
                        self.curr_crd = ''
                        self.curr_ref = ''
                        self.emit_tkn = True
                        return
                    else:
                        self.curr_crd = 0
                        self.curr_ref = self.curr_crd + (self.curr_in_ref * self.meta_dim)
                else:
                    self.curr_crd = ''
                    self.curr_ref = ''
                    self.end_fiber = True
            # run out of coordinates, move to next input reference
            elif self.curr_crd == '' or self.curr_crd == 'D':
                self.curr_crd = ''
                self.curr_ref = ''
            elif self.curr_crd >= self.meta_dim - 1:
                if len(self.in_ref) > 0:
                    next_in = self.in_ref[0]
                    if is_stkn(next_in):
                        self.in_ref.pop(0)
                        stkn = increment_stkn(next_in)
                    else:
                        stkn = 'S0'
                    self.curr_crd = stkn
                    self.curr_ref = stkn
                else:
                    self.curr_crd = ''
                    self.curr_ref = ''
                    self.emit_tkn = True
            else:
                self.curr_crd += 1
                self.curr_ref = self.curr_crd + self.curr_in_ref * self.meta_dim

            if self.debug:
                print("DEBUG: U RD SCAN: \t "
                      "Curr inref:", self.curr_in_ref, "\tEmit tkn:", self.emit_tkn, "\tEnd Fiber", self.end_fiber,
                      "\nCurr crd:", self.curr_crd, "\t curr ref:", self.curr_ref)


def last_stkn(skiplist):
    max_ref = None
    for i, item in enumerate(skiplist):
        if is_stkn(item):
            max_ref = i
    return max_ref


class CompressedCrdRdScan(CrdRdScan):
    def __init__(self, crd_arr=[], seg_arr=[], skip=True, depth=1, tile_size=None, fifo=None, **kwargs):
        super().__init__(**kwargs)

        if tile_size is not None:
            assert len(crd_arr) < tile_size

        # Used for skip list
        self.skip = skip
        self.in_crd_skip = []
        self.curr_skip = None
        self.skip_processed = True
        self.prev_crd = 0
        # [Olivia]: I think this is needed to make sure we are looking at the correct fiber

        self.crd_arr = crd_arr
        self.seg_arr = seg_arr
        self.start_addr = 0
        self.stop_addr = 0

        self.curr_addr = None

        self.end_fiber = False
        self.curr_ref = ''
        self.curr_crd = ''
        self.emit_fiber_stkn = False
        self.meta_clen = len(crd_arr)
        self.meta_slen = len(seg_arr)

        # Statistics
        if self.get_stats:
            self.unique_refs = []
            # self.unique_crds = []
            self.total_outputs = 0
            self.elements_skipped = 0
            self.skip_cnt = 0
            self.intersection_behind_cnt = 0
            self.fiber_behind_cnt = 0
            self.stop_count = 0
            self.empty_tkn_cnt = 0
        self.skip_stkn_cnt = 0  # also used as a statistic
        self.out_stkn_cnt = 0

        self.begin = True
        if self.backpressure_en:
            self.depth = depth
            self.data_valid = True
            self.fifo_avail = True
            self.ready_backpressure = True
        if fifo is not None:
            self.set_fifo(fifo)

    # FIXME (Ritvik): Use reinitialize array isntead of redeclaring the rd scanner
    def reinitialize_arrs(self, seg_arr, crd_arr, fifo):
        # assert False
        self.start_addr = 0
        self.stop_addr = 0
        self.end_fiber = False
        # self.curr_ref = ''
        # self.curr_crd = ''
        self.emit_fiber_stkn = False
        self.meta_clen = len(crd_arr)
        self.meta_slen = len(seg_arr)
        self.skip_stkn_cnt = 0
        self.out_stkn_cnt = 0
        # self.begin = True
        self.seg_arr = seg_arr
        self.crd_arr = crd_arr
        print(fifo)
        for a_ in fifo:
            print(self.in_ref)
            self.in_ref.append(a_)

        # if fifo is not None:
        #     self.set_fifo(fifo)
        # assert False
        self.done = False
        # print("+++++++++")
        return

    def set_fifo(self, fifo):
        for a in fifo:
            self.in_ref.append(a)
        return

    def get_fifo(self):
        return self.in_ref

    def set_in_ref(self, in_ref, parent=None):
        if in_ref != '' and in_ref is not None:
            self.in_ref.append(in_ref)
        if self.backpressure_en and parent != "":
            parent.set_backpressure(self.fifo_avail)

    def out_ref(self, child=None):
        if self.backpressure_en and self.data_valid:
            return self.curr_ref
        elif self.backpressure_en:
            return
        if not self.backpressure_en:
            return self.curr_ref

    def out_crd(self, child=None):
        if self.backpressure_en and self.data_valid:
            return self.curr_crd
        elif self.backpressure_en:
            return
        if not self.backpressure_en:
            return self.curr_crd

    def _emit_stkn_code(self):
        self.end_fiber = True

        if len(self.in_ref) > 0:
            next_in = self.in_ref[0]
            if is_stkn(next_in):
                self.in_ref.pop(0)
                stkn = increment_stkn(next_in)
            else:
                stkn = 'S0'
            self.emit_fiber_stkn = False
        else:
            self.emit_fiber_stkn = True
            stkn = ''
        self.curr_crd = stkn
        self.curr_ref = stkn
        self.curr_addr = None
        self.stop_addr = 0
        self.start_addr = 0

    def _set_curr(self):
        self.curr_ref = self.curr_addr
        self.curr_crd = self.crd_arr[self.curr_addr]
        if self.get_stats:
            # if self.curr_ref not in self.unique_refs:
            #    self.unique_refs.append(self.curr_ref)
            # if self.curr_crd not in self.unique_crds:
            #    self.unique_crds.append(self.curr_crd)
            self.total_outputs += 1

    def return_statistics(self):
        if self.get_stats:
            dic = {"total_size": len(self.crd_arr), "outputs_by_block": self.total_outputs,
                   "unique_refs": len(self.unique_refs),
                   "skip_list_fifo": len(self.in_crd_skip), "total_elements_skipped": self.elements_skipped,
                   "total_skips_encountered": self.skip_cnt, "intersection_behind_rd": self.intersection_behind_cnt,
                   "intersection_behind_fiber": self.fiber_behind_cnt, "stop_tokens": self.stop_count,
                   "skip_stp_tkn_cnt": self.skip_stkn_cnt}
            dic.update(super().return_statistics())
        else:
            dic = {}
        return dic

    def check_backpressure(self):
        if self.backpressure_en:
            copy_backpressure = self.ready_backpressure
            self.ready_backpressure = True
            return copy_backpressure
        return True

    def update(self):
        self.update_done()
        self.update_ready()
        if self.backpressure_en:
            self.data_valid = False
        if (self.backpressure_en and self.check_backpressure()) or not self.backpressure_en:
            if self.backpressure_en:
                self.data_valid = True
            if len(self.in_ref) > 0:
                self.block_start = False

            # Process skip token first and save
            if len(self.in_crd_skip) > 0 and self.skip_processed:
                self.curr_skip = self.in_crd_skip.pop(0)
                if self.skip_stkn_cnt == self.out_stkn_cnt and isinstance(self.curr_skip, int) \
                        and self.curr_skip < self.prev_crd:
                    # ignore the skip if it's too small
                    self.skip_processed = True
                    if self.get_stats:
                        self.intersection_behind_cnt += 1
                elif self.skip_stkn_cnt < self.out_stkn_cnt:
                    # ignore the skip if it's a fiber behind
                    self.skip_processed = True
                    if self.get_stats:
                        self.fiber_behind_cnt += 1
                else:
                    self.skip_processed = False

                if is_stkn(self.curr_skip):
                    self.skip_stkn_cnt += 1

            curr_in_ref = None
            # After Done token has been seen and outputted, do nothing
            if self.curr_crd == 'D' or self.curr_ref == 'D' or self.done:
                self.curr_addr = 0
                self.stop_addr = 0
                self.start_addr = 0
                self.curr_crd = ''
                self.curr_ref = ''
                return
                # Reset
                # self.done = False

            # Scanner needs to emit stop token and the next element has finally arrived.
            # Previously set emit_fiber_stkn to True but wait on next in_ref
            elif len(self.in_ref) > 0 and self.emit_fiber_stkn:
                next_in = self.in_ref[0]
                if is_stkn(next_in):
                    self.in_ref.pop(0)
                    stkn = increment_stkn(next_in)
                else:
                    stkn = 'S0'
                self.curr_crd = stkn
                self.curr_ref = stkn

                self.curr_addr = None
                self.stop_addr = 0
                self.start_addr = 0
                self.emit_fiber_stkn = False

            # There exists another input reference at the segment and
            # either at the start of computation or end of fiber
            elif len(self.in_ref) > 0 and (self.end_fiber or self.begin):
                self.begin = False
                self.end_fiber = False

                curr_in_ref = self.in_ref.pop(0)

                # Input reference is out of bounds
                if isinstance(curr_in_ref, int) and curr_in_ref + 1 > self.meta_slen:
                    print(curr_in_ref, self.meta_slen, self.in_ref)
                    raise Exception('Not enough elements in seg array')

                # Input reference is a done token, so forward that token (and set done if done token)
                elif curr_in_ref == 'D':
                    self.curr_addr = None
                    self.stop_addr = 0
                    self.start_addr = 0
                    self.curr_crd = curr_in_ref
                    self.curr_ref = curr_in_ref
                    self.end_fiber = True
                    self.done = True

                # Input reference is a stop token, so increment and forward that token
                elif is_stkn(curr_in_ref):
                    self.curr_addr = None
                    self.stop_addr = 0
                    self.start_addr = 0
                    self.curr_crd = increment_stkn(curr_in_ref)
                    self.curr_ref = increment_stkn(curr_in_ref)
                    self.end_fiber = True

                # See 'N' 0-token which immediately emits a stop token and ends the fiber
                elif is_0tkn(curr_in_ref):
                    self.curr_crd = 'N'
                    self.curr_ref = 'N'
                    self.end_fiber = True
                    self.emit_fiber_stkn = True
                    self.curr_addr = None
                    self.stop_addr = 0
                    self.start_addr = 0

                # Default case where input reference is an integer value
                # which means to get the segment
                else:
                    self.start_addr = self.seg_arr[curr_in_ref]
                    self.stop_addr = self.seg_arr[curr_in_ref + 1]
                    self.curr_addr = self.start_addr

                    # This case is if the segment has no coordinates (i.e. 5, 5)
                    if self.curr_addr >= self.stop_addr:
                        # End of fiber, get next input reference
                        self._emit_stkn_code()

                    # Default behave normally and emit the coordinates in the segment
                    else:
                        if self.skip and not self.skip_processed:
                            # assert self.out_stkn_cnt == self.skip_stkn_cnt
                            curr_range = self.crd_arr[self.start_addr: self.stop_addr]
                            # Skip to next coordinate
                            if isinstance(self.curr_skip, int) \
                                    and self.curr_skip > self.prev_crd:
                                print("RD SCAN: SKIP HERE")
                                # If coordinate skipped to exists, emit that
                                if self.curr_skip in curr_range:
                                    self.curr_addr = curr_range.index(self.curr_skip) + self.start_addr
                                    self._set_curr()
                                    if self.get_stats:
                                        self.elements_skipped += curr_range.index(self.curr_skip) + 1
                                        self.skip_cnt += 1

                                # Else emit smallest coordinate larger than the one provided by skip
                                else:
                                    larger = [i for i in curr_range if i > self.curr_skip]
                                    if not larger:
                                        self._emit_stkn_code()
                                        if self.get_stats:
                                            self.elements_skipped += len(curr_range)
                                            self.skip_cnt += 1
                                    else:
                                        val_larger = min(larger)
                                        self.curr_addr = curr_range.index(val_larger) + self.start_addr
                                        self._set_curr()
                                        if self.get_stats:
                                            self.elements_skipped += curr_range.index(val_larger) + 1
                                            self.skip_cnt += 1

                            # Early exit from skip
                            elif is_stkn(self.curr_skip):
                                self._emit_stkn_code()
                            self.skip_processed = True
                        # Else behave normally
                        else:
                            self._set_curr()

            # Finished emitting coordinates and have reached the end of the fiber for this level
            elif (self.curr_addr == self.stop_addr - 1 or self.curr_addr == self.meta_clen - 1) and \
                    not self.begin:
                # End of fiber, get next input reference
                self._emit_stkn_code()

            # Base case: increment address and reference by 1 and get next coordinate
            elif len(self.in_ref) > 0 and not self.begin:
                default_behavior = True
                if self.skip and not self.skip_processed:
                    # assert self.out_stkn_cnt == self.skip_stkn_cnt
                    curr_range = self.crd_arr[self.start_addr: self.stop_addr]
                    if isinstance(self.curr_skip, int) \
                            and self.curr_skip > self.prev_crd:
                        print("RD SCAN: SKIP HERE")
                        # If coordinate skipped to exists, emit that
                        if self.curr_skip in curr_range:
                            self.curr_addr = curr_range.index(self.curr_skip) + self.start_addr
                            self._set_curr()
                            if self.get_stats:
                                self.elements_skipped += curr_range.index(self.curr_skip) + 1
                                self.skip_cnt += 1

                        # Else emit smallest coordinate larger than the one provided by skip
                        else:
                            larger = [i for i in curr_range if i > self.curr_skip]
                            if not larger:
                                self._emit_stkn_code()
                                if self.get_stats:
                                    self.elements_skipped += len(curr_range)
                                    self.skip_cnt += 1
                            else:
                                val_larger = min(larger)
                                self.curr_addr = curr_range.index(val_larger) + self.start_addr
                                self._set_curr()
                                if self.get_stats:
                                    self.elements_skipped += curr_range.index(val_larger) + 1
                                    self.skip_cnt += 1

                        default_behavior = False
                    elif is_stkn(self.curr_skip):
                        self._emit_stkn_code()
                        default_behavior = False
                    self.skip_processed = True

                if default_behavior:
                    self.curr_addr += 1
                    self._set_curr()

            # Default stall (when done)
            elif not self.begin:
                self.curr_ref = ''
                self.curr_crd = ''

            # Needed for skip lists
            if is_stkn(self.curr_crd):
                self.out_stkn_cnt += 1
            # Needed for skip lists
            if self.skip_stkn_cnt < self.out_stkn_cnt:
                # ignore the skip if it's a fiber behind
                self.skip_processed = True
            # Needed for skip lists
            if isinstance(self.curr_crd, int):
                self.prev_crd = self.curr_crd

            if self.get_stats and is_stkn(self.curr_crd):
                self.stop_count += 1

            # Debugging print statements
            if self.debug and self.backpressure_en:
                print("DEBUG: C RD SCAN:"
                      "\n \t"
                      "name: ", self.name, "\t ref", len(self.in_ref), " :: ", self.in_ref,
                      "\t Curr crd:", self.curr_crd, "\t curr ref:", self.curr_ref,
                      "\n curr addr:", self.curr_addr, "\t start addr:", self.start_addr, "\t stop addr:", self.stop_addr,
                      "\n end fiber:", self.end_fiber, "\t curr input:", curr_in_ref,
                      "\n skip in:", self.curr_skip, "\t skip processed", self.skip_processed, "\t prev crd:",
                      self.prev_crd,
                      "\n Out stkn cnt:", self.out_stkn_cnt, "\t Skip stkn cnt:", self.skip_stkn_cnt, "\t Bakcpressure: ",
                      self.check_backpressure(), "\t backpressure_len: ",
                      self.data_valid, self.done)
            elif self.debug:
                print("DEBUG: C RD SCAN:"
                      "\n \t"
                      "name: ", self.name, "\t ref", len(self.in_ref), " :: ", self.in_ref,
                      "\t Curr crd:", self.curr_crd, "\t curr ref:", self.curr_ref,
                      "\n curr addr:", self.curr_addr, "\t start addr:", self.start_addr, "\t stop addr:", self.stop_addr,
                      "\n end fiber:", self.end_fiber, "\t curr input:", curr_in_ref,
                      "\n skip in:", self.curr_skip, "\t skip processed", self.skip_processed, "\t prev crd:",
                      self.prev_crd,
                      "\n Out stkn cnt:", self.out_stkn_cnt, "\t Skip stkn cnt:", self.skip_stkn_cnt, self.done)
        else:
            # Debugging print statements
            if self.debug and self.backpressure_en:
                print("DEBUG: C RD SCAN:"
                      "\n \t"
                      "name: ", self.name, "\t ref", len(self.in_ref), " :: ", self.in_ref,
                      "\t Curr crd:", self.curr_crd, "\t curr ref:", self.curr_ref,
                      "\n curr addr:", self.curr_addr, "\t start addr:", self.start_addr, "\t stop addr:", self.stop_addr,
                      "\n skip in:", self.curr_skip, "\t skip processed", self.skip_processed, "\t prev crd:",
                      self.prev_crd,
                      "\n Out stkn cnt:", self.out_stkn_cnt, "\t Skip stkn cnt:", self.skip_stkn_cnt, "\t Bakcpressure: ",
                      self.data_valid)
            elif self.debug:
                print("DEBUG: C RD SCAN:"
                      "\n \t"
                      "name: ", self.name, "\t ref", len(self.in_ref), " :: ", self.in_ref,
                      "\t Curr crd:", self.curr_crd, "\t curr ref:", self.curr_ref,
                      "\n curr addr:", self.curr_addr, "\t start addr:", self.start_addr, "\t stop addr:", self.stop_addr,
                      "\n skip in:", self.curr_skip, "\t skip processed", self.skip_processed, "\t prev crd:",
                      self.prev_crd,
                      "\n Out stkn cnt:", self.out_stkn_cnt, "\t Skip stkn cnt:", self.skip_stkn_cnt, self.done)
            # # Debugging print statements
            # if self.debug:
            #    print("DEBUG: C RD SCAN:"
            #          "\n \t"
            #          "name: ", self.name, "\t ref", len(self.in_ref), " :: ", self.in_ref,
            #          "\t Curr crd:", self.curr_crd, "\t curr ref:", self.curr_ref,
            #          "\n curr addr:", self.curr_addr, "\t start addr:", self.start_addr, "\t stop addr:", self.stop_addr,
            #          "\n end fiber:", self.end_fiber, "\t curr input:", curr_in_ref,
            #          "\n skip in:", self.curr_skip, "\t skip processed", self.skip_processed, "\t prev crd:",
            #          self.prev_crd,
            #          "\n emit_fiber_stkn:", self.emit_fiber_stkn,
            #          "\n Out stkn cnt:", self.out_stkn_cnt, "\t Skip stkn cnt:", self.skip_stkn_cnt)

    def set_crd_skip(self, in_crd, parent=None):
        assert in_crd is None or is_valid_crd(in_crd)
        if in_crd != '' and in_crd is not None:
            if is_stkn(in_crd):
                idx = last_stkn(self.in_crd_skip)
                if idx is not None:
                    # Flush coordinates
                    self.in_crd_skip = self.in_crd_skip[:idx + 1]
            self.in_crd_skip.append(in_crd)
        if self.backpressure_en and parent != "":
            parent.set_backpressure(self.fifo_avail)


class VirtualBuffer():

    def __init__(self,
                 depth) -> None:
        self._fifo = list()
        self.depth = depth

    def full(self):
        return len(self._fifo) == self.depth

    def empty(self):
        return len(self._fifo) == 0

    def valid(self):
        if not self.empty():
            timestamp = self._fifo[0][1]
            if timestamp is None:
                return True
            else:
                return timestamp == 0
        else:
            return False

    def data(self):
        if self.empty():
            return 0
        else:
            return self._fifo[0][0]

    def time(self):
        if self.empty():
            return None
        else:
            return self._fifo[0][1]

    def push(self, data, time=None):

        use_time = 1

        if time is not None:
            assert time >= 1
            use_time = time

        self._fifo.append((data, use_time))

    def pop(self):
        self._fifo.pop(0)

    def tick(self):

        for idx, (data, time) in enumerate(self._fifo):
            if time is not None:
                if time != 0:
                    self._fifo[idx] = (data, time - 1)

    def __str__(self) -> str:
        return str(self._fifo)


class CompressedCrdRdScanModel(CrdRdScan):
    def __init__(self, crd_arr=[], seg_arr=[], skip=True, depth=1, tile_size=None, fifo=None, **kwargs):
        super().__init__(**kwargs)

        if tile_size is not None:
            assert len(crd_arr) < tile_size

        # Used for skip list
        self.skip = skip
        self.in_crd_skip = []
        self.curr_skip = None
        self.skip_processed = True
        self.prev_crd = 0
        # [Olivia]: I think this is needed to make sure we are looking at the correct fiber

        self.crd_arr = crd_arr
        self.seg_arr = seg_arr
        self.start_addr = 0
        self.stop_addr = 0

        # self.curr_addr = None
        self.curr_addr = 0

        self.end_fiber = False
        self.curr_ref = ''
        self.curr_crd = ''
        self.emit_fiber_stkn = False
        self.meta_clen = len(crd_arr)
        self.meta_slen = len(seg_arr)

        self.emit_fiber_stkn = False
        self.fresh_fiber = True

        self._seg_vb0 = VirtualBuffer(8)
        self._seg_vb1 = VirtualBuffer(8)
        self._crd_vb = VirtualBuffer(8)
        self._ref_vb = VirtualBuffer(8)

        self.pop_out_crd = False
        self.pop_out_ref = False

        self.all_vbs = [self._seg_vb0, self._seg_vb1, self._crd_vb, self._ref_vb]

        # Statistics
        if self.get_stats:
            self.unique_refs = []
            # self.unique_crds = []
            self.total_outputs = 0
            self.elements_skipped = 0
            self.skip_cnt = 0
            self.intersection_behind_cnt = 0
            self.fiber_behind_cnt = 0
            self.stop_count = 0
            self.empty_tkn_cnt = 0
        self.skip_stkn_cnt = 0  # also used as a statistic
        self.out_stkn_cnt = 0

        self.begin = True
        if self.backpressure_en:
            self.depth = depth
            self.data_valid = True
            self.fifo_avail = True
            self.ready_backpressure = True
        if fifo is not None:
            self.set_fifo(fifo)

    def reinitialize_arrs(self, seg_arr, crd_arr, fifo):
        # assert False
        self.start_addr = 0
        self.stop_addr = 0
        self.end_fiber = False
        # self.curr_ref = ''
        # self.curr_crd = ''
        self.emit_fiber_stkn = False
        self.meta_clen = len(crd_arr)
        self.meta_slen = len(seg_arr)
        self.skip_stkn_cnt = 0
        self.out_stkn_cnt = 0
        # self.begin = True
        self.seg_arr = seg_arr
        self.crd_arr = crd_arr
        print(fifo)
        for a_ in fifo:
            print(self.in_ref)
            self.in_ref.append(a_)

        # if fifo is not None:
        #     self.set_fifo(fifo)
        # assert False
        self.done = False
        # print("+++++++++")
        return

    def set_fifo(self, fifo):
        for a in fifo:
            self.in_ref.append(a)
        return

    def get_fifo(self):
        return self.in_ref

    def fifo_available(self, br=""):
        if self.backpressure_en and len(self.in_ref) > self.depth:
            return False
        return True

    def set_in_ref(self, in_ref, parent=None):
        if in_ref != '' and in_ref is not None:
            self.in_ref.append(in_ref)
        if self.backpressure_en and parent != "":
            parent.set_backpressure(self.fifo_avail)

    def out_ref(self, child=None):
        # if self.backpressure_en and self.data_valid:
        if self.backpressure_en and self._ref_vb.valid() and self._crd_vb.valid():
            # return self.curr_ref
            self.pop_out_ref = True
            return self._ref_vb.data()
        elif self.backpressure_en:
            return
        if not self.backpressure_en:
            # return self.curr_ref
            # if self._ref_vb.valid():
            if self._crd_vb.valid() and self._ref_vb.valid():
                dat_ret = self._ref_vb.data()
                self.pop_out_ref = True
                # self._ref_vb.pop()
                return dat_ret
            else:
                return

    def out_crd(self, child=None):
        # if self.backpressure_en and self.data_valid:
        if self.backpressure_en and self._crd_vb.valid() and self._ref_vb.valid():
            # return self.curr_crd
            self.pop_out_crd = True
            return self._crd_vb.data()
        elif self.backpressure_en:
            return
        if not self.backpressure_en:
            # return self.curr_crd
            # return self._crd_vb.data()
            # if self._crd_vb.valid():
            if self._crd_vb.valid() and self._ref_vb.valid():
                dat_ret = self._crd_vb.data()
                # self._crd_vb.pop()
                self.pop_out_crd = True
                return dat_ret
            else:
                return

    def _emit_stkn_code(self):
        self.end_fiber = True

        if len(self.in_ref) > 0:
            next_in = self.in_ref[0]
            if is_stkn(next_in):
                self.in_ref.pop(0)
                stkn = increment_stkn(next_in)
            else:
                stkn = 'S0'
            self.emit_fiber_stkn = False
        else:
            self.emit_fiber_stkn = True
            stkn = ''
        self.curr_crd = stkn
        self.curr_ref = stkn
        self.curr_addr = None
        self.stop_addr = 0
        self.start_addr = 0

    def _set_curr(self):
        self.curr_ref = self.curr_addr
        self.curr_crd = self.crd_arr[self.curr_addr]
        if self.get_stats:
            # if self.curr_ref not in self.unique_refs:
            #    self.unique_refs.append(self.curr_ref)
            # if self.curr_crd not in self.unique_crds:
            #    self.unique_crds.append(self.curr_crd)
            self.total_outputs += 1

    def return_statistics(self):
        if self.get_stats:
            dic = {"total_size": len(self.crd_arr), "outputs_by_block": self.total_outputs,
                   "unique_refs": len(self.unique_refs),
                   "skip_list_fifo": len(self.in_crd_skip), "total_elements_skipped": self.elements_skipped,
                   "total_skips_encountered": self.skip_cnt, "intersection_behind_rd": self.intersection_behind_cnt,
                   "intersection_behind_fiber": self.fiber_behind_cnt, "stop_tokens": self.stop_count,
                   "skip_stp_tkn_cnt": self.skip_stkn_cnt}
            dic.update(super().return_statistics())
        else:
            dic = {}
        return dic

    def check_backpressure(self):
        if self.backpressure_en:
            copy_backpressure = self.ready_backpressure
            self.ready_backpressure = True
            return copy_backpressure
        return True

    def update(self):
        self.update_done()
        self.update_ready()

        # if len(self.in_ref) > 0:

        input_valid = len(self.in_ref) > 0
        # Basically if we have already emitted something, we know we should either emit S0 or increment stop token
        if input_valid:
            next_in = self.in_ref[0]

            # If stop token, we increment and push the stop token if there is room
            if is_stkn(next_in):
                # self.in_ref.pop(0)
                stkn = increment_stkn(next_in)
                if (not self._seg_vb0.full()) and (not self._seg_vb1.full()):
                    self._seg_vb0.push((stkn, 1))
                    self._seg_vb1.push((stkn, 1))
                    self.in_ref.pop(0)
                    self.emit_fiber_stkn = False
            elif next_in == 'D' and not self.emit_fiber_stkn:
                # Basically in the done phase if we are not emitting something from before
                if (not self._seg_vb0.full()) and (not self._seg_vb1.full()):
                    self._seg_vb0.push(('D', 1))
                    self._seg_vb1.push(('D', 1))
                    self.in_ref.pop(0)
                    self.emit_fiber_stkn = False
            else:
                # If input is valid not stop token, we want to spend this cycle forwarding the S0 if we marked that
                stkn = 'S0'
                # Emit the stkn if we need to
                if self.emit_fiber_stkn and (not self._seg_vb0.full()) and (not self._seg_vb1.full()):
                    self._seg_vb0.push((stkn, 1))
                    self._seg_vb1.push((stkn, 1))
                    self.emit_fiber_stkn = False
                # Otherwise, make our two requests and put them in the seg vbs
                elif (not self._seg_vb0.full()) and (not self._seg_vb1.full()):
                    self._seg_vb0.push((self.seg_arr[next_in], 0), 5)
                    self._seg_vb1.push((self.seg_arr[next_in + 1], 0), 6)
                    self.emit_fiber_stkn = True
                    self.in_ref.pop(0)

        # Now check if the seg vbs are valid
        if self._seg_vb0.valid() and self._seg_vb1.valid():
            # If it is a stop token, we pass it through if room
            if (self._seg_vb0.data()[1] == 1) and (not self._crd_vb.full()) and (not self._ref_vb.full()):
                data_to_pass = self._seg_vb0.data()[0]
                self._crd_vb.push(data_to_pass)
                self._ref_vb.push(data_to_pass)
                self._seg_vb0.pop()
                self._seg_vb1.pop()
                self.fresh_fiber = True
            # Otherwise, go through the fiber
            else:
                if self.fresh_fiber:
                    self.curr_addr = self._seg_vb0.data()[0]
                    self.fresh_fiber = False
                # Only make movement if there is reservation room
                if (not self._crd_vb.full()) and (not self._ref_vb.full()):
                    self._crd_vb.push(self.crd_arr[self.curr_addr], 6)
                    self._ref_vb.push(self.curr_addr)
                    # Increase the addr, if it is equal to the bound, we can
                    # pop it, set fresh fiber to True
                    self.curr_addr += 1
                    if self.curr_addr == self._seg_vb1.data()[0]:
                        self._seg_vb0.pop()
                        self._seg_vb1.pop()
                        self.fresh_fiber = True

        # Update the vbs
        for vb in self.all_vbs:
            vb.tick()

        print(f'seg_vb0: {self._seg_vb0}')
        print(f'seg_vb1: {self._seg_vb1}')
        print(f'crd_vb: {self._crd_vb}')
        print(f'ref_vb: {self._ref_vb}')
        print(f'seg_vb0 valid: {self._seg_vb0.valid()}')
        print(f'seg_vb1 valid: {self._seg_vb1.valid()}')
        print(f'crd_vb valid: {self._crd_vb.valid()}')
        print(f'ref_vb valid: {self._ref_vb.valid()}')
        print(self.curr_addr)
        # print(self.data_valid)
        print(self.backpressure_en)

        if self.pop_out_ref:
            self._ref_vb.pop()
            self.pop_out_ref = False

        if self.pop_out_crd:
            self._crd_vb.pop()
            self.pop_out_crd = False

        # Debugging print statements
        if self.debug and self.backpressure_en:
            print("DEBUG: C RD SCAN:"
                  "\n \t"
                  "name: ", self.name, "\t ref", len(self.in_ref), " :: ", self.in_ref,
                  "\t Curr crd:", self.curr_crd, "\t curr ref:", self.curr_ref,
                  "\n curr addr:", self.curr_addr, "\t start addr:", self.start_addr, "\t stop addr:", self.stop_addr,
                  "\n end fiber:", self.end_fiber, "\t curr input:", curr_in_ref,
                  "\n skip in:", self.curr_skip, "\t skip processed", self.skip_processed, "\t prev crd:",
                  self.prev_crd,
                  "\n Out stkn cnt:", self.out_stkn_cnt, "\t Skip stkn cnt:", self.skip_stkn_cnt, "\t Bakcpressure: ",
                  self.check_backpressure(), "\t backpressure_len: ",
                  self.data_valid, self.done)
        elif self.debug:
            print("DEBUG: C RD SCAN:"
                  "\n \t"
                  "name: ", self.name, "\t ref", len(self.in_ref), " :: ", self.in_ref,
                  "\t Curr crd:", self.curr_crd, "\t curr ref:", self.curr_ref,
                  "\n curr addr:", self.curr_addr, "\t start addr:", self.start_addr, "\t stop addr:", self.stop_addr,
                  "\n end fiber:", self.end_fiber, "\t curr input:", curr_in_ref,
                  "\n skip in:", self.curr_skip, "\t skip processed", self.skip_processed, "\t prev crd:",
                  self.prev_crd,
                  "\n Out stkn cnt:", self.out_stkn_cnt, "\t Skip stkn cnt:", self.skip_stkn_cnt, self.done)
    # else:
        # Debugging print statements
        if self.debug and self.backpressure_en:
            print("DEBUG: C RD SCAN:"
                  "\n \t"
                  "name: ", self.name, "\t ref", len(self.in_ref), " :: ", self.in_ref,
                  "\t Curr crd:", self.curr_crd, "\t curr ref:", self.curr_ref,
                  "\n curr addr:", self.curr_addr, "\t start addr:", self.start_addr, "\t stop addr:", self.stop_addr,
                  "\n skip in:", self.curr_skip, "\t skip processed", self.skip_processed, "\t prev crd:",
                  self.prev_crd,
                  "\n Out stkn cnt:", self.out_stkn_cnt, "\t Skip stkn cnt:", self.skip_stkn_cnt, "\t Bakcpressure: ",
                  self.data_valid)
        elif self.debug:
            print("DEBUG: C RD SCAN:"
                  "\n \t"
                  "name: ", self.name, "\t ref", len(self.in_ref), " :: ", self.in_ref,
                  "\t Curr crd:", self.curr_crd, "\t curr ref:", self.curr_ref,
                  "\n curr addr:", self.curr_addr, "\t start addr:", self.start_addr, "\t stop addr:", self.stop_addr,
                  "\n skip in:", self.curr_skip, "\t skip processed", self.skip_processed, "\t prev crd:",
                  self.prev_crd,
                  "\n Out stkn cnt:", self.out_stkn_cnt, "\t Skip stkn cnt:", self.skip_stkn_cnt, self.done)
            # # Debugging print statements
            # if self.debug:
            #    print("DEBUG: C RD SCAN:"
            #          "\n \t"
            #          "name: ", self.name, "\t ref", len(self.in_ref), " :: ", self.in_ref,
            #          "\t Curr crd:", self.curr_crd, "\t curr ref:", self.curr_ref,
            #          "\n curr addr:", self.curr_addr, "\t start addr:", self.start_addr, "\t stop addr:", self.stop_addr,
            #          "\n end fiber:", self.end_fiber, "\t curr input:", curr_in_ref,
            #          "\n skip in:", self.curr_skip, "\t skip processed", self.skip_processed, "\t prev crd:",
            #          self.prev_crd,
            #          "\n emit_fiber_stkn:", self.emit_fiber_stkn,
            #          "\n Out stkn cnt:", self.out_stkn_cnt, "\t Skip stkn cnt:", self.skip_stkn_cnt)

    def set_crd_skip(self, in_crd, parent=None):
        assert in_crd is None or is_valid_crd(in_crd)
        if in_crd != '' and in_crd is not None:
            if is_stkn(in_crd):
                idx = last_stkn(self.in_crd_skip)
                if idx is not None:
                    # Flush coordinates
                    self.in_crd_skip = self.in_crd_skip[:idx + 1]
            self.in_crd_skip.append(in_crd)
        if self.backpressure_en and parent != "":
            parent.set_backpressure(self.fifo_avail)


# ---------------- BV --------------#
class BVRdScanSuper(Primitive, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.curr_ref = 'S0'
        self.curr_bv = 'S0'

        self.in_ref = []
        if self.backpressure_en:
            self.fifo_avail = True
            self.ready_backpressure = True
            self.data_valid = True

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
        if self.backpressure_en:
            if len(self.in_ref) > self.depth:
                self.fifo_avail = False
            else:
                self.fifo_avail = True

    def set_in_ref(self, in_ref, parent=None):
        if in_ref != '' and in_ref is not None:
            self.in_ref.append(in_ref)
        if self.backpressure_en and parent != "":
            parent.set_backpressure(self.fifo_avail)

    def out_ref(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.curr_ref

    def out_bv(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.curr_bv


class BVRdScan(BVRdScanSuper):
    def __init__(self, bv_arr=None, dim=4, nbits=4, depth=4, **kwargs):
        super().__init__(**kwargs)

        self.bv_arr = bv_arr if bv_arr is not None else [2 ** nbits - 1] * dim

        self.curr_addr = 0

        self.end_fiber = False
        self.curr_ref = ''
        self.curr_bv = ''
        self.emit_fiber_stkn = False
        self.begin = True

        self.meta_blen = len(bv_arr)
        self.meta_nbits = nbits
        self.meta_dim = dim
        if self.backpressure_en:
            self.depth = depth
            self.data_valid = True
            self.fifo_avail = True
            self.ready_backpressure = True

    def _get_bv_ref(self, addr):
        assert isinstance(addr, int), "Addresses must be integers"
        if addr <= 0:
            return 0
        bits = sum(map(popcount, self.bv_arr[:addr]))
        return bits

    def update(self):
        self.update_done()
        self.update_ready()
        if self.backpressure_en:
            self.data_valid = False
        if (self.backpressure_en and self.check_backpressure()) or not self.backpressure_en:
            if self.backpressure_en:
                self.data_valid = True
            if len(self.in_ref) > 0:
                self.block_start = False

            curr_in_ref = None
            if self.curr_bv == 'D' or self.curr_ref == 'D' or self.done:
                self.curr_addr = 0
                self.curr_bv = ''
                self.curr_ref = ''
            elif len(self.in_ref) > 0 and self.emit_fiber_stkn:
                next_in = self.in_ref[0]
                if is_stkn(next_in):
                    self.in_ref.pop(0)
                    stkn = increment_stkn(next_in)
                else:
                    stkn = 'S0'
                self.curr_bv = stkn
                self.curr_ref = stkn

                self.curr_addr = 0
                self.emit_fiber_stkn = False
            # There exists another input reference at the segment and
            # either at the start of computation or end of fiber
            elif len(self.in_ref) > 0 and (self.end_fiber or self.begin):
                self.end_fiber = False
                self.begin = False
                curr_in_ref = self.in_ref.pop(0)
                if isinstance(curr_in_ref, int) and curr_in_ref + 1 > self.meta_blen:
                    raise Exception('Not enough elements in bv array(' + str(self.meta_blen) + ')')

                # See 'N' 0-token which immediately emits a stop token and ends the fiber
                # TODO: need to figure out how this will work with bitvectors
                elif is_0tkn(curr_in_ref):
                    self.end_fiber = True

                    if len(self.in_ref) > 0:
                        next_in = self.in_ref[0]
                        if is_stkn(next_in):
                            self.in_ref.pop(0)
                            stkn = increment_stkn(next_in)
                        else:
                            stkn = 'S0'
                    else:
                        self.emit_fiber_stkn = True
                        stkn = ''
                    self.curr_bv = stkn
                    self.curr_ref = stkn
                    self.curr_addr = 0

                elif is_stkn(curr_in_ref) or curr_in_ref == 'D':
                    self.curr_addr = 0
                    self.curr_bv = curr_in_ref
                    self.curr_ref = curr_in_ref
                    self.end_fiber = True
                    if curr_in_ref == 'D':
                        self.done = True
                else:
                    self.curr_addr = curr_in_ref
                    # End of fiber, get next input reference
                    self.end_fiber = True

                    self.emit_fiber_stkn = True
                    self.curr_bv = self.bv_arr[self.curr_addr]
                    self.curr_ref = self._get_bv_ref(self.curr_addr)

            else:  # elif self.curr_bv is not None and self.curr_ref is not None:
                # Default stall (when done)
                self.curr_ref = ''
                self.curr_bv = ''

            if self.debug:
                print("DEBUG: C RD SCAN: \t "
                      "Curr bv:", self.curr_bv, "\t curr ref:", self.curr_ref, "\t curr addr:", self.curr_addr,
                      "\t end fiber:", self.end_fiber, "\t curr input:", curr_in_ref)
