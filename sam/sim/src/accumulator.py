from .base import *
from .crd_manager import CrdPtConverter


class Reduce(Primitive):
    def __init__(self, depth=1, **kwargs):
        super().__init__(**kwargs)

        self.in_val = []
        self.curr_out = ""
        self.in_val_size = 0
        self.sum = 0
        self.emit_stkn = False
        self.curr_in_val = None

        if self.get_stats:
            self.reduction_count = 0
            self.num_inputs = 0
            self.num_outputs = 0
            self.stop_token_out = 0
            self.drop_token_out = 0
            self.valid_token_out = 0
            self.zero_out = 0
            self.nonzero_out = 0

        if self.backpressure_en:
            self.ready_backpressure = True
            self.data_valid = True
            self.depth = depth
            self.fifo_avail = True

    def check_backpressure(self):
        if self.backpressure_en:
            copy_backpressure = self.ready_backpressure
            self.ready_backpressure = True
            return copy_backpressure
        return True

    def set_backpressure(self, backpressure):
        if not backpressure:
            self.ready_backpressure = False

    def update_ready(self):
        if self.backpressure_en:
            if len(self.in_val) > self.depth:
                self.fifo_avail = False
            else:
                self.fifo_avail = True

    def update(self):
        self.update_ready()
        self.update_done()
        if self.backpressure_en:
            self.data_valid = False
        if (self.backpressure_en and self.check_backpressure()) or not self.backpressure_en:
            if self.backpressure_en:
                self.data_valid = True
            if (len(self.in_val) > 0):
                self.block_start = False

            curr_in_val = ""
            if self.done:
                self.curr_out = ""
                # Reset state
                self.in_val_size = 0
                self.sum = 0
                self.emit_stkn = False
                self.done = False
            elif self.emit_stkn:
                self.curr_out = decrement_stkn(self.curr_in_val)
                self.emit_stkn = False
            elif len(self.in_val) > 0:
                self.curr_in_val = self.in_val.pop(0)
                if is_stkn(self.curr_in_val) and stkn_order(self.curr_in_val) == 0:
                    self.curr_out = self.sum
                    self.sum = 0
                elif is_stkn(self.curr_in_val) and stkn_order(self.curr_in_val) > 0:
                    self.curr_out = self.sum
                    self.sum = 0
                    self.emit_stkn = True
                elif self.curr_in_val == 'D':
                    self.done = True
                    self.curr_out = 'D'
                else:
                    if self.get_stats:
                        self.reduction_count += 1
                    self.sum += self.curr_in_val
                    self.curr_out = ""
            else:
                self.curr_out = ""

            if self.get_stats:
                if self.curr_out == "":
                    self.drop_token_out += 1
                elif is_stkn(self.curr_out):
                    self.stop_token_out += 1
                else:
                    if (isinstance(self.curr_out, float) or isinstance(self.curr_out, int)) and self.curr_out == 0:
                        self.zero_out += 1
                    else:
                        self.nonzero_out += 1
                    self.valid_token_out += 1

            if self.get_stats:
                self.compute_fifos()
        if self.debug:
            print("DEBUG: REDUCE:", "\t CurrIn:", self.curr_in_val, "\tCurrOut:", self.curr_out,
                  "\t Sum:", self.sum)

    def set_in_val(self, val, parent=None):
        if val != '' and val is not None:
            if self.get_stats:
                self.num_inputs += 1
            self.in_val.append(val)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail)

    def out_val(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            if self.get_stats:
                self.num_outputs += 1
            return self.curr_out

    def compute_fifos(self):
        self.in_val_size = max(self.in_val_size, len(self.in_val))

    def print_fifos(self):
        if self.get_stats:
            print("Reduction counts- total inputs ", self.num_inputs, " total outputs ", self.num_outputs,
                  " reduction values ", self.reduction_count)
            print("FiFO Val size for Reduce block: ", self.in_val_size)

    def return_statistics(self):
        if self.get_stats:
            stats_dict = {"red_count": self.reduction_count, "total_inputs": self.num_inputs,
                          "total_outputs": self.num_outputs, "stkn_outs": self.stop_token_out,
                          "drop_outs": self.drop_token_out, "valid_outs": self.valid_token_out,
                          "zero_outs": self.zero_out, "nonzero_outs": self.nonzero_out}
            stats_dict.update(super().return_statistics())
        else:
            stats_dict = {}
        return stats_dict


class SparseCrdPtAccumulator1(Primitive):
    def __init__(self, maxdim=100, valtype=float, fifos=None, **kwargs):
        super().__init__(**kwargs)

        self.outer_crdpt = []
        self.inner_crdpt = []
        self.in_val = []

        self.curr_in_val = None
        self.curr_in_inner_crdpt = None
        self.curr_in_outer_crdpt = None

        self.emit_output = []
        self.curr_inner_crdpt = ''
        self.curr_outer_crdpt = ''
        self.curr_val = ''

        # Maximum possible dimension for this index level
        self.maxdim = maxdim
        self.order = 1

        self.seen_done = False
        # Accumulation scratchpad storage
        self.storage = dict()
        self.valtype = valtype

        if fifos is not None and len(fifos) == 3:
            self.outer_crdpt = fifos[0]
            self.inner_crdpt = fifos[1]
            self.in_val = fifos[2]

        if self.get_stats:
            self.hits_tracker = {}
            self.stop_token_out = 0
            self.drop_token_out = 0
            self.valid_token_out = 0
            self.zero_out = 0
            self.nonzero_out = 0
            self.out_crd_fifo = 0
            self.in_crd_fifo = 0
            self.in_val_fifo = 0

    def return_fifo(self):
        return self.outer_crdpt, self.inner_crdpt, self.in_val

    def update(self):
        self.update_done()
        if self.debug:
            if self.seen_done or self.done:
                print(self.seen_done, self.done)
                print("current point value", self.outer_crdpt, self.inner_crdpt, self.in_val, self.emit_output,
                      self.curr_in_outer_crdpt, self.curr_in_inner_crdpt, self.curr_val)
                self.print_debug()
            if len(self.in_val) > 0 and self.in_val[0] == "D":
                print("val", self.outer_crdpt, self.inner_crdpt, self.in_val, self.emit_output,
                      self.curr_in_outer_crdpt, self.curr_in_inner_crdpt, self.curr_val)
                self.print_debug()
            if len(self.inner_crdpt) > 0 and self.inner_crdpt[0] == "D":
                print("innercrd", self.outer_crdpt, self.inner_crdpt, self.in_val, self.emit_output,
                      self.curr_in_outer_crdpt, self.curr_in_inner_crdpt, self.curr_val)
                self.print_debug()
            if len(self.outer_crdpt) > 0 and self.outer_crdpt[0] == "D":
                print("outercrd", self.outer_crdpt, self.inner_crdpt, self.in_val, self.emit_output,
                      self.curr_in_outer_crdpt, self.curr_in_inner_crdpt, self.curr_val)
                self.print_debug()

        if len(self.outer_crdpt) > 0 or len(self.inner_crdpt) > 0:
            self.block_start = False

        if self.get_stats:
            self.out_crd_fifo = max(self.out_crd_fifo, len(self.outer_crdpt))
            self.in_crd_fifo = max(self.in_crd_fifo, len(self.inner_crdpt))
            self.in_val_fifo = max(self.in_val_fifo, len(self.in_val))

        if self.done:
            self.curr_outer_crdpt = ''
            self.curr_inner_crdpt = ''
            self.curr_val = ''
            if self.get_stats:
                self.drop_token_out += 1
            return

        if len(self.in_val) > 0 and len(self.outer_crdpt) > 0 and len(self.inner_crdpt) > 0 and not self.seen_done:
            self.curr_in_val = self.in_val.pop(0)
            self.curr_in_inner_crdpt = self.inner_crdpt.pop(0)

            ocrd = self.outer_crdpt.pop(0)
            emit_output = ocrd != self.curr_in_outer_crdpt and self.curr_in_outer_crdpt is not None and \
                self.curr_in_outer_crdpt != "D"
            if emit_output:
                self.emit_output.append([self.curr_in_outer_crdpt, -1])
                # print("@@@@@", self.curr_in_outer_crdpt)
            self.curr_in_outer_crdpt = ocrd
            if self.curr_in_outer_crdpt in self.storage.keys():
                inner_dict = self.storage[self.curr_in_outer_crdpt]
                if self.get_stats:
                    for k in inner_dict.keys():
                        self.hits_tracker[k] = 1
                if self.curr_in_inner_crdpt in inner_dict.keys():
                    if self.get_stats:
                        self.hits_tracker[self.curr_in_inner_crdpt] += 1
                    inner_dict[self.curr_in_inner_crdpt] += self.valtype(self.curr_in_val)
                else:
                    if self.get_stats:
                        self.hits_tracker[self.curr_in_inner_crdpt] = 1
                    inner_dict[self.curr_in_inner_crdpt] = self.valtype(self.curr_in_val)
            # If a done token is seen, cannot emit done until all coordinates have been written out
            elif self.curr_in_outer_crdpt == 'D':
                assert self.curr_in_inner_crdpt == 'D' and self.curr_in_val == 'D', \
                    "If one item is a 'D' token, then all inputs must be"
                self.seen_done = True
            else:
                self.storage[self.curr_in_outer_crdpt] = {self.curr_in_inner_crdpt: self.valtype(self.curr_in_val)}

        if len(self.emit_output) > 0:
            fiber = self.emit_output[0]

            self.curr_outer_crdpt = fiber[0]
            self.curr_inner_crdpt = min(
                [item for item in self.storage[self.curr_outer_crdpt].keys() if item > fiber[1]])
            self.curr_val = self.storage[self.curr_outer_crdpt][self.curr_inner_crdpt]

            if not [item for item in self.storage[self.curr_outer_crdpt].keys() if item > self.curr_inner_crdpt]:
                self.emit_output.pop(0)
            else:
                self.emit_output[0][1] = self.curr_inner_crdpt
        elif self.seen_done:
            self.done = True
            self.seen_done = False
            self.curr_outer_crdpt = 'D'
            self.curr_inner_crdpt = 'D'
            self.curr_val = 'D'
        else:
            self.curr_outer_crdpt = ''
            self.curr_inner_crdpt = ''
            self.curr_val = ''
        if self.get_stats:
            if self.curr_val == "":
                self.drop_token_out += 1
            elif is_stkn(self.curr_val):
                self.stop_token_out += 1
            else:
                if (isinstance(self.curr_val, float) or isinstance(self.curr_val, int)) and self.curr_val == 0:
                    self.zero_out += 1
                else:
                    self.nonzero_out += 1
                self.valid_token_out += 1

        if self.debug:
            print("Done ptaccum:", self.out_done(), self.done,
                  "\n Curr in ocrd: ", self.curr_in_outer_crdpt, "\t Curr in icrd", self.curr_in_inner_crdpt,
                  "\t Curr in val", self.curr_in_val,
                  "\n Curr out ocrd: ", self.curr_outer_crdpt, "\t Curr out icrd: ", self.curr_inner_crdpt,
                  "\t Curr out val: ", self.curr_val,
                  "\n Emit crds: ", self.emit_output,
                  "\n Storage: ", self.storage,
                  "\n f: ", self.outer_crdpt, self.inner_crdpt, self.in_val)

    def print_debug(self):
        print("Crdptaccum_debug Done:", self.out_done(), self.done,
              "\n Curr in ocrd: ", self.curr_in_outer_crdpt, "\t Curr in icrd", self.curr_in_inner_crdpt,
              "\t Curr in val", self.curr_in_val,
              "\n Curr out ocrd: ", self.curr_outer_crdpt, "\t Curr out icrd: ", self.curr_inner_crdpt,
              "\t Curr out val: ", self.curr_val,
              "\n Emit crds: ", self.emit_output,
              "\n Storage: ", self.storage,
              "\n Fifos: ", self.outer_crdpt, self.inner_crdpt, self.in_val)

    def set_inner_crdpt(self, crdpt):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '' and crdpt is not None:
            self.inner_crdpt.append(crdpt)

    def set_outer_crdpt(self, crdpt):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '' and crdpt is not None:
            self.outer_crdpt.append(crdpt)

    def set_val(self, val):
        assert not is_stkn(val), 'Values associated with points should not have stop tokens'
        if val != '' and val is not None:
            self.in_val.append(val)

    def out_outer_crdpt(self):
        return self.curr_outer_crdpt

    def out_inner_crdpt(self):
        return self.curr_inner_crdpt

    def out_val(self):
        return self.curr_val

    def return_statistics(self):
        if self.get_stats:
            stats_dict = {"stkn_outs": self.stop_token_out,
                          "drop_outs": self.drop_token_out, "valid_outs": self.valid_token_out,
                          "zero_outs": self.zero_out, "nonzero_outs": self.nonzero_out}
            stats_dict.update(super().return_statistics())
        else:
            stats_dict = {}
        return stats_dict

    def return_hits(self):
        i = 0
        cnt_gt_zero = 0
        cnt_total = 0
        total_sum = 0
        if self.get_stats:
            for k in self.hits_tracker.keys():
                if self.hits_tracker[k] > i:
                    i = self.hits_tracker[k]
                if self.hits_tracker[k] > 1:
                    cnt_gt_zero += 1
                total_sum += self.hits_tracker[k]
                cnt_total += 1
        return i, cnt_gt_zero, cnt_total, total_sum


# Accumulation into a vector
class SparseAccumulator1(Primitive):
    def __init__(self, maxdim=100, valtype=float, last_level=True, val_stkn=False, depth=1, **kwargs):
        super().__init__(**kwargs)
        self.in_outer_crdpt = []
        self.in_inner_crdpt = []
        self.in_val = []
        self.crdpt_spacc = SparseCrdPtAccumulator1(maxdim=maxdim, valtype=valtype, debug=self.debug,
                                                   statisics=self.get_stats, name="", back_en=False)
        self.crdpt_converter = CrdPtConverter(last_level=last_level, debug=self.debug,
                                              statisics=self.get_stats, name="", back_en=False)

        self.crdpt_spacc_out_val = []

        self.curr_outer_crd = None
        self.curr_inner_crd = None
        self.curr_val = None

        self.outer_crdpt = []
        self.inner_crdpt = []

        self.val_stkn = val_stkn

        if self.get_stats:
            self.in_outer_crd_pt_fifo = 0
            self.in_inner_crd_pt_fifo = 0
            self.in_val_fifo = 0

        if self.backpressure_en:
            self.ready_backpressure = True
            self.data_valid = True
            self.depth = depth
            self.fifo_avail_inner = True
            self.fifo_avail_outer = True
            self.fifo_avail_val = True

        self.temp_maxdim = maxdim
        self.temp_valtype = valtype
        self.temp_last_level = last_level

    def check_backpressure(self):
        if self.backpressure_en:
            copy_backpressure = self.ready_backpressure
            self.ready_backpressure = True
            return copy_backpressure
        return True

    def set_backpressure(self, backpressure):
        if not backpressure:
            self.ready_backpressure = False

    def update_ready(self):
        if self.backpressure_en:
            if len(self.in_inner_crdpt) > self.depth:
                self.fifo_avail_inner = False
            else:
                self.fifo_avail_inner = True
            if len(self.in_outer_crdpt) > self.depth:
                self.fifo_avail_outer = False
            else:
                self.fifo_avail_outer = True
            if len(self.in_val) > self.depth:
                self.fifo_avail_val = False
            else:
                self.fifo_avail_val = True

    def update(self):
        self.update_done()
        self.update_ready()
        if self.backpressure_en:
            self.data_valid = False
        if (self.backpressure_en and self.check_backpressure()) or not self.backpressure_en:
            if self.backpressure_en:
                self.data_valid = True
            if self.debug:
                print(self.in_outer_crdpt, self.in_inner_crdpt, self.in_val)
                print(self.crdpt_spacc.print_debug())
                print(self.crdpt_converter.print_debug())
            if self.done and self.memory_model_en:
                f1, f2, f3 = self.crdpt_spacc.return_fifo()
                f4, f5 = self.crdpt_converter.return_fifo()
                self.crdpt_spacc = SparseCrdPtAccumulator1(maxdim=self.temp_maxdim,
                                                           valtype=self.temp_valtype, fifos=[f1, f2, f3])
                self.crdpt_converter = CrdPtConverter(last_level=self.temp_last_level, fifos=[f4, f5])

            if len(self.in_outer_crdpt) > 0 or len(self.in_inner_crdpt) > 0:
                self.block_start = False

            # What to do for drop tokens?
            if self.get_stats:
                self.in_outer_crd_pt_fifo = max(self.in_outer_crd_pt_fifo, len(self.in_outer_crdpt))
                self.in_inner_crd_pt_fifo = max(self.in_inner_crd_pt_fifo, len(self.in_inner_crdpt))
                self.in_val_fifo = max(self.in_val_fifo, len(self.in_val))

            if len(self.in_outer_crdpt) > 0:
                self.crdpt_spacc.set_outer_crdpt(self.in_outer_crdpt.pop(0))

            if len(self.in_inner_crdpt) > 0:
                self.crdpt_spacc.set_inner_crdpt(self.in_inner_crdpt.pop(0))

            if len(self.in_val) > 0:
                self.crdpt_spacc.set_val(self.in_val.pop(0))

            self.crdpt_spacc.update()
            # print(">>>>>>>>>>>>SPACC:", self.crdpt_spacc.out_outer_crdpt(), self.crdpt_spacc.out_inner_crdpt())
            self.crdpt_converter.set_outer_crdpt(self.crdpt_spacc.out_outer_crdpt())
            self.crdpt_converter.set_inner_crdpt(self.crdpt_spacc.out_inner_crdpt())

            if self.crdpt_spacc.out_val() != '':
                self.crdpt_spacc_out_val.append(self.crdpt_spacc.out_val())

            self.crdpt_converter.update()
            self.curr_outer_crd = self.crdpt_converter.out_crd_outer()
            self.curr_inner_crd = self.crdpt_converter.out_crd_inner()

            if self.val_stkn:
                self.curr_val = self.crdpt_spacc_out_val.pop(0) if isinstance(self.curr_inner_crd, int) and \
                    len(self.crdpt_spacc_out_val) > 0 else self.curr_inner_crd
            else:
                self.curr_val = self.crdpt_spacc_out_val.pop(0) if len(self.crdpt_spacc_out_val) > 0 else ''

            if self.debug:
                print(self.in_val)

            self.done = self.crdpt_spacc.out_done() and self.crdpt_converter.out_done()

        if self.debug:
            print("Vals: ", self.in_val, "\n Done:", self.done,
                  "\n SpCrdPt Accum Done:", self.crdpt_spacc.out_done(),
                  "\t CrdPtConverter Done:", self.crdpt_converter.out_done()
                  )

    def set_inner_crdpt(self, crdpt, parent=None):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '' and crdpt is not None:
            self.in_inner_crdpt.append(crdpt)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail_inner)

    def set_outer_crdpt(self, crdpt, parent=None):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '' and crdpt is not None:
            self.in_outer_crdpt.append(crdpt)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail_outer)

    def crd_in_inner(self, crdpt, parent=None):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '' and crdpt is not None:
            self.in_inner_crdpt.append(crdpt)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail_inner)

    def crd_in_outer(self, crdpt, parent=None):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '' and crdpt is not None:
            self.in_outer_crdpt.append(crdpt)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail_outer)

    def set_crd_inner(self, crdpt, parent=None):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '' and crdpt is not None:
            self.in_inner_crdpt.append(crdpt)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail_inner)

    def set_crd_outer(self, crdpt, parent=None):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '' and crdpt is not None:
            self.in_outer_crdpt.append(crdpt)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail_outer)

    def set_val(self, val, parent=None):
        assert not is_stkn(val), 'Values associated with points should not have stop tokens'
        if val != '' and val is not None:
            self.in_val.append(val)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail_val)

    def out_outer_crd(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.curr_outer_crd

    def out_inner_crd(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.curr_inner_crd

    def out_val(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.curr_val

    def out_crd_outer(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.curr_outer_crd

    def out_crd_inner(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.curr_inner_crd

    def return_statistics(self):
        if self.get_stats:
            stats_dict = {}
            stats_dict["in_outer_fifo"] = self.in_outer_crd_pt_fifo
            stats_dict["in_inner_fifo"] = self.in_inner_crd_pt_fifo
            stats_dict["in_val_fifo"] = self.in_val_fifo
            hits_info = self.crdpt_spacc.return_hits()
            stats_dict["max_hits"] = hits_info[0]
            stats_dict["hits_gt_one"] = hits_info[1]
            stats_dict["total_elems"] = hits_info[2]
            stats_dict["rmw_ops"] = hits_info[3]
            stats_dict.update(self.crdpt_spacc.return_statistics())
            stats_dict.update(super().return_statistics())
        else:
            stats_dict = {}
        return stats_dict

    def print_fifos(self):
        print("Spaccumulator: None available")


class SparseCrdPtAccumulator2(Primitive):
    def __init__(self, maxdim=100, valtype=float, **kwargs):
        super().__init__(**kwargs)
        self.in_crdpt0 = []
        self.in_crdpt1 = []
        self.in_val = []

        self.curr_in_val = None
        self.curr_in0_crdpt = None
        self.curr_in1_crdpt = None

        self.emit_output = []
        self.curr_crdpt0 = ''
        self.curr_crdpt1 = ''
        self.curr_val = ''

        # Maximum possible dimension for this index level
        self.maxdim = maxdim
        self.order = 1

        self.seen_done = False
        # Accumulation scratchpad storage
        self.storage = dict()
        self.valtype = valtype

        if self.get_stats:
            self.hits_tracker = {}
            self.stop_token_out = 0
            self.drop_token_out = 0
            self.valid_token_out = 0
            self.zero_out = 0
            self.nonzero_out = 0

    def return_fifo(self):
        return self.in_crdpt0, self.in_crdpt1, self.in_val

    def update(self):
        self.update_done()
        if len(self.in_crdpt0) > 0 or len(self.in_crdpt0) > 0 or len(self.in_val) > 0:
            self.block_start = False

        if self.done:
            self.curr_crdpt0 = ''
            self.curr_crdpt1 = ''
            self.curr_val = ''
            if self.get_stats:
                self.drop_token_out += 1
            return

        if len(self.in_val) > 0 and len(self.in_crdpt1) > 0 and len(self.in_crdpt0) > 0:
            self.curr_in_val = self.in_val.pop(0)
            self.curr_in0_crdpt = self.in_crdpt0.pop(0)
            self.curr_in1_crdpt = self.in_crdpt1.pop(0)

            emit_output = self.curr_in1_crdpt == 'D'
            if emit_output:
                self.emit_output.append([-1, -1])
                assert self.curr_in1_crdpt == 'D' and self.curr_in0_crdpt == 'D' and self.curr_in_val == 'D', \
                    "If one item is a 'D' token, then all inputs must be"
                self.seen_done = True
            else:
                if self.curr_in1_crdpt in self.storage.keys():
                    inner_dict = self.storage[self.curr_in1_crdpt]
                    if self.get_stats:
                        for k in inner_dict.keys():
                            self.hits_tracker[k] = 1
                    if self.curr_in0_crdpt in inner_dict.keys():
                        if self.get_stats:
                            self.hits_tracker[self.curr_in0_crdpt] += 1
                        inner_dict[self.curr_in0_crdpt] += self.valtype(self.curr_in_val)
                    else:
                        if self.get_stats:
                            self.hits_tracker[self.curr_in0_crdpt] = 1
                        inner_dict[self.curr_in0_crdpt] = self.valtype(self.curr_in_val)
                else:
                    self.storage[self.curr_in1_crdpt] = {self.curr_in0_crdpt: self.valtype(self.curr_in_val)}

        if len(self.emit_output) > 0:
            fiber = self.emit_output.pop(0)
            #
            key1 = min(
                [item for item in self.storage.keys() if item > fiber[0]])
            key0 = min(
                [item for item in self.storage[key1].keys() if item > fiber[1]])

            self.curr_crdpt1 = key1
            self.curr_crdpt0 = key0
            self.curr_val = self.storage[key1][key0]

            # Finished inner coordinates, increment outer coordinate
            if not [item for item in self.storage[key1].keys() if item > key0]:
                # Do not increment outer coordinate if it's the last one
                if [item for item in self.storage.keys() if item > key1]:
                    self.emit_output.append([key1, -1])
            # Do inner coordinates
            else:
                self.emit_output.append([fiber[0], key0])

        elif self.seen_done:
            self.done = True
            self.seen_done = False
            self.curr_crdpt0 = 'D'
            self.curr_crdpt1 = 'D'
            self.curr_val = 'D'
        else:
            self.curr_crdpt0 = ''
            self.curr_crdpt1 = ''
            self.curr_val = ''

        if self.get_stats:
            if self.curr_val == "":
                self.drop_token_out += 1
            elif is_stkn(self.curr_val):
                self.stop_token_out += 1
            else:
                if (isinstance(self.curr_val, float) or isinstance(self.curr_val, int)) and self.curr_val == 0:
                    self.zero_out += 1
                else:
                    self.nonzero_out += 1
                self.valid_token_out += 1

        if self.debug:
            print("Done:", self.out_done(),
                  "\n Curr in crd1: ", self.curr_in1_crdpt,
                  "\t Curr in crd0", self.curr_in0_crdpt,
                  "\t Curr in val", self.curr_in_val,
                  "\n Curr out crd1: ", self.curr_crdpt1,
                  "\t Curr out crd0: ", self.curr_crdpt0,
                  "\t Curr out val: ", self.curr_val,
                  "\n Emit crds: ", self.emit_output,
                  "\n Storage: ", self.storage)

    def set_inner_crdpt(self, crdpt):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '' and crdpt is not None:
            self.in_crdpt0.append(crdpt)

    def set_outer_crdpt(self, crdpt):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '' and crdpt is not None:
            self.in_crdpt1.append(crdpt)

    def set_val(self, val):
        assert not is_stkn(val), 'Values associated with points should not have stop tokens'
        if val != '' and val is not None:
            self.in_val.append(val)

    def out_outer_crdpt(self):
        return self.curr_crdpt1

    def out_inner_crdpt(self):
        return self.curr_crdpt0

    def out_val(self):
        return self.curr_val

    def return_hits(self):
        i = 0
        cnt_gt_zero = 0
        cnt_total = 0
        for k in self.hits_tracker.keys():
            if self.hits_tracker[k] > i:
                i = self.hits_tracker[k]
            if self.hits_tracker[k] > 1:
                cnt_gt_zero += 1
            cnt_total += 1
        return i, cnt_gt_zero, cnt_total

    def return_statistics(self):
        if self.get_stats:
            stats_dict = {"stkn_outs": self.stop_token_out,
                          "drop_outs": self.drop_token_out, "valid_outs": self.valid_token_out,
                          "zero_outs": self.zero_out, "nonzero_outs": self.nonzero_out}
            stats_dict.update(super().return_statistics())
        else:
            stats_dict = {}
        return stats_dict


# Accumulation into a matrix (2D)
class SparseAccumulator2(Primitive):
    def __init__(self, maxdim=100, valtype=float, last_level=True, val_stkn=False, depth=1, **kwargs):
        super().__init__(**kwargs)
        self.in1_crdpt = []
        self.in0_crdpt = []
        self.in_val = []

        self.crdpt_spacc = SparseCrdPtAccumulator2(maxdim=maxdim, valtype=valtype, **kwargs)
        self.crdpt_converter = CrdPtConverter(last_level=True, **kwargs)

        self.crdpt_spacc_out_val = []

        self.curr_1_crd = None
        self.curr_0_crd = None
        self.curr_val = None

        self.outer_crdpt = []
        self.inner_crdpt = []

        self.val_stkn = val_stkn
        if self.get_stats:
            self.in1_fifo = 0
            self.in0_fifo = 0
            self.inval_fifo = 0

        if self.backpressure_en:
            self.ready_backpressure = True
            self.data_valid = True
            self.depth = depth
            self.fifo_avail_inner = True
            self.fifo_avail_outer = True
            self.fifo_avail_val = True
        self.temp_maxdim = maxdim
        self.temp_valtype = valtype
        self.temp_last_level = last_level

    def check_backpressure(self):
        if self.backpressure_en:
            copy_backpressure = self.ready_backpressure
            self.ready_backpressure = True
            return copy_backpressure
        return True

    def set_backpressure(self, backpressure):
        if not backpressure:
            self.ready_backpressure = False

    def update_ready(self):
        if self.backpressure_en:
            if len(self.in0_crdpt) > self.depth:
                self.fifo_avail_inner = False
            else:
                self.fifo_avail_inner = True
            if len(self.in1_crdpt) > self.depth:
                self.fifo_avail_outer = False
            else:
                self.fifo_avail_outer = True
            if len(self.in_val) > self.depth:
                self.fifo_avail_val = False
            else:
                self.fifo_avail_val = True

    def update(self):
        self.update_done()
        self.update_ready()
        if self.backpressure_en:
            self.data_valid = False
        if (self.backpressure_en and self.check_backpressure()) or not self.backpressure_en:
            if self.done and self.memory_model_en:
                f1, f2, f3 = self.crdpt_spacc.return_fifo()
                f4, f5 = self.crdpt_converter.return_fifo()
                self.crdpt_spacc = SparseCrdPtAccumulator2(maxdim=self.temp_maxdim, valtype=self.temp_valtype,
                                                           fifos=[f1, f2, f3])
                self.crdpt_converter = CrdPtConverter(last_level=self.temp_last_level, fifos=[f4, f5])
            if self.backpressure_en:
                self.data_valid = True
            if (len(self.in1_crdpt) > 0 or len(self.in0_crdpt) > 0 or len(self.in_val) > 0):
                self.block_start = False

            if self.get_stats:
                self.compute_fifo()

            if len(self.in1_crdpt) > 0:
                self.crdpt_spacc.set_outer_crdpt(self.in1_crdpt.pop(0))

            if len(self.in0_crdpt) > 0:
                self.crdpt_spacc.set_inner_crdpt(self.in0_crdpt.pop(0))

            if len(self.in_val) > 0:
                self.crdpt_spacc.set_val(self.in_val.pop(0))

            self.crdpt_spacc.update()

            self.crdpt_converter.set_inner_crdpt(self.crdpt_spacc.out_inner_crdpt())
            self.crdpt_converter.set_outer_crdpt(self.crdpt_spacc.out_outer_crdpt())
            self.crdpt_converter.update()

            if self.crdpt_spacc.out_val() != '':
                self.crdpt_spacc_out_val.append(self.crdpt_spacc.out_val())

            self.curr_1_crd = self.crdpt_converter.out_crd_outer()
            self.curr_0_crd = self.crdpt_converter.out_crd_inner()

            if self.val_stkn:
                self.curr_val = self.crdpt_spacc_out_val.pop(0) if isinstance(self.curr_0_crd, int) and \
                    len(self.crdpt_spacc_out_val) > 0 else self.curr_0_crd
            else:
                self.curr_val = self.crdpt_spacc_out_val.pop(0) if len(self.crdpt_spacc_out_val) > 0 else ''

            if self.debug:
                print(self.in_val)

            self.done = self.crdpt_spacc.out_done() and self.crdpt_converter.out_done()

        if self.debug:
            print("Done:", self.done,
                  "\n SpCrdPt Accum Done:", self.crdpt_spacc.out_done(),
                  "\t CrdPtConv 01 Done:", self.crdpt_converter.out_done()
                  )

    def compute_fifo(self):
        self.in1_fifo = max(self.in1_fifo, len(self.in1_crdpt))
        self.in0_fifo = max(self.in0_fifo, len(self.in0_crdpt))
        self.inval_fifo = max(self.inval_fifo, len(self.in_val))

    def set_crd_inner(self, crdpt, parent=None):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '' and crdpt is not None:
            self.in0_crdpt.append(crdpt)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail_inner)

    def set_crd_outer(self, crdpt, parent=None):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '' and crdpt is not None:
            self.in1_crdpt.append(crdpt)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail_outer)

    def set_val(self, val, parent=None):
        assert not is_stkn(val), 'Values associated with points should not have stop tokens'
        if val != '' and val is not None:
            self.in_val.append(val)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail_val)

    def out_crd_outer(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.curr_1_crd

    def out_crd_inner(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.curr_0_crd

    def out_val(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.curr_val

    def return_statistics(self):
        if self.get_stats:
            hits_info = self.crdpt_spacc.return_hits()
            stats_dict = {"in1_fifo": self.in1_fifo, "in0_fifo": self.in0_fifo,
                          "inval_fifo": self.inval_fifo, "max_hits": hits_info[0], "gt_one": hits_info[1],
                          "total_elems": hits_info[2]}
            stats_dict.update(self.crdpt_spacc.return_statistics())
            stats_dict.update(super().return_statistics())
        else:
            stats_dict = {}
        return stats_dict
