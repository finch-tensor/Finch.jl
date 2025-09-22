from .base import *


# Drops tokens
class StknDrop(Primitive):
    def __init__(self, depth=1, **kwargs):
        super().__init__(**kwargs)

        self.in_stream = []

        self.curr_out = ''
        if self.backpressure_en:
            self.data_valid = True
            self.depth = depth
            self.fifo_avail = True
            self.ready_backpressure = True

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
            if len(self.in_stream) > self.depth:
                self.fifo_avail = False
            else:
                self.fifo_avail = True

    def update(self):
        self.update_done()
        self.update_ready()
        self.data_valid = False
        if (self.backpressure_en and self.check_backpressure()) or not self.backpressure_en:
            if self.backpressure_en:
                self.data_valid = True
            if len(self.in_stream) > 0:
                self.block_start = False

            ival = ''

            if self.done:
                self.curr_out = ''
                # return

            if len(self.in_stream) > 0:
                ival = self.in_stream.pop(0)
                if ival == 'D':
                    self.done = True
                    self.curr_out = 'D'
                    return
                else:
                    self.done = False

            self.curr_out = '' if is_stkn(ival) else ival
        if self.debug:
            print("Curr InnerCrd:", ival, "\t Curr OutputCrd:", self.curr_out)

    def set_in_stream(self, val, parent=None):
        if val != '' and val is not None:
            self.in_stream.append(val)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail)

    def out_val(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.curr_out


class EmptyFiberStknDrop(Primitive):
    def __init__(self, depth=4, **kwargs):
        super().__init__(**kwargs)

        self.in_stream = []

        self.largest_stkn = None
        self.prev_stkn = False
        self.leading_stkn = True

        self.emit_ival = False
        self.prev_ival = None
        self.curr_out = ''

        if self.backpressure_en:
            self.ready_backpressure = True
            self.data_valid = True
            self.fifo_avail = True
            self.depth = depth

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
            if len(in_stream) > self.depth:
                self.fifo_avail = False
            else:
                self.fifo_avail = True

    def update(self):
        self.update_done()
        self.update_ready()
        if len(self.in_stream) > 0:
            self.block_start = False
        if self.backpressure_en:
            self.data_valid = False
        if (self.backpressure_en and self.check_backpressure()) or not self.backpressure_en:
            if self.backpressure_en:
                self.data_valid = True
            ival = ''

            if self.done:
                self.curr_out = ''
                # return

            if self.emit_ival:
                self.curr_out = self.prev_ival
                self.emit_ival = False

                if self.prev_ival == 'D':
                    self.curr_out = self.prev_ival
                    self.done = True
                    self.prev_stkn = False
                    self.leading_stkn = False
                return

            if len(self.in_stream) > 0:
                ival = self.in_stream.pop(0)
                if is_stkn(ival) and not self.leading_stkn:
                    self.largest_stkn = ival if self.largest_stkn is None else larger_stkn(self.largest_stkn, ival)
                    self.curr_out = ''
                    self.prev_stkn = True
                    self.prev_stkn = True
                    self.done = False
                elif self.prev_stkn and not self.leading_stkn:
                    self.curr_out = self.largest_stkn
                    self.largest_stkn = None
                    self.prev_stkn = False
                    self.prev_ival = ival
                    self.emit_ival = True
                    self.done = False
                elif ival == 'D':
                    self.done = True
                    self.curr_out = 'D'
                    self.prev_stkn = False
                    self.leading_stkn = False
                elif isinstance(ival, int):
                    self.leading_stkn = False
                    self.curr_out = ival
                    self.done = False
            else:
                self.curr_out = ''

            if self.debug:
                print("Curr InnerCrd:", ival, "\t Curr OutputCrd:", self.curr_out)

    # This can be both val or crd
    def set_in_stream(self, val, parent=None):
        if val != '' and val is not None:
            self.in_stream.append(val)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail)

    def out_val(self):
        return self.curr_out
