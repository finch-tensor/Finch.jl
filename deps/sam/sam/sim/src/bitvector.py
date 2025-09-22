from abc import ABC

from .base import *
from .token import EmptyFiberStknDrop

from functools import reduce


# Compresses entire fiber with splitting of Bitvectors into "width" widths
class ChunkBV(Primitive):
    def __init__(self, width=4, size=16, depth=4, **kwargs):
        super().__init__(**kwargs)

        self.meta_width = width
        self.meta_size = size
        self.curr_in_bv = None
        self.curr_bv = ''
        self.curr_ref = ''
        self.ref_sum = 0

        self.in_bv = []
        self.emit_chunk = False
        self.count = 0
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
            if len(self.in_bv) > self.depth:
                self.fifo_avail = False
            else:
                self.fifo_avail = True

    def update(self):
        self.update_done()
        self.update_ready()
        if self.backpressure_en:
            self.data_valid = False
        if (self.backpressure_en and self.check_backpressure()) or not self.backpressure_en:
            if (len(self.in_bv) > 0):
                self.block_start = False
            if self.backpressure_en:
                self.data_valid = True
            # bin(bv)   # gives 0bx number
            if self.done:
                self.curr_bv = ''
                self.curr_ref = ''
            elif self.emit_chunk:
                mask = (1 << self.meta_width) - 1
                self.curr_bv = self.curr_in_bv & mask
                self.curr_in_bv = self.curr_in_bv >> self.meta_width
                self.count += 1
                self.emit_chunk = self.count * self.meta_width < self.meta_size
                self.curr_ref = self.ref_sum
                self.ref_sum += popcount(self.curr_bv)

            elif len(self.in_bv) > 0:
                self.curr_in_bv = self.in_bv.pop(0)
                if isinstance(self.curr_in_bv, int):

                    mask = (1 << self.meta_width) - 1
                    self.curr_bv = self.curr_in_bv & mask
                    self.curr_in_bv = self.curr_in_bv >> self.meta_width
                    self.count += 1
                    self.ref_sum = 0
                    self.curr_ref = self.ref_sum
                    self.ref_sum += popcount(self.curr_bv)

                    self.emit_chunk = self.count * self.meta_width < self.meta_size
                elif is_stkn(self.curr_in_bv):
                    self.curr_bv = self.curr_in_bv
                    self.curr_ref = self.curr_in_bv
                else:
                    self.curr_bv = 'D'
                    self.curr_ref = 'D'
                    self.done = True
            else:
                self.curr_bv = ''
                self.curr_ref = ''

            if self.debug:
                print("Curr OutBV:", self.curr_bv, "\tCurr InBV:", self.curr_in_bv, "\tCount:", self.count,
                      "\t EmitChunk:", self.emit_chunk)

    def set_in_bv(self, bv, parent=None):
        if bv != '' and bv is not None:
            self.in_bv.append(bv)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail)

    def out_bv_int(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.curr_bv

    def out_bv(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            result = bin(self.curr_bv) if isinstance(self.curr_bv, int) else self.curr_bv
            return result

    def out_ref(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.curr_ref


# Always compresses entire fiber with no splitting
class BV(Primitive):
    def __init__(self, depth=4, **kwargs):
        super().__init__(**kwargs)

        self.in_crd = []
        self.crds = []
        self.curr_bv = None

        self.stkn = None
        self.emit_stkn = False
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
            if len(self.in_crd) > self.depth:
                self.fifo_avail = False
            else:
                self.fifo_avail = True

    def update(self):
        self.update_done()
        self.update_ready()
        if self.backpressure_en:
            self.data_valid = False
        if (self.backpressure_en and self.check_backpressure) or not self.backpressure_en:
            if (len(self.in_crd) > 0):
                self.block_start = False
            if self.backpressure_en:
                self.data_valid = True
            # bin(bv)   # gives 0bx number
            if self.done:
                self.curr_bv = ''
            elif self.emit_stkn:
                self.curr_bv = self.stkn
                self.emit_stkn = False
            elif len(self.in_crd) > 0:
                in_crd = self.in_crd.pop(0)
                if isinstance(in_crd, int):
                    self.curr_bv = ''
                    self.crds.append(in_crd)
                elif is_stkn(in_crd):
                    self.stkn = in_crd
                    self.curr_bv = self.stkn if not self.crds else reduce(lambda a, b: a | b, [0b1 << c for c in self.crds])
                    self.emit_stkn = bool(self.crds)
                    self.crds = []
                else:
                    self.curr_bv = 'D'
                    self.done = True
            else:
                self.curr_bv = ''

    def set_in_crd(self, crd, parent=None):
        if crd != '' and crd is not None:
            self.in_crd.append(crd)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail)

    def out_bv_int(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.curr_bv

    def out_bv(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            result = bin(self.curr_bv) if isinstance(self.curr_bv, int) else self.curr_bv
            return result


class BVDropSuper(Primitive, ABC):
    def __init__(self, depth=4, **kwargs):
        super().__init__(**kwargs)

        self.outer_bv = []
        self.inner_bv = []

        self.curr_obv = ''
        if self.backpressure_en:
            self.ready_backpressure = True
            self.data_valid = True
            self.depth = depth
            self.fifo_avail_outer = True
            self.fifo_avail_inner = True

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
            if len(self.outer_bv) > self.depth:
                self.fifo_avail_outer = False
            else:
                self.fifo_avail_outer = True
            if len(self.inner_bv) > self.depth:
                self.fifo_avail_inner = False
            else:
                self.fifo_avail_inner = True

    def set_outer_bv(self, bv, parent=None):
        if bv != '' and bv is not None:
            self.outer_bv.append(bv)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail_outer)

    def set_inner_bv(self, bv, parent=None):
        if bv != '' and bv is not None:
            self.inner_bv.append(bv)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail_inner)

    def out_bv_outer(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.curr_obv


class BVDropOnly(BVDropSuper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.get_ibv_count = 0

        self.running_obv = ''
        self.orig_obv = ''

        # FIXME: finish doing this
        self.curr_ibv = ''

        self.has_bv = False
        self.get_next_ibv = False
        self.get_next_obv = True

    def update(self):
        self.update_done()
        self.update_ready()
        if self.backpressure_en:
            self.data_valid = False
        if (self.backpressure_en and self.check_backpressure()) or not self.backpressure_en:
            if self.backpressure_en:
                self.data_valid = True
            if len(self.inner_bv) > 0 or len(self.outer_bv) > 0:
                self.block_start = False

            ibv = ""
            if self.debug:
                print("Outer BV:", self.outer_bv)
                print("Inner BV:", self.inner_bv)

            if self.done:
                self.curr_obv = ''
                return

            if len(self.outer_bv) > 0 and self.get_next_obv:
                obv = self.outer_bv.pop(0)
                if isinstance(obv, int):
                    self.running_obv = obv
                    self.orig_obv = self.running_obv
                    self.get_ibv_count = popcount(self.running_obv)
                    self.get_next_ibv = self.get_ibv_count > 0
                    self.get_next_obv = False
                else:
                    self.running_obv = obv
                    self.curr_obv = obv
                    self.get_next_ibv = False
                    self.get_next_obv = True
                    if obv == 'D':
                        self.done = True
                        self.get_next_ibv = True
                        self.curr_obv = 'D'
                self.has_bv = False
            elif self.get_next_obv:
                self.curr_obv = ''

            if self.debug:
                print("\t before getting ibv: GetNext InnerBV:", self.get_next_ibv)

            if len(self.inner_bv) > 0 and self.get_next_ibv:
                ibv = self.inner_bv.pop(0)
                if isinstance(ibv, int):
                    self.has_bv = True
                    self.curr_obv = ''
                    self.get_next_obv = False
                    self.get_next_ibv = True
                elif is_stkn(ibv) and is_stkn(self.running_obv):
                    self.get_next_obv = True
                    self.curr_obv = self.running_obv
                    self.get_next_ibv = False
                elif is_stkn(ibv) and self.get_ibv_count and self.has_bv:
                    self.get_ibv_count -= 1
                    self.get_next_obv = self.get_ibv_count == 0
                    self.curr_obv = self.running_obv if self.get_ibv_count == 0 else ''
                    self.get_next_ibv = self.get_ibv_count > 0
                    self.has_bv = False
                elif is_stkn(ibv) and self.get_ibv_count:
                    self.get_ibv_count -= 1
                    bitn = popcount(self.orig_obv) - self.get_ibv_count - 1
                    self.running_obv &= ~get_nth_bit(self.orig_obv, bitn)
                    self.get_next_obv = self.get_ibv_count == 0
                    self.curr_obv = self.running_obv if self.get_ibv_count == 0 and self.running_obv != 0 else ''
                    self.get_next_ibv = self.get_ibv_count > 0
                    self.has_bv = False
                elif self.done:
                    # assert (ibv == 'D'), ibv
                    self.curr_obv = 'D'
                    self.get_next_ibv = False
                    self.get_next_obv = False
                else:
                    self.curr_obv = ''
                    self.get_next_ibv = False
                    self.get_next_obv = True
            elif self.get_next_ibv:
                self.curr_ibv = ''

            if self.debug:
                print("DEBUG: BVDROP: Curr OuterBV:", self.running_obv, "Orig OuterBV:", self.orig_obv,
                      "\n Curr Output BV:", self.curr_obv, "\t GetNext OuterBV:", self.get_next_obv,
                      "\n HasBV", self.has_bv, "\t BitCount", self.get_ibv_count, "\t GetNext InnerBV:", self.get_next_ibv)


class BVDrop(BVDropSuper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.bv_drop = BVDropOnly(back_en=False)
        self.outer_stkn_drop = EmptyFiberStknDrop(back_en=False)
        self.inner_stkn_drop = EmptyFiberStknDrop(back_en=False)  # **kwargs

        self.curr_ibv = ''

    def update(self):
        self.update_done()
        self.update_ready()
        if self.backpressure_en:
            self.data_valid = False
        if (self.backpressure_en and self.check_backpressure()) or not self.backpressure_en:
            if self.backpressure_en:
                self.data_valid = True
            if len(self.outer_bv) > 0 or len(self.inner_bv) > 0:
                self.block_start = False

            if len(self.outer_bv) > 0:
                self.bv_drop.set_outer_bv(self.outer_bv.pop(0))

            if len(self.inner_bv) > 0:
                ibv = self.inner_bv.pop(0)
                self.bv_drop.set_inner_bv(ibv)
                self.inner_stkn_drop.set_in_stream(ibv)

            self.bv_drop.update()

            self.outer_stkn_drop.set_in_stream(self.bv_drop.out_bv_outer())
            self.outer_stkn_drop.update()

            self.inner_stkn_drop.update()

            self.curr_obv = self.outer_stkn_drop.out_val()
            self.curr_ibv = self.inner_stkn_drop.out_val()

            self.done = self.outer_stkn_drop.out_done() and self.inner_stkn_drop.out_done() and self.bv_drop.out_done()

    def out_bv_inner(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.curr_ibv
