from abc import ABC, abstractmethod
import numpy as np
import warnings

# warnings.simplefilter(action='ignore', category=FutureWarning)


def gen_stkns(dim=10):
    return ['S' + str(i) for i in range(dim)]


valid_tkns = ['', 'D', 'N']
valid_tkns += gen_stkns()


def is_valid_crd(elem, dim=10):
    valid_tkns = ['', 'D'] + gen_stkns(dim)
    return isinstance(elem, int) or elem in valid_tkns


def is_valid_ref(elem, dim=10):
    valid_tkns = ['', 'D', 'N'] + gen_stkns(dim)
    return isinstance(elem, int) or elem in valid_tkns


def is_valid_crdpt(elem):
    valid_tkns = ['', 'D']
    return isinstance(elem, int) or elem in valid_tkns


def is_valid_num(elem, dim=10):
    return isinstance(elem, int) or isinstance(elem, float)


def is_valid_val(elem, dim=10):
    valid_tkns = ['', 'D'] + gen_stkns(dim)
    return isinstance(elem, int) or isinstance(elem, float) or elem in valid_tkns


def is_0tkn(elem):
    return elem == 'N'


# Checks if a token is a non-control (numerical) token
def is_nc_tkn(elem, datatype=int):
    return isinstance(elem, datatype)


def is_stkn(elem):
    if isinstance(elem, str):
        return elem.startswith('S') and (len(elem) == 2)
    return False


def is_dtkn(elem):
    if isinstance(elem, str):
        return elem == 'D'
    return False


def stkn_order(elem):
    assert is_stkn(elem)
    return int(elem[1])


def increment_stkn(elem):
    if elem == '':
        return 'S0'
    return 'S' + str(stkn_order(elem) + 1)


def decrement_stkn(elem):
    assert (stkn_order(elem) >= 0)
    if stkn_order(elem) > 0:
        return 'S' + str(stkn_order(elem) - 1)
    else:
        return ''


def smaller_stkn(a, b):
    return a if stkn_order(a) < stkn_order(b) else b


def larger_stkn(a, b):
    return a if stkn_order(a) > stkn_order(b) else b


class Primitive(ABC):
    def __init__(self, debug=False, statistics=False, name="", back_en=False, memory_model_en=False, **kwargs):
        self.name = name
        self.done = False
        self.debug = debug
        self.done_cycles = 0
        self.start_cycle = ''
        self.total_cycles = 0
        self.block_start = True
        self.get_stats = statistics

        self.backpressure_en = back_en
        self.memory_model_en = memory_model_en

    def out_done(self):
        return self.done

    def is_debug(self):
        return self.debug

    @abstractmethod
    def update(self):
        pass

    # Check the input token of something
    def valid_token(self, element, datatype=int):
        return element != "" and element is not None and \
            (is_dtkn(element) or is_stkn(element) or is_nc_tkn(element, datatype) or is_0tkn(element))

    def reset(self):
        self.done = False

    def get_done_cycle(self):
        if self.done:
            return self.done_cycles
        else:
            return ''

    def update_done(self):
        self.total_cycles += 1
        if not self.done:
            self.done_cycles += 1
        if not self.block_start and self.start_cycle == '':
            self.start_cycle = self.total_cycles

    def return_statistics(self):
        return {"done_cycles": self.done_cycles, "start_cycle": self.start_cycle, "total_cycle": self.total_cycles}

    def return_statistics_base(self):
        return {"done_cycles": self.done_cycles, "start_cycle": self.start_cycle, "total_cycle": self.total_cycles}


def remove_emptystr(stream):
    return [x for x in stream if x != '']


def remove_stoptkn(stream):
    return [x for x in stream if x != 'S']


def remove_donetkn(stream):
    return [x for x in stream if x != 'D']


# ----------- Bitvector ------------------
def right_bit_set(bits):
    pos = 0
    m = 1

    while not (bits & m):
        # left shift
        m = m << 1
        pos += 1

    return 1 << pos


def popcount(bits):
    return bin(bits).count('1')


def get_nth_bit(bits, n):
    assert popcount(bits) > n
    rbit = None
    for i in range(n + 1):
        rbit = right_bit_set(bits)
        bits &= ~rbit
    return rbit
