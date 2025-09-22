import copy
import random
import numpy as np
from functools import reduce

from sam.sim.src.wr_scanner import WrScan, CompressWrScan
from sam.sim.src.array import Array

TIMEOUT = 10000000000000


def check_arr(arr_obj, gold):
    assert (isinstance(arr_obj, WrScan) or isinstance(arr_obj, Array))
    # Assert the array stores values with the rest of the memory initialized to initial value
    assert (arr_obj.get_arr() == gold + [arr_obj.fill] * (arr_obj.size - len(gold)))
    # Assert the array stores only the values
    if isinstance(arr_obj, WrScan):
        arr_obj.resize_arr(len(gold))
    else:
        arr_obj.resize(len(gold))
    assert (arr_obj.get_arr() == gold)


def check_seg_arr(cwrscan, gold):
    assert (isinstance(cwrscan, CompressWrScan))
    # Assert the array stores values with the rest of the memory initialized to initial value
    assert (cwrscan.get_seg_arr() == (gold + [0] * (cwrscan.seg_size - len(gold))))
    # Assert the array stores only the values
    cwrscan.resize_seg_arr(len(gold))
    assert (cwrscan.get_seg_arr() == gold)


# FIXME: Need to generate streams with hierarchical stop tokens or
#           just delete this function
def gen_stream(n=1, max_val=10, max_nnz=10):
    assert (max_val >= max_nnz)

    num_s = 0
    result = []
    end = False
    while not end:
        t = []
        num_v = random.randint(1, max_nnz - 1)
        for _ in range(num_v):
            t.append(random.randint(0, max_val))
            t = sorted(set(t))
        for el in t:
            result.append(el)

        num_s = random.randint(1, n)
        result.append('S' + str(num_s - 1))
        end = num_s == n

    for _ in range(n - num_s):
        result.append('S')

    result.append('D')
    return result


def dedup_adj(olist):
    result = copy.deepcopy(olist)
    for i in range(len(result) - 1, 0, -1):
        if result[i] == result[i - 1]:
            del result[i]
    return result


def gen_crd_arr(dim=4):
    result = [i for i in range(dim) if bool(random.getrandbits(1))]
    if len(result) == 0:
        return [0]
    return result


# Returns crd_arr, seg_arr given a list of coordinates
def gen_comp_arrs(crd_lists, count=0, max_el=4):
    assert len(crd_lists) <= max_el, "Invalid: " + str(len(crd_lists)) + " is not <= " + str(max_el)

    if len(crd_lists) == 0:
        return [], []
    if len(crd_lists) == 1:
        return crd_lists[0], [count, count + len(crd_lists[0])]
    rest = gen_comp_arrs(crd_lists[1:], count=count + len(crd_lists[0]), max_el=max_el)
    return crd_lists[0] + rest[0], [count] + rest[1]


# Returns crd_arr, seg_arr given a list of coordinates
def gen_uncomp_arrs(crd_lists, count=0, max_el=4):
    assert len(crd_lists) == max_el, "Invalid: " + str(len(crd_lists)) + "!=" + str(max_el)

    if len(crd_lists) == 0:
        return [], []
    if len(crd_lists) == 1:
        return crd_lists[0], [count, count + len(crd_lists[0])]
    rest = gen_comp_arrs(crd_lists[1:], count=count + len(crd_lists[0]), max_el=max_el)
    return crd_lists[0] + rest[0], [count] + rest[1]


# Generates a n-dim CSF datastructure of dim**4 to populate arrays
def gen_n_comp_arrs(n=1, dim=4):
    crd_arrs = []
    seg_arrs = []
    return gen_n_comp_arrs_helper(1, crd_arrs, seg_arrs, n, n, dim)


def gen_n_comp_arrs_helper(prev_nnz=1, crd_arrs=[], seg_arrs=[], lvl=1, n=1, dim=4):
    if lvl <= 0:
        return crd_arrs, seg_arrs

    crds = []
    new_nnz = 0
    for i in range(prev_nnz):
        c = gen_crd_arr(dim)
        crds.append(c)
        new_nnz += len(c)
    max_el = dim ** (n + 1 - lvl)
    crd_arr, seg_arr = gen_comp_arrs(crds, max_el=max_el)
    crd_arrs.append(crd_arr)
    seg_arrs.append(seg_arr)
    return gen_n_comp_arrs_helper(new_nnz, crd_arrs, seg_arrs, lvl=lvl - 1, n=n, dim=dim)


def repeat_crds(crd_arr, seg_arr):
    assert (len(crd_arr) + 1 == len(seg_arr))

    result = []
    for i in range(len(crd_arr)):
        n = seg_arr[i + 1] - seg_arr[i]
        crd = crd_arr[i]
        result += [crd] * n
    return result


# Given coordinate and segment arrays,
# get points in a struct of arrays format
def get_point_list(crd_arrs, seg_arrs, val_arr=None):
    assert (len(crd_arrs) == len(seg_arrs))

    repeat_list = []
    for i in range(1, len(seg_arrs)):
        repeat_list.append(crd_arrs[i - 1])
        new_repeat_list = []
        for rl in repeat_list:
            result = repeat_crds(rl, seg_arrs[i])
            new_repeat_list.append(result)
        repeat_list = copy.deepcopy(new_repeat_list)

    repeat_list.append(crd_arrs[-1])

    if val_arr is not None:
        repeat_list.append(val_arr)
    return repeat_list


# Convert points into array of struct format
def convert_point_tuple(pt_list):
    n = len(pt_list)
    nnz = len(pt_list[0])

    # Check lengths match
    for p in pt_list:
        assert len(p) == nnz, str(p) + " does not have length " + str(nnz)

    pt_tup = []
    for i in range(nnz):
        point = []
        for j in range(n):
            point.append(pt_list[j][i])
        pt_tup.append(tuple(point))
    return pt_tup


# Remove all zero values from the point tuple
def remove_zeros(pt_tup):
    tup = sorted(pt_tup)
    ret_tup = []
    for x in tup:
        if x[-1] != 0:
            ret_tup.append(x)
    return ret_tup


# Given two array of struct format point lists,
# make sure they are equivalent
def check_point_tuple(pt_tup1, pt_tup2, err=1e-12):
    tup1 = sorted(pt_tup1)
    tup2 = sorted(pt_tup2)
    assert len(tup1) == len(tup2), "Pts1 length (" + str(len(tup1)) + " != Pts2 length (" + str(len(tup2)) + ")"
    assert len(tup1[0]) == len(tup2[0]), str(len(tup1[0])) + " != " + str(len(tup2[0]))

    for i in range(len(tup1)):
        if tup1[i] != tup2[i]:
            if abs(tup1[i][-1] - tup2[i][-1]) > max(abs(tup1[i][-1]) * err, err):
                print(str(i) + ":", tup1[i], "!=", tup2[i])
                # print(tup1)
                # print(tup2)
                return False
    return True


# Given two array of struct format point lists,
# make sure they are equivalent
def print_point_tuple(out_pt_tup, gold_pt_tup, err=1e-12):
    tup1 = sorted(out_pt_tup)
    tup2 = sorted(gold_pt_tup)

    for i in range(len(tup1)):
        if tup1[i] != tup2[i]:
            if abs(tup1[i][-1] - tup2[i][-1]) > max(abs(tup1[i][-1]) * err, err):
                print(str(i) + ":", tup1[i], "!=", tup2[i])


def convert_point_tuple_ndarr(pt_tup, dim=4):
    n = len(pt_tup[0]) - 1
    shape = [dim] * n
    result = np.zeros(shape)
    for pt in pt_tup:
        result[tuple(pt[:-1])] = pt[-1]

    return result


def convert_ndarr_point_tuple(ndarr):
    return [(*idx, val) for idx, val in np.ndenumerate(ndarr) if val != 0]


def gen_val_arr(size=4, max_val=100, min_val=-100):
    result = [random.randint(min_val, max_val) for _ in range(size)]
    result = [x if x != 0 else 1 for x in result]
    return result


def read_combined_inputs(filename, formatlist):
    return_list = []
    with open(filename) as file:
        file.readline()
        shape = file.readline().split()
        print(shape)
        shape = list(map(int, shape))
        return_list.append(shape)

        for f in formatlist:
            file.readline()
            if f == 'd':
                dimension = int(file.readline())
                return_list.append(dimension)
            elif f == 's':
                seg = file.readline().split()
                seg = list(map(int, seg))
                crd = file.readline().split()
                crd = list(map(int, crd))
                return_list.append((seg, crd))
            else:
                assert False

        file.readline()
        vals = file.readline().split()
        vals = list(map(float, vals))
        return_list.append(vals)

    return return_list


def read_inputs(filename, intype=int, base=10, early_terminate=None, positive_only=False):
    return_list = []
    with open(filename) as f:
        for line in f:
            if early_terminate is not None:
                if early_terminate in line:
                    break
            if base == 16:
                return_list.append(intype(line, base))
            else:
                # Convert to positive if needed?
                if positive_only:
                    temp_token = intype(float(line.strip()))
                    if temp_token < 0:
                        temp_token = temp_token * -1
                else:
                    temp_token = intype(float(line.strip()))
                return_list.append(temp_token)
    return return_list


# --------------Bitvector Helper Functions------------------------ #
def get_bv(crd):
    gold_bv = []
    temp = []
    for x in crd:
        if isinstance(x, int):
            temp.append(x)
        else:
            if temp:
                gold_bv.append(bin(reduce(lambda a, b: a | b, [0b1 << i for i in temp])))
                temp = []
            gold_bv.append(x)
    return gold_bv


def bv(ll):
    result = 0
    for elem in ll:
        result |= 1 << elem
    return result


def inner_bv(ll, size, sf):
    result = []
    for i in range(int(size / sf) + 2):
        temp = bv([elem % sf for elem in ll if max((i - 1) * sf, 0) <= elem < i * sf])
        if temp:
            result.append(temp)
    return result
