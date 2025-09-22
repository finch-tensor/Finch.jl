import sys
import numpy as np
import scipy.sparse
import scipy.io
import os
import argparse
import ast
import yaml
import copy
import pickle
import random
import sparse
import sys

from pathlib import Path

from sam.util import SUITESPARSE_PATH, SuiteSparseTensor, InputCacheSuiteSparse, PydataTensorShifter, ScipyTensorShifter, \
    FROSTT_PATH, FrosttTensor, PydataSparseTensorDumper, InputCacheTensor, constructOtherMatKey, constructOtherVecKey
from sam.sim.src.tiling.process_expr import parse_all

# TODO: tensor3_innerprod output X() vs X(i,j,k)
SAM_STRS = {"matmul_kij": "X(i,j)=B(i,k)*C(k,j) -f=X:ss -f=B:ss:1,0 -f=C:ss -s=reorder(k,i,j)",
            "matmul_ikj": "X(i,j)=B(i,k)*C(k,j) -f=X:ss -f=B:ss -f=C:ss -s=reorder(i,k,j)",
            "matmul_ijk": "X(i,j)=B(i,k)*C(k,j) -f=X:ss -f=B:ss -f=C:ss:1,0  -s=reorder(i,j,k)",
            "mat_elemadd": "X(i,j)=B(i,j)+C(i,j) -f=X:ss -f=B:ss -f=C:ss:1,0  -s=reorder(i,j,k)",
            "mat_elemadd3": "X(i,j)=B(i,j)+C(i,j)+D(i,j) -f=X:ss -f=B:ss -f=C:ss -f=D:ss",
            "mat_elemmul": "X(i,j)=B(i,j)*C(i,j) -f=X:ss -f=B:ss -f=C:ss:1,0  -s=reorder(i,j,k)",
            "mat_mattransmul": "X(i,j)=C(j,i)*c(j)+d(i) -f=X:ss -f=B:ss -f=c:ss:0 -f=d:ss:0  -s=reorder(i,j)",
            "mat_vecmul_ij": "X(i,j)=B(i,j)*c(j) -f=X:ss -f=B:ss -f=c:ss:0  -s=reorder(i,j)",
            # "tensor3_elemadd": "X(i,j,k)=B(i,j,k)+C(i,j,k) -f=X:sss -f=B:sss -f=C:sss:2,1,0  -s=reorder(i,j,k)",
            # "tensor3_elemadd": "X(i,j,k)=B(i,j,k)+C(i,j,k) -f=X:sss -f=B:sss -f=C:sss:0  -s=reorder(i,j,k)",
            "tensor3_elemadd": "X(i,j,k)=B(i,j,k)+C(i,j,k) -f=X:sss -f=B:sss -f=C:sss",
            "tensor3_ttm": "X(i,j,k)=B(i,j,l)*C(k,l) -f=X:sss -f=B:sss -f=C:ss",
            "tensor3_ttv": "X(i,j)=B(i,j,k)*c(k) -f=X:ss -f=B:sss -f=c:s",
            "tensor3_innerprod": "X(i,j,k)=B(i,j,k)*C(i,j,k) -f=X:sss -f=B:sss -f=C:sss",
            "tensor3_mttkrp": "X(i,j)=B(i,k,l)*C(j,k)*D(j,l) -f=X:ss -f=B:sss -f=C:ss -f=D:ss"}


def print_dict(dd):
    for k, v in dd.items():
        print(k, ":", v)


def print_ast(node):
    for child in ast.iter_child_nodes(node):
        print_ast(child)
    print(node)


def get_ivars(names, expr):
    [lhs, rhs] = expr.split("=")

    tree = ast.parse(rhs)
    analyzer = IvarAnalyzer(names, tree)
    return analyzer.get_ivars()


class IvarAnalyzer(ast.NodeVisitor):
    def __init__(self, tensor_names, tree):
        self.tree = tree
        self.tensor_names = tensor_names
        self.name = None
        self.call = False
        self.ivars = {}

    def visit_Call(self, node):
        self.call = True
        if self.name == node.func.id:
            freevars = []
            for arg in node.args:
                ivar = self.visit(arg)
                if ivar is not None:
                    freevars.append(ivar)
            self.ivars[self.name] = freevars
        self.call = False

    def visit_Name(self, node):
        if self.call:
            return node.id
        else:
            return None

    def get_ivars(self):
        for name in self.tensor_names:
            self.name = name
            self.visit(self.tree)
        return self.ivars


def parse_sam_input(string):
    sam_str = SAM_STRS[string]

    str_arr = sam_str.split(" ")
    dictionary = parse_all(str_arr, has_quotes=False)

    # Assume there are no repeat tensors...
    tensors = dictionary["rhs_tensors"]
    permutations = [list(map(int, dictionary[tensor]["perm"])) for tensor in tensors]
    ivars = get_ivars(tensors, str_arr[0])
    ivars = [ivars[tensor] for tensor in tensors]

    print("PARSE SAM INPUTS", tensors)
    return tensors, permutations, ivars


# Outputs Pydata/sparse tensor tiles, given a pydata/sparse tensor (DOK or COO)
# ASSUME: tensor is a scipy.sparse.coo_matrix
# TODO: new_ivar_order right now is assumed to be one fixed order
#       In the future, will have to take into acocunt all reorderings
def tile_tensor(tensor, ivar_map, split_map, new_ivar_order=None, tensor_name=""):
    human_readable = False

    tiles = dict()
    tile_sizes = dict()
    order = len(tensor.shape)

    tensor_coo = sparse.COO(tensor)
    tensor_points = sparse.DOK.from_coo(tensor_coo)

    print("ivar_map: ", ivar_map)
    print("split_map: ", split_map)
    print("order = ", order)

    new_shape = []
    for lvl in range(order):
        ivar = ivar_map[lvl]
        sf = split_map[ivar]
        new_shape.append(sf)

    for crds, val in tensor_points.data.items():
        point = list(crds)

        new_point = []
        tile_id = []
        for lvl in range(order):
            ivar = ivar_map[lvl]
            sf = split_map[ivar]

            new_point.append(point[lvl] % sf)
            tile_id.append(int(point[lvl] / sf))

        # Add in value to the new_point as well
        new_point.append(val)
        tile_id = tuple(tile_id)

        if tile_id in tiles:
            tiles[tile_id].append(new_point)
        else:
            tiles[tile_id] = [new_point]

    # sort the new coo lists
    for key, val in tiles.items():
        if human_readable:
            dok = sorted(val)
        else:
            dok = sparse.DOK(tuple(new_shape))
            for point in val:
                dok[tuple(point[0:-1])] = point[-1]

        tiles[key] = dok

    for tile_id, tile_dok in tiles.items():
        tile = tile_dok.to_coo()
        # FIXME: This size number isn't correct for tensor tiles
        nonempty_rows = tile.nnz
        nonempty_row_ind = np.where(nonempty_rows > 0)[0]
        tile_sizes[tile_id] = tile.nnz * 2 + 2 * len(nonempty_row_ind) + 3

    return tiles, tile_sizes


# Outputs COO tiles, given a COO tensor
# ASSUME: tensor is a scipy.sparse.coo_matrix
# TODO: new_ivar_order right now is assumed to be one fixed order
#       In the future, will have to take into acocunt all reorderings
def tile_coo(tensor, ivar_map, split_map, new_ivar_order=None, tensor_name=""):
    human_readable = False

    tiles = dict()
    tile_sizes = dict()
    order = len(tensor.shape)

    tensor_coo = scipy.sparse.coo_matrix(tensor)
    tensor_points = tensor_coo.todok()

    print("ivar_map: ", ivar_map)
    print("split_map: ", split_map)
    print("order = ", order)

    new_shape = []
    for lvl in range(order):
        ivar = ivar_map[lvl]
        sf = split_map[ivar]
        new_shape.append(sf)

    for crds, val in tensor_points.items():
        point = list(crds)

        new_point = []
        tile_id = []
        for lvl in range(order):
            ivar = ivar_map[lvl]
            sf = split_map[ivar]

            new_point.append(point[lvl] % sf)
            tile_id.append(int(point[lvl] / sf))

        # Add in value to the new_point as well
        new_point.append(val)
        tile_id = tuple(tile_id)

        if tile_id in tiles:
            tiles[tile_id].append(new_point)
        else:
            tiles[tile_id] = [new_point]

    # sort the new coo lists
    for key, val in tiles.items():
        if human_readable:
            dok = sorted(val)
        else:
            dok = scipy.sparse.dok_matrix(tuple(new_shape))
            for point in val:
                dok[tuple(point[0:-1])] = point[-1]

        tiles[key] = dok

    for tile_id, tile_dok in tiles.items():
        tile = tile_dok.tocoo()
        nonempty_rows = tile.getnnz(axis=1)
        nonempty_row_ind = np.where(nonempty_rows > 0)[0]
        tile_sizes[tile_id] = tile.nnz * 2 + 2 * len(nonempty_row_ind) + 3

    return tiles, tile_sizes


# tensor_names: list of tensor names [B,C,D] (from SAM)
# tensors: list of sparse COO tensors (either Scipy or Pydata/Sparse) following tensor_names (from SAM)
# permutation_strs: list of permutation_strs [ss01, ss10] following tensor_names (from SAM)
# ivar_strs: list of ivar_strs ["ik", "kj"] following tensor_names (from SAM)
# split_map: dictionary of split factors (from hardware)
def cotile_coo(tensor_names, tensors, permutation_strs, ivar_strs, split_map, higher_order=False):
    tiled_tensors = dict()
    tiled_tensor_sizes = dict()

    print(tensor_names, tensors, permutation_strs, ivar_strs, split_map)
    for i, tensor in enumerate(tensors):
        tensor_name = tensor_names[i]
        tensor_format = permutation_strs[i]
        ivar_map = dict()
        order = len(tensor.shape)
        print("order is ", order)
        for dim in range(order):
            print("tensor format: ", tensor_format)

            print("dim is ", dim)
            print("tensor_format[dim:dim+1] is ", tensor_format[dim:dim + 1])
            lvl_permutation = tensor_format[dim:dim + 1][0]
            ivar = ivar_strs[i][dim]
            ivar_map[lvl_permutation] = ivar
            print("ivar_map is ", ivar_map)

        if higher_order:
            tiles, tile_sizes = tile_tensor(tensor, ivar_map, split_map, tensor_name=tensor_name)
        else:
            tiles, tile_sizes = tile_coo(tensor, tensor_name, ivar_map, split_map, tensor_name=tensor_name)

        tiled_tensors[tensor_name] = tiles
        tiled_tensor_sizes[tensor_name] = tile_sizes

    return tiled_tensors, tiled_tensor_sizes


def get_other_tensors(app_str, tensor, other_nonempty=True):
    tensors = [tensor]

    if "matmul" in app_str:
        print("Writing shifted...")
        shifted = ScipyTensorShifter().shiftLastMode(tensor)
        trans_shifted = shifted.transpose()
        tensors.append(trans_shifted)

    elif "mat_elemadd3" in app_str:
        print("Writing shifted...")
        shifted = ScipyTensorShifter().shiftLastMode(tensor)
        tensors.append(shifted)

        print("Writing  shifted2...")
        shifted2 = ScipyTensorShifter().shiftLastMode(shifted)
        tensors.append(shifted2)
    elif "mat_elemadd" in app_str or "mat_elemmul" in app_str:
        print("Writing shifted...")
        shifted = ScipyTensorShifter().shiftLastMode(tensor)
        tensors.append(shifted)

    elif "mat_sddmm" in app_str:
        pass
    elif "mat_mattransmul" in app_str:
        print("Writing other tensors...")
        rows, cols = tensor.shape  # i,j
        tensor_c = scipy.sparse.random(cols, 1, data_rvs=np.ones).toarray().flatten()
        # tensor_d = scipy.sparse.random(rows, 1, density=1.0, data_rvs=np.ones).toarray().flatten()
        tensor_d = scipy.sparse.random(rows, 1, data_rvs=np.ones).toarray().flatten()

        if other_nonempty:
            tensor_c[0] = 1
            tensor_d[0] = 1

        # import pdb; pdb.set_trace()

        tensors.append(tensor_c)
        tensors.append(tensor_d)

    elif "mat_residual" in app_str:
        pass
    elif "mat_vecmul" in app_str:
        print("Writing other tensors...")
        rows, cols = tensor.shape
        tensor_c = scipy.sparse.random(cols, 1, data_rvs=np.ones).toarray().flatten()

        if other_nonempty:
            tensor_c[0] = 1

        tensors.append(tensor_c)

    elif "tensor3_ttv" in app_str:
        print("Writing other tensors...")

        tensorName = args.input_tensor
        variant = "mode2"  # k dimension is mode 2
        path = constructOtherVecKey(tensorName, variant)
        tensor_c_loader = FrosttTensor(path)
        tensor_c = tensor_c_loader.load().todense()
        print(tensor_c)
        # size_i, size_j, size_k = tensor.shape  # i,j,k
        # print("OTHER SIZES: ", size_i, size_j, size_k)
        # tensor_c = scipy.sparse.random(size_k, 1, data_rvs=np.ones).toarray().flatten()

        if other_nonempty:
            tensor_c[0] = 1

        tensors.append(tensor_c)

    elif "tensor3_ttm" in app_str:
        print("IN TILE.PY NOW")
        print("Writing other tensors...")

        tensorName = args.input_tensor
        variant = "mode2_ttm"
        path = constructOtherMatKey(tensorName, variant)
        matrix_c_loader = FrosttTensor(path)
        matrix_c = matrix_c_loader.load().todense()
        print(matrix_c)
        # size_i, size_j, size_l = tensor.shape  # i,j,k
        # print("OTHER SIZES: ", size_i, size_j, size_l)
        # # dimension_k = random.randint(min(tensor.shape), 10)
        # dimension_k = 3
        # tensor_c = scipy.sparse.random(dimension_k, size_l, density=0.25, data_rvs=np.ones).toarray()
        # tensor_c = scipy.sparse.random(dimension_k, size_l, data_rvs=np.ones).toarray().flatten()

        if other_nonempty:
            matrix_c[0] = 1

        tensors.append(matrix_c)

    elif "tensor3_elemadd" in app_str:
        print("Writing shifted...")
        # shifted = ScipyTensorShifter().shiftLastMode(tensor)
        shifted = PydataTensorShifter().shiftLastMode(tensor)
        tensors.append(shifted)
    elif "tensor3_innerprod" in app_str:
        print("Writing shifted...")

        print("Writing shifted...")
        print("tensor.shape = ", tensor.shape)
        # shifted = ScipyTensorShifter().shiftLastMode(tensor)
        shifted = PydataTensorShifter().shiftLastMode(tensor)
        tensors.append(shifted)
    elif "tensor3_mttkrp" in app_str:
        print("Writing shifted...")
        size_i, size_j, size_l = tensor.shape  # i,j,k

        tensorName = args.input_tensor
        variant = "mode1_mttkrp"
        path = constructOtherMatKey(tensorName, variant)
        matrix_c_loader = FrosttTensor(path)
        matrix_c = matrix_c_loader.load().todense()
        print(matrix_c)

        tensorName = args.input_tensor
        variant = "mode2_mttkrp"
        path = constructOtherMatKey(tensorName, variant)
        matrix_d_loader = FrosttTensor(path)
        matrix_d = matrix_d_loader.load().todense()
        print(matrix_d)
        # size_k = random.randint(min(tensor.shape), 10)
        # # C & D are dense according to TACO documentation
        # matrix_c = scipy.sparse.random(size_j, size_k, density=1, data_rvs=np.ones).toarray()
        # matrix_d = scipy.sparse.random(size_j, size_l, density=1, data_rvs=np.ones).toarray()

        if other_nonempty:
            matrix_c[0] = 1
            matrix_d[0] = 1

        tensors.append(matrix_c)
        tensors.append(matrix_d)
    else:
        # tensor2 = scipy.sparse.random(tensor.shape[0], tensor.shape[1])
        # tensors.append(tensor2)
        raise NotImplementedError

    return tensors


def cotile_multilevel_coo(app_str, hw_config_fname, tensors, output_dir_path, higher_order=False):
    tensors = get_other_tensors(app_str, tensors[0])

    names, format_permutations, ivars = parse_sam_input(args.cotile)

    print("cotile_multilevel_coo tensors: ", names, "\n", tensors)

    # import pdb
    # pdb.set_trace()

    sizes_dict = {}
    for i, name in enumerate(names):
        tensor = tensors[i]
        sizes_dict[name] = tensor.shape
    tensor_sizes_fname = os.path.join(output_dir_path, "tensor_sizes")
    with open(tensor_sizes_fname, "wb+") as pickle_fname:
        pickle.dump(sizes_dict, pickle_fname)

    with open(hw_config_fname, "r") as stream:
        try:
            hw_config = yaml.safe_load(stream)

            n_levels = hw_config["n_levels"]
            level_names = hw_config["level_names"]
            assert len(level_names) == n_levels

            cotiled = None
            cotiled_sizes = None
            split_map = {}
            for hw_lvl in range(n_levels - 1):
                cotiled_sizes_fname = os.path.join(output_dir_path, "hw_level_" + str(hw_lvl) + "_sizes")
                level_names_left = level_names[hw_lvl + 1:]

                sf = 1
                for level_name in level_names_left:
                    hw_key = level_name + "_tile_size"
                    tile_size = hw_config[hw_key]
                    sf *= tile_size

                unique_ivars = list(set(sum(ivars, [])))
                for ivar in unique_ivars:
                    split_map[ivar] = sf

                if cotiled is None:
                    # First iteration of tiling
                    cotiled, cotiled_sizes = cotile_coo(names, tensors, format_permutations, ivars, split_map,
                                                        higher_order)
                else:
                    # recursively tile the blocks
                    new_cotiled = {}
                    new_cotiled_sizes = {}
                    for i, name in enumerate(names):

                        new_cotiled[name] = {}
                        new_cotiled_sizes[name] = {}
                        for tile_id, tile in cotiled[name].items():
                            if higher_order:
                                tile_in_coo = tile.to_coo()
                            else:
                                tile_in_coo = tile.tocoo()
                            new_cotiled_temp, new_cotiled_sizes_temp = cotile_coo(name, [tile_in_coo],
                                                                                  [format_permutations[i]], [ivars[i]],
                                                                                  split_map, higher_order)

                            for kk, vv in copy.deepcopy(new_cotiled_temp)[name].items():
                                new_tile_id = tuple(list(tile_id) + list(kk))
                                new_cotiled[name][new_tile_id] = vv

                            for kk, vv in copy.deepcopy(new_cotiled_sizes_temp)[name].items():
                                new_tile_id = tuple(list(tile_id) + list(kk))
                                new_cotiled_sizes[name][new_tile_id] = vv
                    cotiled = copy.deepcopy(new_cotiled)
                    cotiled_sizes = copy.deepcopy(new_cotiled_sizes)

                with open(cotiled_sizes_fname, "wb+") as pickle_fname:
                    pickle.dump(cotiled_sizes, pickle_fname)

            return cotiled
        except yaml.YAMLError as exc:
            print(exc)


inputCacheSuiteSparse = InputCacheSuiteSparse()
inputCacheTensor = InputCacheTensor()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='script that tiles tensors')
    parser.add_argument("--tensor_type", choices=['ex', 'gen', 'file', 'ss', 'frostt'], help='The \
        type of tensor to tile: extensor(ex), generated (gen), \
        SuiteSparse (ss), FROSTT (frostt), or input file (file)')
    parser.add_argument("--higher_order", action="store_true", help="If \
    true then we want to process a higher-order tensor. With higher-order set to true, if \
    'tensor_type' is: \
    \n 'gen' then a 3-tensor is generated instead of matrix. \
    \n 'file' then a .tns file is read instead of a .mtx file. \
    \n 'ss' then other matrices used with SuiteSparse are .tns instead of .mtx files. \
    \n 'frostt' should always have 'higher_order' set as true.")

    parser.add_argument("--input_tensor", type=str, default=None,
                        help="Input tensor NAME if tensor_type is set to 'file'. \
            This is for use with SuiteSparse or FROSTT")
    parser.add_argument("--input_path", type=str, default=None, help="Input tensor path")
    parser.add_argument("--output_dir_path", type=str, default="./tiles",
                        help='Output path, directory where tiles get written to')
    parser.add_argument("--hw_config", type=str, default=None,
                        help='Path to the hardware config yaml')

    parser.add_argument("--cotile", type=str, default=None, help='If \
            this is true cotile multiple tensors, else tile one tensor only')
    parser.add_argument("--multilevel", action="store_true", help='If \
            multilevel is true there will exist more than one level of tiles, \
            else only tile once')
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--other_nonempty", action="store_true",
                        help="If this is enabled, the 'other' tensors will have at least one nonzero value")

    args = parser.parse_args()
    print("Args:", args)

    random.seed(args.seed)
    np.random.seed(args.seed)

    tensor = None
    cwd = os.getcwd()
    inputCache = None

    if args.tensor_type == "gen":
        if args.higher_order:
            tensor = sparse.COO(sparse.random((16, 16, 16)))
        else:
            tensor = scipy.sparse.random(16, 16)
    elif args.tensor_type == "ex":
        tensor = scipy.io.mmread(args.input_path)
    elif args.tensor_type == "ss":
        assert args.input_tensor is not None

        inputCache = inputCacheSuiteSparse
        tensor_path = os.path.join(SUITESPARSE_PATH, args.input_tensor + ".mtx")
        ss_tensor = SuiteSparseTensor(tensor_path)
        tensor = inputCache.load(ss_tensor, False)
    elif args.tensor_type == "frostt":
        assert args.input_tensor is not None
        assert args.higher_order

        inputCache = inputCacheTensor
        tensor_path = os.path.join(FROSTT_PATH, args.input_tensor + ".tns")
        print("tensor_path", tensor_path)

        # FIXME: This is broken
        frostt_tensor = FrosttTensor(tensor_path)
        tensor = inputCache.load(frostt_tensor, False)

    else:
        raise ValueError("This choice of 'tensor_type' is unreachable")

    split_map = {"i": 16, "j": 16, "k": 16}

    if args.cotile is None:
        print("ORIG:", tensor)
        print("SPLIT MAP", split_map)
        tiles, tile_sizes = tile_coo(tensor, {0: "i", 1: "j"}, split_map)

        print("TILES:")
        print_dict(tiles)
    else:
        output_mtx_name = os.path.join(args.output_dir_path, "mtx")
        output_mtx_path = Path(output_mtx_name)

        output_mtx_path.mkdir(parents=True, exist_ok=True)
        print(os.path.exists(output_mtx_path))

        if args.multilevel:
            assert args.cotile is not None
            cotiled_tensors = cotile_multilevel_coo(args.cotile, args.hw_config, [tensor],
                                                    os.path.join(args.output_dir_path),
                                                    args.higher_order)
        elif args.cotile is not None:
            tensor2 = scipy.sparse.random(tensor.shape[0], tensor.shape[1])
            names, format_permutations, ivars = parse_sam_input(args.cotile)

            cotiled_tensors = cotile_coo(names, [tensor, tensor2],
                                         format_permutations, ivars, split_map, args.higher_order)
            # print(cotiled_tensors)

        names = cotiled_tensors.keys()
        for name in names:
            for tile_id, tile in cotiled_tensors[name].items():
                [str(item) for item in tile_id]
                filename = "tensor_" + name + "_tile_" + "_".join([str(item) for item in tile_id])
                filename += ".tns" if args.higher_order and args.tensor_type == "frostt" else ".mtx"
                # filename += ".mtx"
                mtx_path_name = os.path.join(output_mtx_name, filename)
                print(tile)
                print("Output path:", mtx_path_name)

                if args.higher_order:
                    if args.tensor_type == "frostt":
                        tns_dumper = PydataSparseTensorDumper()
                        print(tile.shape)
                        print(tile)
                        tns_dumper.dump(tile, mtx_path_name, True)

                    # FIXME: (owhsu) Why did avb03 add this in?
                    elif len(tile.shape) == 1:
                        # print(np.array(tile.todense()).reshape(1,-1))
                        scipy.io.mmwrite(mtx_path_name, scipy.sparse.coo_matrix(tile.todense()))
                    else:
                        # print(tile.todense())
                        scipy.io.mmwrite(mtx_path_name, scipy.sparse.coo_matrix(tile.todense()))

                else:
                    scipy.io.mmwrite(mtx_path_name, tile)
