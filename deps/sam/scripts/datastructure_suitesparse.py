import argparse
import os
import shutil
import scipy.sparse
import numpy as np

from pathlib import Path

from util import FormatWriter, SuiteSparseTensor, InputCacheSuiteSparse
from sam.util import SUITESPARSE_FORMATTED_PATH, ScipyTensorShifter

all_formats = ["coo", "cooT", "csr", "dcsr", "dcsc", "csc", "dense", "denseT"]
formats = ["coo", "cooT", "csr", "dcsr", "dcsc", "csc", "dense"]
scipy_formats = ["coo", "csr", "csc"]


def write_datastructure_tiles(args, tensor, out_path, tile_name):
    print("Writing " + args.name + " for test " + args.benchname + "...")

    dirname = args.output_dir_path if args.output_dir_path is not None else os.path.join(out_path, args.name, args.benchname)
    dirname = os.path.join(dirname, tile_name)
    dirpath = Path(dirname)
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True, mode=0o777)

    print(tile_name)
    tensorname = tile_name.split("_")[1]

    coo = inputCache.load(tensor, False)
    formatWriter.writeout_separate_sparse_only(coo, dirname, tensorname, format_str="ss01", hw=False)


def write_datastructure_bench(args, tensor, out_path, tiles=None):
    shifter = ScipyTensorShifter()

    print("Writing " + args.name + " for test " + args.benchname + "...")

    dirname = args.output_dir_path if args.output_dir_path is not None else os.path.join(out_path, args.name, args.benchname)
    if tiles is not None:
        dirname = os.path.join(dirname, tiles)
    dirpath = Path(dirname)
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True, mode=0o777)

    if "mat_mattransmul" in args.benchname or "mat_residual" in args.benchname:
        tensorname = "C"
    else:
        tensorname = "B"

    coo = inputCache.load(tensor, False)
    shape = coo.shape

    # These benchmarks need format_str == "ss10"
    if args.benchname not in ["matmul_kij", "matmul_kji", "matmul_jki", "mat_vecmul", "mat_vecmul_ji", "mat_mattransmul"]:
        formatWriter.writeout_separate_sparse_only(coo, dirname, tensorname, format_str="ss01")

    if "matmul_ijk" in args.benchname:
        shifted = shifter.shiftLastMode(coo)

        print("Writing " + args.name + " shifted and transposed...")
        tensorname = "C"
        trans_shifted = shifted.transpose()
        formatWriter.writeout_separate_sparse_only(trans_shifted, dirname, tensorname, format_str="ss10")

    elif "matmul_jik" in args.benchname:
        shifted = shifter.shiftLastMode(coo)

        print("Writing " + args.name + " shifted and transposed...")
        tensorname = "C"
        trans_shifted = shifted.transpose()
        formatWriter.writeout_separate_sparse_only(trans_shifted, dirname, tensorname, format_str="ss10")
    elif "matmul_ikj" in args.benchname:
        shifted = shifter.shiftLastMode(coo)

        print("Writing " + args.name + " shifted and transposed...")
        tensorname = "C"
        trans_shifted = shifted.transpose()
        formatWriter.writeout_separate_sparse_only(trans_shifted, dirname, tensorname, format_str="ss01")

    elif "matmul_jki" in args.benchname:
        formatWriter.writeout_separate_sparse_only(coo, dirname, tensorname, format_str="ss10")

        shifted = shifter.shiftLastMode(coo)

        print("Writing " + args.name + " shifted and transposed...")
        tensorname = "C"
        trans_shifted = shifted.transpose()
        formatWriter.writeout_separate_sparse_only(trans_shifted, dirname, tensorname, format_str="ss10")

    elif "matmul_kij" in args.benchname:
        formatWriter.writeout_separate_sparse_only(coo, dirname, tensorname, format_str="ss10")

        shifted = shifter.shiftLastMode(coo)

        print("Writing " + args.name + " shifted and transposed...")
        tensorname = "C"
        trans_shifted = shifted.transpose()
        formatWriter.writeout_separate_sparse_only(trans_shifted, dirname, tensorname, format_str="ss01")

    elif "matmul_kji" in args.benchname:
        formatWriter.writeout_separate_sparse_only(coo, dirname, tensorname, format_str="ss10")

        shifted = shifter.shiftLastMode(coo)

        print("Writing " + args.name + " shifted and transposed...")
        tensorname = "C"
        trans_shifted = shifted.transpose()
        formatWriter.writeout_separate_sparse_only(trans_shifted, dirname, tensorname, format_str="ss01")

    elif "mat_elemadd3" in args.benchname:
        print("Writing " + args.name + " shifted...")
        tensorname = "C"
        shifted = shifter.shiftLastMode(coo)
        formatWriter.writeout_separate_sparse_only(shifted, dirname, tensorname, format_str="ss01")

        print("Writing " + args.name + " shifted2...")
        tensorname = "D"
        shifted2 = shifter.shiftLastMode(shifted)
        formatWriter.writeout_separate_sparse_only(shifted2, dirname, tensorname, format_str="ss01")

    elif "mat_elemadd" in args.benchname or "mat_elemmul" in args.benchname:
        print("Writing " + args.name + " shifted...")
        tensorname = "C"
        shifted = shifter.shiftLastMode(coo)
        formatWriter.writeout_separate_sparse_only(shifted, dirname, tensorname, format_str="ss01")

    elif "mat_mattransmul" in args.benchname:
        formatWriter.writeout_separate_sparse_only(coo, dirname, tensorname, format_str="ss10")
        if not args.no_gen_other:
            tensorname = 'd'
            vec = scipy.sparse.random(shape[0], 1, density=args.density, data_rvs=np.ones)
            vec = vec.toarray().flatten()
            formatWriter.writeout_separate_vec(vec, dirname, tensorname)

            tensorname = 'f'
            vec = scipy.sparse.random(shape[1], 1, density=args.density, data_rvs=np.ones)
            vec = vec.toarray().flatten()
            formatWriter.writeout_separate_vec(vec, dirname, tensorname)
    elif "mat_vecmul" == args.benchname or "mat_vecmul_ji" in args.benchname:
        formatWriter.writeout_separate_sparse_only(coo, dirname, tensorname, format_str="ss10")
        if not args.no_gen_other:
            tensorname = 'c'
            vec = scipy.sparse.random(shape[1], 1, density=args.density, data_rvs=np.ones)
            vec = vec.toarray().flatten()
            formatWriter.writeout_separate_vec(vec, dirname, tensorname)
    elif "mat_vecmul_ij" in args.benchname:
        pass
    elif "mat_sddmm" in args.benchname:
        pass
    elif "mat_residual" in args.benchname:
        if not args.no_gen_other:
            tensorname = 'b'
            vec = scipy.sparse.random(shape[0], 1, density=args.density, data_rvs=np.ones)
            vec = vec.toarray().flatten()
            formatWriter.writeout_separate_vec(vec, dirname, tensorname)

            tensorname = 'd'
            vec = scipy.sparse.random(shape[1], 1, density=args.density, data_rvs=np.ones)
            vec = vec.toarray().flatten()
            formatWriter.writeout_separate_vec(vec, dirname, tensorname)
    elif "mat_identity" in args.benchname:
        pass
    else:
        raise NotImplementedError


parser = argparse.ArgumentParser(description="Process some suitesparse matrices into per-level datastructures")
parser.add_argument('-n', '--name', metavar='ssname', type=str, action='store', help='tensor name to run format '
                                                                                     'conversion on one SS tensor')
parser.add_argument('-f', '--format', metavar='ssformat', type=str, action='store', help='The format that the tensor '
                                                                                         'should be converted to')
parser.add_argument('-comb', '--combined', action='store_true', default=False, help='Whether the formatted datastructures '
                    'should be in separate files')
parser.add_argument('-o', '--omit-dense', action='store_true', default=False, help='Do not create fully dense format')
parser.add_argument('-cast', '--cast', action='store_true', default=False, help='Safe sparsity cast to int for values')
parser.add_argument('-hw', '--hw', action='store_true', default=False,
                    help='Only generate formats used for hardware testing (all sparse'
                         'levels, concordant)')
parser.add_argument('-b', '--benchname', type=str, default=None, help='test name to run format '
                                                                      'conversion on')
parser.add_argument('--input_path', type=str, default=None)
parser.add_argument('--output_dir_path', type=str, default=None)
parser.add_argument('--tiles', action='store_true')
parser.add_argument('--no_gen_other', action='store_true', help="Whether this"
                    "script should generate the randmo 'other' tensors")
parser.add_argument('--seed', type=int, default=0, help='Random seed needed for gen_other')
parser.add_argument('--density', type=int, default=0.25, help='If gen_other, used for density of "other" tensor')
args = parser.parse_args()

np.random.seed(args.seed)

inputCache = InputCacheSuiteSparse()
formatWriter = FormatWriter(args.cast)

cwd = os.getcwd()
if args.output_dir_path is None:
    out_dirname = SUITESPARSE_FORMATTED_PATH
else:
    out_dirname = args.output_dir_path

out_path = Path(out_dirname)
out_path.mkdir(parents=True, exist_ok=True, mode=0o777)

if args.name is None:
    print("Please enter a matrix name")
    exit()

if args.input_path is None:
    SS_PATH = os.getenv('SUITESPARSE_TENSOR_PATH', default=os.path.join(cwd, 'suitesparse'))

else:
    SS_PATH = args.input_path

tensor = None
mtx_files = None
if args.tiles:
    # get all mtx tile files from args.input_path
    mtx_files = [os.path.join(args.input_path, fname) for fname in os.listdir(args.input_path) if fname.endswith(".mtx")]

    tensor = [SuiteSparseTensor(mtx_file) for mtx_file in mtx_files]
elif args.input_path is not None:
    tensor = SuiteSparseTensor(args.input_path)
else:
    print(SS_PATH)
    tensor = SuiteSparseTensor(SS_PATH)

if args.format is not None:
    assert args.format in formats
    filename = os.path.join(out_path, args.name + "_" + args.format + ".txt")

    coo = inputCache.load(tensor, False)
    formatWriter.writeout(coo, args.format, filename)
elif args.combined:
    for format_str in formats:
        filename = os.path.join(out_path, args.name + "_" + format_str + ".txt")
        print("Writing " + args.name + " " + format_str + "...")

        coo = inputCache.load(tensor, False)
        formatWriter.writeout(coo, format_str, filename)

        shifted_filename = os.path.join(out_path, args.name + "_shifted_" + format_str + ".txt")
        shifted = ScipyTensorShifter().shiftLastMode(coo)
        formatWriter.writeout(shifted, format_str, shifted_filename)

        trans_filename = os.path.join(out_path, args.name + "_trans_shifted_" + format_str + ".txt")
        trans_shifted = shifted.transpose()
        formatWriter.writeout(trans_shifted, format_str, trans_filename)
elif args.hw:
    if args.tiles and tensor is not None:
        for i, ten in enumerate(tensor):
            tile_name = os.path.split(mtx_files[i])[1].split(".")[0]
            write_datastructure_tiles(args, ten, out_path, tile_name)
    else:
        write_datastructure_bench(args, tensor, out_path)

else:
    print("Writing " + args.name + " original...")
    dirname = os.path.join(out_path, args.name, "orig")
    dirpath = Path(dirname)
    dirpath.mkdir(parents=True, exist_ok=True, mode=0o777)
    tensorname = "B"
    coo = inputCache.load(tensor, False)
    formatWriter.writeout_separate(coo, dirname, tensorname, omit_dense=args.omit_dense)

    print("Writing " + args.name + " shifted...")
    dirname = os.path.join(out_path, args.name, "shift")
    dirpath = Path(dirname)
    dirpath.mkdir(parents=True, exist_ok=True, mode=0o777)
    tensorname = "C"
    shifted = ScipyTensorShifter().shiftLastMode(coo)
    formatWriter.writeout_separate(shifted, dirname, tensorname, omit_dense=args.omit_dense)

    print("Writing " + args.name + " shifted and transposed...")
    dirname = os.path.join(out_path, args.name, "shift-trans")
    dirpath = Path(dirname)
    dirpath.mkdir(parents=True, exist_ok=True, mode=0o777)
    tensorname = "C"
    trans_shifted = shifted.transpose()
    formatWriter.writeout_separate(trans_shifted, dirname, tensorname, omit_dense=args.omit_dense)
