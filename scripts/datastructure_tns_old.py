import argparse
import os
from pathlib import Path
from util import parse_taco_format

cwd = os.getcwd()


formats = ["sss012", "ss01", "dss", "dds", "ddd", "dsd", "sdd", "sds", "ssd"]

parser = argparse.ArgumentParser(description="Process some Frostt tensors into per-level datastructures")
parser.add_argument('-n', '--name', metavar='fname', type=str, action='store',
                    help='tensor name to run format conversion on one frostt tensor')
parser.add_argument('-f', '--format', metavar='fformat', type=str, action='store',
                    help='The format that the tensor should be converted to')
parser.add_argument('-i', '--int', action='store_false', default=True, help='Safe sparsity cast to int for values')
parser.add_argument('-s', '--shift', action='store_false', default=True, help='Also format shifted tensor')
parser.add_argument('-o', '--other', action='store_true', default=False, help='Format other tensor')
parser.add_argument('-ss', '--suitesparse', action='store_true', default=False, help='Format suitesparse other tensor')
parser.add_argument('-hw', '--hw', action='store_true', default=False,
                    help='Format filenames as in AHA SCGRA <tensor_<name>_mode_<n|type>')
parser.add_argument('-np', '--numpy', action='store_true', default=False, help='Format numpy tensors')
parser.add_argument('-b', '--bench', type=str, default=None, help='Name of benchmark')

args = parser.parse_args()

if args.other:
    if args.suitesparse:
        outdir_name = os.getenv('SUITESPARSE_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))
    else:
        outdir_name = os.getenv('FROSTT_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))
    taco_format_dirname = os.getenv('TACO_TENSOR_PATH')
    if taco_format_dirname is None:
        print("Please set the TACO_TENSOR_PATH environment variable")
        exit()
    taco_format_dirname = os.path.join(taco_format_dirname, "other")
    # taco_format_dirname = os.path.join(taco_format_dirname, "other-formatted-taco")
else:
    outdir_name = os.getenv('FROSTT_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))
    taco_format_dirname = os.getenv('FROSTT_FORMATTED_TACO_PATH')
    if taco_format_dirname is None:
        print("Please set the FROSTT_FORMATTED_TACO_PATH environment variable")
        exit()

out_path = Path(outdir_name)
out_path.mkdir(parents=True, exist_ok=True)

print("args.name is ", args.name)

if args.name is None:
    print("Please enter a tensor name")
    exit()

print("\nhere after Please enter tensor name\n")

if args.format is not None:
    assert args.format in formats
    levels = args.format[:-3]
    if args.other:
        assert args.bench is not None

        print("here to get other file names\n")

        otherfileNames = [f for f in os.listdir(taco_format_dirname) if
                          os.path.isfile(os.path.join(taco_format_dirname, f)) and args.name in f]

        print("have otherfileNames\n")
        print(os.listdir(outdir_name))
        print("length of otherfilenames is: ", len(otherfileNames), "\n")

        for otherfile in otherfileNames:
            print("iterate thru otherfileNames\n")
            taco_format_orig_filename = os.path.join(taco_format_dirname, otherfile)
            # outdir_other_name = os.path.join(outdir_name, args.name, args.bench)
            outdir_other_name = os.path.join(outdir_name, args.name, 'other', otherfile[:-4])
            outdir_orig_path = Path(outdir_other_name)
            outdir_orig_path.mkdir(parents=True, exist_ok=True)

            name = None
            if args.bench == "mat_residual":
                if "mode0" in otherfile:
                    name = 'b'
                elif "mode1" in otherfile:
                    name = 'd'
                else:
                    raise NotImplementedError
            elif args.bench == "mat_mattransmul":
                if "mode0" in otherfile:
                    name = 'd'
                elif "mode1" in otherfile:
                    name = 'f'
                else:
                    raise NotImplementedError
            elif "mat_vecmul" in args.bench:
                if "mode1" in otherfile:
                    name = 'c'
                elif "mode0" in otherfile:
                    continue
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

            assert name is not None, "Other tensor name was not set properly and is None"
            parse_taco_format(taco_format_orig_filename, outdir_other_name, name, args.format, hw_filename=args.hw)

    else:
        print("in else statement\n")
        taco_format_orig_filename = os.path.join(taco_format_dirname, args.name + "_" + levels + '.txt')
        taco_format_shift_filename = os.path.join(taco_format_dirname, args.name + '_shift_' + levels + '.txt')

        # Original
        outdir_orig_name = os.path.join(outdir_name, args.name, 'orig', args.format)
        outdir_orig_path = Path(outdir_orig_name)
        outdir_orig_path.mkdir(parents=True, exist_ok=True)

        print("parse taco format\n")

        parse_taco_format(taco_format_orig_filename, outdir_orig_name, 'B', args.format, hw_filename=args.hw)

        # Shifted
        if args.shift:
            outdir_shift_name = os.path.join(outdir_name, args.name, 'shift', args.format)
            outdir_shift_path = Path(outdir_shift_name)
            outdir_shift_path.mkdir(parents=True, exist_ok=True)

            parse_taco_format(taco_format_shift_filename, outdir_shift_name, 'C', args.format, hw_filename=args.hw)
