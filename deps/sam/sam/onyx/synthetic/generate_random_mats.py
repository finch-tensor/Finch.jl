import numbers
from sam.onyx.generate_matrices import *
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate matrices - random')
    parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--transpose", type=bool, default=False)
    parser.add_argument("--transpose", action="store_true")
    # parser.add_argument("--number", type=int, default=1)
    parser.add_argument("--sparsity", type=float, nargs="*", default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--name", type=str, default='B')
    parser.add_argument("--shape", type=int, nargs="*", default=[10, 10])
    parser.add_argument("--output_format", type=str, default='CSF')
    # parser.add_argument("--num_trials", type=int, default=1000)
    # parser.add_argument("--output_csv", type=str, default="runs.csv")
    args = parser.parse_args()

    seed = args.seed
    sparsity = args.sparsity
    output_dir = args.output_dir
    name = args.name
    shape = args.shape
    # number = args.number
    output_format = args.output_format
    transpose = args.transpose

    # densities = [.00625, .0125, .025, .05, .1, .2, .4, .80]

    if sparsity is None:
        sparsities = [.99375, .9875, .975, .95, .9, .8, .6, .2]
    else:
        sparsities = sparsity

    numpy.random.seed(seed)
    random.seed(seed)

    # for random_mat in range(number):
    for idx, sparsity in enumerate(sparsities):
        print(sparsity)
        tmp_mat = MatrixGenerator(name=f'{name}', shape=shape, sparsity=sparsity,
                                  format='CSF', dump_dir=f"{output_dir}/{name}_random_sp_{sparsity}", tensor=None)
        # print(tmp_mat)
        tmp_mat.dump_outputs(format=output_format, tpose=transpose)
