import numpy
from sam.onyx.generate_matrices import *
import argparse
import random


def generate_runs_vectors(vec_shape, run_length, nnz):

    vec_size = 2000

    vec1 = numpy.zeros(vec_size)
    vec2 = numpy.zeros(vec_size)

    # First create one item in first 5% of the second vector
    # inject_idx = random.randint(0, int(0.05 * vec_size))
    inject_idx = 0
    vec2[inject_idx] = random.randint(0, 1000)

    inject_idx = inject_idx + 1

    done = False
    nnz_1 = 0
    nnz_2 = 0

    while not done:

        for run_idx in range(run_length):
            if nnz_1 < nnz:
                # break
                vec1[inject_idx + run_idx] = random.randint(0, 1000)
                nnz_1 += 1

        # Now create a run on the other side
        # inject_idx = inject_idx + run_idx + 1

        # Now put a number in both
        inject_idx = inject_idx + run_idx + 1
        vec1[inject_idx] = random.randint(0, 1000)
        vec2[inject_idx] = random.randint(0, 1000)
        nnz_1 += 1
        nnz_2 += 1

        inject_idx += 1

        for run_idx in range(run_length):
            if nnz_2 < nnz:
                # break
                vec2[inject_idx + run_idx] = random.randint(0, 1000)
                nnz_2 += 1

        inject_idx = inject_idx + run_idx + 1
        vec1[inject_idx] = random.randint(0, 1000)
        vec2[inject_idx] = random.randint(0, 1000)
        nnz_1 += 1
        nnz_2 += 1

        # inject_idx = inject_idx + run_idx + 1
        inject_idx += 1

        # nnz_1 += run_length
        # nnz_2 += run_length

        if nnz_1 >= nnz and nnz_2 >= nnz:
            done = True

    print(f"{nnz_1}, {nnz_2}")

    vec1[inject_idx] = random.randint(0, 1000)

    return vec1, vec2


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate matrices - random')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--number_nonzeros", type=int, default=1000)
    # parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default=None)
    # parser.add_argument("--run_lengths", type=int, nargs=2, default=[10, 10])
    parser.add_argument("--run_length", type=int, default=10)
    parser.add_argument("--shape", type=int, nargs="*", default=[2000])
    parser.add_argument("--output_format", type=str, default='CSF')

    # parser.add_argument("--shape", type=int, nargs="*", default=[10, 10])
    # parser.add_argument("--num_trials", type=int, default=1000)
    # parser.add_argument("--output_csv", type=str, default="runs.csv")
    args = parser.parse_args()

    seed = args.seed
    nnz = args.number_nonzeros
    # sparsity = args.sparsity
    output_dir = args.output_dir
    # name = args.name
    # shape = args.shape
    # run_lengths = args.run_lengths
    run_length = args.run_length
    output_format = args.output_format
    shape = args.shape

    random.seed(seed)
    numpy.random.seed(seed)

    # for random_mat in range(number):

    v1, v2 = generate_runs_vectors(shape, run_length, nnz)

    tmp_v1 = MatrixGenerator(name=f'B', format='CSF',
                             dump_dir=f"{output_dir}/runs_rl_{run_length}_nnz_{nnz}",
                             # dump_dir=f"{output_dir}/runs_{random_mat}_{run_lengths[0]}_{run_lengths[1]}",
                             tensor=v1)

    tmp_v2 = MatrixGenerator(name=f'C', format='CSF',
                             dump_dir=f"{output_dir}/runs_rl_{run_length}_nnz_{nnz}",
                             # dump_dir=f"{output_dir}/runs_{random_mat}_{run_lengths[0]}_{run_lengths[1]}",
                             tensor=v2)

    print(tmp_v1)
    print(tmp_v2)

    tmp_v1.dump_outputs(format=output_format)
    tmp_v2.dump_outputs(format=output_format)
