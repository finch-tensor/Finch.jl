import random
import argparse


def get_matrix_indices(seed, n=5):
    random.seed(seed)
    matrices = []
    with open('suitesparse_valid_small50.txt', 'r') as small50:
        randlist = random.sample(range(50), n)
        print(randlist)
        lines = small50.read().splitlines()
        chosen_matrices = [lines[i] for i in randlist]
        matrices += chosen_matrices
    with open('suitesparse_valid_mid50.txt', 'r') as mid50:
        randlist = random.sample(range(50), n)
        print(randlist)
        lines = mid50.read().splitlines()
        chosen_matrices = [lines[i] for i in randlist]
        matrices += chosen_matrices
    with open('suitesparse_valid_large50.txt', 'r') as large50:
        randlist = random.sample(range(50), n)
        print(randlist)
        lines = large50.read().splitlines()
        chosen_matrices = [lines[i] for i in randlist]
        matrices += chosen_matrices
    return matrices


def write_temp(matrices, out_path):
    with open(out_path, "w+") as ff:
        ff.write('\n'.join(matrices))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get benchmark matrices")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num', type=int, default=5)
    parser.add_argument('--out_path', type=str, default="suitesparse_benchmarks.txt")
    args = parser.parse_args()

    matrices = get_matrix_indices(args.seed, args.num)
    write_temp(matrices, args.out_path)
