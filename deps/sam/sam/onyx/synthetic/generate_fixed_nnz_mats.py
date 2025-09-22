import scipy.io
import scipy.sparse
import numpy as np
import argparse


def generate_mat(nnz, dim):
    return scipy.sparse.random(dim, dim, nnz / (dim**2), data_rvs=np.ones)


def write_mtx(path, t):
    scipy.io.mmwrite(path, t)


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)

    parser = argparse.ArgumentParser(description="Create some random matrices of given nnz and dim")
    parser.add_argument('--nnz', type=int, nargs='+', help='nnz')
    parser.add_argument('--dim', type=int, nargs='+', help='dim')
    parser.add_argument('--extensor', action='store_true', help='generate extensor dims and nnzs')
    args = parser.parse_args()

    if args.extensor:
        dims = list(range(1024, 15721, 1336))
        nnzs = [5000, 10000, 25000, 50000]
    else:
        dims = args.dim
        nnzs = args.nnz
    print("RUNNING:", dims, nnzs)

    for nnz in nnzs:
        for dim in dims:
            print("nnz:", nnz, "dim", dim)
            tensor = generate_mat(nnz, dim)
            write_mtx("extensor_" + str(nnz) + "_" + str(dim) + ".mtx", tensor)
