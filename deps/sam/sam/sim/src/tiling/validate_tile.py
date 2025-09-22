import scipy.sparse
import scipy.io
import argparse
import os

from sam.util import ScipyTensorShifter


def validate_simple(args):
    cwd = os.getcwd()

    SS_PATH = os.getenv('SUITESPARSE_PATH', default=os.path.join(cwd, 'suitesparse'))
    print("PATH:", SS_PATH)
    tensor_path = os.path.join(SS_PATH, args.input_tensor + ".mtx")
    tensor = scipy.io.mmread(tensor_path).toarray()

    tensor_shifted = ScipyTensorShifter().shiftLastMode(tensor).toarray()
    tensor_shifted = tensor_shifted.transpose()

    for i0 in range(4):
        for k0 in range(4):
            for j0 in range(4):
                for i00 in range(2):
                    for k00 in range(2):
                        for j00 in range(2):
                            B_tile_id = [i0, k0, i00, k00]
                            #                            C_tile_id = [j0, k0, j00, k00]
                            C_tile_id = [k0, j0, k00, j00]
                            B_tile_id = [str(item) for item in B_tile_id]
                            C_tile_id = [str(item) for item in C_tile_id]

                            B_tile_path = os.path.join(args.input_dir_path,
                                                       "tensor_B_tile_" + "_".join(B_tile_id) + ".mtx")
                            C_tile_path = os.path.join(args.input_dir_path,
                                                       "tensor_C_tile_" + "_".join(C_tile_id) + ".mtx")

                            if os.path.exists(B_tile_path):
                                B_tile = scipy.io.mmread(B_tile_path)

                                B0 = 16 * i0 + 8 * i00
                                B1 = 16 * k0 + 8 * k00

                                B_tile_og = tensor[B0:B0 + 8, B1:B1 + 8]

                                assert (B_tile.toarray() == B_tile_og).all(), ",".join(B_tile_id) + "\n" + str(
                                    B_tile.toarray()) + "\n" + str(B_tile_og)

                            if os.path.exists(C_tile_path):
                                C_tile = scipy.io.mmread(C_tile_path)

                                C0 = 16 * k0 + 8 * k00
                                C1 = 16 * j0 + 8 * j00

                                C_tile_og = tensor_shifted[C0:C0 + 8, C1:C1 + 8]
                                assert (C_tile.toarray() == C_tile_og).all(), ",".join(C_tile_id) + "\n" + str(
                                    C_tile.toarray()) + "\n" + str(C_tile_og)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate tiled matrices')
    parser.add_argument("--input_tensor", type=str, default=None)
    parser.add_argument("--input_dir_path", type=str, default=None)
    args = parser.parse_args()

    validate_simple(args)
