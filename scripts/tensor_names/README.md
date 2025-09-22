# Tensor Names

the `scripts/tensor_names` folder contains text files with lists of SuiteSparse matrix
names that are used for different scenarios

The scripts in this folder are:
1. `suitesparse.txt` - All (ANY type) suitesparse matrices (alphabetical order)
2. `suitesparse_ci.txt` - All suitesparse matrices that are in the `sam/data/` folder in the SAM repo for CI purposes
3. `suitesparse_real.txt` - All real suitesparse matrices (alphabetical order)
4. `suitesparse_valid.txt` - All REAL and INTEGER suitesparse matrices that fit
   in memory on LANKA (MIT) for at least ONE test from the original SAM paper
   passed (unordered) 
5. `suitesparse_valid_all.txt` - All real and integer suitesparse matrices that
   fit in memory on LANKA (MIT) for ALL tests from the original SAM paper
   (ordered by dense dimension) 
6. `suitesparse_valid_large50.txt` - The largest 50 (by dense dimension) suitesparse matrices that passed ALL tests from the original SAM paper
7. `suitesparse_valid_mid50.txt` - The median 50 (by dense dimension) suitesparse matrices that passed ALL tests from the original SAM paper
8. `suitesparse_valid_small50.txt` - The smallest 50 (by dense dimension) suitesparse matrices that passed ALL tests from the original SAM paper
9. `temp_*.txt` - `suitesparse_valid.txt` split into various files for running tests in parallel
