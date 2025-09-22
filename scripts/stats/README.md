# Statistics Scripts

The `scripts/stats/` folder contains scripts used to get general sparse statistics about the datasets. This is useful for the designs

1. `get_tensor_arrlen.py` - Script that gets the length of the datastructure arrays from the input datasets (to populate CSVs).
2. `suitesparse_stats.sh` - Script that calls `suitesparse_states.py` 
3. `suitesparse_stats.py` - File that calcultes certain statistics (e.g size,
   len, nnz) of the SuiteSparse data structure arrays (e.g. seg/crd)
