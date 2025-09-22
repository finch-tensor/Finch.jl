# Get Data

The `scripts/get_data` folder contains scripts used to download and unpack
datasets (SuiteSparse matrices and FROSTT tensors) 

1. `download_frostt.sh` - Download and unpack FROSTT tns files into
   `$FROSTT_PATH`  
2. `download_suitesparse.sh` - Download SuiteSparse mtx files into
   `$SUITESPARSE_PATH` 
3. `unpack_suitesparse.sh` - Unpack SuiteSparse mtx files in `$SUITESPARSE_PATH` based on a <tensor_names.txt> file 
4. `unpack_suitesparse_all.sh` - Unpack SuiteSparse mtx files in `$SUITESPARSE_PATH` for all `*.tar.gz` files that exist
