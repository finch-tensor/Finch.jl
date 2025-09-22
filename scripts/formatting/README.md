# Formatting

The `scripts/formatting/` folder contains scripts used to format the datasets from mtx/tns files to the seg/crd/vals arrays for CSF. The CSF files are expected by both the Onyx AHA flow and the SAM simulator.

1. `datastructure_suitesparse.py` - Python script used by
   `generate_suitesparse_formats.sh` to format from mtx to CSF files. 
2. `datastructure_tns.py` - Python script used by
   `generate_frostt_formats.sh` to format from tns to CSF files. 
3. `download_unpack_format_suitesparse.sh` - Script that downloads, unpacks,
   and formats a list of suitesparse matrices. To download and unpack it
   calls scripts in `scripts/get_data`.
4. `generate_frostt_formats.sh` - Formats all FROSTT datasets. FIXME: This file needs fixing as it uses the old CSF formatting (e.g. `matrix_name/B_seg0`) instead of the new one (e.g. `app/tensor_B_mode_0_seg`)  
5. `generate_suitesparse_formats.sh` - Formats all SuiteSparse datasets 

Formatted CSF files should reside in `$SUITESPARSE_FORMATTED_PATH` for SuiteSparse matrices. 
