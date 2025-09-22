#!/bin/bash

# ======================================================================
# sparse_tile_flow.sh
#   This script create tile files for the sparse datasets and put them 
#   into the docker container.
#   See: https://docs.google.com/document/d/1Y8Jj_tVUiuQ9uSFHLlzFVsL-GoyUUwTHjjBkRjYUHUI
# ======================================================================

# docker container path
docker_path=rupertlu-opal-codegen

# tilesize
tilesize=300

# setup env variables
export SPARSE_DATASETS_PATH=/nobackup/owhsu/sparse-datasets
export SUITESPARSE_PATH=${SPARSE_DATASETS_PATH}/suitesparse
export TACO_TENSOR_PATH=${SPARSE_DATASETS_PATH}/other
export FROSTT_PATH=${SPARSE_DATASETS_PATH}/frostt

export SAM_PATH=/nobackup/rupertlu/workspace/sam
export SUITESPARSE_FORMATTED_PATH=${SAM_PATH}/SUITESPARSE_FORMATTED
export FROSTT_FORMATTED_TACO_PATH=${SAM_PATH}/FROST_FORMATTED_TACO
export FROSTT_FORMATTED_PATH=${SAM_PATH}/FROST_FORMATTED

# sparse application lists
# ElemAdd3   : mat_elemadd3
# ElemMult   : mat_elemmul
# SpM*SpV    : mat_vecmul_ij
# SpM*SpM    : matmul_ijk / matmul_ikj
# TTV        : tensor3_ttv
# TTM        : tensor3_ttm
# InnerProd. : tensor3_innerprod
# MTTKRP     : tensor3_mttkrp
# app_names=()
# app_names+=("mat_elemadd3")
# app_names+=("mat_elemmul")
# app_names+=("mat_vecmul_ij")
# app_names+=("matmul_ijk")
# app_names+=("matmul_ikj")
# app_names+=("tensor3_ttv")
# app_names+=("tensor3_ttm")
# app_names+=("tensor3_innerprod")
# app_names+=("tensor3_mttkrp")

# sparse dataset list
# top-5 largest number of nnz in small50
# ch7-6-b1: 1260
# relat5  : 1058
# mk9-b1  :  756
# rel5    :  656
# n4c6-b1 :  420
# datasets=()
# datasets+=("ch7-6-b1")
# datasets+=("mk9-b1")
# datasets+=("n4c6-b1")
# datasets+=("rel5")
# datasets+=("relat5")
# datasets+=("cage3")
# datasets+=("bcsstm26") # for mat_elemadd3, mat_elemmul, mat_vecmul_ij, matmul_ijk, matmul_ikj
# datasets+=("rand_large_tensor5") # for tensor3_ttv, tensor3_ttm, tensor3_innerprod, tensor3_mttkrp

# Onyx VLSI
# tilesize=300
# app_names=("mat_elemadd3")
# datasets=("bcsstm26")


# ==============================================================================================
# Init
# ==============================================================================================

git submodule update --init
make taco/build


# ==============================================================================================
# SuiteSparse Dataset 
# ==============================================================================================

# Datasets, can selectively comment in/out
datasets=()
datasets+=("cage3")
datasets+=("bcsstm26")
datasets+=("qiulp")
datasets+=("tols2000")
datasets+=("west2021")
datasets+=("adder_dcop_30")
datasets+=("watt_2")
datasets+=("rajat12")
datasets+=("adder_trans_02")
datasets+=("G42")
datasets+=("G30")
datasets+=("N_biocarta")
datasets+=("bwm2000")
datasets+=("G33")
datasets+=("progas")
datasets+=("Ip_25fv47")

# Loop through the datasets
for dataset in "${datasets[@]}"; do

    case $dataset in
        "cage3")
            app_names=();                 tilesizes=()
            app_names+=("mat_elemadd3");  tilesizes+=("30")
            app_names+=("mat_elemmul");   tilesizes+=("30")
            app_names+=("mat_vecmul_ij"); tilesizes+=("30")
            app_names+=("matmul_ijk");    tilesizes+=("30")
            app_names+=("matmul_ikj");    tilesizes+=("30")
            ;;
        "bcsstm26")
            app_names=();                 tilesizes=()
            app_names+=("mat_elemadd3");  tilesizes+=("300")
            app_names+=("mat_elemmul");   tilesizes+=("760")
            app_names+=("mat_vecmul_ij"); tilesizes+=("760")
            app_names+=("matmul_ijk");    tilesizes+=("30")
            app_names+=("matmul_ikj");    tilesizes+=("30")
            ;;
        "qiulp")
            app_names=();                 tilesizes=()
            app_names+=("mat_elemadd3");  tilesizes+=("130")
            app_names+=("mat_elemmul");   tilesizes+=("497")
            app_names+=("mat_vecmul_ij"); tilesizes+=("497")
            app_names+=("matmul_ijk");    tilesizes+=("40")
            app_names+=("matmul_ikj");    tilesizes+=("40")
            ;;
        "tols2000")
            app_names=();                 tilesizes=()
            app_names+=("mat_elemadd3");  tilesizes+=("190")
            app_names+=("mat_elemmul");   tilesizes+=("360")
            app_names+=("mat_vecmul_ij"); tilesizes+=("512")
            app_names+=("matmul_ijk");    tilesizes+=("30")
            app_names+=("matmul_ikj");    tilesizes+=("30")
            ;;
        "west2021")
            app_names=();                 tilesizes=()
            app_names+=("mat_elemadd3");  tilesizes+=("130")
            app_names+=("mat_elemmul");   tilesizes+=("429")
            app_names+=("mat_vecmul_ij"); tilesizes+=("429")
            app_names+=("matmul_ijk");    tilesizes+=("60")
            app_names+=("matmul_ikj");    tilesizes+=("60")
            ;;
        "adder_dcop_30")
            app_names=();                 tilesizes=()
            app_names+=("mat_elemadd3");  tilesizes+=("120")
            app_names+=("mat_elemmul");   tilesizes+=("300")
            app_names+=("mat_vecmul_ij"); tilesizes+=("300")
            ;;
        "watt_2")
            app_names=();                 tilesizes=()
            app_names+=("mat_elemadd3");  tilesizes+=("80")
            app_names+=("mat_elemmul");   tilesizes+=("150")
            app_names+=("mat_vecmul_ij"); tilesizes+=("175")
            app_names+=("matmul_ijk");    tilesizes+=("30")
            app_names+=("matmul_ikj");    tilesizes+=("30")
            ;;
        "rajat12")
            app_names=();                 tilesizes=()
            app_names+=("mat_elemadd3");  tilesizes+=("80")
            app_names+=("mat_elemmul");   tilesizes+=("175")
            app_names+=("mat_vecmul_ij"); tilesizes+=("150")
            ;;
        "adder_trans_02")
            app_names=();                 tilesizes=()
            app_names+=("mat_elemadd3");  tilesizes+=("120")
            app_names+=("mat_elemmul");   tilesizes+=("250")
            app_names+=("mat_vecmul_ij"); tilesizes+=("250")
            ;;
        "G42")
            app_names=();                 tilesizes=()
            app_names+=("mat_elemadd3");  tilesizes+=("40")
            app_names+=("mat_elemmul");   tilesizes+=("75")
            app_names+=("mat_vecmul_ij"); tilesizes+=("75")
            ;;
        "G30")
            app_names=();                 tilesizes=()
            app_names+=("mat_elemadd3");  tilesizes+=("160")
            app_names+=("mat_elemmul");   tilesizes+=("300")
            app_names+=("mat_vecmul_ij"); tilesizes+=("270")
            ;;
        "N_biocarta")
            app_names=();                 tilesizes=()
            app_names+=("matmul_ijk");    tilesizes+=("60")
            app_names+=("matmul_ikj");    tilesizes+=("60")
            ;;
        "bwm2000")
            app_names=();                 tilesizes=()
            app_names+=("matmul_ijk");    tilesizes+=("60")
            app_names+=("matmul_ikj");    tilesizes+=("60")
            ;;
        "G33")
            app_names=();                 tilesizes=()
            app_names+=("matmul_ijk");    tilesizes+=("30")
            app_names+=("matmul_ikj");    tilesizes+=("30")
            ;;
        "progas")
            app_names=();                 tilesizes=()
            app_names+=("matmul_ijk");    tilesizes+=("40")
            app_names+=("matmul_ikj");    tilesizes+=("40")
            ;;
        "Ip_25fv47")
            app_names=();                 tilesizes=()
            app_names+=("matmul_ijk");    tilesizes+=("50")
            app_names+=("matmul_ikj");    tilesizes+=("50")
            ;;
        *)
            echo "Invalid dataset: $dataset"
            exit 1
            ;;
    esac

    # Get the length of the arrays
    length=${#app_names[@]}

    # Loop through the arrays
    for ((i=0; i<$length; i++)); do

        app_name=${app_names[$i]}
        tilesize=${tilesizes[$i]}

        python setup_tiling_mat.py \
            ${app_name} \
            ${dataset} \
            ${tilesize} \
            ${docker_path} |& tee run_vlsi_${app_name}_${dataset}.log

    done

done

# ==============================================================================================
# Randomized Tensor Dataset
# ==============================================================================================

# Datasets, can selectively comment in/out
datasets=()
datasets+=("rand_large_tensor5")
datasets+=("rand_large_tensor7") 
datasets+=("rand_large_tensor9")
datasets+=("rand_large_tensor6")
datasets+=("rand_large_tensor1")
datasets+=("rand_large_tensor3")
datasets+=("rand_large_tensor2")
datasets+=("rand_large_tensor8")
datasets+=("rand_large_tensor10")
datasets+=("rand_large_tensor4")

RUN_SCRIPT="./scripts/formatting/generate_tensor_tile_formats_onyx.sh"
MEM_CONFIG="./sam/sim/src/tiling/memory_config_opal.yaml"
TMP_TENSOR="tensor_names.txt"
# tilesize=10 : this information is coded in the Mem_tile_size field in $MEM_CONFIG

# store the app names and other formats in associate arrays
app_names=();                     other_formats=()
app_names+=("tensor3_ttv");       other_formats+=("s")
app_names+=("tensor3_ttm");       other_formats+=("ss")
app_names+=("tensor3_innerprod"); other_formats+=("sss")
app_names+=("tensor3_mttkrp");    other_formats+=("ss")

# Get the length of the arrays
length=${#app_names[@]}

# Loop through the datasets
for dataset in "${datasets[@]}"; do

    # Loop through the arrays
    for ((i=0; i<$length; i++)); do

        app_name=${app_names[$i]}
        other_format=${other_formats[$i]}

        rm $TMP_TENSOR
        echo $dataset > $TMP_TENSOR

        "$RUN_SCRIPT" \
            ${TMP_TENSOR} \
            ${MEM_CONFIG} \
            ${app_name} \
            ${other_format} \
            ${docker_path} |& tee run_vlsi_${app_name}_${dataset}.log

    done

done

# ==============================================================================================
# Error Logs for tiled matrices and tensors (sam branch: as is in container (7d5cc49))
# ==============================================================================================
# for apps involving vectors, go hack the c_shape.txt and delete the 1 dimension
# fix the name matching bug by grabbing the latest sam/sam/onyx/generate_matrices.py

# dataset   : bcsstm26
# app_name  : mat_elemadd3
# tilesize  : 300

# PASSED

# Note:
#   - glb output exceed the default 1024 words
#   - make it 2048 in garnet/tests/test_memory_core/build_tb.py:prepare_glb_collateral()

# ---------------------------------------------------------------------------------------------------

# dataset   : bcsstm26
# app_name  : mat_elemmul
# tilesize  : 760

# PASSED

# ---------------------------------------------------------------------------------------------------

# dataset   : bcsstm26
# app_name  : mat_vecmul_ij
# tilesize  : 760

# PASSED

# Note:
#   - hack the c_shape.txt and delete the 1 dimension
#   - get the latest name matching bug fix in sam/sam/onyx/generate_matrices.py

# ---------------------------------------------------------------------------------------------------

# dataset   : bcsstm26
# app_name  : matmul_ijk / matmul_ikj
# tilesize  : 30

# PASSED

# ---------------------------------------------------------------------------------------------------

# dataset   : rand_large_tensor5
# app_name  : tensor3_ttv

# PASSED

# Note:
#   - update this piece of codes: https://github.com/StanfordAHA/aha/blob/master/aha/util/regress.py#L441-L458
#   - update the copy_formatted_tensor_tiling.py to the latest version
#   - make sure you run `aha regress` in the /aha/garnet folder (copy_formatted_tensor_tiling.py assumes it)

# ---------------------------------------------------------------------------------------------------

