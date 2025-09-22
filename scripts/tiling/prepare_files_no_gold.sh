#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive

# ./scripts/tiling/prepare_files_no_gold.sh extensor_<NNZ>_<DIM>.mtx

basedir=$(pwd)
rm -rf $basedir/tiles/*

$basedir/scripts/tiling/tile_ext.sh $1 memory_config_extensor_17M_llb.yaml
