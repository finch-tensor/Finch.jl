#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive

# ./scripts/tiling/prepare_files.sh extensor_<NNZ>_<DIM>.mtx

appname=$3
testname=$4

basedir=$(pwd)

rm -rf $basedir/tiles/*

./scripts/tiling/tile_ext.sh $1 memory_config_extensor_17M_llb.yaml $appname $testname

# python3 scripts/tiling/generate_gold_matmul_tiled.py --yaml_name memory_config_extensor_17M_llb.yaml
