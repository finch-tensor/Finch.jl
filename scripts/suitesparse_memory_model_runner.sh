#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive

benchout=memory_model_out

basedir=$(pwd)
# bench=matmul_ijk_tile_pipeline_final
yaml_fname=memory_config_onyx.yaml
path=$basedir/$benchout

fname=$1

appname=$2

echo "Running for suitesparse $fname"

export SAM_HOME=$basedir
# export TILED_SUITESPARSE_FORMATTED_PATH=${SAM_HOME}/tiles/matmul_ijk/formatted
# export TILED_OUTPUT_PATH=${SAM_HOME}/tiles/matmul_ijk/output/

export TILED_SUITESPARSE_FORMATTED_PATH=${SAM_HOME}/tiles/${appname}/formatted
export TILED_OUTPUT_PATH=${SAM_HOME}/tiles/${appname}/output/

pushd .

mkdir -p $path

mkdir -p $basedir/tiles/
rm -rf $basedir/tiles/*

./scripts/tiling/prepare_files.sh $fname.mtx $yaml_fname $fname $appname

cd $basedir/sam/sim
# python3 -m pytest test/advanced-simulator/test_$bench.py --ssname $fname -s --check-gold --skip-empty --nbuffer --yaml_name=$yaml_fname  --benchmark-json=$path/mem_model_$fname.json 
# pytest test/advanced-simulator/test_$bench.py --ssname $fname -s --check-gold --skip-empty --nbuffer --yaml_name=$yaml_fname  --benchmark-json=$path/mem_model_$fname.json 

# python3 $basedir/scripts/converter.py --json_name $path/mem_model_$fname.json	

python3 $basedir/scripts/util/bench_csv_aggregator.py $path $basedir/$benchout/$bench.csv

popd
