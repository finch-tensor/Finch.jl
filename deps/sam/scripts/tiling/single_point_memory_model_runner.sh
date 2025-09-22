#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive

# ./scripts/tiling/single_point_memory_model_runner.sh extensor_<NNZ>_<DIMSIZE>.mtx

benchout=memory_model_out

basedir=$(pwd)
bench=matmul_ikj_tile_pipeline_final
yaml_fname=memory_config_extensor_17M_llb.yaml
line=random_sparsity
path=$basedir/$benchout

fname=$1
temp_fname=${fname#*_}
temp_fname=${temp_fname%%.*}
nnz=${temp_fname%_*}
dim=${temp_fname#*_}

echo "Running for point NNZ=$nnz and DIMSIZE=$dim"

export SAM_HOME=$basedir
export TILED_SUITESPARSE_FORMATTED_PATH=${SAM_HOME}/tiles/matmul_ikj/formatted
export TILED_OUTPUT_PATH=${SAM_HOME}/tiles/matmul_ikj/output/

pushd .

mkdir -p $path

mkdir -p $basedir/tiles/
rm -rf $basedir/tiles/*

./scripts/tiling/prepare_files.sh $fname 

cd $basedir/sam/sim
pytest test/advanced-simulator/test_$bench.py --ssname $line -s --check-gold --skip-empty --nbuffer --yaml_name=$yaml_fname --memory-model --nnz-value=$nnz --benchmark-json=$path/${line}_${nnz}_${dim}.json 

python $basedir/scripts/util/converter.py --json_name $path/${line}_${nnz}_${dim}.json	

python3 $basedir/scripts/util/bench_csv_aggregator.py $path $basedir/$benchout/$bench.csv

popd
