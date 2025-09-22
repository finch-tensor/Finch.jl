#!/bin/bash
#sbatch -n 1
#sbatch --mem 120000
#sbatch -p lanka-v3
#sbatch --exclusive


basedir=$(pwd)
yaml_fname=memory_config_onyx.yaml
line=random_sparsity

nnz=$1
dim=$2
echo "running for point nnz=$nnz and dimsize=$dim"

export sam_home=$basedir
export tiled_suitesparse_formatted_path=${sam_home}/tiles/matmul_ikj/formatted
export tiled_output_path=${sam_home}/tiles/matmul_ikj/output/

pushd .

mkdir extensor_mtx
cd extensor_mtx
python ../sam/onyx/synthetic/generate_fixed_nnz_mats.py --nnz $nnz --dim $dim
cd ..

mkdir -p $path

mkdir -p $basedir/tiles/
rm -rf $basedir/tiles/*

./scripts/prepare_files.sh extensor_${nnz}_${dim}.mtx $yaml_fname 

