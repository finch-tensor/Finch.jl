#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive

# ./scripts/tiling/full_ext_runner.sh

NNZ=(
  5000
  10000
  25000
  50000
)

DIMENSIONS=(
 1024
 2360
 3028
 5032
 7704
 9040
 11712
 15720
)


for nnz in ${!NNZ[@]}; do
	for dim in ${!DIMENSIONS[@]}; do
		filename=${NNZ[$nnz]}_${DIMENSIONS[$dim]}
		./scripts/tiling/ext_runner.sh extensor_${NNZ[$nnz]}_${DIMENSIONS[$dim]}.mtx
	done
done 
