#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive

# ./scripts/tiling/advanced_simulator_runner.sh <tensor_names.txt> <machine>
# where machine is either 0(local), 1(Lanka), or 2(Kiwi/Neva/Lagos) 

set -u

BENCHMARKS=(
#  mat_vecmul_FINAL
#  matmul_FINAL
#  mat_identity
#  mat_identity_back
#  matmul_ikj_memory_back
#   matmul_ikj_sparse_tiling2
#  matmul_ikj_glb_tile
#  matmul_ikj_glb_tile2
matmul_ikj_tile_pipeline_final
#  matmul_ikj_glb_tile_pipeline
#  matmul_ikj_glb_no_pipe
#  matmul_ikj_input_only
#  matmul_ikj_tiled_bcsstm02
#  matmul_ikj_check
#  matmul_ikj_tiling
#  matmul_ikj_back
#  mat_elemmul_FINAL
#  mat_elemadd_FINAL
#  mat_elemadd3_FINAL
#  mat_residual_FINAL
#  mat_mattransmul_FINAL
)

errors=()
RED='\033[0;31m'
NC='\033[0m' # No Color

basedir=$(pwd)

sspath=$SUITESPARSE_PATH
benchout=suitesparse-bench_simulator/sam
format_outdir=${SUITESPARSE_FORMATTED_PATH} 

source $basedir/../venv/bin/activate

mkdir -p "$benchout"
mkdir -p $format_outdir
mkdir -p $TACO_TENSOR_PATH/other-formatted-taco

for b in ${!BENCHMARKS[@]}; do
	bench=${BENCHMARKS[$b]}
	path=$basedir/$benchout/$bench
	mkdir -p $basedir/$benchout/$bench
	echo "Testing $bench..."

	while read line; do
		cd $format_outdir

		if [ $2 -eq 1 ]; then
			matrix="$sspath/$line/$line.mtx"
		elif [ $2 -eq 2 ]; then
			matrix="$sspath/$line.mtx"
		else
			matrix="$sspath/$line.mtx"
		fi

		if [ "$bench" == "matmul_ikj" ]; then
			echo "Generating input format files for $line..."
			SUITESPARSE_TENSOR_PATH=$matrix python $basedir/scripts/formatting/datastructure_suitesparse.py -n $line 

			SUITESPARSE_TENSOR_PATH=$matrix $basedir/compiler/taco/build/bin/taco-test sam.pack_other_ss    
			python $basedir/scripts/formatting/datastructure_frostt.py -n $line -f ss01 --other -ss
		fi

		cd $basedir/sam/sim
		#python -m cProfile -o test/final-apps/test_$bench.py --ssname $line -s --benchmark-json=$path/$line.json 
		pytest test/advanced-simulator/test_$bench.py --ssname $line -s  --report-stats --check-gold --skip-empty --nbuffer --yaml_name=$3 --benchmark-json=$path/$line.json 
		# pytest test/advanced-simulator/test_$bench.py --ssname $line -s --report-stats --back --depth=1 --debug-sim --check-gold --benchmark-json=$path/$line.json
		# python $basedir/scripts/util/converter.py --json_name $path/$line.json	
		    
		status=$?
		if [ $status -gt 0 ]
		then 
		  errors+=("${line}, ${bench}")
		fi

		cd $basedir
	done <$1

	python $basedir/scripts/util/bench_csv_aggregator.py $path $basedir/$benchout/suitesparse_$bench.csv

	echo -e "${RED}Failed tests:"
	for i in ${!errors[@]}; do
	    error=${errors[$i]} 
	    echo -e "${RED}$error,"
	done
	echo -e "${NC}"
done
