#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive

# ./scripts/run_sam_sim/sam_suitesparse_runner.sh <tensor_names.txt> 

set -u

BENCHMARKS=(
  mat_vecmul_FINAL
  matmul_FINAL
  mat_elemadd_FINAL
  mat_elemadd3_FINAL
  mat_residual_FINAL
  mat_mattransmul_FINAL
)

errors=()
RED='\033[0;31m'
NC='\033[0m' # No Color

format_outdir=${SUITESPARSE_FORMATTED_PATH} 
basedir=$(pwd)
sspath=$SUITESPARSE_PATH
benchout=suitesparse-bench/sam

mkdir -p "$benchout"
mkdir -p $format_outdir
mkdir -p $TACO_TENSOR_PATH/other-formatted-taco

# make -j8 taco/build NEVA=$neva LANKA=$lanka GEN=ON
make -j8 taco/build GEN=ON

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

		if [ "$bench" == "mat_vecmul_FINAL" ]; then
			echo "Generating input format files for $line..."
			SUITESPARSE_TENSOR_PATH=$matrix python $basedir/scripts/formatting/datastructure_suitesparse.py -n $line 

			SUITESPARSE_TENSOR_PATH=$matrix $basedir/compiler/taco/build/bin/taco-test sam.pack_other_ss    
			python $basedir/scripts/formatting/datastructure_tns.py -n $line -f ss01 --other -ss
		fi

		cd $basedir/sam/sim

		pytest test/final-apps/test_$bench.py --ssname $line -s --report-stats --benchmark-json=$path/$line.json 
		python $basedir/scripts/util/converter.py --json_name $path/$line.json	
		    
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

