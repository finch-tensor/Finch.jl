#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive

# Script steps
# 1. Formats data
# 2. Runs SAM sim in Pytest
# 3. Converts data to CSV
# 4. Aggregates CSV

# ./scripts/run_sam_sim/sam_frostt_runner.sh <tensor_names.txt>

set -u

BENCHMARKS=(
  tensor3_innerprod_FINAL
  tensor3_elemadd_FINAL
  tensor3_ttv_FINAL
  tensor3_ttm_FINAL
  tensor3_mttkrp_FINAL
)

errors=()
RED='\033[0;31m'
NC='\033[0m' # No Color

format_outdir=${FROSTT_FORMATTED_PATH} 
basedir=$(pwd)
frosttpath=$FROSTT_PATH
benchout=frostt-bench/sam

mkdir -p "$benchout"
mkdir -p $format_outdir
mkdir -p $TACO_TENSOR_PATH/other-formatted-taco

make -j8 taco/build NEVA=$neva LANKA=$lanka GEN=ON

for b in ${!BENCHMARKS[@]}; do
	bench=${BENCHMARKS[$b]}
	path=$basedir/$benchout/$bench
	mkdir -p $basedir/$benchout/$bench
	echo "Testing $bench..."

	while read line; do
		name=$line
		cd $format_outdir


		if [ "$bench" == "tensor3_innerprod_FINAL" ]; then
			echo "Generating input format files for $name..."

			$basedir/compiler/taco/build/bin/taco-test sam.pack_sss012
			$basedir/compiler/taco/build/bin/taco-test sam.pack_other_frostt
			python $basedir/scripts/formatting/datastructure_tns.py -n $name -f sss012
			python $basedir/scripts/formatting/datastructure_tns.py -n $name -f sss012 --other
			chmod -R 775 $FROSTT_FORMATTED_PATH
		fi

		cd $basedir/sam/sim

		pytest test/final-apps/test_$bench.py --frosttname $name --benchmark-json=$path/$name.json 
		python $basedir/scripts/util/converter.py --json_name $path/$name.json	
		    
		status=$?
		if [ $status -gt 0 ]
		then 
		  errors+=("${name}, ${bench}")
		fi

		cd $basedir
	done <$1 

	python $basedir/scripts/util/bench_csv_aggregator.py $path $basedir/$benchout/frostt_$bench.csv

	echo -e "${RED}Failed tests:"
	for i in ${!errors[@]}; do
	    error=${errors[$i]} 
	    echo -e "${RED}$error,"
	done
	echo -e "${NC}"
done

