#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360

# ./scripts/run_sam_sim/pytest_frostt_with_benchmarks.sh <tensor_names.txt>

BENCHMARKS=(
  tensor3_elemmul
  tensor3_identity
  tensor3_elemadd
  tensor3_ttm
  tensor3_ttv
  tensor3_innerprod
  tensor_mttkrp
)

outdir=$FROSTT_FORMATTED_PATH

errors=()
RED='\033[0;31m'
NC='\033[0m' # No Color


basedir=$(pwd)
resultdir=results


cd ./sam/sim

for b in ${!BENCHMARKS[@]}; do
    bench=${BENCHMARKS[$b]}
    path=$basedir/$resultdir/$bench

    mkdir -p $basedir/$resultdir/$bench
    echo "Testing $bench..."

    while read line; do
        name=$line 

        echo "Testing $name..."

        pytest test/apps/test_$bench.py --ssname $name -s --benchmark-json=$path/$name.json 
        python $basedir/scripts/util/converter.py --json_name $path/$name.json	
            
        status=$?
        if [ $status -gt 0 ]
        then 
          errors+=("${name}, ${bench}")
        fi
    done <$1
    
    python $basedir/scripts/util/bench_csv_aggregator.py $path $basedir/suitesparse_$bench.csv

done

echo -e "${RED}Failed tests:"
for i in ${!errors[@]}; do
    error=${errors[$i]} 
    echo -e "${RED}$error,"
done
echo -e "${NC}"
