#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360

# ./scripts/run_sam_sim/pytest_suitesparse_with_benchmarks.sh <tensor_names.txt>

BENCHMARKS=(
  matmul_kij
  matmul_kji
  matmul_ikj
  matmul_jki
  matmul_ijk
  matmul_jik
  mat_elemmul
  mat_identity
#  vecmul_ij
#  vecmul_ji
#  vec_elemmul
#  vec_identity
#  vec_elemadd
#  vec_scalar_mul
#  tensor3_elemmul
#  tensor3_identity
)


errors=()
RED='\033[0;31m'
NC='\033[0m' # No Color


cwd=$(pwd)
resultdir=results


cd ./sam/sim

for b in ${!BENCHMARKS[@]}; do
    bench=${BENCHMARKS[$b]}
    path=$cwd/$resultdir/$bench

    mkdir -p $cwd/$resultdir/$bench
    echo "Testing $bench..."

    while read line; do
        name=$line

        echo "Testing $name..."

        pytest test/apps/test_$bench.py --ssname $name -s --benchmark-json=$path/$name.json 
        python $cwd/scripts/util/converter.py --json_name $path/$name.json	
            
        status=$?
        if [ $status -gt 0 ]
        then 
          errors+=("${name}, ${bench}")
        fi
    done
    
    python $cwd/scripts/util/bench_csv_aggregator.py $path $cwd/suitesparse_$bench.csv

done < $1

echo -e "${RED}Failed tests:"
for i in ${!errors[@]}; do
    error=${errors[$i]} 
    echo -e "${RED}$error,"
done
echo -e "${NC}"
