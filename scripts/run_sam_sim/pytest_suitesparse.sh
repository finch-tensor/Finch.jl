#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360

# ./scripts/run_sam_sim/pytest_suitesparse.sh <tensor_names.txt>


outdir=$SUITESPARSE_FORMATTED_PATH

errors=()
RED='\033[0;31m'
NC='\033[0m' # No Color


mkdir -p $outdir
cd ./sam/sim

while read line; do
    name=$line 

    echo "Testing $name..."

    pytest ./test/final-apps --ssname $name -s --check-gold #--debug-sim 
    status=$?
    if [ $status -gt 0 ]
    then 
      errors+=("${name}")
    fi
done < $1

echo -e "${RED}Failed tests:"
for i in ${!errors[@]}; do
    error=${errors[$i]} 
    echo -e "${RED}$error,"
done
echo -e "${NC}"
