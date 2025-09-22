#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360

# Run from sam/ repo
# ./scripts/run_sam_sim/pytest_frostt.sh <tensor_names.txt> 

# Script that runs ALL test_tensor* pytest tests under sam/sim/test

outdir=$FROSTT_FORMATTED_PATH

errors=()
RED='\033[0;31m'
NC='\033[0m' # No Color

mkdir -p $outdir
cd ./sam/sim

while read line; do
    name=$line 

    echo "Testing $name..."
 
    pytest -k test_tensor --frosttname $name -s -vv #--debug-sim 
    status=$?
    if [ $status -gt 0 ]
    then 
      errors+=("${name} test")
    fi
done <$1

echo -e "${RED}Failed tests:"
for i in ${!errors[@]}; do
    error=${errors[$i]} 
    echo -e "${RED}$error,"
done
echo -e "${NC}"
