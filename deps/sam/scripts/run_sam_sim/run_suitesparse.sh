#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360

# 1. Formats input files 
# 2. Runs suitesparse sam sims in pytest

# ./scripts/run_sam_sim/run_suitesparse.sh <tensor_names.txt>

# THIS FILE MUST BE RUN FROM sam/ location
outdir=$SUITESPARSE_FORMATTED_PATH
basedir=$(pwd)

errors=()
RED='\033[0;31m'
NC='\033[0m' # No Color


mkdir -p $outdir

while read line; do
    name=$line

    cd $outdir
    echo "Generating input format files for $name..."
    python $basedir/scripts/formatting/datastructure_suitesparse.py -n $name 
    chgrp -R sparsity $outdir
    chmod -R 777 $outdir

    cd $basedir/sam/sim/test/apps
    echo "Testing $name..."
    pytest --ssname $name
    status=$?

    if [ $status -gt 0 ]
    then 
      errors+=($name)
    fi

    cd $outdir
    echo "Removing format files for $name..."
    rm ./$name*.txt
done < $1

echo -e "${RED}Failed tests:"
for i in ${!errors[@]}; do
    error=${errors[$i]} 
    echo -e "${RED}$error,"
done
echo -e "${NC}"

