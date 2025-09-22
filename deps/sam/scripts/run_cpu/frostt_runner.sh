#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive

# ./frostt_runner.sh <machine> 
# Arg1 <machine> - Which machine is being used (0:local, 1:Lanka, 2:Kiwi/Neva) 

DATASET_NAMES=(
  fb1k
  fb10k
  facebook
  nell-2
  nell-1
)

sspath=$SUITESPARSE_PATH
cwd=$(pwd)
# LANKA
if [ $1 -eq 1 ]; then
	lanka=ON
	neva=OFF
elif [ $1 -eq 2 ]; then
	lanka=OFF
	neva=ON
else
	lanka=OFF
	neva=OFF
fi

out=frostt-bench/taco

mkdir -p "$out"

for i in ${!DATASET_NAMES[@]}; do
	name=${DATASET_NAMES[$i]} 
	csvout="$out/result-$name.csv"
	make -j8 taco-bench BENCHES="$name" NEVA=$neva LANKA=$lanka GEN=ON
done

