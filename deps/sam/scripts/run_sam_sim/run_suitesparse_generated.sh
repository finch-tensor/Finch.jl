#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive

pushd .
cd sam/sim

echo $1

while read line; do
	pytest test/apps/ --ssname $line --check-gold
done < $1

popd
