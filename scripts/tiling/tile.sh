#!/bin/bash

# ./scripts/tiling/tile.sh <tensor_names.txt>

BENCHMARKS=(
  matmul_ikj
)

# THIS FILE MUST BE RUN FROM sam/ location
sspath=$SUITESPARSE_PATH

basedir=$(pwd)


for b in ${!BENCHMARKS[@]}; do
	bench=${BENCHMARKS[$b]}
	path=$basedir/$benchout/$bench
	mkdir -p $basedir/$benchout/$bench
	echo "Testing $bench..."

	while read line; do

		echo "Tiling mtx file"
                python $basedir/sam/sim/src/tiling/tile.py --input_tensor $line --cotile $bench --multilevel --hw_config $basedir/sam/sim/src/tiling/$2 

                echo "Generating input format files for $line..."
                python $basedir/scripts/formatting/datastructure_suitesparse.py -n $line -hw -b $bench --input $basedir/tiles/$bench/mtx/ --output_dir_path $basedir/tiles/$bench/formatted --tiles

	done <$1
done

