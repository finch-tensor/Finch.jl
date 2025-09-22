#!/bin/bash
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p lanka-v3
#SBATCH --exclusive

# Command: ./scripts/sam_hw_suitesparse_runner.sh <suitesparse_names.txt> <0|1|2>
# Where 0 = local, 1 = Lanka, 2 = kiwi/neva
# This should be run from the sam/ directory

# This script:
# 1. Formats teh matrices
# 2. Then runs build_tb in garnet

set -u

BENCHMARKS=(
  matmul_FINAL
  mat_elemadd_FINAL
  mat_elemadd3_FINAL
  mat_vecmul_FINAL
  mat_residual_FINAL
  mat_mattransmul_FINAL
)

errors=()
RED='\033[0;31m'
NC='\033[0m' # No Color

mkdir -p $TACO_TENSOR_PATH
mkdir -p $SUITESPARSE_FORMATTED_PATH
mkdir -p $FROSTT_FORMATTED_TACO_PATH
mkdir -p $FROSTT_FORMATTED_PATH

# LANKA
if [ $2 -eq 1 ]; then
	lanka=ON
	neva=OFF
# KIWI/NEVA
elif [ $2 -eq 2 ]; then
	# NEVA/KIWI
	lanka=OFF
	neva=ON
# Local Machine
else
	lanka=OFF
	neva=OFF
fi

format_outdir=${SUITESPARSE_FORMATTED_PATH} 
basedir=$(pwd)
sspath=$SUITESPARSE_PATH
benchout=suitesparse-bench/sam

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
		cd $format_outdir

		if [ $2 -eq 1 ]; then
			matrix="$sspath/$line/$line.mtx"
		elif [ $2 -eq 2 ]; then
			matrix="$sspath/$line.mtx"
		else
			matrix="$sspath/$line.mtx"
		fi

		echo "Generating input format files for $line..."
		SUITESPARSE_TENSOR_PATH=$matrix python $basedir/scripts/datastructure_suitesparse.py -n $line -t $bench -hw

		SUITESPARSE_TENSOR_PATH=$matrix $basedir/compiler/taco/build/bin/taco-test sam.pack_other_ss    
		python $basedir/scripts/datastructure_tns.py -n $line -f ss01 --other -ss

		cd $basedir/garnet

		python tests/test_memory_core/build_tb.py --ic_fork --sam_graph /aha/sam/compiler/sam-outputs/dot/matmul_ijk.gv --output_dir /aha/garnet/OUTPUT_DIR --input_dir /aha/garnet/INPUT_DIR/ --test_dump_dir DUMP_DIR --matrix_tmp_dir $TMP_MAT --seed 0 --trace --gold_dir GOLD_OUT --fifo_depth 8 --gen_pe --dump_bitstream --add_pond --combined --pipeline_scanner --dump_glb --glb_dir GLB_DIR/matmul_ijk_combined
	done <$1

done

