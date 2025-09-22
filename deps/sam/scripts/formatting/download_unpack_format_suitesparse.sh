#!/bin/bash

# Command: ./scripts/formatting/download_unpack_format_suitesparse.sh <tensor_names.txt>

basedir=$(pwd)
path=$basedir/jsons
download_script=scripts/get_data/download_suitesparse_partial.sh

mkdir -p $path

BENCHMARKS=(
	matmul_jki
	matmul_ikj
	matmul_kji
	matmul_kij
	matmul_ijk
	matmul_jik
	mat_elemadd
	mat_elemmul
	mat_identity
	mat_elemadd3
	mat_mattransmul
	mat_residual
	mat_vecmul_ij
	mat_vecmul_ji
	mat_vecmul
	mat_sddmm
)

# Create download_script that downloads ONLY the suitesparse matrices listed in the text file that is passed in as the first argument of this script
[ -e $download_script ] && rm $download_script
echo "mkdir -p ${SUITESPARSE_PATH}" >> $download_script
echo "pushd ." >> $download_script
echo "cd ${SUITESPARSE_PATH}" >> $download_script
grep -F -f $1 scripts/get_data/download_suitesparse.sh >> $download_script 
echo "popd" >> $download_script

# Make it an executable
chmod ugo+x $download_script

# Call the download_script (created above)
./$download_script

# Unpack the downloaded suitesparse files since they come in .tar format
./scripts/get_data/unpack_suitesparse.sh $(realpath $1)

for b in ${!BENCHMARKS[@]}; do
	bench=${BENCHMARKS[$b]}
	while read line; do
		echo "Generating input format files for $line..."
		sspath=${SUITESPARSE_PATH}/$line
		SUITESPARSE_TENSOR_PATH=$sspath python $basedir/scripts/formatting/datastructure_suitesparse.py -n $line -hw -b $bench 
	done <$(realpath $1)
done
