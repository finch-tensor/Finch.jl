#!/bin/bash

# Vars
if [ -z "$2" ]
then
    export SYNTHETIC_PATH="$(pwd)/synthetic/"
else
    export SYNTHETIC_PATH="$2"
fi

export SRC_PATH="$(pwd)/sam/onyx/synthetic/"

# Create the main directories
mkdir -p $SYNTHETIC_PATH
pushd $SYNTHETIC_PATH
for vectype in "random" "blocks" "runs"
do
    mkdir -p "${SYNTHETIC_PATH}/${vectype}/compressed/"
    mkdir -p "${SYNTHETIC_PATH}/${vectype}/uncompressed/"
    case $vectype in
        random)
            python ${SRC_PATH}/generate_random_mats.py --seed 0 --output_dir ${SYNTHETIC_PATH}/${vectype}/compressed/ --name B --shape 2000 --output_format CSF
            python ${SRC_PATH}/generate_random_mats.py --seed 1 --output_dir ${SYNTHETIC_PATH}/${vectype}/compressed/ --name C --shape 2000 --output_format CSF
            python ${SRC_PATH}/generate_random_mats.py --seed 0 --output_dir ${SYNTHETIC_PATH}/${vectype}/uncompressed/ --name B --shape 2000 --output_format UNC
            python ${SRC_PATH}/generate_random_mats.py --seed 1 --output_dir ${SYNTHETIC_PATH}/${vectype}/uncompressed/ --name C --shape 2000 --output_format UNC
            ;;
        blocks)
            for bs in 1 2 5 10 20 30 40 50 75 100 200 300 400
            do
                nnz=400
                python ${SRC_PATH}/generate_blocks.py --seed 0 --output_dir ${SYNTHETIC_PATH}/${vectype}/compressed/ --number_nonzeros $nnz --len_blocks $bs --shape 2000 --output_format CSF
                python ${SRC_PATH}/generate_blocks.py --seed 0 --output_dir ${SYNTHETIC_PATH}/${vectype}/uncompressed/ --number_nonzeros $nnz --len_blocks $bs --shape 2000 --output_format UNC
                # python ${SRC_PATH}/generate_blocks.py --seed 0 --output_dir ${SYNTHETIC_PATH}/${vectype}/compressed/ --shape 2000 --output_format CSF
                # python ${SRC_PATH}/generate_blocks.py --seed 0 --output_dir ${SYNTHETIC_PATH}/${vectype}/uncompressed/ --shape 2000 --output_format UNC
            done
            ;;
        runs)
            for rl in 1 2 5 10 20 30 40 50 75 100 200 300 400
            do
                python ${SRC_PATH}/generate_runs.py --seed 0 --output_dir ${SYNTHETIC_PATH}/${vectype}/compressed/ --shape 2000  --number_nonzeros 400 --run_length $rl --output_format CSF
                python ${SRC_PATH}/generate_runs.py --seed 0 --output_dir ${SYNTHETIC_PATH}/${vectype}/uncompressed/ --shape 2000  --number_nonzeros 400 --run_length $rl --output_format UNC
            # python ${SRC_PATH}/generate_runs.py --seed 0 --output_dir ${SYNTHETIC_PATH}/${vectype}/compressed/ --output_format CSF --run_lengths 100 200
            # python ${SRC_PATH}/generate_runs.py --seed 0 --output_dir ${SYNTHETIC_PATH}/${vectype}/uncompressed/ --output_format UNC --run_lengths 100 200
            done
            ;;
    esac
done
popd

# Now generate the matrices in both DCSR/DCSC formats
pushd $SYNTHETIC_PATH

mkdir -p "${SYNTHETIC_PATH}/matrix/DCSR"
mkdir -p "${SYNTHETIC_PATH}/matrix/DCSC"
mkdir -p "${SYNTHETIC_PATH}/matrix/DENSE"

i=250
j=250
k=100

sparsity="0.95"

python ${SRC_PATH}/generate_random_mats.py --seed 0 --sparsity $sparsity --output_dir ${SYNTHETIC_PATH}/matrix/DCSR/ --name B --shape $i $k --output_format CSF
python ${SRC_PATH}/generate_random_mats.py --seed 0 --sparsity $sparsity --output_dir ${SYNTHETIC_PATH}/matrix/DCSC/ --name B --shape $i $k --output_format CSF --transpose
python ${SRC_PATH}/generate_random_mats.py --seed 0 --sparsity $sparsity --output_dir ${SYNTHETIC_PATH}/matrix/DENSE/ --name B --shape $i $k --output_format UNC

python ${SRC_PATH}/generate_random_mats.py --seed 1 --sparsity $sparsity --output_dir ${SYNTHETIC_PATH}/matrix/DCSR/ --name C --shape $k $j --output_format CSF
python ${SRC_PATH}/generate_random_mats.py --seed 1 --sparsity $sparsity --output_dir ${SYNTHETIC_PATH}/matrix/DCSC/ --name C --shape $k $j --output_format CSF --transpose
python ${SRC_PATH}/generate_random_mats.py --seed 1 --sparsity $sparsity --output_dir ${SYNTHETIC_PATH}/matrix/DENSE/ --name C --shape $k $j --output_format UNC

popd