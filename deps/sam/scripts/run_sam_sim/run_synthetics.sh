#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360

# ./scripts/run_sam_sim/run_synthetics.sh <synthetic_path>

# Req: Need to run this after synthetic/ is generated
# 1. Runs all of the synthetic tests from the ASPLOS 2023 SAM paper

basedir=$(pwd)
resultdir=results

# Vars
if [ -z "$1" ]
then
    export SYNTHETIC_PATH="$basedir/synthetic/"
else
    export SYNTHETIC_PATH="$1"
fi

BENCHMARKS=(
  test_vec_elemmul_bittree
  test_vec_elemmul_bitvector
  test_vec_elemmul_compress_skip
  test_vec_elemmul_compress_split
  test_vec_elemmul_compressed
  test_vec_elemmul_uncompressed
)


for b in ${!BENCHMARKS[@]}; do
    bench=${BENCHMARKS[$b]}
    path=$resultdir/

    mkdir -p $resultdir/
    echo "Testing $bench..."

    pytest sam/sim/test/study-apps/$bench.py --synth --check-gold -k "random-40 or 0.2-blocks or 0.2-runs" --benchmark-json="$path/$bench.json"
    python $basedir/scripts/util/converter.py --json_name $path/$bench.json
    python $basedir/scripts/util/bench_csv_aggregator.py $path $basedir/SYNTH_OUT_ACCEL.csv

done

BENCHMARKS=(
    test_reorder_matmul_ijk
    test_reorder_matmul_ikj
    test_reorder_matmul_jik
    test_reorder_matmul_jki
    test_reorder_matmul_kij
    test_reorder_matmul_kji
)

basedir=$(pwd)
resultdir=results_reorder

for b in ${!BENCHMARKS[@]}; do
    bench=${BENCHMARKS[$b]}
    path=$resultdir/

    mkdir -p $resultdir/
    echo "Testing $bench..."

    pytest sam/sim/test/reorder-study/$bench.py --synth --check-gold --benchmark-json="$path/$bench.json"
    python $basedir/scripts/util/converter.py --json_name $path/$bench.json
    python $basedir/scripts/util/bench_csv_aggregator.py $path $basedir/SYNTH_OUT_REORDER.csv

done

BENCHMARKS=(
    test_mat_sddmm_coiter_fused
    test_mat_sddmm_locate_fused
    test_mat_sddmm_unfused
)

basedir=$(pwd)
resultdir=results_fusion

for b in ${!BENCHMARKS[@]}; do
    bench=${BENCHMARKS[$b]}
    path=$resultdir/

    mkdir -p $resultdir/
    echo "Testing $bench..."

    pytest sam/sim/test/fusion-study/$bench.py --synth --check-gold --benchmark-json="$path/$bench.json"
    python $basedir/scripts/util/converter.py --json_name $path/$bench.json
    python $basedir/scripts/util/bench_csv_aggregator.py $path $basedir/SYNTH_OUT_FUSION.csv

done

# echo -e "${RED}Failed tests:"
# for i in ${!errors[@]}; do
#     error=${errors[$i]} 
#     echo -e "${RED}$error,"
# done
# echo -e "${NC}"
