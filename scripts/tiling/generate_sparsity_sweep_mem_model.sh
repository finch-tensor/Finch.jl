# ./scripts/tiling/generate_sparsity_sweep_mem_model.sh 

SECONDS=0

mkdir extensor_mtx
cd extensor_mtx
python ../sam/onyx/synthetic/generate_fixed_nnz_mats.py --extensor
cd ..
ELAPSED="Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
printf "$ELAPSED"
