# Setup for generating files for getting plots that model memory and allow fior a simulation result that recreates some of the results that extensor captures
# Generates files and stores in extensor_mtx folder to be run in the main directory
# This creates a extensor_mtx diorectory in the home directory of the project and creates matrixes sweeping across dimensions for different number of non-zeros
# ~8 mins to build it
./scripts/generate_sparsity_sweep_mem_model.sh

# OLD: mkdir extensor_mtx && cd extensor_mtx && python ../sam/onyx/synthetic/generate_fixed_nnz_mats.py && cd ..

# To run the entire test that generates a csv file with numbers for figure 15:
# Put 0 as the second command if you dont want to check against gold as you go along the simulation (results in faster simulation)
# Put 1 as second command if want to check agqinst gold
# These will generate a directory called "Tiles" with the pre-tiled matrix for the current test and then create a directory called suitesparse-bench_simulator/sam/"test_name" (test name is matmul_ikj_tile_pipeline_final)
# Inside this directory a json and a csv for each # of nnz and dimension size is created
# After all such matrices are done they are aggregated into a single csv (which is used by the script to create fig 15) the aggregated csv is suitesparse_matmul_ikj_tile_pipeline_final.csv in suitesparse-bench_simulator/sam/ under home dir
# Runs stuff from smallest to greatest
# Without gold check this takes it takes 63 hrs 9 mins to run
./scripts/full_memory_model_runner2.sh memory_config_extensor_17M_llb.yaml 0 

# Generate the figure 15:
# TODO: add the matplotlib script

# To run a restricted set of tests (from the nnz we choose only 5000 and 25000 for dimension sizes 1024, 3696, 9040, 15720)
# Without gold takes 8hrs 2min hrs to run 
# With gold takes 19hrs 46min hrs to run
./scripts/few_points_model_runner.sh memory_config_extensor_17M_llb.yaml 0

# TODO:matplotlib script for above? 

# Verify any single point (by default gold check is on)
# Smallest case 5000 nnz and dimension size 1024: 23 mins 22 sec
# Largest case 50000 nnz and dimension size 15720: 16hrs 59min 22sec
# Memory requirement with gold is for the largest matrix is 1.8 Gb (15720 dim and 50000 nnzs)
# Doesnt add anything o the csv currently just prints out the number of cycles  
./scripts/ext_runner.sh extensor_"# of nnzs"_"dim size".mtx
