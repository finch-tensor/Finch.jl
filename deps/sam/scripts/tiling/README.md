# Tiling Scripts

The `scripts/tiling/` folder contains scripts used to tile datasets and run tiling benchmarks. 

1. `advanced_simulator_runner.sh` - Script that formats, runs, and generates a
   CSV for the tiled simulation (aka advanced simulator). 
2. `clean_memory_model.sh` - Helper script to remove all generated files from
   the tiled SAM flow 
3. `ext_runner.sh` - Script that runs the Extensor configuration of for
   inner-product matmul, used to recreate a graph in the ASPLOS 2023 SAM paper.
4. `few_points_memory_model_runner.sh` - Script that runs a restricted set of
   experiments (8 points) from Figure 15 on pg. 12 of the SAM ASPLOS 2023 paper
(used in the ASPLOS 2023 artifact evaluation).  
5. `full_ext_runner.sh` - Script that runs `ext_runner.sh` for all combinations
   of NNZ and Dimension points.  
6. `full_memory_model_runner.sh` - Script that runs the full set of experiments
   to generate Figure 15 on pg. 12 of the SAM ASPLOS 2023 paper (used in the
ASPLOS 2023 artifact evaluation). 
7. `generate_gold_matmul_tiled.py` - Script that generates the golden matmul
   partial sums for each tile. 
8. `generate_sparsity_sweep_mem_model.sh` - Script that generates pre-tiled
   synthetic matrices. Used in the ASPLOS 2023 SAM artifact evaluation. 
9. `prepare_files_no_gold.sh` - Script that runs `tile_ext.sh` for the extensor
   configuration 
10. `prepare_files.sh` - Script that runs `tile_ext.sh` and also prepares the
    gold files using `generate_gold_matmul_tiled.py 
11. `single_point_memory_model_runner.sh` - Script that runs a single point
    from Figure 15 on pg. 12 of the SAM ASPLOS 2023 paper (Used in the ASPLOS
2023 artifact evaluation). 
12. `tile_ext.sh` - Script that tiles the input matrices from a directory (like
    extensor_mtx). 
13. `tile.sh` - Script that tiles the input matrices from a tensor name (like
    SuiteSparse matrices). 
