# Scripts to Run SAM Simulations 

All scripts should ultimately run pytest to test the SAM
simulator applications 

the `scripts/run_sam_sim/` folder contains scripts that run the sam simulator for the following datasets:
1. SuiteSparse
2. FROSTT
3. Synthetically generated data

1. `pytest_frostt.sh` - Script that runs ALL pytest tests beginning with the
   name `test_tensor*` under `sam/sim/test/` with the FROSTT tensors.
2. `pytest_frostt_with_benchmarks.sh` - Script that runs only select pytest
   benchmarks under `sam/sim/test/` with the FROSTT tensors.
3. `pytest_suitesparse.sh` - Script that runs ALL pytest tests in
   `sam/sim/test/final-apps` with gold checking enabled for the SuiteSparse
matrices provided in `tensor_names.txt`. 
4. `pytest_suitesparse_with_benchmarks.sh` - Script that runs runs select
   SuiteSparse pytest benchmarks under `sam/sim/test/apps/`. This script has gold checking
disabled and aggregates results into a CSVs.  
5. `run_suitesparse_final.sh` - Script that runs ALL SuiteSparse final tests in
   `sam/sim/test/final-apps/`
6. `run_suitesparse_generated.sh` - Script that runs ALL SuiteSparse generated tests in
   `sam/sim/test/apps/`
7. `run_suitesparse.sh` - Script that formats input SuiteSparse matrices and then runs
   pytest on all SuiteSparse benchmarks in `sam/sim/test/apps`
8. `run_synthetics.sh` - Script that runs all of the synthetic benchmarks from
   the ASPLOS 2023 SAM paper. 
9. `sam_frostt_runner.sh` - Script that formats, runs, and generates CSVs for
   all frostt benchmarks.
10. `sam_suitesparse_runner.sh` - Script that formats, runs, and generates CSVs
    for all SuiteSparse benchmarks in `final-apps`.
11. `sam_suitesparse_runner_sddmmonly.sh` - Script that formats, runs, and
    generates CSVs for the `final-apps` SDDMM SuiteSparse benchmark only. 
12. `suitesparse_validator.sh` - Script that runs the CPU benchmarks and then
    the SAM pytest benchmarks in `apps` on SuiteSparse data.
