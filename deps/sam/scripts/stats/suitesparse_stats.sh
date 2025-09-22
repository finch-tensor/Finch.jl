#!/bin/sh
#SBATCH -N 1 
#SBATCH --exclusive

basedir=$(cwd)

rm -rf $basedir/scripts/logs 

python suitesparse_stats.py --overall -nstop 250 
