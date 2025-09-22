#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360

# Command: ./scripts/formatting/generate_frostt_formats.sh 

DATASET_NAMES=(
  fb1k
  fb10k
  facebook
  nell-2
  nell-1
)

IGNORED_NAMES=(
  amazon-reviews
  reddit
  patents
)

FORMATS=(
  sss012
)

basedir=$(pwd)

for i in ${!FORMATS[@]}; do
    format=${FORMATS[@]};
    echo "Generating files for format $format..."
    
    $basedir/compiler/taco/build/bin/taco-test sam.pack_$format
    $basedir/compiler/taco/build/bin/taco-test sam.pack_other_frostt

    for j in ${!DATASET_NAMES[@]}; do
        
        name=${DATASET_NAMES[$j]} 
        echo "Generating input format files for $name..."
        python $basedir/scripts/formatting/datastructure_tns.py -n $name -f $format
        python $basedir/scripts/formatting/datastructure_tns.py -n $name -f $format --other
        chmod -R 775 $FROSTT_FORMATTED_PATH
    done
done
