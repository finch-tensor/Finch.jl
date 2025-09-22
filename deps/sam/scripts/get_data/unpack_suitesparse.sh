#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360

# Command: ./scripts/get_data/unpack_suitesparse.sh <tensor_names.txt>

pushd .
cd $SUITESPARSE_PATH

# Uncompress tar file
for f in *.tar.gz; do
    tar -xvf "$f" --strip=1
    rm "$f"
done

while read line; do
	for f in ${line}_*.mtx; do
		rm "$f"
	done
done <$1

# Change file permissions of .mtx file
for f in *.mtx; do
    chmod ugo+r "$f"
done
popd
