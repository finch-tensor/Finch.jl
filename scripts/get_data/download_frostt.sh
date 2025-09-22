#!/bin/bash
#SBATCH -N 1
#SBATCH -t 360
#SBATCH -p lanka-v3

# Command: ./scripts/get_data/download_frostt.sh

set -e

TENSOR_NAMES=(
  "chicago-crime"
  "lbnl-network"
  "nips"
  "uber-pickups"
  "amazon-reviews"
  "delicious"
  "enron"
  "flickr"
  "nell-1"
  "nell-2"
  "patents"
  "reddit"
  "vast"
)

TENSOR_URLS=(
  "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/chicago-crime/comm/chicago-crime-comm.tns.gz"
  "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/lbnl-network/lbnl-network.tns.gz"
  "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nips/nips.tns.gz"
  "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/uber-pickups/uber.tns.gz"
  "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/amazon/amazon-reviews.tns.gz"
  "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/delicious/delicious-4d.tns.gz"
  "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/enron/enron.tns.gz"
  "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/flickr/flickr-4d.tns.gz"
  "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nell/nell-1.tns.gz"
  "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/nell/nell-2.tns.gz"
  "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/patents/patents.tns.gz"
  "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/reddit-2015/reddit-2015.tns.gz"
  "https://s3.us-east-2.amazonaws.com/frostt/frostt_data/vast-2015-mc1/vast-2015-mc1-5d.tns.gz"
)

outdir=$FROSTT_PATH

mkdir -p $outdir

for i in ${!TENSOR_URLS[@]}; do
    name=${TENSOR_NAMES[$i]}
    url=${TENSOR_URLS[$i]}
    out="$outdir/$name.tns"
    if [ -f "$out" ]; then
        continue
    fi
    echo "Downloading tensor $name to $out"
    curl $url | gzip -d -c > "$out"
done
