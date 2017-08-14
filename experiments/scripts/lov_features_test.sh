#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0


for model in "$@"
do
    # train FCN for single frames
    ./tools/test_net.py --gpu 0 \
    --model "$model" \
    --imdb lov_val \
    --cfg experiments/cfgs/lov_features.yml
done
