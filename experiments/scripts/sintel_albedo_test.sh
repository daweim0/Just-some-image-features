#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0
export LD_PRELOAD=/usr/lib/libtcmalloc.so.4


for model in "$@"
do
    # train FCN for single frames
    ./tools/test_net.py --gpu 0 \
    --network vgg16_flow \
    --model "$model" \
    --imdb sintel_albedo_val \
    --cfg experiments/cfgs/sintel_albedo.yml
done
