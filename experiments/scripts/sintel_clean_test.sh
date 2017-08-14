#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0


for model in "$@"
do
    # train FCN for single frames
    ./tools/test_net.py --gpu 0 \
    --network vgg16_flow \
    --model "$model" \
    --imdb sintel_clean_val \
    --cfg experiments/cfgs/sintel_clean.yml
done
