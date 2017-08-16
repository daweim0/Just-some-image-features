#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

# train FCN for single frames
./tools/test_net.py --gpu 0 \
--model "$2" \
--imdb lov_synthetic_val-59_60_long \
--cfg experiments/cfgs/lov_features.yml \
--calc_EPE_all
