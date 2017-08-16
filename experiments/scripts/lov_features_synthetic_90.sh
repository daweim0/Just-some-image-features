#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

LOG="experiments/logs/lov_single_color.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# train FCN for single frames
export LD_PRELOAD=/usr/lib/libtcmalloc.so.4
./tools/train_net.py --gpu 0 \
  --weights data/imagenet_models/vgg16_convs.npy \
  --imdb lov_synthetic_train-45_90 \
  --cfg experiments/cfgs/lov_features.yml \
  --iters 40000
