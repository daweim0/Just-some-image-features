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
  --weights $2 \
  --imdb lov_synthetic_train \
  --cfg experiments/cfgs/lov_features.yml \
  --iters 80000

if [ -f $PWD/output/sintel_albedo/sintel_albedo_val/vgg16_fcn_color_single_frame_lov_iter_40000/segmentations.pkl ]
then
  rm $PWD/output/sintel_albedo/sintel_albedo_val/vgg16_fcn_color_single_frame_lov_iter_40000/segmentations.pkl
fi
