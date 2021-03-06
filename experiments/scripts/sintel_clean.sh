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
  --network vgg16_flow_features \
  --weights data/imagenet_models/vgg16_convs.npy \
  --imdb sintel_clean_train \
  --cfg experiments/cfgs/sintel_clean.yml \
  --iters 40000

if [ -f $PWD/output/sintel_albedo/sintel_albedo_val/vgg16_fcn_color_single_frame_lov_iter_40000/segmentations.pkl ]
then
  rm $PWD/output/sintel_albedo/sintel_albedo_val/vgg16_fcn_color_single_frame_lov_iter_40000/segmentations.pkl
fi
