#!/usr/bin/env bash
clear
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
echo $TF_INC

CUDA_PATH=/usr/local/cuda

nvcc -std=c++11 -g -G -c -o triplet_flow_loss_op.cu.o triplet_flow_loss_op_gpu.cu.cc \
	-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_30 --expt-relaxed-constexpr

g++ -std=c++11 -shared -g3 -o triplet_flow_loss.so triplet_flow_loss_op.cc \
	triplet_flow_loss_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64 -D_GLIBCXX_USE_CXX11_ABI=0 -w

