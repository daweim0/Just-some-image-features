all:
	#!/usr/bin/env bash
	OLD_PATH=$(pwd)
	cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
	TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
	echo $TF_INC

	CUDA_PATH=/usr/local/cuda

	#nvcc -std=c++11 -c -o triplet_flow_loss_op.cu.o triplet_flow_loss_op_gpu.cu.cc \
	#	-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_50 --expt-relaxed-constexpr

	clang++ -std=c++11 -g3 -shared -o triplet_flow_loss.so triplet_flow_loss_op.cc \
		triplet_flow_loss_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64 -D_GLIBCXX_USE_CXX11_ABI=0

	cd $OLD_PATH
