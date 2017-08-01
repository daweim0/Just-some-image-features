

nvcc -std=c++11 -c -o triplet_loss_op.cu.o triplet_loss_op_gpu.cu.cc \
	-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_50

g++ -std=c++11 -shared -o triplet_loss.so triplet_loss_op.cc \
triplet_loss_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64 -D_GLIBCXX_USE_CXX11_ABI=0
