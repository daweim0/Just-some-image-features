#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_USER_OPS_BACKPROJECTING_OP_GPU_H_
#define TENSORFLOW_USER_OPS_BACKPROJECTING_OP_GPU_H_

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Run the forward pass of max pooling, optionally writing the argmax indices to
// the mask array, if it is not nullptr. If mask is passed in as nullptr, the
// argmax indices are not written.
bool TripletFlowForwardLaucher(
    const float* left_data, const float* right_data, const float* flow_tensor, const int* occluded_tensor,
    const int batch_size, const int height, const int width, const int channels, const int n_triplets,
    const float margin, const int negative_radius_, float* top_data, float* bottom_left_diff, float* bottom_right_diff, const Eigen::GpuDevice& d);

bool TripletFlowBackwardLaucher(const float* top_diff, const float* bottom_left_diff, const float* bottom_right_diff, const int batch_size,
                            const int height, const int width, const int channels, float* output_l, float* output_r, const Eigen::GpuDevice& d);

}  // namespace tensorflow

#endif
