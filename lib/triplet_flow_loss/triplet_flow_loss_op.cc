/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Lifted Structured Loss Op

#include <stdio.h>
#include <cfloat>
#include <math.h>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <unistd.h>

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;


REGISTER_OP("TripletFlow")
	.Attr("T: {float, double}")
	.Attr("margin: float")
	.Input("left_data: T")
	.Input("right_data: T")
	.Input("gt_flow: T")
	.Input("occluded_mask: int32")
	.Output("loss: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->Scalar());
        return Status::OK();
        })
	.Output("bottom_left_diff: T")
	.Output("bottom_right_diff: T");

REGISTER_OP("TripletFlowGrad")
	.Attr("T: {float, double}")
	.Attr("margin: float")
	.Input("bottom_left_diff: T")
	.Input("bottom_right_diff: T")
	.Input("grad: T")
	.Output("output_left: T")
    .Output("output_right: T");

template <typename Device, typename T>
class TripletFlowOp : public OpKernel {
	public:
	explicit TripletFlowOp(OpKernelConstruction* context) : OpKernel(context) {
	// Get the margin
	OP_REQUIRES_OK(context,
				   context->GetAttr("margin", &margin_));
	// Check that margin is positive
	OP_REQUIRES(context, margin_ >= 0,
				errors::InvalidArgument("Need margin >= 0, got ", margin_));
	}

	// bottom_data: (batch_size, height, width, channels)
	// bottom_label: (batch_size, height, width, num_classes)
	void Compute(OpKernelContext* context) override
	{
		// Grab the input tensor
		const Tensor& left_data = context->input(0);
		const T* left_data_array = left_data.flat<T>().data();

		const Tensor& right_data = context->input(1);
		const T* right_data_array = right_data.flat<T>().data();

		const Tensor& flow_tensor = context->input(2);
		const T* flow_array = flow_tensor.flat<T>().data();

		const Tensor& occluded = context->input(3);
		const int* occluded_array = occluded.flat<int>().data();

		// data should have 4 dimensions.
		OP_REQUIRES(context, left_data.dims() == 4,
					errors::InvalidArgument("left data must be 3-dimensional"));

		OP_REQUIRES(context, right_data.dims() == 4,
					errors::InvalidArgument("right data must be 3-dimensional"));

		OP_REQUIRES(context, flow_tensor.dims() == 4,
					errors::InvalidArgument("flow must be 3-dimensional"));

		OP_REQUIRES(context, occluded.dims() == 4,
					errors::InvalidArgument("occlusions must be 3-dimensional"));

		// batch size
		int batch_size = left_data.dim_size(0);
		// height
		int height = left_data.dim_size(1);
		// width
		int width = left_data.dim_size(2);
		// number of channels
		int feature_depth = left_data.dim_size(3);

//        std::cout << "Image batch_size: " << batch_size << " height: " << height << " width: " << width << " feature_depth: " << feature_depth << std::endl;
//        std::cout << "flow batch_size: " << flow_tensor.dim_size(0) << " height: " << flow_tensor.dim_size(1) << " width: " << flow_tensor.dim_size(2) << " feature_depth: " << flow_tensor.dim_size(3) << std::endl;
//        std::cout << "occluded batch_size: " << occluded.dim_size(0) << " height: " << occluded.dim_size(1) << " width: " << occluded.dim_size(2) << std::endl;

        assert(batch_size == flow_tensor.dim_size(0));
        assert(batch_size == occluded.dim_size(0));
        assert(height == flow_tensor.dim_size(1));
        assert(height == occluded.dim_size(1));
        assert(width == flow_tensor.dim_size(2));
        assert(width == occluded.dim_size(2));

        int feature_arr_size = batch_size * height * width * feature_depth;

        int flow_arr_size = flow_tensor.dim_size(0) * flow_tensor.dim_size(1) * flow_tensor.dim_size(2) * flow_tensor.dim_size(3);

        // Create output loss tensor
		int dim = 1;
		TensorShape output_shape;
//		TensorShapeUtils::MakeShape(&dim, 1, &output_shape);

		Tensor* top_data_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &top_data_tensor));
		auto top_data = top_data_tensor->template flat<T>();

		// bottom diff
		TensorShape output_shape_left_diff = left_data.shape();
		Tensor* bottom_left_diff_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(1, output_shape_left_diff, &bottom_left_diff_tensor));
		T* bottom_left_diff = bottom_left_diff_tensor->template flat<T>().data();
		memset(bottom_left_diff, 0, batch_size * height * width * feature_depth *sizeof(T));

		TensorShape output_shape_right_diff = left_data.shape();
		Tensor* bottom_right_diff_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(2, output_shape_right_diff, &bottom_right_diff_tensor));
		T* bottom_right_diff = bottom_right_diff_tensor->template flat<T>().data();
		memset(bottom_right_diff, 0, batch_size * height * width * feature_depth *sizeof(T));

		// sample triplets to define the loss
		// compute label indexes

//		// classes in the batch

		int saftey_margin = 2;  // the radius around the target where we shouldn't sample
		// sampling
		std::srand(unsigned (std::time(0)));
		std::vector<int> triplets;  // don't initialize it with a size because some points might be occluded
		triplets.reserve(batch_size * height * width * 3);
		for (int n = 0; n < batch_size; n++)
		{
			for (int h = 0; h < height; h++)
			{
				for (int w = 0; w < width; w++)
				{
					// anchor
					int index = n * height * width + h * width + w;
					if(occluded_array[index] != 0)
					{
//                        std::cout << "point occluded, h: " << h << " w: " << w << std::endl;
						continue;  // don't use this point
					}

					// find the corresponding pixel
                    if(index*2+1 >= flow_arr_size) {
                        std::cout << "index greater than flow array" << std::endl;
                        std::cout << "\tflow_arr_size: " << flow_arr_size << std::endl;
                        std::cout << "\tindex*2+1: " << index*2+1 << std::endl;
                        assert(false);
                    }
					float w_new = w + round(flow_array[index * 2]);
					float h_new = h + round(flow_array[index * 2 + 1]);
                    if(0 > w_new || 0 > h_new || width <= w_new || height <= h_new) {
//                        std::cout << "flow goes outside image, h: " << h << " w: " << w << std::endl;
                        continue;  // corresponding right image location is outside image
                    }
                    int index_positive = n * height * width + h_new * width + w_new;

					// sample a negative pixel
					// check the predicted label of this pixel for hard negative

					int possible_h = std::rand() % (height - saftey_margin*2);
                    assert(0 <= possible_h < height - saftey_margin*2);
					int possible_w = std::rand() % (width - saftey_margin*2);
                    assert(0 <= possible_w < width - saftey_margin*2);

					if(possible_h + saftey_margin > h)
						possible_h += saftey_margin * 2;
					if(possible_w + saftey_margin > w)
						possible_w += saftey_margin * 2;

					int index_negative = ((n) * height + possible_h) * width + possible_w;

					// store the triplet
					triplets.push_back(index);
					triplets.push_back(index_positive);
					triplets.push_back(index_negative);
                    assert(-1 < index && index < batch_size * height * width);
                    assert(-1 < index_positive && index_positive < batch_size * height * width);
                    assert(-1 < index_negative && index_negative < batch_size * height * width);
				}
			}
		}

//        std::cout << "hi2" << std::endl;

		double loss = 0;
		// for each triplet
		int num_triplets = triplets.size() / 3;
//        std::cout << "num_triplets: " << num_triplets << std::endl;
		for (int triplet_num = 0; triplet_num < num_triplets; triplet_num++)
		{
            const int index_i = triplets.at(triplet_num * 3 + 0);
            const int index_j = triplets.at(triplet_num * 3 + 1);
            const int index_k = triplets.at(triplet_num * 3 + 2);
            assert(-1 < index_j);

            // compute the distances
            T D_ij = 0;
            T D_ik = 0;
            for (int c = 0; c < feature_depth; c++)
            {
                int data_i_index = index_i * feature_depth + c;
                int data_j_index = index_j * feature_depth + c;
                int data_k_index = index_k * feature_depth + c;

                if ((0 > data_i_index) || (data_i_index >= feature_arr_size) ||
                    (0 > data_j_index) || (data_j_index >= feature_arr_size) ||
                    (0 > data_k_index) || (data_k_index >= feature_arr_size)) {
                    std::cout << "index out of bounds" << std::endl;
                    std::cout << "\t max index size: " << batch_size * height * width * feature_depth << std::endl;
                    std::cout << "\t data_i_index: " <<  data_i_index << " index_i" << index_i << std::endl;
                    std::cout << "\t data_j_index: " <<  data_j_index << " index_j" << index_j << std::endl;
                    std::cout << "\t data_k_index: " <<  data_k_index << " index_k" << index_k << std::endl;
                    assert(false);
                }

//                std::cout << "hi 2.1" << std::endl;
                D_ij += pow(left_data_array[data_i_index] - right_data_array[data_j_index], 2);
//                std::cout << "hi 2.2" << std::endl;
                D_ik += pow(left_data_array[data_i_index] - left_data_array[data_k_index], 2);
            }
            // add the loss
            double dis = D_ij - D_ik + margin_;
			T old_loss = loss;
            loss += std::max(dis, double(0.0));

            if(num_triplets == 0){
                std::cout << "loss will be nan! (" << loss << ")" << std::endl;
            }

			// std::cout << "dis: " << dis << " D_ij: " << D_ij << " D_ik: " << D_ik << std::endl;
			if(old_loss > loss && dis > 0.0 && 1 < 0) {
				std::cout << "loss overflowed, old_loss: " << old_loss << " loss: " << loss << std::endl;
				assert(false);
			}



            // compute gradients
            if (dis > 0) {
                for (int c = 0; c < feature_depth; c++) 
                {
                    int diff_i_index = index_i * feature_depth + c;
                    int diff_j_index = index_j * feature_depth + c;
                    int diff_k_index = index_k * feature_depth + c;

                    if ((0 > diff_i_index) || (diff_i_index >= feature_arr_size) ||
                        (0 > diff_j_index) || (diff_j_index >= feature_arr_size) ||
                        (0 > diff_k_index) || (diff_k_index >= feature_arr_size)) {
                        std::cout << "index out of bounds" << std::endl;
                        std::cout << "\t max index size: " << batch_size * height * width * feature_depth << std::endl;
                        std::cout << "\t diff_i_index: " <<  diff_i_index << std::endl;
                        std::cout << "\t diff_j_index: " <<  diff_j_index << std::endl;
                        std::cout << "\t diff_k_index: " <<  diff_k_index << std::endl;
                        assert(false);
                    }

                    // update x_i
//                    std::cout << "hi 2.3, diff_i_index =" << diff_i_index << std::endl;
                    bottom_left_diff[diff_i_index] +=
                            (left_data_array[diff_k_index] - right_data_array[diff_j_index]) / num_triplets;
                    // update x_j
//                    std::cout << "hi 2.4, diff_j_index =" << diff_j_index << std::endl;
                    bottom_right_diff[diff_j_index] +=
                            (right_data_array[diff_j_index] - left_data_array[diff_i_index]) / num_triplets;
                    // update x_k
//                    std::cout << "hi 2.5, diff_k_index =" << diff_k_index << std::endl;
                    bottom_left_diff[diff_k_index] +=
                            (left_data_array[diff_i_index] - left_data_array[diff_k_index]) / num_triplets;
                }
            }
		}
//        std::cout << "pre-scaled loss: " << loss;
		loss /= num_triplets * 2.0;
//        std::cout << " scaled loss: " << loss << std::endl;
//		top_data(0) = T(loss);
        top_data.setConstant(T(loss));
//        std::cout << "hi3" << std::endl;
	}

	private:
	float margin_;
};

REGISTER_KERNEL_BUILDER(Name("TripletFlow").Device(DEVICE_CPU).TypeConstraint<float>("T"), TripletFlowOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("TripletFlow").Device(DEVICE_CPU).TypeConstraint<double>("T"), TripletFlowOp<CPUDevice, double>);


//// GPU implementation for forward pass
//bool TripletForwardLaucher(
//	const float* bottom_data, const float* bottom_label, const int* bottom_prediction,
//	const int batch_size, const int height, const int width, const int channels, const int num_classes,
//	const float margin, float* top_data, float* bottom_diff, const Eigen::GpuDevice& d);
//
//static void TripletKernel(
//	OpKernelContext* context, const Tensor* bottom_data, const Tensor* bottom_label, const Tensor* bottom_prediction,
//	const int batch_size, const int height, const int width, const int channels, const int num_classes,
//	const float margin, const TensorShape& tensor_output_shape, const TensorShape& tensor_output_shape_diff)
//{
//  Tensor* top_data = nullptr;
//  Tensor* bottom_diff = nullptr;
//  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &top_data));
//  OP_REQUIRES_OK(context, context->allocate_output(1, tensor_output_shape_diff, &bottom_diff));
//
//  if (!context->status().ok()) {
//	return;
//  }
//
//  TripletForwardLaucher(
//	bottom_data->flat<float>().data(), bottom_label->flat<float>().data(), bottom_prediction->flat<int>().data(),
//	batch_size, height, width, channels, num_classes, margin,
//	top_data->flat<float>().data(), bottom_diff->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
//}
//
//template <class T>
//class TripletOp<Eigen::GpuDevice, T> : public OpKernel {
// public:
//  typedef Eigen::GpuDevice Device;
//
//  explicit TripletOp(OpKernelConstruction* context) : OpKernel(context) {
//	// Get the margin
//	OP_REQUIRES_OK(context,
//				   context->GetAttr("margin", &margin_));
//	// Check that margin is positive
//	OP_REQUIRES(context, margin_ >= 0,
//				errors::InvalidArgument("Need margin >= 0, got ", margin_));
//  }
//
//  void Compute(OpKernelContext* context) override
//  {
//	// Grab the input tensor
//	const Tensor& bottom_data = context->input(0);
//	const Tensor& bottom_label = context->input(1);
//	const Tensor& bottom_prediction = context->input(2);
//
//	// data should have 4 dimensions.
//	OP_REQUIRES(context, bottom_data.dims() == 4,
//				errors::InvalidArgument("data must be 4-dimensional"));
//
//	OP_REQUIRES(context, bottom_label.dims() == 4,
//				errors::InvalidArgument("label must be 4-dimensional"));
//
//	OP_REQUIRES(context, bottom_prediction.dims() == 3,
//				errors::InvalidArgument("prediction must be 3-dimensional"));
//
//	// batch size
//	int batch_size = bottom_data.dim_size(0);
//	// height
//	int height = bottom_data.dim_size(1);
//	// width
//	int width = bottom_data.dim_size(2);
//	// number of channels
//	int num_channels = bottom_data.dim_size(3);
//	int num_classes = bottom_label.dim_size(3);
//
//	// Create output tensors
//	// loss
//	int dim = 1;
//	TensorShape output_shape;
//	TensorShapeUtils::MakeShape(&dim, 1, &output_shape);
//
//	// bottom diff
//	TensorShape output_shape_diff = bottom_data.shape();
//
//	TripletKernel(context, &bottom_data, &bottom_label, &bottom_prediction, batch_size, height,
//	  width, num_channels, num_classes, margin_, output_shape, output_shape_diff);
//  }
// private:
//  float margin_;
//};
//
//REGISTER_KERNEL_BUILDER(Name("TripletFlow").Device(DEVICE_GPU).TypeConstraint<float>("T"), TripletOp<Eigen::GpuDevice, float>);


// compute gradient
template <class Device, class T>
class TripletFlowGradOp : public OpKernel {
public:
    explicit TripletFlowGradOp(OpKernelConstruction* context) : OpKernel(context) {
        // Get the margin
        OP_REQUIRES_OK(context,
                       context->GetAttr("margin", &margin_));
        // Check that margin is positive
        OP_REQUIRES(context, margin_ >= 0,
                    errors::InvalidArgument("Need margin >= 0, got ", margin_));
    }

    void Compute(OpKernelContext* context) override
    {
        const Tensor& bottom_left_diff = context->input(0);
        auto bottom_left_diff_flat = bottom_left_diff.flat<T>();

        const Tensor& bottom_right_diff = context->input(1);
        auto bottom_right_diff_flat = bottom_right_diff.flat<T>();

        const Tensor& out_backprop = context->input(2);
        T loss = out_backprop.flat<T>()(0);

        // data should have 4 dimensions.
        OP_REQUIRES(context, bottom_left_diff.dims() == 4,
                    errors::InvalidArgument("bottom diff must be 4-dimensional"));

        // data should have 4 dimensions.
        OP_REQUIRES(context, bottom_right_diff.dims() == 4,
                    errors::InvalidArgument("bottom_right_diff must be 4-dimensional"));

        // batch size
        int batch_size = bottom_left_diff.dim_size(0);
        // height
        int height = bottom_left_diff.dim_size(1);
        // width
        int width = bottom_left_diff.dim_size(2);
        // number of channels
        int num_channels = bottom_left_diff.dim_size(3);

//        std::cout << "batch_size: " << batch_size << " height: " << height << " width: " << width << " num_channels: " << num_channels << std::endl;
//        std::cout << "\tout_backprop dims: " << out_backprop.dims() << std::endl;

        // construct the output shape
        TensorShape output_shape = bottom_left_diff.shape();
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        auto top_data_left = output->template flat<T>();

        OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &output));
        auto top_data_right = output->template flat<T>();

        for (int i = 0; i < batch_size * height * width * num_channels; i++)
        {
            top_data_left(i) = loss * bottom_left_diff_flat(i);
            top_data_right(i) = loss * bottom_right_diff_flat(i);
        }
    }
private:
    float margin_;
};

REGISTER_KERNEL_BUILDER(Name("TripletFlowGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"), TripletFlowGradOp<CPUDevice, float>);
//
//
//bool TripletBackwardLaucher(const float* top_diff, const float* bottom_diff, const int batch_size,
//	const int height, const int width, const int channels, float* output, const Eigen::GpuDevice& d);
//
//static void TripletFlowGradKernel(
//	OpKernelContext* context, const Tensor* bottom_diff, const Tensor* out_backprop,
//	const int batch_size, const int height, const int width, const int channels,
//	const TensorShape& tensor_output_shape)
//{
//  Tensor* output = nullptr;
//  OP_REQUIRES_OK(context, context->allocate_output(0, tensor_output_shape, &output));
//
//  if (!context->status().ok()) {
//	return;
//  }
//
//  TripletBackwardLaucher(
//	out_backprop->flat<float>().data(), bottom_diff->flat<float>().data(),
//	batch_size, height, width, channels, output->flat<float>().data(), context->eigen_device<Eigen::GpuDevice>());
//}


//template <class T>
//class TripletFlowGradOp<Eigen::GpuDevice, T> : public OpKernel {
// public:
//  typedef Eigen::GpuDevice Device;
//
//  explicit TripletFlowGradOp(OpKernelConstruction* context) : OpKernel(context) {
//	// Get the margin
//	OP_REQUIRES_OK(context,
//				   context->GetAttr("margin", &margin_));
//	// Check that margin is positive
//	OP_REQUIRES(context, margin_ >= 0,
//				errors::InvalidArgument("Need margin >= 0, got ", margin_));
//  }
//
//  void Compute(OpKernelContext* context) override
//  {
//	const Tensor& bottom_diff = context->input(0);
//	const Tensor& out_backprop = context->input(1);
//
//	// data should have 4 dimensions.
//	OP_REQUIRES(context, bottom_diff.dims() == 4,
//				errors::InvalidArgument("bottom diff must be 4-dimensional"));
//
//	// batch size
//	int batch_size = bottom_diff.dim_size(0);
//	// height
//	int height = bottom_diff.dim_size(1);
//	// width
//	int width = bottom_diff.dim_size(2);
//	// number of channels
//	int num_channels = bottom_diff.dim_size(3);
//
//	// construct the output shape
//	TensorShape output_shape = bottom_diff.shape();
//
//	// run the kernel
//	TripletFlowGradKernel(
//	  context, &bottom_diff, &out_backprop, batch_size, height, width, num_channels, output_shape);
//  }
// private:
//  float margin_;
//};
//
//REGISTER_KERNEL_BUILDER(Name("TripletFlowGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), TripletFlowGradOp<Eigen::GpuDevice, float>);
